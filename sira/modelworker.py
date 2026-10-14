"""Model worker: generate / train (LoRA or QLoRA) / merge. Runs as its own process.

    python -m sira.modelworker <generate|train|merge> <args.json>

Only fixed, trusted code runs here (no LLM-written code ever reaches the host). Progress events are
printed as lines starting with 'SIRA_EVT ' so the Training Maestro can relay them to the GUI.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time


def evt(**kw):
    print("SIRA_EVT " + json.dumps(kw), flush=True)


def dtype_of(name):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def has_template(tok):
    return bool(getattr(tok, "chat_template", None))


def format_prompt(tok, prompt):
    """Prompt text, identical in generation and training."""
    if has_template(tok):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return f"### Question\n{prompt}\n\n### Answer\n"


def encode_prompt(tok, text):
    # chat templates already contain BOS/special tokens; the plain format needs them added
    return tok(text, add_special_tokens=not has_template(tok))["input_ids"]


def from_pretrained_dtype(cls, path, dt, **kw):
    """transformers >= 4.56 wants `dtype`, older versions `torch_dtype`."""
    try:
        return cls.from_pretrained(path, dtype=dt, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dt, **kw)


def load_model(path, dtype, device, four_bit=False, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw = {"trust_remote_code": False}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_of(dtype))
        kw["device_map"] = {"": 0}
    else:
        kw["dt"] = dtype_of(dtype)
    dt_ = kw.pop("dt", None)
    model = from_pretrained_dtype(AutoModelForCausalLM, path, dt_, **kw) if dt_ is not None else AutoModelForCausalLM.from_pretrained(path, **kw)
    if not four_bit:
        model.to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model


# ------------------------------------------------------------------------------------------- generate
def cmd_generate(a):
    import torch
    tok = load_tokenizer(a["model_path"])
    tok.padding_side = "left"
    model = load_model(a["model_path"], a["dtype"], a["device"], a.get("four_bit", False), a.get("adapter"))
    model.eval()
    probs = json.load(open(a["prompts_path"]))           # [{"id":..., "prompt":...}]
    n, bs = int(a.get("n_samples", 1)), int(a.get("batch_size", 8))
    temp = float(a.get("temperature", 0.0))
    out = {}
    t0 = time.time()
    for i in range(0, len(probs), bs):
        batch = probs[i:i + bs]
        texts = [format_prompt(tok, p["prompt"]) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not has_template(tok)).to(model.device)
        gen_kw = dict(max_new_tokens=int(a["max_new_tokens"]), pad_token_id=tok.pad_token_id,
                      num_return_sequences=n if temp > 0 else 1)
        if temp > 0:
            gen_kw.update(do_sample=True, temperature=temp, top_p=0.95)
        else:
            gen_kw.update(do_sample=False)
        with torch.no_grad():
            seqs = model.generate(**enc, **gen_kw)
        per = gen_kw["num_return_sequences"]
        plen = enc["input_ids"].shape[1]
        for j, p in enumerate(batch):
            out[p["id"]] = [tok.decode(seqs[j * per + k][plen:], skip_special_tokens=True) for k in range(per)]
        evt(event="progress", phase="generate", done=min(i + bs, len(probs)), total=len(probs), secs=round(time.time() - t0, 1))
    json.dump(out, open(a["out_path"], "w"))


# ------------------------------------------------------------------------------------------- held-out NLL (progress signal before exact match moves)
def cmd_nll(a):
    """Mean per-token negative log-likelihood of the RESPONSES of a jsonl file {prompt,response} under a model."""
    import torch
    tok = load_tokenizer(a["model_path"])
    model = load_model(a["model_path"], a["dtype"], a["device"], False, a.get("adapter"))
    model.eval()
    rows = [json.loads(l) for l in open(a["rows_path"]) if l.strip()]
    ex = build_examples(tok, rows, int(a["max_len"]))
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ex), 16):
            ids, lab, att = collate(ex[i:i + 16], tok.pad_token_id)
            ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
            tot += loss.item()
            n += int((lab[:, 1:] != -100).sum())
    json.dump({"mean_nll": tot / max(1, n), "tokens": n, "examples": len(ex)}, open(a["out_path"], "w"))


# ------------------------------------------------------------------------------------------- train
def build_examples(tok, rows, max_len):
    ex = []
    eos = tok.eos_token_id
    for r in rows:
        p_ids = encode_prompt(tok, format_prompt(tok, r["prompt"]))
        r_ids = tok(r["response"], add_special_tokens=False)["input_ids"] + ([eos] if eos is not None else [])
        if len(p_ids) >= max_len - 8:
            continue
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]
        ex.append((ids, labels))
    return ex


def collate(batch, pad_id):
    import torch
    m = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids, lab, att


def linear_leaf_names(model) -> list:
    """Leaf names of every Linear / Conv1D module inside transformer blocks (excluding the output head), for any architecture."""
    import re as _re
    import torch
    try:
        from transformers.pytorch_utils import Conv1D
        kinds = (torch.nn.Linear, Conv1D)
    except Exception:                                      # pragma: no cover
        kinds = (torch.nn.Linear,)
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, kinds) and _re.search(r"\.(?:layers|h|blocks|block|layer)\.\d+\.", n + "."):
            names.add(n.split(".")[-1])
    if not names:
        raise SystemExit("could not find any Linear/Conv1D modules inside transformer blocks for LoRA")
    return sorted(names)


def prepare_model(a, dropout=0.05):
    """Load the model and make it trainable according to a['method'] (lora | qlora | full) and a['lora_variant'] (lora | rslora | dora)."""
    import re as _re
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    random.seed(int(a.get("seed", 0)))
    torch.manual_seed(int(a.get("seed", 0)))
    tok = load_tokenizer(a["model_path"])
    four_bit = a["method"] == "qlora"
    full = a["method"] == "full"
    model = load_model(a["model_path"], a["dtype"], a["device"], four_bit)
    layers = [int(i) for i in a.get("train_layers") or []]
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(a.get("grad_checkpointing")))
    elif a.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if full:
        for n, p_ in model.named_parameters():
            on = True
            if layers:
                m_ = _re.search(r"\.(?:layers|h|blocks)\.(\d+)\.", n)
                on = bool(m_) and int(m_.group(1)) in layers
            if any(pat in n for pat in a.get("freeze_patterns") or []):
                on = False
            p_.requires_grad = on
        if not any(p_.requires_grad for p_ in model.parameters()):
            raise SystemExit("every parameter is frozen; nothing to train")
    else:
        kw = {}
        if layers:
            kw["layers_to_transform"] = layers
            kw["target_modules"] = linear_leaf_names(model)          # architecture-agnostic: Linear and GPT-style Conv1D leaf names
        else:
            kw["target_modules"] = "all-linear"
        variant = a.get("lora_variant", "lora")
        model = get_peft_model(model, LoraConfig(r=int(a["lora_r"]), lora_alpha=int(a["lora_alpha"]), lora_dropout=dropout,
                                                 task_type="CAUSAL_LM", use_rslora=(variant == "rslora"), use_dora=(variant == "dora"), **kw))
    model.train()
    return model, tok, full


def make_optimizer(model, a, full):
    """AdamW (or bitsandbytes AdamW8bit). LoRA+: the B matrices get lr * lora_plus_ratio."""
    import torch
    lr = float(a["learning_rate"])
    ratio = float(a.get("lora_plus_ratio", 1.0))
    named = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
    if ratio > 1.0 and not full:
        groups = [{"params": [p_ for n, p_ in named if "lora_B" in n], "base_lr": lr * ratio},
                  {"params": [p_ for n, p_ in named if "lora_B" not in n], "base_lr": lr}]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [{"params": [p_ for _, p_ in named], "base_lr": lr}]
    for g in groups:
        g["lr"] = g["base_lr"]
    if a.get("optimizer") == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, weight_decay=0.0)
        except Exception as e:
            evt(event="warning", msg=f"adamw8bit unavailable ({e}); using AdamW")
    return torch.optim.AdamW(groups, weight_decay=0.0)


def cosine_factor(s, steps, warm):
    if s < warm:
        return (s + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))


def cmd_train(a):
    import torch
    model, tok, full = prepare_model(a)
    rows = [json.loads(l) for l in open(a["dataset_path"]) if l.strip()]
    ex = build_examples(tok, rows, int(a["max_len"]))
    if not ex:
        raise SystemExit(f"no usable training examples: all {len(rows)} prompts are longer than max_seq_len={a['max_len']} tokens (raise max_seq_len)")
    mb, ga = int(a["micro_batch"]), int(a["grad_accum"])
    steps = max(1, math.ceil(len(ex) * float(a["epochs"]) / (mb * ga)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model, a, full)
    warm = max(1, int(0.05 * steps))

    def lr_at(s):
        return cosine_factor(s, steps, warm)

    dev = model.device if hasattr(model, "device") else a["device"]
    use_amp = a["device"] == "cuda"
    amp_dtype = dtype_of(a["dtype"]) if use_amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and a["dtype"] == "float16"))
    order, pos = [], 0
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = g.get("base_lr", float(a["learning_rate"])) * lr_at(s)
        tot = 0.0
        for _ in range(ga):
            batch = []
            for _ in range(mb):
                if pos >= len(order):
                    order, pos = list(range(len(ex))), 0
                    random.shuffle(order)
                batch.append(ex[order[pos]])
                pos += 1
            ids, lab, att = collate(batch, tok.pad_token_id)
            ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ga
            else:
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ga
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss; aborting training (lower the learning rate)")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        evt(event="progress", phase="train", step=s + 1, total=steps, loss=round(tot, 4), secs=round(time.time() - t0, 1))
    model.save_pretrained(a["out_dir"], **({"safe_serialization": True} if full else {}))
    if full:
        tok.save_pretrained(a["out_dir"])
    evt(event="trained", examples=len(ex), steps=steps)


# ------------------------------------------------------------------------------------------- policy optimisation (GRPO, DPO)
def _tok_logps(model, ids, att):
    """Per-token log-probabilities of ids[:, 1:] given the prefix. Shape [B, T-1]."""
    import torch
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _encode_pair(tok, prompt, completion, max_len):
    p_ids = encode_prompt(tok, format_prompt(tok, prompt))
    c_ids = tok(completion, add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    if len(p_ids) >= max_len - 8:
        return None
    ids = (p_ids + c_ids)[:max_len]
    mask = ([0] * len(p_ids) + [1] * len(c_ids))[:max_len]
    return ids, mask


def _pad(seqs, pad_id, device):
    import torch
    m = max(len(s_) for s_ in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for k, s_ in enumerate(seqs):
        ids[k, :len(s_)] = torch.tensor(s_)
        att[k, :len(s_)] = 1
    return ids.to(device), att.to(device)


def _pad_mask(masks, device):
    import torch
    m = max(len(x) for x in masks)
    out = torch.zeros((len(masks), m), dtype=torch.float32)
    for k, x in enumerate(masks):
        out[k, :len(x)] = torch.tensor(x, dtype=torch.float32)
    return out[:, 1:].to(device)        # aligned with logps (token t predicted from prefix < t)


def cmd_grpo(a):
    """Group Relative Policy Optimisation with verifiable rewards.
    rows: {"prompt","completion","reward","group"}.  A_i = (r_i - mean_g)/(std_g + 1e-4);
    L = -mean_tokens min(rho*A, clip(rho,1-eps,1+eps)*A) [+ beta*KL(k3) to the policy at the start of the round]."""
    import torch
    model, tok, full = prepare_model(a, dropout=0.0)
    rows = [json.loads(l) for l in open(a["dataset_path"]) if l.strip()]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    ex = []
    for g in groups.values():
        rs = torch.tensor([float(x["reward"]) for x in g])
        if rs.std(unbiased=False) < 1e-6:
            continue                                                    # no learning signal inside this group
        adv = (rs - rs.mean()) / (rs.std(unbiased=False) + 1e-4)
        for x, ad in zip(g, adv.tolist()):
            e = _encode_pair(tok, x["prompt"], x["completion"], int(a["max_len"]))
            if e:
                ex.append((e[0], e[1], ad))
    if not ex:
        raise SystemExit("no GRPO signal: every group has identical rewards")
    dev = model.device if hasattr(model, "device") else a["device"]
    mb, ga = int(a["micro_batch"]), int(a["grad_accum"])
    eps, beta = float(a.get("clip_eps", 0.2)), float(a.get("beta_kl", 0.0))
    # old log-probs from the sampling policy (= the weights we start from)
    old = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ex), mb):
            b = ex[i:i + mb]
            ids, att = _pad([e[0] for e in b], tok.pad_token_id, dev)
            lp = _tok_logps(model, ids, att).cpu()
            old += [lp[k, :len(b[k][0]) - 1] for k in range(len(b))]
    model.train()
    opt = make_optimizer(model, a, full)
    epochs = max(1, int(round(float(a["epochs"]))))
    steps_total = max(1, math.ceil(len(ex) * epochs / (mb * ga)))
    warm = max(1, int(0.05 * steps_total))
    order = [i for _ in range(epochs) for i in random.sample(range(len(ex)), len(ex))]
    t0, pos = time.time(), 0
    for s in range(steps_total):
        for g in opt.param_groups:
            g["lr"] = g.get("base_lr", float(a["learning_rate"])) * cosine_factor(s, steps_total, warm)
        tot, nupd = 0.0, 0
        for _ in range(ga):
            idx = order[pos:pos + mb]
            pos += mb
            if not idx:
                break
            b = [ex[i] for i in idx]
            ids, att = _pad([e[0] for e in b], tok.pad_token_id, dev)
            mask = _pad_mask([e[1] for e in b], dev)
            adv = torch.tensor([e[2] for e in b], device=dev).unsqueeze(1)
            ol = torch.zeros_like(mask)
            for k, i in enumerate(idx):
                ol[k, :old[i].shape[0]] = old[i].to(dev)
            lp = _tok_logps(model, ids, att)
            ratio = torch.exp(lp - ol)
            obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
            per_tok = -obj
            if beta > 0:
                d = ol - lp
                per_tok = per_tok + beta * (torch.exp(d) - d - 1)
            loss = ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean() / ga
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss in GRPO")
            loss.backward()
            tot += loss.item()
            nupd += 1
        if nupd:
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        evt(event="progress", phase="grpo", step=s + 1, total=steps_total, loss=round(tot, 5), secs=round(time.time() - t0, 1))
    _save_trained(model, tok, a, full)
    evt(event="trained", algorithm="grpo", examples=len(ex), steps=steps_total)


def cmd_dpo(a):
    """Direct Preference Optimisation. rows: {"prompt","chosen","rejected"}.
    L = -log sigmoid(beta * [(log pi(y_w) - log pi(y_l)) - (log ref(y_w) - log ref(y_l))]); ref = weights at the start of the round."""
    import torch
    import torch.nn.functional as F
    model, tok, full = prepare_model(a, dropout=0.0)
    rows = [json.loads(l) for l in open(a["dataset_path"]) if l.strip()]
    ex = []
    for r in rows:
        w, l_ = _encode_pair(tok, r["prompt"], r["chosen"], int(a["max_len"])), _encode_pair(tok, r["prompt"], r["rejected"], int(a["max_len"]))
        if w and l_:
            ex.append((w, l_))
    if not ex:
        raise SystemExit("no usable preference pairs")
    dev = model.device if hasattr(model, "device") else a["device"]
    beta, mb, ga = float(a.get("beta", 0.1)), int(a["micro_batch"]), int(a["grad_accum"])

    def seq_lp(items, grad):
        ids, att = _pad([e[0] for e in items], tok.pad_token_id, dev)
        mask = _pad_mask([e[1] for e in items], dev)
        with torch.set_grad_enabled(grad):
            return (_tok_logps(model, ids, att) * mask).sum(1)

    ref = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ex), mb):
            b = ex[i:i + mb]
            rw, rl = seq_lp([x[0] for x in b], False), seq_lp([x[1] for x in b], False)
            ref += list(zip(rw.cpu().tolist(), rl.cpu().tolist()))
    model.train()
    opt = make_optimizer(model, a, full)
    epochs = max(1, int(round(float(a["epochs"]))))
    steps_total = max(1, math.ceil(len(ex) * epochs / (mb * ga)))
    warm = max(1, int(0.05 * steps_total))
    order = [i for _ in range(epochs) for i in random.sample(range(len(ex)), len(ex))]
    t0, pos = time.time(), 0
    for s in range(steps_total):
        for g in opt.param_groups:
            g["lr"] = g.get("base_lr", float(a["learning_rate"])) * cosine_factor(s, steps_total, warm)
        tot, nupd, acc = 0.0, 0, 0.0
        for _ in range(ga):
            idx = order[pos:pos + mb]
            pos += mb
            if not idx:
                break
            b = [ex[i] for i in idx]
            pw, pl = seq_lp([x[0] for x in b], True), seq_lp([x[1] for x in b], True)
            rw = torch.tensor([ref[i][0] for i in idx], device=dev)
            rl = torch.tensor([ref[i][1] for i in idx], device=dev)
            margin = beta * ((pw - pl) - (rw - rl))
            loss = -F.logsigmoid(margin).mean() / ga
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss in DPO")
            loss.backward()
            tot += loss.item()
            acc += (margin > 0).float().mean().item() / ga
            nupd += 1
        if nupd:
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        evt(event="progress", phase="dpo", step=s + 1, total=steps_total, loss=round(tot, 5), pref_acc=round(acc, 3), secs=round(time.time() - t0, 1))
    _save_trained(model, tok, a, full)
    evt(event="trained", algorithm="dpo", examples=len(ex), steps=steps_total)


def _save_trained(model, tok, a, full):
    model.save_pretrained(a["out_dir"], **({"safe_serialization": True} if full else {}))
    if full:
        tok.save_pretrained(a["out_dir"])


# ------------------------------------------------------------------------------------------- merge
def cmd_merge(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = from_pretrained_dtype(AutoModelForCausalLM, a["model_path"], dtype_of(a["dtype"]), trust_remote_code=False)
    model = PeftModel.from_pretrained(model, a["adapter"])
    model = model.merge_and_unload()
    model.save_pretrained(a["out_dir"], safe_serialization=True)
    load_tokenizer(a["model_path"]).save_pretrained(a["out_dir"])
    evt(event="merged", out=a["out_dir"])


# ------------------------------------------------------------------------------------------- safetensors helpers
def st_files(d):
    import glob
    import os
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def st_index(d):
    """name -> file for a model directory."""
    from safetensors import safe_open
    idx = {}
    for f in st_files(d):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                idx[k] = f
    return idx


def st_get(idx, name):
    from safetensors import safe_open
    with safe_open(idx[name], framework="pt") as h:
        return h.get_tensor(name)


def copy_non_weights(src, dst):
    import os
    import shutil
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if not fn.endswith(".safetensors") and os.path.isfile(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))


# ------------------------------------------------------------------------------------------- scan (brain map statistics)
def cmd_scan(a):
    """Per-tensor statistics; optional delta (relative change) against another model directory."""
    import os
    idx = st_index(a["model_path"])
    other = st_index(a["compare_to"]) if a.get("compare_to") else None
    out = {}
    for i, name in enumerate(sorted(idx)):
        t = st_get(idx, name).float()
        row = {"shape": list(t.shape), "numel": t.numel(), "mean": t.mean().item(), "std": t.std().item() if t.numel() > 1 else 0.0,
               "absmax": t.abs().max().item(), "norm": t.norm().item(), "frac_small": (t.abs() < 1e-6).float().mean().item()}
        if other is not None and name in other:
            o = st_get(other, name).float()
            if o.shape == t.shape:
                row["delta_rel"] = ((t - o).norm() / (o.norm() + 1e-12)).item()
        out[name] = row
        if i % 20 == 0:
            evt(event="progress", phase="scan", done=i + 1, total=len(idx))
    json.dump(out, open(a["out_path"], "w"))


# ------------------------------------------------------------------------------------------- transplant
def resize_tensor(t, shape):
    """Heuristic weight-space resize (interpolation) that preserves the tensor's scale. NOT an exact function transfer."""
    import torch
    import torch.nn.functional as F
    t = t.float()
    if tuple(t.shape) == tuple(shape):
        return t
    if t.dim() == 1:
        r = F.interpolate(t[None, None], size=shape[0], mode="linear", align_corners=False)[0, 0]
    elif t.dim() == 2:
        r = F.interpolate(t[None, None], size=tuple(shape), mode="bilinear", align_corners=False)[0, 0]
    else:
        raise ValueError("only 1-D and 2-D tensors can be resized")
    if r.std() > 0 and t.std() > 0:
        r = (r - r.mean()) / r.std() * t.std() + t.mean()
    return r


def cmd_transplant(a):
    """ops: [{"target": name, "donor": name, "mode": copy|blend|resize, "alpha": 0..1}] applied to target model -> out_dir."""
    import os
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    donor = st_index(a["donor_path"])
    ops = {o["target"]: o for o in a["ops"]}
    copy_non_weights(a["target_path"], a["out_dir"])
    done = set()
    report = []
    for f in st_files(a["target_path"]):
        tensors = {}
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                if k in ops:
                    o = ops[k]
                    d = st_get(donor, o["donor"])
                    mode, alpha = o.get("mode", "copy"), float(o.get("alpha", 1.0))
                    if tuple(d.shape) != tuple(t.shape):
                        if mode != "resize":
                            raise SystemExit(f"shape mismatch for {k}: target {tuple(t.shape)} vs donor {tuple(d.shape)}; use mode=resize")
                        d = resize_tensor(d, t.shape)
                    d = d.to(torch.float32)
                    new = d if mode in ("copy", "resize") and alpha >= 1.0 else (1 - alpha) * t.float() + alpha * d
                    tensors[k] = new.to(t.dtype).contiguous()
                    report.append({"target": k, "donor": o["donor"], "mode": mode, "alpha": alpha})
                    done.add(k)
                else:
                    tensors[k] = t
        save_file(tensors, os.path.join(a["out_dir"], os.path.basename(f)), metadata={"format": "pt"})
        evt(event="progress", phase="transplant", file=os.path.basename(f))
    missing = set(ops) - done
    if missing:
        raise SystemExit(f"target tensors not found: {sorted(missing)[:5]}")
    json.dump(report, open(os.path.join(a["out_dir"], "transplant_report.json"), "w"), indent=1)
    evt(event="transplanted", tensors=len(done))


# ------------------------------------------------------------------------------------------- model merging
def _slerp(t, v0, v1, eps=1e-8):
    import torch
    a, b = v0.flatten().float(), v1.flatten().float()
    na, nb = a / (a.norm() + eps), b / (b.norm() + eps)
    dot = torch.clamp((na * nb).sum(), -1 + 1e-7, 1 - 1e-7)
    if dot > 0.9995:
        return ((1 - t) * v0.float() + t * v1.float())
    om = torch.acos(dot)
    r = (torch.sin((1 - t) * om) * a + torch.sin(t * om) * b) / torch.sin(om)
    return r.reshape(v0.shape)


def _ties(deltas, weights, density):
    import torch
    trimmed = []
    for d in deltas:
        k = max(1, int(density * d.numel()))
        thr = d.abs().flatten().kthvalue(max(1, d.numel() - k + 1)).values
        trimmed.append(torch.where(d.abs() >= thr, d, torch.zeros_like(d)))
    stack = torch.stack([w * t for w, t in zip(weights, trimmed)])
    sign = torch.sign(stack.sum(0))
    agree = (torch.sign(stack) == sign).float() * (stack != 0).float()
    return (stack * agree).sum(0) / agree.sum(0).clamp(min=1)


def cmd_mergemodels(a):
    """mode: linear | slerp (2 models) | task (base + sum w_i (m_i - base)) | ties.  models: [{"path","weight"}]"""
    import os
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    mode = a["mode"]
    models = a["models"]
    idxs = [st_index(m["path"]) for m in models]
    base_idx = st_index(a["base"]) if a.get("base") else None
    ws = [float(m.get("weight", 1.0)) for m in models]
    if mode == "linear":
        tot = sum(ws)
        ws = [w / tot for w in ws]
    if mode == "slerp" and len(models) != 2:
        raise SystemExit("slerp needs exactly 2 models")
    if mode in ("task", "ties", "dare") and base_idx is None:
        raise SystemExit(f"{mode} needs a base model")
    copy_non_weights(models[0]["path"], a["out_dir"])
    for f in st_files(models[0]["path"]):
        out = {}
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                t0 = h.get_tensor(k)
                if any(k not in ix for ix in idxs) or (base_idx is not None and k not in base_idx):
                    out[k] = t0
                    continue
                ts = [t0 if i == 0 else st_get(ix, k) for i, ix in enumerate(idxs)]
                if any(tuple(x.shape) != tuple(t0.shape) for x in ts) or not t0.is_floating_point():
                    out[k] = t0
                    continue
                fl = [x.float() for x in ts]
                if mode == "linear":
                    r = sum(w * x for w, x in zip(ws, fl))
                elif mode == "slerp":
                    r = _slerp(ws[1], fl[0], fl[1])
                else:
                    b = st_get(base_idx, k).float()
                    deltas = [x - b for x in fl]
                    if mode == "dare":          # random drop with rescale, then task arithmetic
                        pdrop = float(a.get("drop", 0.5))
                        gen = torch.Generator().manual_seed(int(a.get("seed", 0)))
                        deltas = [d * (torch.rand(d.shape, generator=gen) >= pdrop).float() / (1 - pdrop) for d in deltas]
                    r = b + (sum(w * d for w, d in zip(ws, deltas)) if mode in ("task", "dare") else _ties(deltas, ws, float(a.get("density", 0.5))))
                out[k] = r.to(t0.dtype).contiguous()
        save_file(out, os.path.join(a["out_dir"], os.path.basename(f)), metadata={"format": "pt"})
        evt(event="progress", phase="merge", file=os.path.basename(f))
    evt(event="merged_models", out=a["out_dir"], mode=mode)


# ------------------------------------------------------------------------------------------- from scratch
SIZES = {"tiny": dict(hidden_size=128, intermediate_size=344, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2),
         "small": dict(hidden_size=384, intermediate_size=1024, num_hidden_layers=8, num_attention_heads=6, num_key_value_heads=2),
         "base": dict(hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4)}


def cmd_train_tokenizer(a):
    import os
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast
    files = [f for f in a["corpus_files"] if os.path.exists(f)]
    if not files:
        raise SystemExit("no corpus files")
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=int(a["vocab_size"]), special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train(files, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    os.makedirs(a["out_dir"], exist_ok=True)
    tok.save_pretrained(a["out_dir"])
    evt(event="tokenizer", vocab=len(tok))


def cmd_pretrain(a):
    """Random-initialised Llama-style causal LM, trained on raw text (language only)."""
    import os
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    random.seed(int(a.get("seed", 0)))
    torch.manual_seed(int(a.get("seed", 0)))
    tok = load_tokenizer(a["tokenizer_dir"])
    seq = int(a["seq_len"])
    if a.get("resume_from"):
        from transformers import AutoModelForCausalLM as _AM
        model = _AM.from_pretrained(a["resume_from"])
    else:
        from sira.cla_model import make_config, ClaForCausalLM
        sz = dict(SIZES[a["size"]])
        base = dict(vocab_size=len(tok), max_position_embeddings=max(seq, 512), bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id, tie_word_embeddings=True, rope_theta=10000.0, **sz)
        cfg = make_config(base, kv_heads=int(a.get("kv_heads") or 0) or None, cla_factor=int(a.get("cla_factor", 1)))
        model = ClaForCausalLM(cfg) if int(a.get("cla_factor", 1)) > 1 else LlamaForCausalLM(cfg)
    dev = a["device"]
    dt = dtype_of(a["dtype"])
    model.to(dev)
    if dt != torch.float32:
        model.to(dt)
    ids = []
    total_chars = 0
    for f in a["corpus_files"]:
        if not os.path.exists(f):
            continue
        txt = open(f, encoding="utf-8", errors="ignore").read()
        total_chars += len(txt)
        ids.extend(tok(txt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        if total_chars > 400_000_000:
            break
    if len(ids) < seq * 4:
        raise SystemExit(f"corpus too small: {len(ids)} tokens (need at least {seq * 4})")
    data = torch.tensor(ids, dtype=torch.long)
    nblocks = (len(data) - 1) // seq
    steps, bs, lr = int(a["steps"]), int(a["batch_size"]), float(a["learning_rate"])
    evt(event="info", tokens=len(ids), params=sum(p.numel() for p in model.parameters()), blocks=nblocks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, int(0.05 * steps))
    model.train()
    t0 = time.time()
    for s in range(steps):
        f_ = (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * f_
        starts = torch.randint(0, nblocks, (bs,)) * seq
        x = torch.stack([data[i:i + seq] for i in starts]).to(dev)
        loss = model(input_ids=x, labels=x).loss
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss in pretraining")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == steps - 1:
            evt(event="progress", phase="pretrain", step=s + 1, total=steps, loss=round(loss.item(), 4),
                ppl=round(math.exp(min(20, loss.item())), 2), secs=round(time.time() - t0, 1))
    model.save_pretrained(a["out_dir"], safe_serialization=True)
    tok.save_pretrained(a["out_dir"])
    evt(event="pretrained", out=a["out_dir"])


def cmd_convert_arch(a):
    """Weight surgery to another attention architecture (Llama family): fewer KV heads (GQA/MQA by mean-pooling head groups, as in
    the GQA up-training recipe) and/or Cross-Layer Attention (a group of `cla_factor` layers shares the K/V of its first layer;
    K/V weights initialised as the mean over the group, or the source's own with cla_init='first'). A short healing run should follow."""
    import os
    import torch
    from safetensors.torch import save_file
    src = a["model_path"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    mt = cfg.get("model_type")
    GQA_FAMILIES = {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "olmo", "olmo2", "stablelm", "starcoder2", "cohere", "granite", "llama_cla"}
    if mt not in GQA_FAMILIES:
        raise SystemExit(f"'{mt}' does not support grouped-query attention in its configuration; GQA/MQA/CLA conversion is available for: {sorted(GQA_FAMILIES)}")
    idx = st_index(src)
    if not any(k.endswith("self_attn.k_proj.weight") for k in idx):
        raise SystemExit(f"'{mt}' has no separate self_attn.k_proj / v_proj weights (fused or non-standard attention); "
                         f"head pooling / CLA conversion is not supported for it")
    if int(a.get("cla_factor", 1)) > 1 and mt not in ("llama", "mistral", "llama_cla"):
        raise SystemExit(f"Cross-Layer Attention conversion needs the Llama attention implementation (llama, mistral); '{mt}' supports only GQA/MQA conversion")
    L, H, d = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["hidden_size"]
    hd = cfg.get("head_dim") or d // H
    kv = cfg.get("num_key_value_heads", H)
    new_kv = int(a.get("kv_heads") or kv)
    f_old, f_new = int(cfg.get("cla_factor", 1)), int(a.get("cla_factor", 1))
    if f_old != 1 and f_new != f_old:
        raise SystemExit("model is already cross-layer shared; convert from a standard model instead")
    if kv % new_kv or new_kv > kv:
        raise SystemExit(f"kv_heads must divide the current {kv} KV heads (got {new_kv})")
    pool = kv // new_kv
    import re as _re
    kvre = _re.compile(r"(.*\.layers\.)(\d+)(\.self_attn\.)([kv])_proj\.(weight|bias)$")

    def pooled(name):
        w = st_get(idx, name).float()
        if pool == 1:
            return w
        return w.reshape(new_kv, pool, hd, *w.shape[1:]).mean(1).reshape(new_kv * hd, *w.shape[1:])

    out = {}
    for name in idx:
        m = kvre.match(name)
        if not m:
            out[name] = st_get(idx, name)
            continue
        li = int(m.group(2))
        if f_new > 1 and li % f_new != 0:
            continue                                            # shared layer: no K/V of its own any more
        if f_new > 1 and a.get("cla_init", "mean") == "mean":
            members = [i for i in range(li, min(L, li + f_new)) if f"{m.group(1)}{i}{m.group(3)}{m.group(4)}_proj.{m.group(5)}" in idx]
            w = sum(pooled(f"{m.group(1)}{i}{m.group(3)}{m.group(4)}_proj.{m.group(5)}") for i in members) / len(members)
        else:
            w = pooled(name)
        out[name] = w.to(st_get(idx, name).dtype)
    os.makedirs(a["out_dir"], exist_ok=True)
    copy_non_weights(src, a["out_dir"])
    shards, cur, size, n = [], {}, 0, 0
    for k, v in out.items():
        nb = v.numel() * v.element_size()
        if cur and size + nb > 4e9:
            shards.append(cur); cur, size = {}, 0
        cur[k] = v.contiguous(); size += nb
    shards.append(cur)
    wm, total = {}, 0
    for i, sh in enumerate(shards):
        fn = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(a["out_dir"], fn), metadata={"format": "pt"})
        for k, v in sh.items():
            wm[k] = fn; total += v.numel() * v.element_size()
    if len(shards) > 1:
        json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(os.path.join(a["out_dir"], "model.safetensors.index.json"), "w"))
    for fn in os.listdir(a["out_dir"]):
        if fn.endswith(".safetensors") and fn not in set(wm.values()):
            os.remove(os.path.join(a["out_dir"], fn))                # stale shard copied from the source layout
    if os.path.exists(os.path.join(a["out_dir"], "model.safetensors.index.json")) and len(shards) == 1:
        os.remove(os.path.join(a["out_dir"], "model.safetensors.index.json"))
    cfg["num_key_value_heads"] = new_kv
    if f_new > 1:
        cfg.update(model_type="llama_cla", architectures=["ClaForCausalLM"], cla_factor=f_new)
        cfg.pop("sliding_window", None)
    # GQA/MQA only: the family, class and every other setting stay exactly as they were
    json.dump(cfg, open(os.path.join(a["out_dir"], "config.json"), "w"), indent=1)
    evt(event="converted", kv_heads=new_kv, cla_factor=f_new, tensors=len(out))


def ensure_registered():
    try:
        from sira import cla_model
        cla_model.register()
    except Exception as e:                                  # transformers missing: commands that need it will fail with their own message
        print("SIRA_EVT " + json.dumps({"event": "warning", "msg": f"CLA registration skipped: {e}"}), flush=True)


def main(argv):
    cmds = {"generate": cmd_generate, "train": cmd_train, "grpo": cmd_grpo, "dpo": cmd_dpo, "merge": cmd_merge, "scan": cmd_scan, "transplant": cmd_transplant,
            "mergemodels": cmd_mergemodels, "convert_arch": cmd_convert_arch, "nll": cmd_nll, "train_tokenizer": cmd_train_tokenizer, "pretrain": cmd_pretrain}
    if len(argv) != 3 or argv[1] not in cmds:
        print(__doc__)
        return 2
    a = json.load(open(argv[2]))
    ensure_registered()
    cmds[argv[1]](a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
