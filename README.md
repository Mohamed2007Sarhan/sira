# SIRA v2 - Self-Improving Research Agent (pure Python)

Repository: https://github.com/Mohamed2007Sarhan/sira

One Python program that lets an AI system **improve itself**:

* a hierarchy of agents (Grand Maestro -> Research / Training / Safety maestros -> specialised agents) that search the open web for *leads* and **test everything by running real experiments** in a sandbox;
* a "black box" of simulated worlds (physics, chemistry, ecology, epidemiology, evolution, ...) plus an empty world (`terra_incognita`) whose laws the agents write themselves, and a "reality lab" that compares simulations with real data;
* a training engine that improves the *weights* of a model (from Hugging Face, or **created from scratch with one button**) using only answers that an independent verifier confirmed;
* a GUI with full visibility and control (pause, stop, budgets, internet policy, brain map, backups, tests, statistics);
* it **adapts itself at start-up**: probes the machine, installs the training stack, tunes batch sizes and parallelism.

The run can end in exactly three ways: **goal reached**, **budget exhausted**, **you press STOP**.

---------------------------------------------------------------------------------------------------

## What is new in this version (2.1)

* **Tooling:** `--doctor`, `--check-config`, `--report`, `--list-runs`, `--serve` (see section 11); `eval()`-free math verification; CI + LICENSE.
* **Architecture choice (Cross-Layer Attention, GQA, MQA, any custom shape).** Born from scratch with the architecture you pick (`scratch_arch`: `gqa`, `llama`, `mha`, `mqa`, `cla2`, `cla3`, `custom`), or convert a ready model (Llama family) from the **Brain map** tab (*Convert architecture...*): weight surgery + short healing run, kept only if its score is within `arch_tolerance` of the parent. CLA = every `f` consecutive layers share one set of K/V projections, so the KV cache shrinks `f` times (see `sira/cla_model.py`; verified: cached decoding equals the full forward pass to 3e-7).
* **Thinking in every layer** of the architecture from the original design book: Evaluation Core, AI Researcher, Sandbox, Verifier, Failure Analysis, Experience Memory, Improvement Engine and an automated Red Team. Each layer deliberates (analysis, risks, alternatives, decision, confidence) before acting; see the **Thinking** tab and `thinking: off|low|medium|high`. Thoughts are advice - the mechanical checks (evaluator, verifier, sandbox) are never overridden.
* **Environment bootstrap** (installs the training stack, adapts to the machine), **Ollama + every OpenAI-compatible provider**, **GRPO / DPO / DoRA / rsLoRA / LoRA+ / DARE**, a paired statistical test for accepting generations.
* `experiments/` with the real measurements behind the research report (including negative results).

## Hardening in this release (answers to the defect list)

* **Any architecture.** Verified with real runs (intake, generation, LoRA, full fine-tuning, merge, scan, brain map, merging) on 14 families: GPT-2, GPT-J, GPT-NeoX, OPT, Bloom, Falcon, Phi-3, Llama, Mistral, Qwen2, Gemma, OLMo, StableLM, StarCoder2. GQA/MQA conversion works for the 7 families that support grouped-query attention; CLA conversion for Llama/Mistral; others are refused with a clear message.
* **No forgetting.** Retention guard (a child is rejected if a capability the parent had drops by more than `retention_tol`), replay buffer (`replay_fraction`), adaptive learning rate (halved after a rejection), and a signal audit in every training round. A **progress lane** accepts a child whose held-out NLL improves while exact match is not worse (cold start).
* **Harder verifiers.** Mandatory independent holdout, consistency checks, an 18-item junk battery, duplicate detection, answer-size caps, and a separate `open` class: open problems are never reported as solved. Escape and cheating attempts are tested (`tests/test_security.py`).
* **Memory quarantine.** LLM-written lessons start `pending` and are adopted only after a mechanical retest that fixes the error and breaks nothing; every memory item records source and confidence.
* **Blind, paraphrased society tests** with an ambiguous episode; mental-state score reported separately from behaviour score.
* **Real-model validation harness:** `python -m sira --validate-real --config my.json` records every brain input/output (`llm_calls.jsonl`), measures JSON-protocol compliance per role, runs a mini quest and (with a `base_model`) a before/after test. **It has not been run against a real Ollama/API model in the development environment - run it on your machine.**
* **Packaging:** `pip install .` (pyproject), `python -m sira --verify-install`, signal-safe headless shutdown, resume. Fault-injection tests: `tests/test_faults.py`.

## Autonomous quest mode: give a goal, nothing else

Set `mode` to `auto` (or `quest`) and write only the goal, e.g. *"reach AGI"* or *"reach artificial consciousness"* (`examples/quest_agi.json`). No headings are needed. The platform then runs this loop by itself:

1. **Plan** - turns the goal into an operational definition, a capability map and testable criteria, and states what no test can verify (for consciousness: phenomenal experience).
2. **Self-made hard challenges** - it writes its own problems (reference / optimisation / discovery, up to open-ended ones) with a *hidden graded verifier*. The verifier is sanity-checked before use (probes must order correctly, junk must score <= 0.2, deterministic, not gameable) and runs in a separate sandbox call, so the solver can never read it.
3. **Try, get closer, get rewarded or penalised** - reward = progress (`10 x improvement of the best score`), +5 first solve, +1 novelty; penalties for invalid output, *gaming* (visible score far above an independent holdout), *overconfidence*, *repeating a failed approach*, parking. Everything is in the **Challenges & Rewards** tab.
4. **Stuck -> read -> retry** - after two attempts without progress the agent searches the open web for approaches (leads, unverified) and tries again; frontier problems are revisited as skills grow. Difficulty follows a zone-of-proximal-development ladder per domain.
5. **Exam** - when it believes it understands, it is examined. Ground truth is mechanical (experiment database, stored metrics, re-running reality with a new seed, symbolic checks, citation validity), never a model's opinion, so the examiner may be a *smaller* model (`examiner_provider/examiner_model`, weight capped by `examiner_weight`). A blank student (no notes) takes the same exam: the difference is measured learning. Wrong answers get an **autopsy** (why wrong, corrected understanding, what to study next) that feeds memory, new topics and training data.
6. **Life** - a deterministic virtual society of humans (false-belief, hidden need, promises, honesty, deception) scores behaviour against ground truth; you can join it from the **Society** tab.
7. **Position papers** are written every round (`reports/position_NNN.md`) with evidence per criterion.

`GOAL_REACHED` in quest mode means *the system passed this operational examination* (exam streak, criteria coverage, society life score, challenge skill). It is **not** a proof of AGI or consciousness. With a model (`mode: both|auto` plus `base_model` or `scratch_mode`) the platform also trains the weights on verified self-generated data, exam corrections and challenge rollouts (graded rewards for GRPO).

## 1. Requirements

| Item | Needed for | Notes |
|---|---|---|
| Python 3.10+ with `tkinter` | everything / GUI | Debian/Ubuntu: `sudo apt install python3-tk` |
| `bubblewrap` **or** Docker | sandbox (experiment code is untrusted) | Linux: `sudo apt install bubblewrap`. Docker: `python -m sira --build-sandbox` |
| NVIDIA GPU (recommended) | training real models | CPU works for tiny models only |
| An LLM for the agents ("brain") | agent reasoning | Ollama (local, free), any OpenAI-compatible server, Anthropic, or `mock` for an offline demo |

Everything else (PyTorch, transformers, PEFT, accelerate, safetensors, tokenizers, bitsandbytes, ...) is **installed automatically at the first start** (`auto_install`, on by default). To do it by hand: `pip install -r requirements.txt` or run `python -m sira --bootstrap`.

macOS/Windows: the GUI and agents work, but the strong sandbox needs Linux (bubblewrap) or Docker. Without either, SIRA refuses to run experiments unless you explicitly allow the weak backend (`allow_weak_sandbox`).

## 2. Quick start

```bash
cd sira_project
python -m sira --bootstrap          # probe the machine, install missing packages, list usable algorithms
python -m sira --selftest           # proves the sandbox really isolates experiment code (must print passed=True)
python -m sira --test-llm --config examples/ollama.json     # checks the brain connection
python -m sira                      # opens the GUI
```

First try without any API key: in the GUI pick provider **mock**, base model `Qwen/Qwen2.5-0.5B-Instruct` (or any local model folder), press **START RUN**. The mock brain is an offline stand-in so you can watch the whole cycle.

Headless (no GUI): `python -m sira --headless --config my.json` (Ctrl+C = human stop).
Resume a run (or a restored backup): `python -m sira --resume runs/run_YYYYMMDD_HHMMSS`.

### Two start buttons (Setup tab)

* **START RUN (existing model)** - takes a Hugging Face id (`org/name`) or a local folder, analyses architecture / size / license / hardware, chooses LoRA / QLoRA, measures a baseline, then improves it.
* **CREATE MODEL FROM SCRATCH + START** - corpus (your `.txt` folder, or random Wikipedia articles in the chosen language) -> tokenizer -> pre-training (language only) -> a first lesson from verified solvers -> self-improvement with full-weight training. Every stage resumes where it stopped.

## 3. The agents' brain: Ollama, OpenAI-compatible servers, Anthropic

Set `llm_provider`, `llm_model` (and, if needed, `llm_base_url`, `llm_api_key_env`) in the Setup tab or the JSON config. Use **Test brain connection** (GUI) or `python -m sira --test-llm --config file.json` to verify.

| `llm_provider` | Default endpoint | API key variable | Cost counted |
|---|---|---|---|
| `ollama` | `http://localhost:11434` (native `/api/chat`; honours `OLLAMA_HOST`) | none | free |
| `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | yes |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | yes |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | yes |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | yes |
| `together` | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | yes |
| `mistral` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` | yes |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | yes |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` | yes |
| `xai` | `https://api.x.ai/v1` | `XAI_API_KEY` | yes |
| `lmstudio` | `http://localhost:1234/v1` | none | free |
| `vllm` | `http://localhost:8000/v1` | none | free |
| `llamacpp` | `http://localhost:8080/v1` | none | free |
| `openai_compatible` | **you set `llm_base_url`** | `OPENAI_API_KEY` or `llm_api_key_env` | yes (free if the URL is localhost) |
| `mock` | offline demo brain | none | free |

Hosted providers: set the key in your shell before starting, e.g. `export GROQ_API_KEY=...`. Prices for the budget counter are `price_in_per_mtok` / `price_out_per_mtok` (USD per million tokens).

### Ollama step by step

```bash
curl -fsSL https://ollama.com/install.sh | sh        # or install from ollama.com
ollama serve &                                       # skip if it already runs as a service
ollama pull llama3.1:8b                              # any chat model; bigger = better agents
python -m sira --test-llm --config examples/ollama.json
python -m sira --config examples/ollama.json
```

* In the GUI: provider `ollama`, model e.g. `llama3.1:8b`; buttons **Ollama: list models** and **Ollama: pull model** manage the server.
* Remote Ollama: set `llm_base_url` to `http://HOST:11434` (or export `OLLAMA_HOST`).
* **`llm_num_ctx`** (default 16384) is sent to Ollama on every call. Ollama's own default context (2-4k tokens) silently truncates agent prompts, so keep this high enough for your GPU memory.
* `llm_json_mode: true` makes Ollama return strictly JSON (`format: json`); the agents' protocol is JSON, so this helps small models.
* Agents need a model that follows JSON instructions reliably. 7-8B instruct models work for simple topics; 30B+ gives much better research. The brain is **separate** from the model you are improving (the target model).

### Any other OpenAI-compatible server

```json
{ "llm_provider": "openai_compatible", "llm_model": "my-model",
  "llm_base_url": "https://my-server.example.com/v1", "llm_api_key_env": "MY_SERVER_API_KEY" }
```
Works with vLLM, llama.cpp server, LM Studio, text-generation-inference, LiteLLM, Azure/OpenAI-style gateways that expose `/chat/completions`. Set `llm_json_mode: true` only if the server supports `response_format: json_object`.

## 4. Start-up self-adaptation (Environment tab)

At every start SIRA:

1. probes OS, CPU cores, RAM, free disk, NVIDIA GPUs (`nvidia-smi`), CUDA driver version, installed packages;
2. installs what the training stack needs - the right PyTorch wheel for the machine (CPU wheel, or `cu118/cu121/cu124/cu126/cu128` matching the driver), transformers, PEFT, accelerate, safetensors, tokenizers, bitsandbytes (Linux + CUDA), ... Only package names from SIRA's own catalog are ever installed;
3. checks the science libraries **inside the sandbox** (numpy, scipy, sympy) and installs missing ones into the agents' persistent `/station/pylib`;
4. adapts settings to the machine: parallel experiments from CPU count, sandbox memory from RAM, quotas from free disk, micro-batch / gradient accumulation / sequence length from VRAM;
5. shows which learning algorithms are usable (table in the Environment tab).

Turn off with `auto_install: false` / `auto_adapt: false`. On Debian/Ubuntu system Python (PEP 668) the installer retries with `--user --break-system-packages`; a virtual environment is cleaner: `python -m venv .venv && . .venv/bin/activate`.

## 5. Learning algorithms included

| Name | Kind | Needs | Idea |
|---|---|---|---|
| `lora`, `rslora`, `dora`, `lora_plus` | adaptation | peft | low-rank adapters; rank-stabilised scale; weight-decomposed; separate LR for B |
| `qlora` | adaptation | peft + bitsandbytes + CUDA | LoRA on a 4-bit NF4 base model |
| `sft_full` | adaptation | torch | all weights (models born inside SIRA) |
| `adamw8bit` | optimizer | bitsandbytes + CUDA | 8-bit optimizer states |
| `rft` | self-improvement | - | learn only from answers the verifier confirmed |
| `grpo` | self-improvement | - | group-relative policy optimisation with verifiable rewards |
| `dpo` | self-improvement | - | preference learning: verified-correct vs wrong samples |
| `soup`, `slerp`, `task`, `ties`, `dare` | merging | safetensors | weight interpolation / task arithmetic / TIES / DARE |
| `transplant` | surgery | safetensors | copy / blend / resize chosen layers from a donor model |

`algorithm: auto` uses RFT while the model is weak (and adds *bootstrap* lessons from the evaluator's own solvers on **training** problems if it solves almost nothing), then alternates RFT and GRPO once rewards vary inside groups. A new generation is accepted only if it beats its parent on the **same fresh hidden problems** by `min_gain` **and** wins a paired exact test (McNemar, `accept_p`); otherwise everything is rolled back.

## 5b. Choosing the architecture (the CLA option)

| Where | How |
|---|---|
| From scratch | Setup tab -> *From scratch: architecture* = `cla2` / `cla3` / `mqa` / `mha` / `gqa` / `custom` (`arch_kv_heads`, `arch_cla_factor`) -> **CREATE MODEL FROM SCRATCH + START** |
| Existing model | Brain map tab -> *Convert architecture (CLA / GQA / MQA)...* -> preset or custom, init `mean`/`first`, heal yes/no |
| Keep the architecture | just **START RUN**; evolution (RFT / GRPO / DPO + merging) works on any of them, including CLA models |

Conversion supports Llama-family checkpoints (`llama`, `mistral`, already-converted `llama_cla`). Other families (e.g. Qwen with attention biases) are refused with a clear message.

## 6. GUI tour (21 tabs)

Quest (rounds, exams, learning delta, position papers) - Challenges & Rewards - Society - Thinking (every layer's deliberations) - Setup (all settings, brain test, Ollama tools) - Dashboard (live budget bars you can edit, score curve) - Agents (live hierarchy) - Experiments - Knowledge (verified / refuted / lead / inconclusive, Veto) - Live log - Sandbox (isolation self-test) - Model intake - **Brain map** (click a layer header to select it, click a cell to inspect a tensor; colour by norm/std/delta; *Train ONLY selected*, *Freeze selected*, *Transplant from donor*, *Merge with model*) - **Worlds & Tools** (approve/disable what agents create, check a world's laws) - **Computer** (the agents' persistent machine; you can run commands too) - **Tests** (every problem with prompt, answer, verdict) - **Concepts & Reports** - **Backups** (backup now, verify, restore into a new run, roll back a generation) - **Network & Policy** (web open/allowlist/blocklist live, sandbox network, approvals) - Statistics - Environment.

Top bar: **Pause/Resume**, **STOP**, and a box to send a directive to the Grand Maestro.

## 7. Safety model (what the agents can and cannot do)

* All agent-written code runs only in the sandbox (bubblewrap/Docker): no host files, no host environment, no network by default, memory/CPU/time/disk/output caps. `--selftest` proves it by trying to escape.
* The agents **cannot modify** the evaluator, verifier, safety, control or sandbox code (hash-checked every cycle; any change pauses the run and alerts you). They *can* create new tools, worlds, libraries and notes inside the sandbox; set `approve_new_tools` to review each one.
* Web content is treated as untrusted data (never instructions). Private/loopback/metadata addresses are always blocked for the web tools; everything else follows your Network policy.
* Nothing becomes "knowledge" without: a sandbox experiment + machine-checked conditions + replication with other seeds + an independent audit.

## 8. Backups

Snapshots (database, config, station, reports, best weights via hard links, sha256 manifest) are taken every `backup_every_min`, after each accepted generation, and at the end. Restore always goes into a **new** run folder: `Backups` tab, or `BackupManager.restore_to_new_run(...)`, then `python -m sira --resume <folder>`.

## 9. Main configuration keys

`base_model`, `goal_text`, `target_score`, `confirm_rounds`, `budget_usd`, `budget_gpu_hours`, `budget_wall_hours`, `llm_*`, `algorithm`, `lora_variant`, `lora_plus_ratio`, `optimizer`, `accept_p`, `min_gain`, `sandbox_backend` (`auto|docker|bwrap|subprocess`), `sandbox_network` (`none|open`), `web_mode` (`open|allowlist|blocklist`), `approve_new_tools`, `require_approval`, `scratch_*`, `train_layers`, `freeze_patterns`, `backup_*`, `auto_install`, `auto_adapt`. All defaults and comments are in `sira/config.py`.

## 10. Honest limits

* No code can guarantee "stronger than humans". SIRA gives *measurable* progress on verifiable domains (math, logic, physics, chemistry, code). The ceiling is set by model size, GPU and time.
* Simulators are approximations; important claims should be compared with real data (`worldlab.compare_to_data`).
* Tested here: sandbox (bubblewrap), GUI (virtual display), training / merging / transplant / pre-training on tiny local models on CPU, every stop condition, backups, providers against fake servers, real package auto-install. **Not tested here** (offline environment): live LLM APIs and a real Ollama server, live web search, Docker, GPU/QLoRA, downloading real Hugging Face models. Run `--selftest` and `--test-llm` first on your machine.

Tests: `python -m unittest discover -s tests -t . -v` (or `make test`).

## 11. Tooling (new in 2.1)

| Command | What it does |
|---|---|
| `python -m sira --doctor [--fast] [--json] [--config f.json]` | one-shot health check: manifest integrity, Python, tkinter, training stack, sandbox backend **and isolation self-test**, GPU, disk, config, brain connectivity. Exit code 1 on any FAIL |
| `python -m sira --check-config --config f.json` | typos (with "did you mean"), wrong types, out-of-range values, missing API key variable, inconsistent mode/goal settings |
| `python -m sira --report runs/run_X [--format html\|md\|json] [--out f]` | summary of a run: generations + score chart, claims by status, experiments, evals, rewards, challenges, exams |
| `python -m sira --list-runs [runs]` | table of run folders with state, best score, generations |
| `python -m sira --serve runs [--port 8765] [--host 127.0.0.1] [--token T]` | read-only dashboard for headless servers. Loopback by default; any other `--host` **requires** `--token`. Strict CSP, no JavaScript |

Development: `make check` (compile + integrity + tests), `make manifest` after editing code (CI fails if the manifest is stale).
