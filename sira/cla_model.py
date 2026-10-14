"""Cross-Layer Attention (CLA) for Llama-style models.

In CLA every `cla_factor` consecutive layers share ONE set of key/value projections: the first layer of each group
("source") computes K and V, the following layers reuse them and only compute their own queries.
    standard / GQA :  K_l = X_l W_K^(l),  V_l = X_l W_V^(l)              for every layer l
    CLA-f          :  K_l = K_{f*floor(l/f)},  V_l = V_{f*floor(l/f)}    (shared inside a group)
Effects: KV-cache size / f, fewer parameters (k_proj, v_proj only on source layers), same queries / MLP.
Reference: Brandon et al., 2024, "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention".

The classes plug into Hugging Face (`AutoModelForCausalLM`, generate, PEFT/LoRA, safetensors) after register().
Tested with transformers 5.x; needs the attention-interface API (>= 4.48)."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.models.llama import modeling_llama as ml

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:                                            # pragma: no cover
    ALL_ATTENTION_FUNCTIONS = ml.ALL_ATTENTION_FUNCTIONS


def _interface(config):
    impl = config._attn_implementation
    getter = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    if getter:
        return getter(impl, ml.eager_attention_forward)
    return ml.eager_attention_forward if impl == "eager" else ALL_ATTENTION_FUNCTIONS[impl]


class ClaConfig(LlamaConfig):
    model_type = "llama_cla"

    def __init__(self, cla_factor: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.cla_factor = int(cla_factor)


def n_source_layers(num_layers: int, factor: int) -> int:
    return -(-num_layers // max(1, factor))


class ClaAttention(ml.LlamaAttention):
    def __init__(self, config: ClaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        f = max(1, int(getattr(config, "cla_factor", 1)))
        self.cla_factor, self.group = f, layer_idx // f
        self.is_source = layer_idx % f == 0
        self.box: dict | None = None                       # shared by all layers of one model (set by ClaForCausalLM)
        if not self.is_source:                              # shared layers own no K/V projections
            del self.k_proj, self.v_proj

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if self.is_source:
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            q, k = ml.apply_rotary_pos_emb(q, k, cos, sin)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx)
            self.box[self.group] = (k, v)                   # full (cached) K/V for the followers of this group
        else:
            q, _ = ml.apply_rotary_pos_emb(q, q, cos, sin)  # only the query is rotated here; K was rotated in the source layer
            k, v = self.box[self.group]
        out, w = _interface(self.config)(self, q, k, v, attention_mask, dropout=0.0 if not self.training else self.attention_dropout,
                                         scaling=self.scaling, **kwargs)
        return self.o_proj(out.reshape(*input_shape, -1).contiguous()), w


class ClaForCausalLM(ml.LlamaForCausalLM):
    config_class = ClaConfig

    def __init__(self, config: ClaConfig):
        super().__init__(config)
        self._box: dict = {}
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = ClaAttention(config, i)
            layer.self_attn.box = self._box
        self.post_init()

    def forward(self, *a, **k):
        self._box.clear()                                   # never keep activations between forwards
        try:
            return super().forward(*a, **k)
        finally:
            self._box.clear()


_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    AutoConfig.register("llama_cla", ClaConfig, exist_ok=True)
    AutoModelForCausalLM.register(ClaConfig, ClaForCausalLM, exist_ok=True)
    _registered = True


def make_config(base: dict, kv_heads: int | None = None, cla_factor: int = 1) -> LlamaConfig:
    """Config for a Llama (cla_factor=1) or CLA model built from a dict of Llama hyper-parameters."""
    base = dict(base)
    if kv_heads:
        base["num_key_value_heads"] = int(kv_heads)
    return ClaConfig(cla_factor=cla_factor, **base) if cla_factor > 1 else LlamaConfig(**base)


def kv_cache_bytes_per_token(num_layers, kv_heads, head_dim, cla_factor=1, bytes_per=2):
    return 2 * n_source_layers(num_layers, cla_factor) * kv_heads * head_dim * bytes_per
