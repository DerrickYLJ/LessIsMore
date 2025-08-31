import math
import time  # Import time module
import types
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def llama_tidal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    top_k=None,
    sparse_layer_start=2,
    correction_layer=13,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    output_attentions = kwargs.get("output_attentions", False)

    # If output_attentions is requested, fall back to original implementation
    if output_attentions:
        return self.original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_seq_len = key_states.shape[-2]

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if self.layer_idx < sparse_layer_start or q_len == kv_seq_len:
        # non-sparse layers or prefilling
        # Use SDPA for non-sparse layers
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    else:
        # generation with sparse attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        last_dim_size = attn_weights.size(-1)
        token_budget = min(last_dim_size, top_k)

        # decoding
        if self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer:
            # extract top_k mask
            _, top_k_indices = torch.topk(attn_weights, k=token_budget, dim=-1)
            top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
            self.pos_mask = top_k_mask
            self.pos_index = top_k_indices
        else:
            # apply top_k mask
            if not hasattr(self, "pos_mask") or self.pos_mask is None:
                raise ValueError("pos dict should be set up in sparse attn layers")
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(
                self.pos_mask.to(attn_weights.device) == 0, min_value
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, None


def enable_llama_tidal_attention(
    model, top_k, attn_type="tidal", sparse_layer_start=2, correction_layer=13
):
    def wrap_forward(module):
        # Store the original forward method
        module.original_forward = module.forward

        def new_tidal_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ):
            return llama_tidal_attention_forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                top_k=top_k,
                sparse_layer_start=sparse_layer_start,
                correction_layer=correction_layer,
                **kwargs,
            )

        if attn_type == "tidal":
            module.forward = new_tidal_forward

    def apply_to_attention_layers(module):
        for name, child in module.named_children():
            if (
                hasattr(child, "__class__")
                and "LlamaAttention" in child.__class__.__name__
            ):
                wrap_forward(child)
            else:
                apply_to_attention_layers(child)

    apply_to_attention_layers(model)
