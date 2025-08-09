import math
import time  # Import time module
import types
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack


def qwen_tidal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    top_k: int = None,
    sparse_layer_start: int = 2,
    correction_layer: int = 9,
    attention_sink=4,
    tidal_ratio=0.5,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    """
    LessIsMore forward pass for Qwen3Attention

    prefilling: as full-weight attention
    generation:
    - non-sparse layers: full-weight attention #1
    - sparse_layer_start: full-weight attention + top_k selection #2          topk: (1, 32, 1, topk) => fold (1, 32, topk)
    - sattn_layer_start -> correction layer - 1: use the same top-k #3        Q: (1, 32, 1, 128) K: (1, 32, topk, 128)
    - correction layer: full-weight attention + new top_k selection
    - after correction layer: use the same top-k
    """

    # If output_attentions is True, fall back to original implementation
    if output_attentions:
        return self.original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len = input_shape

    # Project and normalize query, key, value
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Update cache
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    kv_seq_len = (
        past_key_value.get_seq_length(self.layer_idx)
        if past_key_value is not None
        else q_len
    )

    # Check if we should use sparse attention
    if self.layer_idx < sparse_layer_start or q_len == kv_seq_len:
        # Non-sparse layers or prefilling - use original attention interface
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                pass  # Stay with eager
            else:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    else:
        # Generation phase with sparse attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention weights
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        last_dim_size = attn_weights.size(-1)
        token_budget = min(last_dim_size, top_k)

        attn_weights_real = attn_weights.clone()

        # Token selection logic
        if (
            self.layer_idx == sparse_layer_start or self.layer_idx == correction_layer
        ) and top_k >= last_dim_size:
            self.pos_mask = torch.ones_like(attn_weights)
        elif (
            self.layer_idx == sparse_layer_start
            or self.layer_idx == correction_layer
        ):
            middle_budget = int(token_budget * tidal_ratio)
            most_recent_amount = token_budget - middle_budget
            if most_recent_amount < attention_sink:
                attention_sink = 0
            else:
                most_recent_amount -= attention_sink
            assert middle_budget + attention_sink + most_recent_amount == token_budget

            sink_indices = torch.arange(attention_sink, device=attn_weights.device)
            sink_indices = sink_indices.expand(
                attn_weights.shape[:-1] + (attention_sink,)
            )

            recent_start = last_dim_size - most_recent_amount
            middle_scores = attn_weights[..., attention_sink:recent_start]
            _, middle_indices = torch.topk(middle_scores, k=middle_budget, dim=-1)
            middle_indices = middle_indices + attention_sink

            ## Union capped by token_budget ###
            union_tensor = middle_indices.transpose(1, 3).contiguous().view(bsz, -1)
            union_list = list(dict.fromkeys(union_tensor[0].tolist()))
            if len(union_list) > middle_budget:
                union_list = union_list[:middle_budget]
            # (k,) -> (1, 32, 1, k) and replace top_k_indices
            middle_indices = torch.tensor(
                union_list, dtype=middle_indices.dtype, device=middle_indices.device
            )
            middle_indices = middle_indices.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            middle_indices = middle_indices.expand(bsz, key_states.shape[1], q_len, -1)
            ## Union capped by token_budget ###

            recent_indices = torch.arange(
                recent_start, last_dim_size, device=attn_weights.device
            )
            recent_indices = recent_indices.expand(
                attn_weights.shape[:-1] + (most_recent_amount,)
            )

            # combine indices
            top_k_indices = torch.cat(
                [sink_indices, middle_indices, recent_indices], dim=-1
            )
            top_k_mask = torch.zeros_like(attn_weights).scatter_(-1, top_k_indices, 1.0)
            self.pos_mask = top_k_mask  # store top_k mask for position persistence
            self.pos_index = top_k_indices

        else:
            # Apply stored top_k mask at sparse layers
            if not hasattr(self, "pos_mask") or self.pos_mask is None:
                raise ValueError("pos_mask should be set up in sparse attention layers")
            min_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(
                self.pos_mask.to(attn_weights.device) == 0, min_value
            )

        # Apply softmax and compute output
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.num_key_value_groups * self.num_key_value_groups,
            q_len,
            self.head_dim,
        ):
            # Handle GQA case - fix the expected size calculation
            expected_heads = key_states.shape[
                1
            ]  # This accounts for repeat_kv expansion
            if attn_output.size() != (bsz, expected_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, expected_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


def enable_qwen_tidal_attention(
    model,
    top_k,
    attn_type="tidal",
    sparse_layer_start=2,
    correction_layer=9,
    attention_sink=0,
    lim_ratio=1.0,
):
    """
    Enable LessIsMore sparse attention for Qwen3 model

    Args:
        model: The Qwen3 model to modify
        top_k: Number of tokens to keep in sparse attention
        attn_type: Type of attention modification (default: "tidal")
        sparse_layer_start: Layer index to start sparse attention
        correction_layer: Layer index for attention correction
    """

    def wrap_forward(module):
        def new_tidal_forward(
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
            **kwargs: Unpack[FlashAttentionKwargs],
        ):
            return qwen_tidal_attention_forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                output_attentions,
                top_k=top_k,
                sparse_layer_start=sparse_layer_start,
                correction_layer=correction_layer,
                attention_sink=attention_sink,
                tidal_ratio=lim_ratio,
                **kwargs,
            )

        module.original_forward = module.forward
        if attn_type == "lim":
            module.forward = new_tidal_forward

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_qwen_tidal_attention(
                module,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attention_sink,
                lim_ratio,
            )

        # Check if this module is a Qwen3Attention module
        if module.__class__.__name__ == "Qwen3Attention":
            print(f"Applying LessIsMore to layer {module.layer_idx}: {name}")
            print(f"  - top_k: {top_k}")
            print(f"  - sparse_layer_start: {sparse_layer_start}")
            print(f"  - correction_layer: {correction_layer}")

            wrap_forward(module)
