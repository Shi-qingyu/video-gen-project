from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, CogVideoXBlock
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.configuration_utils import register_to_config
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, unscale_lora_layers

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .attention_processor import VisAttnMapCogVideoXAttnProcessor2_0, RegionCogVideoXAttnProcessor2_0


class VisAttnMapCogVideoXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        block_idx: int = None,
    ):
        super().__init__()
        self.block_idx = block_idx

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=VisAttnMapCogVideoXAttnProcessor2_0(block_idx=block_idx),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Dict = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class RegionCogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor, region_prompt_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

        if region_prompt_embs is not None:
            region_prompt_embs = self.norm(region_prompt_embs) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]

        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :], region_prompt_embs


class RegionCogVideoXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = RegionCogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=RegionCogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = RegionCogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        region_prompt_embs: Optional[torch.Tensor] = None,
        region_masks: Optional[torch.Tensor] = None,
        base_ratio: Optional[int] = None,
        height: int = None,
        width: int = None,
        frames: int = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        if region_prompt_embs is not None:
            region_prompt_length = region_prompt_embs.size(1)
        
        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa, norm_region_prompt_embs = self.norm1(
            hidden_states, encoder_hidden_states, temb, region_prompt_embs
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states, attn_region_prompt_embs = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            region_prompt_embs=norm_region_prompt_embs,
            region_masks=region_masks,
            base_ratio=base_ratio,
            height=height,
            width=width,
            num_frames=frames,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        if region_prompt_embs is not None:
            region_prompt_embs = region_prompt_embs + enc_gate_msa * attn_region_prompt_embs

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff, norm_region_prompt_embs = self.norm2(
            hidden_states, encoder_hidden_states, temb, region_prompt_embs
        )

        if region_prompt_embs is not None:
        # feed-forward
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_region_prompt_embs, norm_hidden_states], dim=1)
        else:
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        if region_prompt_embs is not None:
            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length + region_prompt_length:]
            region_prompt_embs = region_prompt_embs + enc_gate_ff * ff_output[:, text_seq_length: text_seq_length + region_prompt_length]
        else:
            hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]

        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states, region_prompt_embs