from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class Conv1DModule(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels=None, kernel_size=3):
        super(Conv1DModule, self).__init__()
        output_channels = output_channels if output_channels else input_channels
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channels, output_channels, kernel_size=kernel_size, padding=padding, bias=False)

        self.init_param()
    
    def init_param(self):
        for param in self.conv2.parameters():
            nn.init.zeros_(param)
        for param in self.conv1.parameters():
            nn.init.normal_(param)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels=None):
        super(MLPModule, self).__init__()
        output_channels = output_channels if output_channels else input_channels

        self.linear1 = nn.Linear(input_channels, mid_channels, bias=False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mid_channels, output_channels, bias=False)

        self.init_param()
    
    def init_param(self):
        for param in self.linear1.parameters():
            nn.init.zeros_(param)
        for param in self.linear2.parameters():
            nn.init.normal_(param)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class BaseCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3, module_type="conv1d"):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.module_type = module_type

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)


class QKVConv1dCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3, module_type="conv1d"):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.module_type = module_type

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        if self.module_type == "conv1d":
            hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)    # [BHW, T, C]
            hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)  # [BHW, T, C]
            hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
            hidden_states_qkv = hidden_states_conv1d.flatten(1, 3)   # [B, THW, C]
        elif self.module_type == "mlp":
            hidden_states_mlp = self.temporal_emb(hidden_states)
            hidden_states_qkv = hidden_states_mlp

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query[:, text_seq_length:] = query[:, text_seq_length:] + hidden_states_qkv
        key[:, text_seq_length:] = key[:, text_seq_length:] + hidden_states_qkv
        value[:, text_seq_length:] = value[:, text_seq_length:] + hidden_states_qkv

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states



class KVConv1dCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3, module_type="conv1d"):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.module_type = module_type

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        if self.module_type == "conv1d":
            hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)    # [BHW, T, C]
            hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)  # [BHW, T, C]
            hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
            hidden_states_kv = hidden_states_conv1d.flatten(1, 3)   # []
        elif self.module_type == "mlp":
            hidden_states_mlp = self.temporal_emb(hidden_states)
            hidden_states_kv = hidden_states_mlp

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        key[:, text_seq_length:] = key[:, text_seq_length:] + hidden_states_kv
        value[:, text_seq_length:] = value[:, text_seq_length:] + hidden_states_kv

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class SkipConv1dCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(
        self, 
        height, 
        width, 
        frames, 
        dim, 
        rank=128, 
        kernel_size=3, 
        module_type="conv1d",
        store=None,
        block_index=None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.module_type = module_type

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)
        
        self.store = store
        self.block_index = block_index
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        if self.module_type == "conv1d":
            hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)    # [BHW, T, C]
            hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)  # [BHW, T, C]
            hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
            if self.store is not None:
                self.store[self.block_index] = self.store.get(self.block_index, 0) + hidden_states_conv1d[-1].float().detach().clone().cpu()
            hidden_states_skip = hidden_states_conv1d.flatten(1, 3)
        elif self.module_type == "mlp":
            hidden_states_mlp = self.temporal_emb(hidden_states)
            hidden_states_skip = hidden_states_mlp

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        hidden_states = hidden_states + hidden_states_skip

        return hidden_states, encoder_hidden_states


class AttentionConv1dCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3, module_type="conv1d"):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)
        hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)
        hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
        hidden_states_conv1d = hidden_states_conv1d.flatten(1, 3)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states[:, text_seq_length:] = hidden_states[:, text_seq_length:] + hidden_states_conv1d

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        return hidden_states, encoder_hidden_states


class SkipConv1dLTXVideoAttentionProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3, module_type="conv1d"):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)
        hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)
        hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
        hidden_states_conv1d = hidden_states_conv1d.flatten(1, 3)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states + hidden_states_conv1d


class SkipConv1dHunyuanVideoAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128, kernel_size=3):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim
        self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)
        hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)
        hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
        hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
        hidden_states_conv1d = hidden_states_conv1d.flatten(1, 3)

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        hidden_states = hidden_states + hidden_states_conv1d

        return hidden_states, encoder_hidden_states


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out