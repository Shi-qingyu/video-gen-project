from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class Conv1DModule(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels=None):
        super(Conv1DModule, self).__init__()
        output_channels = output_channels if output_channels else input_channels

        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channels, output_channels, kernel_size=3, padding=1, bias=False)

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


class KVConv1dCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, height, width, frames, dim, rank=128):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim
        self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank)
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

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

        video_key = key[:, text_seq_length:]
        video_value = value[:, text_seq_length:]

        video_key = video_key.reshape(-1, self.frames, self.height, self.width, self.dim)
        video_value = video_value.reshape(-1, self.frames, self.height, self.width, self.dim)

        video_key = video_key.permute(0, 2, 3, 4, 1).flatten(0, 2)
        video_value = video_value.permute(0, 2, 3, 4, 1).flatten(0, 2)

        video_key = self.temporal_emb(video_key).reshape(-1, self.height, self.width, self.dim, self.frames).permute(0, 4, 1, 2, 3).flatten(1, 3)
        video_value = self.temporal_emb(video_value).reshape(-1, self.height, self.width, self.dim, self.frames).permute(0, 4, 1, 2, 3).flatten(1, 3)

        key[:, text_seq_length:] = key[:, text_seq_length:] + video_key
        value[:, text_seq_length:] = value[:, text_seq_length:] + video_value

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
    def __init__(self, height, width, frames, dim, rank=128):
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim
        self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank)
    
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

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        hidden_states = hidden_states + hidden_states_conv1d

        return hidden_states, encoder_hidden_states