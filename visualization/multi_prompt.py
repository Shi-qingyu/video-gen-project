from diffusers.models.attention_processor import Attention
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps, CogVideoXPipeline
from diffusers.utils import export_to_video

from typing import Union, Optional, Tuple, Dict, Any, List
import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiPromptCogVideoXAttnProcessor2_0():
    def __init__(
        self,
        block_index=None,
    ):
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

        uncond_encoder_hidden_states_1, uncond_encoder_hidden_states_2, encoder_hidden_states_1, encoder_hidden_states_2 = encoder_hidden_states.chunk(4)
        if self.block_index < 21:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states_1, uncond_encoder_hidden_states_2, encoder_hidden_states_1, encoder_hidden_states_1], dim=0)
        else:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states_1, uncond_encoder_hidden_states_2, encoder_hidden_states_2, encoder_hidden_states_2], dim=0)

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
        return hidden_states, encoder_hidden_states


pipeline = CogVideoXPipeline.from_pretrained(
    pretrained_model_name_or_path="THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
    revision=None
)

transformer = pipeline.transformer
attn_processors = {}
for key, value in transformer.attn_processors.items():
    block_idx = int(key.split(".")[1])
    if block_idx in list(range(0, 42)):
        attn_processor = MultiPromptCogVideoXAttnProcessor2_0(
            block_index=block_idx
        )
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value
transformer.set_attn_processor(attn_processors)

pipeline = pipeline.to("cuda")
pipeline.vae.enable_tiling()

videos = pipeline(
    prompt=["A tiger is walking on the ground", "A tiger is lying on the ground"],
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42)
).frames

for i, video in enumerate(videos):
    export_to_video(video, f"test{i}_5.mp4", fps=8)