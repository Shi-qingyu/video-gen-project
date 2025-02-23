from typing import Optional
import gc

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention

from src.new.attention_processor import Conv1DModule, MLPModule


class VisAttnMapCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    def __init__(self, block_idx=None):
        super().__init__()
        self.layer_idx = block_idx
        self.attention_store = None
        

    def get_attention_scores(
        self, 
        attn: Attention, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs


    def store_attention_map(
        self,
        attention_map,
        num_frames,
        height,
        width,
        store_text_cross_attention,
    ):
        if store_text_cross_attention:
            attention_map = attention_map.reshape(
                num_frames, height, width, -1 
            )
            _attention_map = attention_map.detach().clone().cpu()
            self.attention_store.store(self.layer_idx, _attention_map)
        else:
            attention_map = attention_map.reshape(
                height, width, -1
            )
            _attention_map = attention_map.detach().clone().cpu()
            self.attention_store.store(self.layer_idx, _attention_map)
        

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        save_text_attention: Optional[bool] = False,
        frame_idx_as_query: Optional[int] = None,
        word_ids: list = None,
        height: int = None,
        width: int = None,
        num_frames: int = None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

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
        
        hidden_state_blocks = []
        block_size = height * width
        block_num = int((query.shape[2] - text_seq_length) / block_size + 1)
        block_ids = [(text_seq_length + i * block_size) for i in range(block_num)]
        
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        attention_map = []
        start_idx = 0

        for i, idx in enumerate(block_ids):
            end_idx = idx
            query_partial = query[:, start_idx: end_idx]
            import time

            attention_probs = self.get_attention_scores(attn, query_partial, key) # (b * h, q_len, k_len)
            
            if i != 0:
                if save_text_attention:
                    attention_map.append(attention_probs[attn.heads:, :, torch.tensor(word_ids, device=query.device, dtype=torch.int32)].mean(0))
                else:
                    if i == frame_idx_as_query:
                        attention_map.append(attention_probs[attn.heads:, :, text_seq_length:].mean(0)) # (1350, 17550)

            hidden_states_partial = torch.bmm(attention_probs, value)
            hidden_states_partial = hidden_states_partial.reshape(-1, attn.heads, *hidden_states_partial.shape[1:])
            hidden_states_partial = hidden_states_partial.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_state_blocks.append(hidden_states_partial)
            start_idx = end_idx
        
        hidden_states = torch.cat(hidden_state_blocks, dim=1)   # (bs, seq_len, dim)
        attention_map = torch.cat(attention_map, dim=0) # concat all of the frames
        self.store_attention_map(
            attention_map=attention_map, 
            num_frames=num_frames, 
            height=height, 
            width=width, 
            store_text_cross_attention=save_text_attention,
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
    

class NewVisAttnMapCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        height,
        width,
        frames,
        dim,
        rank,
        kernel_size,
        module_type,
        block_idx=None,
        word_ids=None,
        query_frame_ids=None,
        attention_store=None,
    ):
        super().__init__()
        self.layer_idx = block_idx

        self.attention_store = attention_store
        self.word_ids = word_ids
        self.query_frame_ids = query_frame_ids

        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.module_type = module_type

        if module_type == "conv1d":
            self.temporal_emb = Conv1DModule(input_channels=dim, mid_channels=rank, kernel_size=kernel_size)
        elif module_type == "mlp":
            self.temporal_emb = MLPModule(input_channels=dim, mid_channels=rank)

    def store_attention_map(
        self,
        attention_map,
        store_text_cross_attention,
    ):
        if store_text_cross_attention:
            attention_map = attention_map.reshape(
                self.frames, self.height, self.width, -1 
            )
            attention_map = attention_map.detach().clone().cpu()
            self.attention_store.store(self.layer_idx, attention_map)
        else:
            attention_map = attention_map.reshape(
                self.height, self.width, -1
            )
            attention_map = attention_map.detach().clone().cpu()
            self.attention_store.store(self.layer_idx, attention_map)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        if self.module_type == "conv1d":
            hidden_states_conv1d = hidden_states.reshape(-1, self.frames, self.height, self.width, self.dim)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 2, 3, 4, 1).flatten(0, 2)    # [BHW, T, C]
            hidden_states_conv1d = self.temporal_emb(hidden_states_conv1d)  # [BHW, T, C]
            hidden_states_conv1d = hidden_states_conv1d.reshape(-1, self.height, self.width, self.dim, self.frames)
            hidden_states_conv1d = hidden_states_conv1d.permute(0, 4, 1, 2, 3)
            hidden_states_skip = hidden_states_conv1d.flatten(1, 3)
        elif self.module_type == "mlp":
            hidden_states_mlp = self.temporal_emb(hidden_states)
            hidden_states_skip = hidden_states_mlp

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

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
        
        hidden_states_chunks = []
        chunk_size = self.height * self.width
        num_chunk = int((query.shape[2] - text_seq_length) / chunk_size + 1)
        chunk_ids = [(text_seq_length + i * chunk_size) for i in range(num_chunk)]
        
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        attention_maps = []
        start_idx = 0

        for i, idx in enumerate(chunk_ids):
            end_idx = idx
            query_chunk = query[:, start_idx: end_idx]

            # (bs * h, len_q, len_k)
            attention_probs = attn.get_attention_scores(query_chunk, key, attention_mask=attention_mask)
            if i != 0:
                if self.word_ids is not None:
                    word_ids = torch.tensor(self.word_ids, device=query.device).to(torch.int32)
                    attention_maps.append(attention_probs[attn.heads:, :, word_ids].mean(0))  # (1350)
                else:
                    if i == self.query_frame_ids:
                        attention_maps.append(attention_probs[attn.heads:, :, text_seq_length:].mean(0)) # (1350, 17550)

            hidden_states_chunk = torch.bmm(attention_probs, value)
            hidden_states_chunk = hidden_states_chunk.reshape(-1, attn.heads, *hidden_states_chunk.shape[1:])
            hidden_states_chunk = hidden_states_chunk.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states_chunks.append(hidden_states_chunk)
            start_idx = end_idx
        
        hidden_states = torch.cat(hidden_states_chunks, dim=1)   # (bs, seq_len, dim)
        attention_map = torch.cat(attention_maps, dim=0) # concat all of the frames
        self.store_attention_map(
            attention_map=attention_map, 
            store_text_cross_attention=self.word_ids is not None,
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        hidden_states = hidden_states + hidden_states_skip
        return hidden_states, encoder_hidden_states


class RegionCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        region_prompt_embs: Optional[torch.Tensor] = None,
        region_masks: torch.Tensor = None,
        base_ratio: Optional[int] = None,
        height: int = None,
        width: int = None,
        num_frames: int = None,
    ):
        """
        Args:
            hidden_states.shape = [1, thw, d]
            encoder_hidden_states = [1, l, d]
            region_prompt_embs = [1, n * l, d]
            region_masks = [1, n, t, h, w]
        """
        base_hidden_states, encoder_hidden_states = super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        if base_ratio is None and region_prompt_embs is None:
            return base_hidden_states, encoder_hidden_states, None
        else:    
            text_seq_length = encoder_hidden_states.size(1)
            num_regions = region_prompt_embs.size(1) // text_seq_length

            hidden_states = torch.cat([region_prompt_embs, hidden_states], dim=1)   # [1, n * l + thw, d]

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

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

                query[:, :, num_regions * text_seq_length:] = apply_rotary_emb(query[:, :, num_regions * text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, num_regions * text_seq_length:] = apply_rotary_emb(key[:, :, num_regions * text_seq_length:], image_rotary_emb)
        
            attention_mask = torch.zeros(size=(query.size(2), key.size(2)), device=query.device)
            background_mask = region_masks[0, -1].flatten()

            for i in range(num_regions):
                attention_mask[i * text_seq_length: (i + 1) * text_seq_length, i * text_seq_length: (i + 1) * text_seq_length] = True
                
                flatten_region_mask = region_masks[0, i].flatten()  # [thw]
                cross_attention_mask = flatten_region_mask[None].repeat(text_seq_length, 1)
                attention_mask[i * text_seq_length: (i + 1) * text_seq_length, num_regions * text_seq_length:] = cross_attention_mask
                attention_mask[num_regions * text_seq_length: , i * text_seq_length: (i + 1) * text_seq_length] = cross_attention_mask.transpose(0, 1)

                # background_mask = torch.logical_xor(background_mask, flatten_region_mask)
                region_attention_mask = flatten_region_mask[:, None] * flatten_region_mask[None, :]
                attention_mask[num_regions * text_seq_length:, num_regions * text_seq_length:] = torch.logical_or(
                    attention_mask[num_regions * text_seq_length:, num_regions * text_seq_length:], region_attention_mask
                )
            
            background_attention_mask = background_mask[:, None] * background_mask[None, :]
            attention_mask[num_regions * text_seq_length:, num_regions * text_seq_length:] = torch.logical_or(
                attention_mask[num_regions * text_seq_length:, num_regions * text_seq_length:], background_attention_mask
            )

            attention_mask = attention_mask.to(torch.bool)

            q_u, q_c = query.chunk(2)
            k_u, k_c = key.chunk(2)
            v_u, v_c = value.chunk(2)
            
            hidden_states_u = F.scaled_dot_product_attention(
                q_u, k_u, v_u, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states_c = F.scaled_dot_product_attention(
                q_c, k_c, v_c, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = torch.cat([hidden_states_u, hidden_states_c], dim=0)

            # hidden_states = F.scaled_dot_product_attention(
            #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )   # [1, h, query_len, head_dim]
            
            del attention_mask
            gc.collect()
            torch.cuda.empty_cache()

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            region_prompt_embs, hidden_states = hidden_states.split(
                [num_regions * text_seq_length, hidden_states.size(1) - num_regions * text_seq_length], dim=1
            )

            hidden_states = (1 - base_ratio) * hidden_states + base_ratio * base_hidden_states

            return hidden_states, encoder_hidden_states, region_prompt_embs