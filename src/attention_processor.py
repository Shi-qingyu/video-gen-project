from typing import Optional
import gc

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention


class CogVideoXAttnProcessor3_0(CogVideoXAttnProcessor2_0):
    def __init__(self, block_idx=None):
        super().__init__()
        self.layer_idx = block_idx
        self.attention_store = None
    

    def get_attention_scores(
        self, attn: Attention, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
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
    

class RegionCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    def prepare_attention_mask(
        self,
        region_masks,
        height,
        width,
        num_frames,
        query_length,
        frame_idx,
        num_regions,
        text_seq_length,
        device,
    ):
        """
        Args:
            region_masks: shape [1, n, t, h, w]
        """
        region_masks = region_masks.to(device=device, dtype=torch.bool)
        attention_mask = torch.zeros(
            (query_length, num_regions * text_seq_length + num_frames * height * width), dtype=torch.bool, device=device
        )
        if query_length == num_regions * text_seq_length:
            for i in range(num_regions):
                attention_mask[i * text_seq_length: (i + 1) * text_seq_length, i * text_seq_length: (i + 1) * text_seq_length] = True
                region_mask = region_masks[0, i]    # [t, h, w]
                region_mask = region_mask.flatten()
                region_mask = region_mask[None, :].repeat(text_seq_length, 1)
                attention_mask[i * text_seq_length: (i + 1) * text_seq_length, num_regions * text_seq_length:] = region_mask
        else:
            flatten_background = torch.ones((height * width), device=device, dtype=torch.bool)
            for i in range(num_regions):
                region_mask = region_masks[0, i, frame_idx]    # [h, w]
                region_mask = region_mask.flatten()    # [hw]
                flatten_background = torch.logical_xor(flatten_background, region_mask)

                region_mask_for_text = region_mask[:, None].repeat(1, text_seq_length)
                attention_mask[:, i * text_seq_length: (i + 1) * text_seq_length] = region_mask_for_text

                region_masks_for_self = region_masks[0, i]  # [t, h, w]
                region_masks_for_self = region_masks_for_self.flatten()
                self_attention_mask = region_mask[:, None] * region_masks_for_self[None, :] # [hw, thw]
                attention_mask[:, num_regions * text_seq_length:] += self_attention_mask
            
            flatten_background = (flatten_background > 0).to(torch.bool)
            flatten_background_all_frame = (region_masks.sum(1).flatten() < 1).to(torch.bool)
            bachground_mask = flatten_background[:, None] * flatten_background_all_frame[None, :]   # [hw, thw]

            attention_mask[:, num_regions * text_seq_length:] += bachground_mask
            attention_mask = (attention_mask > 0).to(torch.bool)
        
        return attention_mask

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
        
            hidden_state_blocks = []
            block_size = height * width
            block_num = int((query.shape[2] - num_regions * text_seq_length) // block_size + 1)
            block_ids = [(num_regions * text_seq_length + i * block_size) for i in range(block_num)]

            start_idx = 0
            for i, idx in enumerate(block_ids):
                end_idx = idx
                query_tmp = query[:, :, start_idx: end_idx]
                query_length = end_idx - start_idx
                import time

                attention_mask = self.prepare_attention_mask(
                    region_masks=region_masks,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    query_length=query_length,
                    frame_idx=i-1,
                    num_regions=num_regions,
                    text_seq_length=text_seq_length,
                    device=query.device,
                )   # [query_len, key_len]

                hidden_states_tmp = F.scaled_dot_product_attention(
                    query_tmp, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )   # [1, h, query_len, head_dim]
                
                del attention_mask
                gc.collect()
                torch.cuda.empty_cache()

                hidden_states_tmp = hidden_states_tmp.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_state_blocks.append(hidden_states_tmp)
                start_idx = end_idx
            
            hidden_states = torch.cat(hidden_state_blocks, dim=1)   # (bs, seq_len, dim)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            region_prompt_embs, hidden_states = hidden_states.split(
                [num_regions * text_seq_length, hidden_states.size(1) - num_regions * text_seq_length], dim=1
            )

            hidden_states = (1 - base_ratio) * hidden_states + base_ratio * base_hidden_states

            return hidden_states, encoder_hidden_states, region_prompt_embs


class RegionCogVideoXAttnProcessor3_0(CogVideoXAttnProcessor2_0):
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