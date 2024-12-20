from typing import Optional
import gc

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention


class CogVideoXAttnProcessor3_0(CogVideoXAttnProcessor2_0):

    def __init__(self, height=30, width=45, block_idx=None):
        super().__init__()
        self.height = height
        self.width = width
        self.block_idx = block_idx

    def prepare_attention_mask(
        self,
        attn: Attention,
        word_ids: list,
        word_lens: list,
        bboxes: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        query_length: int,
        text_seq_length: int,
        batch_size: int,
        dtype,
    ):
        head_size = attn.heads

        attention_mask = torch.zeros(
            size=(query_length, 226 + num_frames * height * width),
            dtype=dtype
        )

        for word_idx, word_len, bbox in zip(word_ids, word_lens, bboxes):
            x1, x2 = [int(x * width) for x in bbox[::2]]
            y1, y2 = [int(y * height) for y in bbox[1::2]]
            
            if query_length == height * width:
                video_mask = torch.zeros(size=(height, width), dtype=dtype)
                video_mask[y1: y2, x1: x2] = 1
                vertical_video_mask = video_mask.flatten()[..., None].repeat(1, word_len)
                vertical_video_mask = (1 - vertical_video_mask) * (-1e5) + 20

                attention_mask[:, word_idx: word_idx+word_len] = vertical_video_mask

            elif query_length == text_seq_length:
                video_mask = torch.zeros(size=(num_frames, height, width), dtype=dtype)
                video_mask[:, y1: y2, x1: x2] = 1
                video_mask = video_mask.flatten()[None].repeat(word_len, 1)
                attention_mask[word_idx: word_idx+word_len, 226:] = video_mask

        if query_length == height * width:
            pos_in_bbox = vertical_video_mask[:, 0].bool()
            attention_mask[pos_in_bbox, :word_idx] = -1e5
            attention_mask[pos_in_bbox, word_idx+word_len:] = -1e5
        elif query_length == text_seq_length:
            attention_mask[word_ids, 226:] = torch.where(attention_mask[word_ids, 226:].bool(), 20, -1e5).to(dtype)

        attention_mask = attention_mask[None].repeat(batch_size * head_size, 1, 1)

        return attention_mask   

    def prepare_attention_mask_optimized(
        self,
        attn: Attention,
        word_ids: torch.Tensor,
        word_lens: torch.Tensor,
        bboxes: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        query_length: int,
        text_seq_length: int,
        batch_size: int,
        dtype,
    ):
        head_size = attn.heads
        device = bboxes.device

        total_seq_length = 226 + num_frames * height * width

        attention_mask = torch.full(
            size=(query_length, total_seq_length),
            fill_value=0,
            dtype=dtype,
            device=device
        )

        x1 = torch.clamp((bboxes[:, 0] * width).long(), 0, width - 1)
        y1 = torch.clamp((bboxes[:, 1] * height).long(), 0, height - 1)
        x2 = torch.clamp((bboxes[:, 2] * width).long(), 1, width)
        y2 = torch.clamp((bboxes[:, 3] * height).long(), 1, height)

        x2 = torch.max(x2, x1 + 1)
        y2 = torch.max(y2, y1 + 1)

        if query_length == height * width:
            for idx in range(len(word_ids)):
                word_id = word_ids[idx]
                word_len = word_lens[idx]

                video_mask = torch.zeros((height, width), dtype=dtype, device=device)
                video_mask[y1[idx]:y2[idx], x1[idx]:x2[idx]] = 1
                video_mask_flat = video_mask.flatten()

                vertical_video_mask = video_mask_flat[:, None].repeat(1, word_len)
                vertical_video_mask = torch.where(
                    vertical_video_mask.bool(), torch.tensor(0.2, dtype=dtype, device=device), torch.tensor(-0.2, dtype=dtype, device=device)
                )
                
                attention_mask[:, word_id:word_id + word_len] = vertical_video_mask

                pos_in_bbox = video_mask_flat.bool()
                attention_mask[pos_in_bbox, :word_id] = torch.tensor(-0.2, dtype=dtype, device=device)
                attention_mask[pos_in_bbox, word_id + word_len:] = torch.tensor(-0.2, dtype=dtype, device=device)

        elif query_length == text_seq_length:
            pass
            # for idx in range(len(word_ids)):
            #     word_id = word_ids[idx]
            #     word_len = word_lens[idx]

            #     video_mask = torch.zeros((num_frames, height, width), dtype=dtype, device=device)
            #     video_mask[:, y1[idx]:y2[idx], x1[idx]:x2[idx]] = 1
            #     video_mask_flat = video_mask.flatten()

            #     video_mask_expanded = video_mask_flat[None, :].repeat(word_len, 1)

            #     attention_mask[word_id:word_id + word_len, 226:] = torch.where(
            #         video_mask_expanded.bool(), torch.tensor(2.5, dtype=dtype, device=device), torch.tensor(-2.5, dtype=dtype, device=device)
            #     )

        attention_mask = attention_mask[None].repeat(batch_size * head_size, 1, 1)

        return attention_mask
    
    def prepare_attention_mask_aug(
        self,
        attn: Attention,
        word_ids: torch.Tensor,
        word_lens: torch.Tensor,
        bboxes: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        query_length: int,
        text_seq_length: int,
        batch_size: int,
        dtype,
    ):
        head_size = attn.heads
        device = bboxes.device

        total_seq_length = 226 + num_frames * height * width

        attention_mask = torch.full(
            size=(query_length, total_seq_length),
            fill_value=0,
            dtype=dtype,
            device=device
        )

        x1 = torch.clamp((bboxes[:, 0] * width).long(), 0, width - 1)
        y1 = torch.clamp((bboxes[:, 1] * height).long(), 0, height - 1)
        x2 = torch.clamp((bboxes[:, 2] * width).long(), 1, width)
        y2 = torch.clamp((bboxes[:, 3] * height).long(), 1, height)

        x2 = torch.max(x2, x1 + 1)
        y2 = torch.max(y2, y1 + 1)

        if query_length == height * width:
            for idx in range(len(word_ids)):
                word_id = word_ids[idx]
                word_len = word_lens[idx]

                video_mask = torch.zeros((height, width), dtype=dtype, device=device)
                video_mask[y1[idx]:y2[idx], x1[idx]:x2[idx]] = 1
                video_mask_flat = video_mask.flatten()

                vertical_video_mask = video_mask_flat[:, None].repeat(1, word_len)
                vertical_video_mask = torch.where(
                    vertical_video_mask.bool(), torch.tensor(2.5, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device)
                )
                
                attention_mask[:, word_id:word_id + word_len] = vertical_video_mask

                # pos_in_bbox = video_mask_flat.bool()
                # attention_mask[pos_in_bbox, :word_id] = torch.tensor(-1e5, dtype=dtype, device=device)
                # attention_mask[pos_in_bbox, word_id + word_len:] = torch.tensor(-1e5, dtype=dtype, device=device)

        elif query_length == text_seq_length:
            for idx in range(len(word_ids)):
                word_id = word_ids[idx]
                word_len = word_lens[idx]

                video_mask = torch.zeros((num_frames, height, width), dtype=dtype, device=device)
                video_mask[:, y1[idx]:y2[idx], x1[idx]:x2[idx]] = 1
                video_mask_flat = video_mask.flatten()

                video_mask_expanded = video_mask_flat[None, :].repeat(word_len, 1)

                attention_mask[word_id:word_id + word_len, 226:] = torch.where(
                    video_mask_expanded.bool(), torch.tensor(2.5, dtype=dtype, device=device), torch.tensor(0, dtype=dtype, device=device)
                )

        attention_mask = attention_mask[None].repeat(batch_size * head_size, 1, 1)

        return attention_mask
    
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
            alpha=self.scale,
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
        width
    ):
        attention_map = attention_map.reshape(
            attention_map.shape[0], num_frames, height, width, -1 
        )
        _attention_map = attention_map.mean(0).detach().clone().cpu()
        self.attention_store.store(self.block_idx, _attention_map)
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        word_ids: list = None,
        word_lens: list = None,
        bboxes: torch.Tensor = None,
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
        block_size = self.height * self.width
        block_num = int((query.shape[2] - text_seq_length) / block_size + 1)
        block_ids = [(text_seq_length + i * block_size) for i in range(block_num)]
        
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        attention_map = []
        start_idx = 0
        for i, idx in enumerate(block_ids):
            end_idx = idx
            query_tmp = query[:, start_idx: end_idx]
            query_length = end_idx - start_idx
            import time

            attention_mask = self.prepare_attention_mask_optimized(
                attn,
                word_ids,
                word_lens,
                bboxes,
                height,
                width,
                num_frames,
                query_length,
                text_seq_length,
                batch_size,
                query.dtype,
            )
            attention_mask = attention_mask.to(device=query.device)

            attention_probs = self.get_attention_scores(attn, query_tmp, key, attention_mask) # (b * h, q_len, k_len)
            
            if i != 0:
                attention_map.append(attention_probs[:, :, torch.tensor(word_ids, device=query.device, dtype=torch.int32)])
            hidden_states_tmp = torch.bmm(attention_probs, value)
            hidden_states_tmp = hidden_states_tmp.reshape(-1, attn.heads, *hidden_states_tmp.shape[1:])
            hidden_states_tmp = hidden_states_tmp.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_state_blocks.append(hidden_states_tmp)
            start_idx = end_idx
        
        hidden_states = torch.cat(hidden_state_blocks, dim=1)   # (bs, seq_len, dim)
        attention_map = torch.cat(attention_map, dim=1)
        self.store_attention_map(attention_map=attention_map, num_frames=num_frames, height=height, width=width)

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

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )   # [1, h, query_len, head_dim]
            
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