import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention


class CogVideoXAttnProcessor3_0(CogVideoXAttnProcessor2_0):

    def __init__(self, height=30, width=45):
        super().__init__()
        self.height = height
        self.width = width

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
        batch_size: int,
    ):
        head_size = attn.heads

        attention_mask = torch.zeros(
            size=(query_length, 226 + num_frames * height * width),
            dtype=torch.float32
        )

        for word_idx, word_len, bbox in zip(word_ids, word_lens, bboxes):
            x1, x2 = [int(x) for x in bbox[::2] * width]
            y1, y2 = [int(y) for y in bbox[1::2] * height]
            
            if query_length == height * width:
                video_mask = torch.zeros(size=(height, width), dtype=torch.float32)
                video_mask[y1: y2, x1: x2] = 1
                video_mask_ = video_mask.flatten()[..., None].repeat(1, word_len)
                video_mask_ = (1 - video_mask_) * (-1e8) + 20

                attention_mask[:, word_idx: word_idx+word_len] = video_mask_

            elif query_length <= height * width:
                video_mask = torch.zeros(size=(num_frames, height, width), dtype=torch.float32)
                video_mask[:, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                video_mask = video_mask.flatten()[None].repeat(word_len, 1)
                attention_mask[word_idx: word_idx+word_len, 226:] = video_mask

        if query_length == height * width:
            pos_in_bbox = video_mask[:, 0].bool()
            attention_mask[pos_in_bbox, :word_idx] = -1e8
            attention_mask[pos_in_bbox, word_idx+word_len:] = -1e8
            # attention_mask[:, :226] = torch.where(attention_mask[:, :226].bool(), 20, -1e8)
        else:
            attention_mask[:, 226:] = torch.where(attention_mask[:, 226:].bool(), 20, -1e8)

        attention_mask = attention_mask[None].repeat(batch_size * head_size, 1, 1)

        return attention_mask
        
        
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

        start_idx = 0
        for idx in block_ids:
            end_idx = idx
            query_tmp = query[:, start_idx: end_idx]
            query_length = end_idx - start_idx
            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attn,
                    word_ids,
                    word_lens,
                    bboxes,
                    height,
                    width,
                    num_frames,
                    query_length,
                    batch_size
                )
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                attention_mask = attention_mask.to(query.device)

            attention_probs = attn.get_attention_scores(query_tmp, key, attention_mask)
            hidden_states_tmp = torch.bmm(attention_probs, value)
            hidden_states_tmp = hidden_states_tmp.reshape(-1, attn.heads, *hidden_states_tmp.shape[1:])
            hidden_states_tmp = hidden_states_tmp.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_state_blocks.append(hidden_states_tmp)
            start_idx = end_idx
        
        hidden_states = torch.cat(hidden_state_blocks, dim=1)   # (bs, seq_len, dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
    

