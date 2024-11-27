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
        attention_mask: torch.Tensor,
        sequence_length: int,
        batch_size: int,
    ):
        head_size = attn.heads
        if attention_mask is None:
            return None
        
        
        

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
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
    

