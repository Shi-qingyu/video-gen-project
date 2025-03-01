import os
from typing import Union, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffusers import CogVideoXPipeline
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, BaseOutput

from src.pipeline import IntermediateCogVideoXPipeline
from src.motion_embedding import inject_and_load_motion_embedding
from src.pipeline import MyCogVideoXPipeline
from src.new.attention_processor import SkipConv1dCogVideoXAttnProcessor2_0

from utils import save_tensor_as_images


SEED = 42

prompt = "A man riding a lion is jumping over a fence."
device = "cuda"
ckpt_path = "checkpoints/lr_1e-3_spatial_temporal_horsejump-high/checkpoint-500/motion_embedding.pth"
version = "spatial_temporal"

intermediate_layer = 1


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821
    intermediate: "torch.Tensor"


def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 1. Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # 2. Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    p = self.config.patch_size
    intermediate = None
    # 3. Transformer blocks
    for i, block in enumerate(self.transformer_blocks):
        hidden_states, encoder_hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=emb,
            image_rotary_emb=image_rotary_emb,
        )
        if i == intermediate_layer - 1:
            intermediate = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1)

    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # 4. Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    # Note: we use `-1` instead of `channels`:
    #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
    #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output, intermediate=intermediate)


pipe = IntermediateCogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

transformer = pipe.transformer
transformer.forward = forward.__get__(transformer, transformer.__class__)

height = transformer.config.sample_height // transformer.config.patch_size
width = transformer.config.sample_width // transformer.config.patch_size
frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

store = {}
attn_processors = {}
for key, value in transformer.attn_processors.items():
    block_idx = int(key.split(".")[1])
    if block_idx in list(range(0, 15)):
        attn_processor = SkipConv1dCogVideoXAttnProcessor2_0(
            height=height, 
            width=width, 
            frames=frames, 
            dim=dim, 
            rank=128,
            kernel_size=3,
            module_type="",
            store=store,
            block_index=str(block_idx)
        ).to(dtype=transformer.dtype)
        attn_processors[key] = attn_processor
    else:
        attn_processors[key] = value

transformer.set_attn_processor(attn_processors)
transformer.load_state_dict(torch.load(ckpt_path), strict=False)

inject_and_load_motion_embedding(
    transformer,
    ckpt_path=ckpt_path,
    version=version,
    train=True,
    interpolate_layers=list(range(42)),
    complexity=None,
)

pipe.transformer = transformer
pipe.to(device)
pipe.vae.enable_tiling()

videos, intermediate = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device=device).manual_seed(SEED),
)   # intermediate.shape = (b, f, h, w, c)

save_path = "test.mp4"
export_to_video(videos[0], save_path, fps=8)

intermediate = intermediate[-1].to(torch.float32)
save_tensor_as_images(intermediate=intermediate, root="visualization")

for key, value in store.items():
    value = value.to(torch.float32)
    root = "visualization/" + f"block_{key}"
    save_tensor_as_images(intermediate=value, root=root)