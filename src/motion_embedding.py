import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

class SpatialEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim
        self.appearance_emb = nn.Parameter(torch.zeros(size=(height, width, dim)))
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, dim = hidden_states.shape

        spatial_emb = self.appearance_emb[None].repeat(self.frames, 1, 1, 1)
        spatial_emb = spatial_emb.flatten(0, 2)

        if seq_len <= spatial_emb.shape[0]:
            spatial_emb = spatial_emb[:seq_len]
        else:
            spatial_emb = F.interpolate(spatial_emb.unsqueeze(0), size=(seq_len, dim), mode='linear', align_corners=False).squeeze(0)
        
        assert spatial_emb.shape == hidden_states[0].shape, f"expect emb.shape = {hidden_states[0].shape} but got {spatial_emb.shape}!"
        
        spatial_emb = spatial_emb.to(dtype=hidden_states.dtype, device=hidden_states.device)
        return hidden_states + spatial_emb[None]


class TemporalEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.motion_emb = nn.Parameter(torch.zeros(size=(frames, dim)))
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, dim = hidden_states.shape

        motion_emb = self.motion_emb.reshape(self.frames, 1, 1, self.dim).repeat(1, self.height, self.width, 1)
        emb = motion_emb
        emb = emb.flatten(0, 2)

        if seq_len <= emb.shape[0]:
            emb = emb[:seq_len]
        else:
            emb = F.interpolate(emb.unsqueeze(0), size=(seq_len, dim), mode='linear', align_corners=False).squeeze(0)
        
        assert emb.shape == hidden_states[0].shape, f"expect emb.shape = {hidden_states[0].shape} but got {emb.shape}!"
        
        emb = emb.to(dtype=hidden_states.dtype, device=hidden_states.device)
        return hidden_states + emb[None]
    

class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.appearance_emb = nn.Parameter(torch.zeros(size=(height, width, dim)))
        self.motion_emb = nn.Parameter(torch.zeros(size=(frames, dim)))
    
    def forward(self, hidden_states: torch.Tensor, train=True):
        batch, seq_len, dim = hidden_states.shape

        motion_emb = self.motion_emb.reshape(self.frames, 1, 1, self.dim).repeat(1, self.height, self.width, 1)
        appearance_emb = self.appearance_emb.reshape(1, self.height, self.width, self.dim).repeat(self.frames, 1, 1, 1)

        if train:
            emb = motion_emb + appearance_emb
        else:
            emb = motion_emb
 
        emb = emb.flatten(0, 2)

        if seq_len <= emb.shape[0]:
            emb = emb[:seq_len]
        else:
            emb = F.interpolate(emb.unsqueeze(0), size=(seq_len, dim), mode='linear', align_corners=False).squeeze(0)
        
        assert emb.shape == hidden_states[0].shape, f"expect emb.shape = {hidden_states[0].shape} but got {emb.shape}!"
        
        emb = emb.to(dtype=hidden_states.dtype, device=hidden_states.device)

        return hidden_states + emb[None]
    

def inject_motion_embedding(transformer: CogVideoXTransformer3DModel, train=True, version=""):
    def CogVideoXBlock_forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        norm_hidden_states = self.motion_embedding(norm_hidden_states, train=train)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
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

    height = transformer.config.sample_height // transformer.config.patch_size
    width = transformer.config.sample_width // transformer.config.patch_size
    frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
    dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    trainable_parameters = []
    for module in transformer.modules():
        if module.__class__.__name__ == "CogVideoXBlock":
            module.forward = CogVideoXBlock_forward.__get__(module, CogVideoXBlock)
            if version == "spatial":
                motion_embedding = SpatialEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "temporal":
                motion_embedding = TemporalEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "spatial_frozen_temporal":
                motion_embedding = SpatialTemporalEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "spatial_temporal":
                motion_embedding = SpatialTemporalEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)

            module.add_module("motion_embedding", motion_embedding)

            if train:
                for name, param in motion_embedding.named_parameters():
                    if version == "spatial_frozen_temporal":
                        if "appearance_emb" in name:
                            param.requires_grad_(False)
                            continue

                    param.requires_grad_(True)
                    trainable_parameters.append(param)
                assert len(trainable_parameters) > 0, f"There is no trainable parameter!"
    return trainable_parameters


def save_motion_embedding(
    transformer: CogVideoXTransformer3DModel, 
    save_path: str
):
    motion_embedding_state_dict = {}
    for name, param in transformer.state_dict().items():
        if "motion_embedding" in name:
            motion_embedding_state_dict[name] = param.detach().cpu()

    torch.save(motion_embedding_state_dict, save_path)


def inject_and_load_motion_embedding(
    transformer: CogVideoXTransformer3DModel, 
    ckpt_path: str, 
    version: str, 
    train: bool,
):
    trainable_parameters = inject_motion_embedding(transformer=transformer, train=train, version=version)
    ckpt = torch.load(ckpt_path)
    _, unexpected_keys = transformer.load_state_dict(ckpt, strict=False)
    assert len(unexpected_keys) == 0, f"Something wrong with the checkpoint!"
    print("Loading motion embedding sucessfully!")
    return trainable_parameters


if __name__ == "__main__":
    pretrained_model_name_or_path = "THUDM/CogVideoX-5b"

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer = pipe.transformer

    trainable_parameters = inject_motion_embedding(transformer)

    save_motion_embedding(transformer, "test.pth")
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer"
    )
    inject_and_load_motion_embedding(transformer, "test.pth")

    print("Finish!")

    # pipe.vae.enable_tiling()
    # prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
    # video = pipe(
    #     prompt=prompt,
    #     num_videos_per_prompt=1,
    #     num_inference_steps=50,
    #     num_frames=49,
    #     guidance_scale=6,
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # ).frames[0]
    # export_to_video(video, "output.mp4", fps=8)