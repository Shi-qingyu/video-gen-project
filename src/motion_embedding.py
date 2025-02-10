import math

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from .utils import mask2bbox


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
    
    def forward(self, hidden_states: torch.Tensor, train=True, local_trajectories=None, **kwargs):
        batch, seq_len, dim = hidden_states.shape

        motion_emb = self.motion_emb.reshape(self.frames, 1, 1, self.dim).repeat(1, self.height, self.width, 1)
        appearance_emb = self.appearance_emb.reshape(1, self.height, self.width, self.dim).repeat(self.frames, 1, 1, 1)
        emb = motion_emb + appearance_emb
        emb = emb.flatten(0, 2)

        if seq_len <= emb.shape[0]:
            emb = emb[:seq_len]
        else:
            emb = F.interpolate(emb.unsqueeze(0), size=(seq_len, dim), mode='linear', align_corners=False).squeeze(0)
        
        assert emb.shape == hidden_states[0].shape, f"expect emb.shape = {hidden_states[0].shape} but got {emb.shape}!"
        
        emb = emb.to(dtype=hidden_states.dtype, device=hidden_states.device)

        return hidden_states + emb[None]


class AdaptiveSpatialTemporalEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim, complexity=8) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        size = round(math.sqrt(complexity))
        self.motion_emb = nn.Parameter(torch.zeros(size=(frames, size, size, dim)))
        self.appearance_emb = nn.Parameter(torch.zeros(size=(height, width, dim)))
    
    def forward(self, hidden_states: torch.Tensor, train: bool, **kwargs):
        batch_size, seq_len, dim = hidden_states.shape

        spatial_emb = self.appearance_emb[None].repeat(self.frames, 1, 1, 1)
        spatial_emb = spatial_emb.flatten(0, 2).to(dtype=hidden_states.dtype, device=hidden_states.device)
        
        motion_emb = self.motion_emb.permute(0, 3, 1, 2)
        motion_emb = F.interpolate(
            motion_emb, size=(self.height, self.width), mode="bilinear", align_corners=True
        )
        motion_emb = motion_emb.permute(0, 2, 3, 1).flatten(0, 2)

        if train:
            hidden_states = hidden_states + spatial_emb[None] + motion_emb[None]
        else:
            hidden_states = hidden_states + motion_emb[None]
        
        return hidden_states



class AdaptiveMaskTemporalEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim, complexity=8) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        size = round(math.sqrt(complexity))
        self.motion_emb = nn.Parameter(torch.zeros(size=(frames, size, size, dim)))
        self.appearance_emb = nn.Parameter(torch.zeros(size=(height, width, dim)))

    def forward(self, hidden_states: torch.Tensor, train: bool, masks: torch.Tensor, **kwargs):
        batch_size, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, self.frames, self.height, self.width, -1)

        spatial_emb = self.appearance_emb[None].repeat(self.frames, 1, 1, 1)
        spatial_emb = spatial_emb.flatten(0, 2).to(dtype=hidden_states.dtype, device=hidden_states.device)
        
        batch_size, _, h, w = masks.shape
        masks = masks.reshape(-1, 1, h, w)
        masks = F.interpolate(masks, size=(self.height, self.width), mode="nearest")
        masks = masks.reshape(batch_size, -1, self.height, self.width)
        # bbox: [B, F, 4]
        bbox = mask2bbox(mask=masks)
        motion_emb = self.motion_emb[None].repeat(batch_size, 1, 1, 1, 1)

        for b in range(motion_emb.shape[0]):
            for f in range(motion_emb.shape[1]):
                motion_emb_t = motion_emb[b, f] # [complexity, complexity, dim]
                bbox_t = bbox[b, f] # [top_left_y, top_left_x, bottom_right_y, bottom_right_x]
                motion_emb_t = motion_emb_t.permute(2, 0, 1)[None]
                motion_emb_t = F.interpolate(
                    motion_emb_t, size=(bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]), mode="bilinear", align_corners=True
                )[0]
                motion_emb_t = motion_emb_t.permute(1, 2, 0)
                hidden_states[b, f, bbox_t[0]: bbox_t[2], bbox_t[1]: bbox_t[3]] = hidden_states[b, f, bbox_t[0]: bbox_t[2], bbox_t[1]: bbox_t[3]] + motion_emb_t

        if train:
            hidden_states = hidden_states.flatten(1, 3) + spatial_emb[None]
        else:
            hidden_states = hidden_states.flatten(1, 3)
        
        return hidden_states


class ScaleShiftEmbedding(nn.Module):
    def __init__(self, height, width, frames, dim) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.frames = frames
        self.dim = dim

        self.shift_emb = nn.Parameter(torch.zeros(size=(height, width, dim)))
        self.scale_emb = nn.Parameter(torch.zeros(size=(frames, dim)))
    
    def forward(self, hidden_states: torch.Tensor, train=True, **kwargs):
        batch, seq_len, dim = hidden_states.shape

        scale_emb = self.scale_emb.reshape(self.frames, 1, 1, self.dim).repeat(1, self.height, self.width, 1)
        scale_emb = scale_emb.flatten(0, 2)[None]
        shift_emb = self.shift_emb.reshape(1, self.height, self.width, self.dim).repeat(self.frames, 1, 1, 1)
        shift_emb = shift_emb.flatten(0, 2)[None]

        scale_emb = scale_emb.to(dtype=hidden_states.dtype, device=hidden_states.device)
        shift_emb = shift_emb.to(dtype=hidden_states.dtype, device=hidden_states.device)

        hidden_states = hidden_states * (1 + scale_emb) + shift_emb
        return hidden_states
    

def inject_motion_embedding(
    transformer: CogVideoXTransformer3DModel, 
    train=True, 
    version="",
    interpolate_layers=[],
    **kwargs
):
    def CogVideoXBlock_forward(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        temb, 
        image_rotary_emb, 
        motion_module_kwargs=None,
    ):
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        if hasattr(self, "motion_embedding"):
            if motion_module_kwargs is not None:
                # motion injection
                norm_hidden_states = self.motion_embedding(norm_hidden_states, train=train, **motion_module_kwargs)
            else:
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
    for name, module in transformer.named_modules():
        if module.__class__.__name__ == "CogVideoXBlock" or module.__class__.__name__ == "MyCogVideoXBlock":
            module.forward = CogVideoXBlock_forward.__get__(module, CogVideoXBlock)

            layer_index = int(name.split(".")[1])
            if len(interpolate_layers) > 0 and layer_index not in interpolate_layers:
                continue
            
            if version == "spatial":
                motion_embedding = SpatialEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "temporal":
                motion_embedding = TemporalEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "spatial_frozen_temporal" or version == "spatial_temporal":
                motion_embedding = SpatialTemporalEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "scale_shift":
                motion_embedding = ScaleShiftEmbedding(height=height, width=width, frames=frames, dim=dim).to(transformer.device)
            elif version == "adaptive_spatial_temporal":
                complexity = kwargs.get("complexity", None)
                assert complexity is not None, "complexity can't be None in adaptive temporal version!"
                motion_embedding = AdaptiveSpatialTemporalEmbedding(height=height, width=width, frames=frames, dim=dim, complexity=complexity)
            elif version == "adaptive_mask_temporal":
                complexity = kwargs.get("complexity", None)
                assert complexity is not None, "complexity can't be None in adaptive mask temporal version!"
                motion_embedding = AdaptiveMaskTemporalEmbedding(height=height, width=width, frames=frames, dim=dim, complexity=complexity)                
            else:
                raise ValueError(f"Unexpected motion embedding version: {version}")

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
    **kwargs,
):
    trainable_parameters = inject_motion_embedding(
        transformer=transformer, train=train, version=version, **kwargs,
    )
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