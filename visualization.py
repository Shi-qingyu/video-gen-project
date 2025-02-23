from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import BaseOutput

from utils import save_tensor_as_images


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


def read_video_from_file(file_path, max_num_frames=49):
    """
    Args:
        file_path: path to video file
    Return:
        video tensor: [F, C, H, W]
    """
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
        ]
    )

    import decord
    decord.bridge.set_bridge("torch")

    video_reader = decord.VideoReader(file_path, width=720, height=480)
    video_length = len(video_reader)

    video_num_frames = len(video_reader)
    if video_num_frames <= max_num_frames:
        frames = video_reader.get_batch(list(range(0, video_num_frames)))
    else:
        indices = list(range(0, video_num_frames, (video_num_frames - 0) // max_num_frames))
        frames = video_reader.get_batch(indices)

    frames = frames.float()
    frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
    return frames.permute(0, 3, 1, 2)


def load_models(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )

    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder="vae", revision=None
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, subfolder="scheduler"
    )

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision=None
    )

    return tokenizer, text_encoder, vae, scheduler, transformer


def get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    with torch.no_grad():
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    intermediate_layer: int = 28,
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


if __name__ == "__main__":
    noise_timestep = 400
    intermediate_layer = 1

    with torch.no_grad():
        tokenizer, text_encoder, vae, scheduler, transformer = load_models("THUDM/CogVideoX-5b")

        vae.enable_slicing()
        vae.enable_tiling()

        text_encoder.to("cuda", dtype=torch.float32)
        vae.to("cuda", dtype=torch.float32)
        transformer.to("cuda", dtype=torch.bfloat16)

        file_path = "MTBench_subset/MTBench_hard/dog-agility/videos/dog-agility.mp4"
        video = read_video_from_file(file_path)
        num_frames, _, height, width = video.shape
        video = video[None]

        latent_dist = vae.encode(video.permute(0, 2, 1, 3, 4).to(vae.device)).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor   # [B, C, f, h, w]
        
        model_input = latent.permute(0, 2, 1, 3, 4).to(transformer.dtype)
        batch_size, num_frames, num_channels, _, _ = model_input.shape

        noise = torch.randn_like(model_input)
        timesteps = torch.tensor([noise_timestep], device=noise.device)
        noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

        vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height,
                width=width,
                num_frames=num_frames,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                patch_size=transformer.config.patch_size,
                attention_head_dim=transformer.config.attention_head_dim,
                device="cuda",
            )
            if transformer.config.use_rotary_positional_embeddings
            else None
        )

        prompt = "A dog running through a series of red and white barriers."
        prompt_embeds = get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt,
            device=text_encoder.device,
            dtype=text_encoder.dtype,
        )

        prompt_embeds = prompt_embeds.to(transformer.dtype)

        transformer.forward = forward.__get__(transformer, transformer.__class__)

        model_output = transformer(
            hidden_states=noisy_model_input,    # [B, F, C, H, W]
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=True,
            intermediate_layer=intermediate_layer,
        )

        intermediate = model_output.intermediate.to(torch.float32)    # [B, F, H, W, C]
        intermediate = intermediate[-1]  # [F, H, W, C]
        t, h, w, c = intermediate.shape
        save_tensor_as_images(intermediate, root="visualization")

        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt
        # import numpy as np

        # n_samples = 1
        # indices = torch.randint(h * w // 2 - 5, h * w // 2 + 5, size=(n_samples,))

        # # 将 (h, w, c) 展开为 (h*w, c)
        # x_reshaped = intermediate.permute(1, 2, 0, 3).flatten(0, 1)[indices]  # 形状为 (h*w, t, c)
        # global_x = intermediate.permute(1, 2, 0, 3).flatten(0, 1).mean(0, keepdim=True)
        # x_reshaped = torch.cat([x_reshaped, global_x], dim=0)
        # x_reshaped = x_reshaped.cpu().numpy()

        # # 使用 PCA 将 c 维度降维到 2
        # pca = PCA(n_components=2)  # 降维到 2
        # x_pca = np.zeros((n_samples + 1, t, 2))  # 初始化结果数组

        # # 对每个 (h, w) 位置进行 PCA
        # for i in range(n_samples + 1):
        #     x_pca[i] = pca.fit_transform(x_reshaped[i])  # 形状为 (t, 2)

        # # 将结果以折线图的形式画出来
        # plt.figure(figsize=(12, 8))
        # for i in range(n_samples + 1):
        #     plt.plot(x_pca[i, :, 0], x_pca[i, :, 1], linestyle='-', linewidth=0.5)  # 画每条曲线

        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.title(f'{n_samples + 1} PCA Curves')
        # plt.grid(False)

        # # 保存图像到本地
        # plt.savefig('pca_curves.png', dpi=300, bbox_inches='tight')  # 保存为 pca_curves.png