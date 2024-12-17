import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm
from transformers import logging

from diffusers.utils import export_to_video


# suppress partial model loading warning
logging.set_verbosity_error()

class Preprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = "cuda"
        self.config = config
        self.use_depth = False

        print("Loading video model")
        self.model_key = "THUDM/CogVideoX-5b"
        self.resolution = (720, 480)

        self.pipeline = CogVideoXPipeline.from_pretrained(self.model_key, torch_dtype=torch.bfloat16)
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.vae.enable_tiling()

        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.transformer = self.pipeline.transformer
        self.scheduler = self.pipeline.scheduler
        print("video model loaded")

    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self, 
        prompt,
        max_sequence_length,
    ):
        device = self.pipeline._execution_device
        dtype = self.text_encoder.dtype

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt,
        negative_prompt,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 226,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt
            )
        else:
            negative_prompt_embeds = None
        
        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True):
        reversed_timesteps = reversed(self.scheduler.timesteps).to(latent.device)
        for i, t in enumerate(tqdm(reversed_timesteps)):
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_plus_1 = (
                self.scheduler.alphas_cumprod[reversed_timesteps[i + 1]] if i < len(reversed_timesteps) - 1 else self.scheduler.alphas_cumprod[-1]
            )

            mu_t = alpha_prod_t ** 0.5
            mu_next = alpha_prod_t_plus_1 ** 0.5
            sigma_t = (1 - alpha_prod_t) ** 0.5
            sigma_next = (1 - alpha_prod_t_plus_1) ** 0.5

            t = t.expand(latent.shape[0])
            v_predict = self.transformer(
                hidden_states=latent, 
                timestep=t, 
                encoder_hidden_states=cond,
            )[0]

            pred_x0 = mu_t * latent - sigma_t * v_predict
            eps_pred = mu_t * v_predict + sigma_t * latent

            latent = sigma_next * eps_pred + mu_next * pred_x0
            if save_latents:
                torch.save(latent.clone().detach().cpu(), os.path.join(save_path, f"noisy_latents_{t.item()}.pt"))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps.to(x.device)
        for i, t in enumerate(tqdm(timesteps)):
            prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_t]
                if prev_t >= 0
                else self.scheduler.final_alpha_cumprod
            )

            beta_prod_t = 1 - alpha_prod_t

            t = t.expand(x.shape[0])
            v_pred = self.transformer(
                hidden_states=x, 
                timestep=t, 
                encoder_hidden_states=cond,
            )[0]
            pred_original_sample = (alpha_prod_t ** 0.5) * x - (beta_prod_t ** 0.5) * v_pred

            a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
            b_t = alpha_prod_t_prev ** 0.5 - alpha_prod_t ** 0.5 * a_t

            x = a_t * x + b_t * pred_original_sample

        return x

    @staticmethod
    def preprocess_video(video):
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        video = video.float()
        video = torch.stack([train_transforms(frame) for frame in video], dim=0)
        video = video.unsqueeze(0)
        return video

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, inversion_prompt="", negative_prompt=""):
        self.scheduler.set_timesteps(num_steps)

        cond = self.encode_prompt(inversion_prompt, negative_prompt, do_classifier_free_guidance=False)[0]
        print(f"cond.shape = {cond.shape}")
        if data_path.endswith(".mp4"):
            import decord
            decord.bridge.set_bridge("torch")

            video_reader = decord.VideoReader(data_path, width=self.resolution[0], height=self.resolution[1])
            batch_ids = torch.linspace(0, len(video_reader) - 1, self.transformer.config.sample_frames)
            video = video_reader.get_batch(batch_ids)
            video = video.permute(0, 3, 1, 2)   # [F, C, H, W]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem))
            video = [Image.open(img).resize(self.resolution).convert("RGB") for img in images]

        video = self.preprocess_video(video)    # [B, F, C, H, W]
        video = video.transpose(1, 2)
        latent = self.vae.encode(video.to(device=self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor    # [B, C, F, H, W]

        latent = latent.transpose(1, 2).to(dtype=self.transformer.dtype, device=self.transformer.device)
        inverted_x = self.ddim_inversion(cond, latent, save_path, save_latents=True)

        if self.config["save_ddim_reconstruction"]:
            latent_reconstruction = self.ddim_sample(inverted_x, cond)
            video = self.pipeline.decode_latents(latent_reconstruction)
            video = self.pipeline.video_processor.postprocess_video(video=video, output_type="pil")
            export_to_video(video[0], "reconstruct.mp4")


def run(opt):
    save_path = opt.save_dir
    os.makedirs(save_path, exist_ok=True)

    model = Preprocess(opt)
    model.extract_latents(
        data_path=opt.video_path,
        num_steps=config["n_timesteps"],
        save_path=save_path,
        inversion_prompt=opt.prompt,
        negative_prompt=opt.negative_prompt,
    )


if __name__ == "__main__":
    # # ==============this code added==================================================================:
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace("132.76.81.120", port=12345, stdoutToServer=True, stderrToServer=True)
    # # ================================================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inversion_config.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    run(config)
