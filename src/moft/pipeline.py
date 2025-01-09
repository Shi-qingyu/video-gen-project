import os

import decord

import torch
from torchvision import transforms

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from .ddim_inversion import DDIMInversion


def get_tensor_from_video(video_path, width=720, height=480, max_num_frames=49):

    decord.bridge.set_bridge("torch")
    
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
        ]
    )

    video_reader = decord.VideoReader(video_path, width=width, height=height)
    num_frames = len(video_reader)

    frames = video_reader.get_batch(list(range(0, num_frames)))
    frames = frames[: max_num_frames]
    selected_num_frames = frames.shape[0]

    remainder = (3 + (selected_num_frames % 4)) % 4
    if remainder != 0:
        frames = frames[:-remainder]
    
    frames = frames.float()
    frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
    frames = frames.permute(3, 0, 1, 2) # (C, F, H, W)
    return frames


class MyCogVideoXPipeline(CogVideoXPipeline):
    def save_inter_feat(self, video_path, prompts, layer_idx, outpath):
        video_tensor = get_tensor_from_video(video_path)
        video = video_tensor[None].to(self._execution_device)  # (B, C, F, H, W)

        latents, inter_feat = self.run_ddim_inversion(video, prompts, self.scheduler, layer_idx=layer_idx, return_intermediates=True)

        last_latent = latents[-1]
        print(f"last latent's shape: {last_latent.shape}")
        torch.save(last_latent, outpath)
        # inter_feat = sum([inter_feat[i][0] for i in range(len(inter_feat))]) / len(inter_feat)   # [B * F, C, H, W]
        # torch.save(inter_feat, outpath)

    def run_ddim_inversion(self, image, prompt, scheduler, pred_step=None, layer_idx=[0], return_intermediates=False):
        inversioner = DDIMInversion(self,
                                    inversion_reg_steps = 5,
                                    inversion_ac_rolls = 5,
                                    inversion_kl_weight = 20,
                                    inversion_auto_coor_weight = 20,
                                    scheduler=scheduler,
                                    cfg=False)
        inversioner.init_prompt(prompt)
        latents, inter_feat = inversioner.ddim_inversion(image, pred_step=pred_step, layer_idx=layer_idx, return_intermediates=return_intermediates)

        encoder_hidden_state = inversioner.context
        _, cond = torch.chunk(encoder_hidden_state, 2)
        self.cond_embedding = cond
        # embedding = self.model.compute_text_embeddings(prompt)
        return latents, inter_feat