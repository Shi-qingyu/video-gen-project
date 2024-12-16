from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils import is_torch_version

import torch

from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import gc


class MyCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        return_layer_ids: List[int] = [10],
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

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
        hidden_states = hidden_states[:, text_seq_length:]  # (B, THW, D)

        ft = {}
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

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
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return output   # [b, f, c, h, w]
    

class OneStepPipeline(CogVideoXPipeline):
    @torch.no_grad()
    def __call__(
        self,
        video_tensor,   # [b, c, f, h, w]
        t,
        return_layer_ids,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):
        device = self._execution_device
        latents = self.vae.encode(video_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        latents = latents.transpose(1, 2).to(self.transformer.device)   # [b, f, c, h, w]

        t = torch.tensor(t, dtype=torch.long, device=device).expand(video_tensor.shape[0])
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(480, 720, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        negative_prompt = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=226,
            device=device,
        )

        ft_output = self.transformer(
            hidden_states=latents_noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=t,
            image_rotary_emb=image_rotary_emb,
            return_layer_ids=return_layer_ids,
        )   # (b, f, c, h, w)

        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        eps = (alpha_prod_t ** 0.5) * ft_output + (1 - alpha_prod_t) ** 0.5 * latents_noisy
        pred_original_sample = (alpha_prod_t ** 0.5) * latents_noisy - (1 - alpha_prod_t) ** 0.5 * ft_output

        return pred_original_sample
    

class SDFeaturizer:
    def __init__(self, sd_id='THUDM/CogVideoX-5b', null_prompt=''):
        transformer = MyCogVideoXTransformer3DModel.from_pretrained(sd_id, subfolder="transformer")
        onestep_pipe = OneStepPipeline.from_pretrained(sd_id, transformer=transformer, safety_checker=None)
        onestep_pipe.vae.decoder = None
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(
        self,
        video_tensor,
        prompt='',
        t=261,
        return_layer_ids=1,
        ensemble_size=8
    ):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        video_tensor = video_tensor.repeat(ensemble_size, 1, 1, 1, 1).cuda() # ensem, c, f, h, w
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt = prompt * video_tensor.shape[0]

        assert -1 <= video_tensor.min() and video_tensor.max() <= 1

        output = self.pipe(
            video_tensor=video_tensor,  # [b, c, f, h, w]
            t=t,
            return_layer_ids=[return_layer_ids],
            prompt=prompt
        )
        transformer_ft = output.mean(0) # f, c, h, w

        return transformer_ft