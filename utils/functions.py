import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

import sys
sys.path.append("..") 

from train_image_transformer import validation
from utils.image_transformer import Image_Transformer, inj_forward_crossattention, value_local_list
from utils.domain_adaptor import Domain_Adaptor, inj_forward_text

def save_progress(mapper, accelerator, args, step=None, name='domain_adaptor'):
    state_dict = accelerator.unwrap_model(mapper).state_dict()

    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, name + '_' +str(step)+'.pt'))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, name+".pt"))


def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def _pil_from_latents(vae, latents):
    _latents = 1 / 0.18215 * latents.clone()
    image = vae.decode(_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    ret_pil_images = [Image.fromarray(image) for image in images]

    return ret_pil_images


def pww_load_tools(
    device: str = "cuda:0",
    scheduler_type=LMSDiscreteScheduler,
    domain_adaptor_path: Optional[str] = None,
    image_transformer_path: Optional[str] = None,
    diffusion_model_path: Optional[str] = None,
    model_token: Optional[str] = None,
):

    # 'CompVis/stable-diffusion-v1-4'
    # local_path_only = diffusion_model_path is not None
    local_path_only = False
    vae = AutoencoderKL.from_pretrained(
        diffusion_model_path,
        subfolder="vae",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16,)


    # Load models and create wrapper for stable diffusion
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    unet = UNet2DConditionModel.from_pretrained(
        diffusion_model_path,
        subfolder="unet",
        use_auth_token=model_token,
        torch_dtype=torch.float16,
        local_files_only=local_path_only,
    )

    mapper = Domain_Adaptor(input_dim=1024, output_dim=768)

    mapper_local = Image_Transformer(input_dim=1024, output_dim=768)

    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            to_v_local = nn.Linear(shape[1], shape[0], bias=False)
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_v', to_v_local)

            to_k_local = nn.Linear(shape[1], shape[0], bias=False)
            mapper_local.add_module(f'{_name.replace(".", "_")}_to_k', to_k_local)

    mapper.load_state_dict(torch.load(domain_adaptor_path, map_location='cpu'))
    mapper.half()

    mapper_local.load_state_dict(torch.load(image_transformer_path, map_location='cpu'))
    mapper_local.half()

    for _name, _module in unet.named_modules():
        if 'attn1' in _name: continue
        if _module.__class__.__name__ == "CrossAttention":
            _module.add_module('to_k_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
            _module.add_module('to_v_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))
            _module.add_module('to_v_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_v'))
            _module.add_module('to_k_local', getattr(mapper_local, f'{_name.replace(".", "_")}_to_k'))

    vae.to(device), unet.to(device), text_encoder.to(device), image_encoder.to(device), mapper.to(device), mapper_local.to(device)

    scheduler = scheduler_type(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    vae.eval()
    unet.eval()
    image_encoder.eval()
    text_encoder.eval()
    mapper.eval()
    mapper_local.eval()
    return vae, unet, text_encoder, tokenizer, image_encoder, mapper, mapper_local, scheduler
