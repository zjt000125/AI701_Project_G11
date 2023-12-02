import argparse
import itertools
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


from typing import Optional
from utils.functions import *
from utils.image_transformer import value_local_list
from datasets import OpenImagesDatasetWithMask

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--domain_adaptor_path", type=str, default=None,
        help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--image_transformer_path", type=str, default=None,
        help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, domain_adaptor, image_transformer, vae, device, guidance_scale, seed=None, llambda=1, num_steps=100):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    # the number of images to generate one time
    num_sample = 4
    uncond_input = tokenizer(
        [''] * num_sample,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (num_sample, unet.in_channels, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (num_sample, unet.in_channels, 64, 64), generator=generator,
        )

    # latents = vae.encode(example["pixel_values"].type(torch.float16)).latent_dist.sample().detach()
    # latents = latents.repeat(4, 1, 1, 1)

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]

    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = domain_adaptor(image_embeddings)

    inj_embedding = inj_embedding[:, 0:1, :]
    
    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]
    
    # print('*************************************')
    # print(example["input_ids"].shape)
    # print(example["input_ids"])
    # print(inj_embedding.shape)
    # print(placeholder_idx.detach())
    # print(encoder_hidden_states.shape)

    image_obj = F.interpolate(example["pixel_values_obj"], (224, 224), mode='bilinear')
    image_features_obj = image_encoder(image_obj, output_hidden_states=True)
    image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8],
                            image_features_obj[2][12], image_features_obj[2][16]]
    image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

    inj_embedding_local = image_transformer(image_embeddings_obj)
    mask = F.interpolate(example["pixel_values_seg"], (16, 16), mode='nearest')
    mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
    inj_embedding_local = inj_embedding_local * mask

    encoder_hidden_states = encoder_hidden_states.repeat(num_sample, 1, 1)

    inj_embedding_local = inj_embedding_local.repeat(num_sample, 1, 1)

    placeholder_idx = placeholder_idx.repeat(num_sample)

    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)

        # print(latent_model_input.shape)
        # print(encoder_hidden_states.shape)
        # print(inj_embedding_local.shape)
        # print(placeholder_idx.shape)

        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
                "LOCAL": inj_embedding_local,
                "LOCAL_INDEX": placeholder_idx.detach(),
                "LAMBDA": llambda
            }
        ).sample
        value_local_list.clear()
        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample
        value_local_list.clear()
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Load the tokenizer and add the placeholder token as a additional special token

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    domain_adaptor = Domain_Adaptor(input_dim=1024, output_dim=768)
    image_transformer = Image_Transformer(input_dim=1024, output_dim=768)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # replace the forward method of the crossattention to finetune the to_k and to_v layers
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            to_k_global.weight.data = _module.to_k.weight.data.clone()
            domain_adaptor.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            to_v_global.weight.data = _module.to_v.weight.data.clone()
            domain_adaptor.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            to_k_local = nn.Linear(shape[1], shape[0], bias=False)
            to_k_local.weight.data = _module.to_k.weight.data.clone()
            image_transformer.add_module(f'{_name.replace(".", "_")}_to_k', to_k_local)
            _module.add_module('to_k_local', to_k_local)

            to_v_local = nn.Linear(shape[1], shape[0], bias=False)
            to_v_local.weight.data = _module.to_v.weight.data.clone()
            image_transformer.add_module(f'{_name.replace(".", "_")}_to_v', to_v_local)
            _module.add_module('to_v_local', to_v_local)

            if args.domain_adaptor_path is None:
                _module.add_module('to_k_global', to_k_global)
                _module.add_module('to_v_global', to_v_global)

            if args.image_transformer_path is None:
                _module.add_module('to_k_local', to_k_local)
                _module.add_module('to_v_local', to_v_local)

    if args.domain_adaptor_path is not None:
        domain_adaptor.load_state_dict(torch.load(args.domain_adaptor_path, map_location='cpu'))
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                _module.add_module('to_k_global', getattr(domain_adaptor, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_global', getattr(domain_adaptor, f'{_name.replace(".", "_")}_to_v'))

    if args.image_transformer_path is not None:
        image_transformer.load_state_dict(torch.load(args.image_transformer_path, map_location='cpu'))
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                _module.add_module('to_k_local', getattr(image_transformer, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_local', getattr(image_transformer, f'{_name.replace(".", "_")}_to_v'))

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())
    unfreeze_params(image_transformer.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(image_transformer.parameters()),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = OpenImagesDatasetWithMask(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        set="test"
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    image_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        image_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    image_encoder.to(accelerator.device)
    text_encoder.to(accelerator.device)
    domain_adaptor.to(accelerator.device)
    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()
    image_encoder.eval()
    domain_adaptor.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("elite", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        image_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(image_transformer):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                placeholder_idx = batch["index"]
                image = F.interpolate(batch["pixel_values_clip"], (224, 224), mode='bilinear')
                image_obj = F.interpolate(batch["pixel_values_obj"], (224, 224), mode='bilinear')

                mask = F.interpolate(batch["pixel_values_seg"], (16, 16), mode='nearest')
                mask = mask[:, 0].reshape(mask.shape[0], -1, 1)

                image_features = image_encoder(image, output_hidden_states=True)
                image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
                image_embeddings = [emb.detach() for emb in image_embeddings]
                inj_embedding = domain_adaptor(image_embeddings)

                # only use the first word
                inj_embedding = inj_embedding[:, 0:1, :]

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder({'input_ids': batch["input_ids"],
                                                      "inj_embedding": inj_embedding,
                                                      "inj_index": placeholder_idx.detach()})[0]

                image_features_obj = image_encoder(image_obj, output_hidden_states=True)
                image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8], image_features_obj[2][12], image_features_obj[2][16]]
                image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

                inj_embedding_local = image_transformer(image_embeddings_obj)
                inj_embedding_local = inj_embedding_local * mask


                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states={
                    "CONTEXT_TENSOR": encoder_hidden_states,
                    "LOCAL": inj_embedding_local,
                    "LOCAL_INDEX": placeholder_idx.detach()
                }).sample

                mask_values = batch["mask_values"]
                loss_mle = F.mse_loss(noise_pred, noise, reduction="none")
                loss_mle = ((loss_mle*mask_values).sum([1, 2, 3])/mask_values.sum([1, 2, 3])).mean()

                loss_reg = 0
                for vvv in value_local_list:
                    loss_reg += torch.mean(torch.abs(vvv))
                loss_reg = loss_reg / len(value_local_list) * 0.0001

                loss = loss_mle + loss_reg

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(image_transformer.parameters(), 1)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                value_local_list.clear()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_progress(image_transformer, accelerator, args, global_step, 'image_transformer')
                    syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, domain_adaptor, image_transformer, vae, batch["pixel_values_clip"].device, 5)
                    input_images = [th2image(img) for img in batch["pixel_values"]]
                    clip_images = [th2image(img).resize((512, 512)) for img in batch["pixel_values_clip"]]
                    obj_images = [th2image(img).resize((512, 512)) for img in batch["pixel_values_obj"]]
                    input_masks = torch.cat([mask_values, mask_values, mask_values], dim=1)
                    input_masks = [th2image(img).resize((512, 512)) for img in input_masks]
                    obj_masks = [th2image(img).resize((512, 512)) for img in batch["pixel_values_seg"]]
                    img_list = []
                    for syn, input_img, input_mask, clip_image, obj_image, obj_mask in zip(syn_images, input_images, input_masks, clip_images, obj_images, obj_masks):
                        img_list.append(np.concatenate((np.array(syn), np.array(input_img), np.array(input_mask), np.array(clip_image), np.array(obj_image), np.array(obj_mask)), axis=1))
                    img_list = np.concatenate(img_list, axis=0)
                    Image.fromarray(img_list).save(os.path.join(args.output_dir, f"{str(global_step).zfill(5)}.jpg"))

            logs = {"loss_mle": loss_mle.detach().item(), "loss_reg": loss_reg.detach().item(),  "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_progress(image_transformer, accelerator, args, name='image_transformer')

    accelerator.end_training()


if __name__ == "__main__":
    main()