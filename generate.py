import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from utils.image_transformer import Image_Transformer, inj_forward_crossattention
from utils.domain_adaptor import Domain_Adaptor, inj_forward_text
from utils.functions import th2image
from train_image_transformer import validation
import torch.nn as nn
from datasets import CustomDatasetWithBG
import datetime
import matplotlib.pyplot as plt

from utils.functions import _pil_from_latents, pww_load_tools
 
def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--domain_adaptor_path",
        type=str,
        required=True,
        help="Path to pretrained domain adaptor network.",
    )

    parser.add_argument(
        "--image_transformer_path",
        type=str,
        required=True,
        help="Path to pretrained image transformer network.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='outputs',
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="S",
        help="A token to use as a placeholder for the concept.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a {}",
        help="Text prompt for customized genetation.",
    )

    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=True, help="A folder containing the testing data."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="object",
        help="Suffix of save directory.",
    )

    parser.add_argument(
        "--selected_data",
        type=int,
        default=-1,
        help="Data index. -1 for all.",
    )

    parser.add_argument(
        "--sigma",
        type=str,
        default="0.8",
        help="sigma for fuse the domain adaptor outputs and image transformer outputs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for testing.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    save_dir = os.path.join(args.output_dir, current_date_time)
    os.makedirs(save_dir, exist_ok=True)

    vae, unet, text_encoder, tokenizer, image_encoder, domain_adaptor, image_transformer, scheduler = pww_load_tools(
            "cuda:0",
            LMSDiscreteScheduler,
            diffusion_model_path=args.pretrained_model_name_or_path,
            domain_adaptor_path=args.domain_adaptor_path,
            image_transformer_path=args.image_transformer_path,
        )

    train_dataset = CustomDatasetWithBG(
        data_root=args.test_data_dir,
        tokenizer=tokenizer,
        size=512,
        placeholder_token=args.placeholder_token,
        template=args.prompt,
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    for step, batch in enumerate(train_dataloader):
        if args.selected_data > -1 and step != args.selected_data:
            continue
        batch["pixel_values"] = batch["pixel_values"].to("cuda:0")
        batch["pixel_values_clip"] = batch["pixel_values_clip"].to("cuda:0").half()
        batch["pixel_values_obj"] = batch["pixel_values_obj"].to("cuda:0").half()
        batch["pixel_values_seg"] = batch["pixel_values_seg"].to("cuda:0").half()
        batch["input_ids"] = batch["input_ids"].to("cuda:0")
        batch["index"] = batch["index"].to("cuda:0").long()
        print(step, batch['text'])
        syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, domain_adaptor, image_transformer, vae,
                                batch["pixel_values_clip"].device, 5,
                                seed=args.seed, llambda=float(args.sigma))

        # syn_images = [np.ndarray(shape=(3*512*512)).reshape(512, 512, 3)] * 8
        print(np.array(syn_images[0]).shape)
        
        concat = np.concatenate((np.array(syn_images[0]), 
                                 np.array(syn_images[1]), 
                                 np.array(syn_images[2]), 
                                 np.array(syn_images[3]), 
                                #  np.array(syn_images[4]), 
                                #  np.array(syn_images[5]), 
                                #  np.array(syn_images[6]), 
                                #  np.array(syn_images[7]), 
                                 th2image(batch["pixel_values"][0])), 
                                 axis=1)
        plt.imshow(concat)
        plt.show()
        Image.fromarray(concat).save(os.path.join(save_dir, f'{str(step).zfill(5)}_{str(args.seed).zfill(5)}.jpg'))
        save_text_dir = os.path.join(save_dir, batch['text'][0]+'_'+str(step))
        os.makedirs(save_text_dir, exist_ok=True)
        Image.fromarray(np.array(th2image(batch["pixel_values"][0]))).save(os.path.join(save_text_dir, 'ori.jpg'))
        for i in range(len(syn_images)):
            Image.fromarray(np.array(syn_images[i])).save(os.path.join(save_text_dir, str(i)+'_syn'+'.jpg'))

        