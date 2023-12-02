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


class Image_Transformer(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
    ):
        super(Image_Transformer, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:])
            hidden_states += (hidden_state.unsqueeze(0),)
        hidden_states = torch.cat(hidden_states, dim=0).mean(dim=0)
        return hidden_states

value_local_list = []

def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None):

    context = encoder_hidden_states
    hidden_states_local = hidden_states.clone()
    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]
    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    if context is not None:
        key = self.to_k_global(context_tensor)
        value = self.to_v_global(context_tensor)
    else:
        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)


    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    if context is not None and "LOCAL" in context:
        # Perform cross attention with the local context
        query_local = self.to_q(hidden_states_local)
        key_local = self.to_k_local(context["LOCAL"])
        value_local = self.to_v_local(context["LOCAL"])

        query_local = self.reshape_heads_to_batch_dim(query_local)
        key_local = self.reshape_heads_to_batch_dim(key_local)
        value_local = self.reshape_heads_to_batch_dim(value_local)

        attention_scores_local = torch.matmul(query_local, key_local.transpose(-1, -2))
        attention_scores_local = attention_scores_local * self.scale
        attention_probs_local = attention_scores_local.softmax(dim=-1)

        # To extract the attmap of learned [w]
        index_local = context["LOCAL_INDEX"]
        index_local = index_local.reshape(index_local.shape[0], 1).repeat((1, self.heads)).reshape(-1)
        attention_probs_clone = attention_probs.clone().permute((0, 2, 1))
        attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]), index_local]
        # Normalize the attention map
        attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

        if "LAMBDA" in context:
            _lambda = context["LAMBDA"]
        else:
            _lambda = 1
        # print(attention_probs_local.shape)
        # print(attention_probs_mask.shape)
        # print(_lambda)
        attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
        hidden_states += torch.matmul(attention_probs_local, value_local)
        value_local_list.append(value_local)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states