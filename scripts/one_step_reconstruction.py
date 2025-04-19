# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs the main decomposition algorithm."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-function-docstring,missing-class-docstring
import argparse
import glob
import logging
import math
import os
from pathlib import Path
import random
import cv2
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch.backends.cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor


from accelerate.logging import get_logger
from transformers import T5Tokenizer, T5EncoderModel

import diffusers

from diffusers.optimization import get_scheduler
import numpy as np
from packaging import version
import PIL
from PIL import Image
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
from diffusers.utils import export_to_video
import sys


from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
# from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict
from models.model_zoo import LBsimilarity




# === Pil version fitting for interpolation (in case of resizing)===
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {"linear": PIL.Image.Resampling.BILINEAR,
                        "bilinear": PIL.Image.Resampling.BILINEAR,
                        "bicubic": PIL.Image.Resampling.BICUBIC,
                        "lanczos": PIL.Image.Resampling.LANCZOS,
                        "nearest": PIL.Image.Resampling.NEAREST,}
else:
    PIL_INTERPOLATION = {"linear": PIL.Image.LINEAR,
                        "bilinear": PIL.Image.BILINEAR,
                        "bicubic": PIL.Image.BICUBIC,
                        "lanczos": PIL.Image.LANCZOS,
                        "nearest": PIL.Image.NEAREST,}


logger = get_logger(__name__)
template = ["A dog {}"]


def gpu_clean_and_status(stage, print_gpu_status=False):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 
    if print_gpu_status:
        print("GPUs status for:", stage)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {torch.cuda.get_device_name(i)} | Allocated: {torch.cuda.memory_allocated(i)/(1024**3):.2f}GB | Cached: {torch.cuda.memory_reserved(i)/(1024**3):.2f}GB | Total: {props.total_memory/(1024**3):.2f}GB")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default=None,
                        required=True,
                        help=("Path to pretrained model or model identifier from huggingface.co/models."),)
    parser.add_argument("--revision",
                        type=str,
                        default=None,
                        required=False,
                        help=("Revision of pretrained model identifier from huggingface.co/models."),)
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to be explained.")
    parser.add_argument("--train_data_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="A folder containing the training data.",)
    parser.add_argument("--validation_data_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="A folder containing the validation data.",)
    parser.add_argument("--placeholder_token",
                        type=str,
                        default=None,
                        required=True,
                        help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--concept",
                        type=str,
                        default=None,
                        required=True,
                        help="The concept to explain.",)
    parser.add_argument("--repeats",
                        type=int,
                        default=1,
                        help="How many times to repeat the training data.",)
    parser.add_argument("--output_dir",
                        type=str,
                        default="output",
                        help=("The output directory where the model predictions and checkpoints will be written."),)
    parser.add_argument("--seed", type=int, default=42, help="A seed for train videos.")
    parser.add_argument("--validation_seed",
                        type=int,
                        default=42,
                        help="A seed for validation videos.",)
    parser.add_argument("--resolution",
                        type=int,
                        default=(480,720), 
                        help=("The resolution for input videos, all the videos in the"
                                " train/validation dataset will be resized to this resolution"),)
    parser.add_argument("--center_crop",
                        action="store_true",
                        help="Whether to center crop videos before resizing to resolution.",)
    parser.add_argument("--remove_concept_tokens",
                        action="store_true",
                        default=False,
                        help="Whether to remove the concept token from the dictionary.",)
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=2,
                        help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=10000,
                        help=("Total number of training steps to perform.  If provided, overrides"
                                " num_train_epochs."),)
    parser.add_argument("--dictionary_size",
                        type=int,
                        default=5000,
                        help="Number of top tokens to consider as dictionary.",)
    parser.add_argument("--num_explanation_tokens",
                        type=int,
                        default=50,
                        help="Number of words to produce as explanation.",)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help=("Number of updates steps to accumulate before performing a"
                                " backward/update pass."),)
    parser.add_argument("--gradient_checkpointing",
                        action="store_true", 
                        help=("Whether or not to use gradient checkpointing to save memory at the"
                                " expense of slower backward pass."),)
    parser.add_argument("--print_gpu_status",
                        type=bool, 
                        default=False,
                        help=("Whether or not to use print gpu status"),)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--sparsity_coeff",
                        type=float,
                        default=0.001,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--motion_coeff",
                        type=float,
                        default=5,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--scale_lr",
                        action="store_true",
                        default=False,
                        help=("Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."),)
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="constant",
                        help=('The scheduler type to use. Choose between ["linear", "cosine",'
                                ' "cosine_with_restarts", "polynomial", "constant",'
                                ' "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps",
                        type=int,
                        default=500,
                        help="Number of steps for the warmup in the lr scheduler.",)
    parser.add_argument("--dataloader_num_workers",
                        type=int,
                        default=0,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the"
                            " data will be loaded in the main process."),)
    parser.add_argument("--adam_beta1",
                        type=float,
                        default=0.9,
                        help="The beta1 parameter for the Adam optimizer.",)
    parser.add_argument("--adam_beta2",
                        type=float,
                        default=0.999,
                        help="The beta2 parameter for the Adam optimizer.",)
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.",)
    parser.add_argument("--adam_epsilon",
                        type=float,
                        default=1e-08,
                        help="Epsilon value for the Adam optimizer",)
    parser.add_argument("--logging_dir",
                        type=str,
                        default="logs",
                        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
                            " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    parser.add_argument("--mixed_precision",
                        type=str,
                        default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help=("Whether to use mixed precision. Choose"
                                "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                                "and an Nvidia Ampere GPU."),)
    parser.add_argument("--allow_tf32",
                        action="store_true",
                        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up"
                            " training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
    parser.add_argument("--report_to",
                        type=str,
                        default="tensorboard",
                        help=("The integration to report the results and logs to. Supported"
                                ' platforms are `"tensorboard"` (default)'),)
    parser.add_argument("--validation_prompt",
                        type=str,
                        default=None,
                        help=("A prompt that is used during validation to verify that the model is"
                            " learning."),)
    parser.add_argument("--num_validation_videos",
                        type=int,
                        default=5,
                        help=("Number of videos that should be generated during validation with"
                                " `validation_prompt`."),)
    parser.add_argument("--validation_steps",
      type=int,
      default=50,
      help=(
          "Run validation every X epochs. Validation consists of running the"
          " prompt `args.validation_prompt` multiple times:"
          " `args.num_validation_videos` and logging the videos."),)
    parser.add_argument("--path_to_encoder_embeddings", 
                        type=str,
                        default="./LB_text_encoding.pt",
                        help="Path to the saved embeddings matrix of the text encoder",)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank",)
    parser.add_argument("--enable_xformers_memory_efficient_attention",
                        action="store_true",
                        help="Whether or not to use xformers.",)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


class ConceptDataset(Dataset):

    def __init__(self,
                data_root,
                tokenizer,
                width=720,
                height=480,
                repeats=100,
                interpolation="bicubic",
                flip_p=0.5,
                split="train",
                placeholder_token="*",
                center_crop=False,
                original_prompt="A dog walking"):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.original_prompt=original_prompt

        self.videos_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_videos = len(self.videos_paths)
        self._length = self.num_videos

        if split == "train":
            self._length = self.num_videos * repeats 

        self.interpolation = {"linear": PIL_INTERPOLATION["linear"],
                                "bilinear": PIL_INTERPOLATION["bilinear"],
                                "bicubic": PIL_INTERPOLATION["bicubic"],
                                "lanczos": PIL_INTERPOLATION["lanczos"],}[interpolation]

        self.templates = template
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        
        video_path = self.videos_paths[i % self.num_videos]
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not read frames from video {video_path}")
        
        # Convert frames to tensor
        processed_frames = []
        for frame in frames:
            # Convert to PIL for consistent processing
            image = Image.fromarray(frame)
            
            # Process each frame as before
            img = np.array(image).astype(np.uint8)

            if self.center_crop: 
                raise NotImplementedError("Center cropping is not implemented yet for video.") 

            image = Image.fromarray(img)
            image = image.resize((self.width, self.height), resample=self.interpolation)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32) # Normalize to [-1, 1]
            processed_frames.append(image)

        # Stack frames into a single tensor
        video_tensor = torch.from_numpy(np.stack(processed_frames)).permute(3, 0, 1, 2) # TO [C, F, H, W]

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(text,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.tokenizer.model_max_length, 
                                                return_tensors="pt",).input_ids[0]
        example["pixel_values"] = video_tensor
        example['original_prompt'] = self.tokenizer(self.original_prompt,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.tokenizer.model_max_length, 
                                                return_tensors="pt",).input_ids[0]

        return example


def get_language_bind_encodings(data_root, device, batch_size=100):
    # Define the modalities to use with LanguageBind for video
    clip_type = {'video': 'LanguageBind_Video_FT'}
    
    # Initialize the LanguageBind model with the given modalities and cache directory, and move it to the GPU (device)
    language_bind_model = LanguageBind(clip_type=clip_type, cache_dir='/home/joberant/NLP_2425a/baralon1/cache').to(device)
    
    # Create a dictionary of transformation functions for each modality.
    # Each transform is created using the corresponding configuration from the model.
    modality_transform = {c: transform_dict[c](language_bind_model.modality_config[c]) for c in clip_type.keys()}
    
    # Ensure the model set to evaluation mode (disables dropout, etc.)
    language_bind_model.eval()
    
    # Get a sorted list of all .mp4 video file paths from the provided data directory.
    video_paths = sorted(glob.glob(f"{data_root}/*.mp4"))
    
    # Process videos in batches to avoid OOM errors.
    encoding_list = []
    with torch.no_grad():
        for i in range(0, len(video_paths), batch_size):
            # Get the batch of video paths.
            batch_paths = video_paths[i: i + batch_size]
            # Process the current batch using the video modality transform and move it to GPU.
            inputs_batch = {'video': to_device(modality_transform['video'](batch_paths), device)}
            # Get the encodings for the current batch.
            batch_encodings = language_bind_model(inputs_batch)['video']
            # Normalize each encoding vector (each row) to have unit norm.
            batch_encodings /= batch_encodings.norm(dim=-1, keepdim=True)
            encoding_list.append(batch_encodings)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
    
    # Concatenate all batch encodings into a single tensor.
    target_video_encodings = torch.cat(encoding_list, dim=0)
    
    # Clean up: move the model to CPU, delete it, and free GPU cache.
    language_bind_model = language_bind_model.to("cpu")
    del language_bind_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return target_video_encodings

def get_dictionary_indices(args, text_encoder, tokenizer, dictionary_size, device):
    """
    Finds and prints the top-N most similar tokens in embedding space to the concept string.
    Returns token indices of the top matches.
    """
    # Get embedding matrix
    embedding_weight = text_encoder.get_input_embeddings().weight  # [vocab_size, dim]
    embedding_weight = F.normalize(embedding_weight, dim=1)  # normalize for cosine similarity

    # Encode the concept and get its embedding
    concept_inputs = tokenizer([args.concept], return_tensors="pt", padding=True).to(device)
    concept_token_ids = concept_inputs["input_ids"][0]
    concept_embeddings = embedding_weight[concept_token_ids]  # [num_tokens, dim]
    concept_embedding = concept_embeddings.mean(dim=0, keepdim=True)  # [1, dim]
    concept_embedding = F.normalize(concept_embedding, dim=1)

    # Compute cosine similarity
    cosine_sim = torch.matmul(concept_embedding, embedding_weight.T).squeeze(0)  # [vocab_size]
    # === Zero out similarities above threshold ===
    # cosine_sim[cosine_sim > 0.21] = 0

    top_sim, top_idx = torch.topk(cosine_sim, k=dictionary_size)

    # Print top matches
    for rank, (sim, idx) in enumerate(zip(top_sim, top_idx)):
        if rank < 10:
            word = tokenizer.decode([int(idx)])
            print(f"{rank}: Word: -{word}- Token id:{int(idx)} Similarity: {sim.item():.4f}")

    return top_idx

# def get_dictionary_indices(args, target_video_encodings, tokenizer, dictionary_size, device):
#     normalized_text_encodings = torch.load(args.path_to_encoder_embeddings)

#     if args.remove_concept_tokens:
#         # === Remove concept tokens ===
#         clip_type = {'video': 'LanguageBind_Video_FT'}
#         language_bind_model = LanguageBind(clip_type=clip_type, cache_dir='/home/joberant/NLP_2425a/baralon1/cache').to(device)
#         language_bind_model.eval()
#         lb_concept_inputs = tokenizer([args.concept], padding=True, return_tensors="pt").to(device)

#         inputs = {}
#         inputs['language'] = lb_concept_inputs
#         lb_concept_features = language_bind_model(inputs) 

#         concept_words_similarity = torch.cosine_similarity(lb_concept_features['language'], normalized_text_encodings, axis=1)
#         # Print top words and their similarity values
#         topk = 30000 # You can change this number as desired
#         top_sim, top_idx = torch.topk(concept_words_similarity, topk)
#         # print("Top words and their cosine similarities:")
#         for counter, (sim, idx) in enumerate(zip(top_sim, top_idx)):
#             # Decode the token index; note that tokenizer.decode expects a list or tensor.
#             word = tokenizer.decode([int(idx)])
#             print(f"{counter}: Word: -{word}- Token id:{int(idx)} Similarity: {sim.item():.4f}")        
#         # similar_words = (np.array(concept_words_similarity.detach().cpu()) > 0.9).nonzero()[0] # TODO: Fix the threshold
#         # similar_words = [10681, 1482,10801, 12539, 29873, 6412, 3214, 19525, 13521, 24063]
#         # Zero-out similar words
#         # for i in similar_words:
#         #     print("removing similar word", tokenizer.decode(i))
#         #     cosine_similarities[0, i] = 0
#         language_bind_model = language_bind_model.to("cpu")
#         del language_bind_model
#         import gc
#         gc.collect()
#         torch.cuda.empty_cache()

#     # calculate cosine similarities for the average video
#     mean_target_video = target_video_encodings.mean(dim=0).reshape(1, -1)
#     cosine_similarities = torch.cosine_similarity(mean_target_video, normalized_text_encodings).reshape(1, -1)

#     # Average similarities across the videos
#     mean_cosine = torch.mean(cosine_similarities, dim=0)
#     _, sorted_indices = torch.sort(mean_cosine, descending=True)

#     # return the indices of the words to consider in the dictionary
#     return sorted_indices[:dictionary_size]


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.flatten().abs()


def sorted_l1_penalty(alphas):
    values, _ = torch.sort(torch.abs(alphas), descending=True)
    weights = torch.arange(1, len(values) + 1, device=alphas.device).float()
    weights = weights / weights.sum()
    return (values * weights).sum()


def main(): 
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppress verbose logging output 
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    # Load the pipe line
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="balanced")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing() # Enables channel-wise slicing - optimize memory usage
    pipe.vae.enable_tiling() # splits the image into smaller tiles and processes each tile separately - optimize memory usage
    pipe.set_progress_bar_config(disable=True)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
        pipe.vae.enable_gradient_checkpointing()

    weight_dtype = torch.bfloat16 # Check if needed


    # Add the placeholder token in tokenizer
    num_added_tokens = pipe.tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token}."
                            " Please pass a different `placeholder_token` that is not already in"
                            " the tokenizer.")

    placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(args.placeholder_token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer)) # Resize the token embeddings 

    # Freeze  all parameters in pipeline
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.encoder.requires_grad_(False)  # Freeze the entire encoder
    for param in pipe.text_encoder.encoder.block.parameters():
        param.requires_grad = False  # Freeze all encoder blocks
    pipe.text_encoder.encoder.final_layer_norm.requires_grad_(False)  # Freeze final layer norm (if exists)
    pipe.text_encoder.shared.requires_grad_(False)  # Freeze input token embeddings
    pipe.text_encoder.get_input_embeddings().weight.requires_grad_(False)


    # Initialize net
    net = Net().to(dtype=weight_dtype, device='cuda')
    # Print a sanity check
    for name, param in net.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    # Initialize the optimizer - only optimize the embeddings
    optimizer = torch.optim.AdamW(net.parameters(),
                                    lr=args.learning_rate,
                                    betas=(args.adam_beta1, args.adam_beta2),
                                    weight_decay=args.adam_weight_decay,
                                    eps=args.adam_epsilon,)

    # Dataset and DataLoaders creation:
    train_dataset = ConceptDataset(data_root=args.train_data_dir,
                                    tokenizer=pipe.tokenizer,
                                    width=args.resolution[1],
                                    height=args.resolution[0],
                                    placeholder_token=args.placeholder_token,
                                    repeats=args.repeats,
                                    center_crop=args.center_crop,
                                    split="train",
                                    original_prompt=args.prompt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.dataloader_num_workers,)


    lr_scheduler = get_scheduler(args.lr_scheduler,
                                    optimizer=optimizer,
                                    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # Print the arguments
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print("***** Arguments *****")
    for arg, value in vars(args).items():
        print(f"  {arg} = {value}")

    # keep original embeddings as reference
    orig_embeds_params = (pipe.text_encoder.get_input_embeddings().weight.data.clone())

    # Extract the average norm
    norms = [i.norm().item() for i in orig_embeds_params]
    avg_norm = np.mean(norms)

    # Get dictionary
    num_tokens = args.dictionary_size # The number of tokens in the game

    # Get video encoding for train and val
    target_video_encodings = get_language_bind_encodings(args.train_data_dir, device)
    validation_video_encodings = get_language_bind_encodings(args.validation_data_dir, device)

    # Find the most similar tokens to the video encodings
    dictionary_indices = get_dictionary_indices(args, pipe.text_encoder, pipe.tokenizer, num_tokens, device).to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dictionary_indices, f"{args.output_dir}/dictionary.pt")

    # Make sure we don't track back gradients for the new tensors
    target_video_encodings.detach_().requires_grad_(False)
    validation_video_encodings.detach_().requires_grad_(False)
    dictionary = orig_embeds_params[dictionary_indices] # Make sure we didn't change anything

    best_validation_score = 0
    best_alphas = None
    best_epoch = None
    best_words = None
    best_loss = 100000
    best_train_words = None
    best_train_alphas = None
    validation_model = LBsimilarity() 

    # Debug a normal inference
    # with torch.no_grad():
    #     generator = torch.Generator(device).manual_seed(args.validation_seed)
    #     video = pipe(
    #                 width=720,
    #                 height=480,
    #                 prompt="A dog walking",  
    #                 num_videos_per_prompt=1,
    #                 num_inference_steps=50,
    #                 num_frames=81,
    #                 use_dynamic_cfg=True,
    #                 guidance_scale=6.0,
    #                 generator=generator
    #                 ).frames[0]

    for epoch in range(args.num_train_epochs):
        net.train()
        for batch_num, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            print(f"Epoch: {epoch} Batch_num: {batch_num}")
            gpu_clean_and_status('New batch')
            pipe.text_encoder.get_input_embeddings().weight.detach_().requires_grad_(False)

            # Calculate current embeddings
            alphas = net(dictionary) # Pass the embeddings of the encoder throw a net - each get a single number
            _, sorted_indices = torch.sort(alphas.abs(), descending=True) # Abs to get the most important token even negative
            print_words = args.num_explanation_tokens

            # Make the embbedings - take the full dictionary and make a new embedding then apply the known norm
            embedding = torch.matmul(alphas[sorted_indices], dictionary[sorted_indices]) # Make an new embedding to assign to the new token
            embedding = torch.mul(embedding, 1 / embedding.norm())
            embedding = torch.mul(embedding, avg_norm)
            
            # Update embedding
            pipe.text_encoder.get_input_embeddings().weight[placeholder_token_id] = embedding

            # Print the top words for debuging
            top_words = [pipe.tokenizer.decode(dictionary_indices[sorted_indices[i]]) for i in range(print_words)]
            print(" | ".join([f"{i}: {alphas[sorted_indices[i]]} * <{top_words[i]}>" for i in range(print_words)]))            

            
            gpu_clean_and_status('After Net')

            # Get the text embedding for conditioning
            # encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0] # The 0 is for getting the last hidden state
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt([args.validation_prompt]*args.train_batch_size,
                                                                        None,
                                                                        True,
                                                                        num_videos_per_prompt=1,
                                                                        prompt_embeds=None,
                                                                        negative_prompt_embeds=None,
                                                                        max_sequence_length=226,
                                                                        device='cuda',)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            
            gpu_clean_and_status('After Encoder')

            # Sample a random timestep for each video
            timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, 50, 'cuda', None)
            random_indices = torch.randint(0, len(timesteps), (1,), device=device)
            timesteps = timesteps[random_indices]
            
            # Convert videos to latent space
            latents = (pipe.vae.encode(batch["pixel_values"].to(device).contiguous().to(dtype=weight_dtype))
                        .latent_dist.sample().detach())

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
        
            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Create rotary embeddings
            image_rotary_emb = pipe._prepare_rotary_positional_embeddings(args.resolution[0], args.resolution[1], noisy_latents.size(2), device)

            # Prepare noisy latents for uncondtion pred
            noisy_latents = torch.cat([latents] * 2).permute(0, 2, 1, 3, 4)
            noisy_latents = pipe.scheduler.scale_model_input(noisy_latents, timesteps) # added from the pipeline code
            
            # Prepare timesteps to tranformer forward
            timesteps = timesteps.expand(noisy_latents.shape[0])

            gpu_clean_and_status("Before Transformer")

            # Predict the noise residual
            model_pred = pipe.transformer(hidden_states=noisy_latents, 
                                            encoder_hidden_states=prompt_embeds, 
                                            timestep=timesteps,
                                            image_rotary_emb=image_rotary_emb)[0].permute(0, 2, 1, 3, 4) 
            
            # Predict the noise residual
            # model_pred = pipe.transformer(hidden_states=noisy_latents, 
            #                                 encoder_hidden_states=prompt_embeds, 
            #                                 timestep=timesteps,
            #                                 image_rotary_emb=image_rotary_emb).sample.permute(0, 2, 1, 3, 4) 
            

            model_pred = model_pred.float()
            
            # Handle guidance with uncoditioned
            guidance_scale = 1 + 6 * (
                        (1 - math.cos(math.pi * ((num_inference_steps - timesteps[0].item()) / num_inference_steps) ** 5.0)) / 2)
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            gpu_clean_and_status('After Transformer')

            # Get the target for loss depending on the prediction type (classic ddpm vs flow matching)
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction": # Cog using flow matching
                target = pipe.scheduler.get_velocity(latents, noise, timesteps[:args.train_batch_size])
            else:
                raise ValueError("Unknown prediction type"
                f" {pipe.scheduler.config.prediction_type}")

            mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            ############## START CODE FOR MULTI‑SCALE MOTION LOSS (EXPLICIT) ################

            # Permute to [B, T, H, W, C]
            pred_latents = model_pred.permute(0, 2, 3, 4, 1)
            orig_latents = latents.permute(0, 2, 3, 4, 1)

            # 1‑frame motion
            pred_diff1 = torch.abs(pred_latents[:, 1:] - pred_latents[:, :-1])   # [B, T-1, H, W, C]
            orig_diff1 = torch.abs(orig_latents[:, 1:] - orig_latents[:, :-1])   # [B, T-1, H, W, C]
            loss1     = F.mse_loss(pred_diff1.float(), orig_diff1.float(), reduction="mean")

            # 4‑frame motion
            pred_diff4 = torch.abs(pred_latents[:, 4:] - pred_latents[:, :-4])   # [B, T-4, H, W, C]
            orig_diff4 = torch.abs(orig_latents[:, 4:] - orig_latents[:, :-4])   # [B, T-4, H, W, C]
            loss4     = F.mse_loss(pred_diff4.float(), orig_diff4.float(), reduction="mean")

            # 8‑frame motion
            pred_diff8 = torch.abs(pred_latents[:, 8:] - pred_latents[:, :-8])   # [B, T-8, H, W, C]
            orig_diff8 = torch.abs(orig_latents[:, 8:] - orig_latents[:, :-8])   # [B, T-8, H, W, C]
            loss8     = F.mse_loss(pred_diff8.float(), orig_diff8.float(), reduction="mean")

            # Equal‑weight average across scales
            motion_loss = (loss1 + loss4 + loss8)

            ############## END CODE FOR MULTI‑SCALE MOTION LOSS #############################

            sparsity_loss = torch.norm(alphas, p=1) # l1 loss
            # sparsity_loss = sorted_l1_penalty(alphas)

            # calculate final loss
            loss = mse_loss + args.sparsity_coeff * sparsity_loss + args.motion_coeff * motion_loss

            print(f"Total loss: {loss.item():.3f} = "
                f"MSE Loss: {mse_loss:.3f}, "
                f"{args.sparsity_coeff:.3f} * Sparsity Loss: {sparsity_loss.item():.3f}, "
                f"{args.motion_coeff:.3f} * Motion Loss: {motion_loss:.3f}")
            top_indices = [sorted_indices[i].item() for i in range(args.num_explanation_tokens)]
            top_embedding = torch.matmul(alphas[top_indices], dictionary[top_indices]) # Now take only the top embbedings
            top_embedding = torch.mul(top_embedding, 1 / top_embedding.norm())
            top_embedding = torch.mul(top_embedding, avg_norm)

            if loss < best_loss:
                best_train_words = top_words
                best_train_alphas = alphas
                best_train_top_embedding = top_embedding
                best_loss = loss
                print('Best currently')
            

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.arange(len(pipe.tokenizer)) != placeholder_token_id
            with torch.no_grad():
                pipe.text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
            gpu_clean_and_status("Done with batch")

        if (args.validation_prompt):
            with torch.no_grad():
                validation_dir = f"{args.output_dir}/validation/epoch_{epoch}"
                validation_dir = Path(validation_dir)
                validation_dir.mkdir(exist_ok=True, parents=True)
                for name, embedding in [("latest", top_embedding), ("best", best_train_top_embedding)]:
                    pipe.text_encoder.get_input_embeddings().weight[placeholder_token_id] = embedding
                    print("Running validation... \n Generating"
                        f" {args.num_validation_videos} videos with prompt:"
                        f" {args.validation_prompt} using {name} embeddings")

                    # run inference
                    generator = torch.Generator(device).manual_seed(args.validation_seed) # Create a deterministic random number generator 
                    probabilities = []
                    for i in range(args.num_validation_videos):
                        video = pipe(
                                width=720,
                                height=480,
                                prompt=args.validation_prompt,  
                                num_videos_per_prompt=1,
                                num_inference_steps=50,
                                num_frames=81,
                                use_dynamic_cfg=True,
                                guidance_scale=6.0,
                                generator=generator
                                ).frames[0]
                        video_path = f"{validation_dir}/{name}_{i}.mp4"
                        print(f"Video generated {video_path}")
                        export_to_video(video, video_path, fps=16)
                        
                        probability = validation_model.get_probability(video_path, target_videos=validation_video_encodings[i : i + 1])
                        # TODO: Check for mean instead of choosing
                        probabilities.append(probability.item())

                    validation_probability = np.mean(probabilities)
                    print("validation probability: ", validation_probability)

                    if validation_probability > best_validation_score:
                        print("replacing best alphas")
                        best_validation_score = validation_probability
                        best_alphas = alphas.detach()
                        best_words = top_words
                        best_epoch = epoch


                    print("saving alphas from step: ", epoch)
                    torch.save(alphas, f"{args.output_dir}/{epoch}_alphas.pt")
            print(f"saving best alphas from validation epoch {best_epoch}, words = ", best_words)
            torch.save(best_alphas, f"{args.output_dir}/best_alphas.pt")


if __name__ == "__main__":
  main()
