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

from accelerate.logging import get_logger
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


def gpu_status(stage):
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
                        default=100,
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
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=1000,
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
                        action="store_true", # TODO: Fix in code to use this
                        help=("Whether or not to use gradient checkpointing to save memory at the"
                                " expense of slower backward pass."),)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--sparsity_coeff",
                        type=float,
                        default=0.001,
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


def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents # TODO: Find the proper unscaling to replace 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1) # TODO: Force from [-1,1] to [0,1] - Find out if needed
    image = image.permute(0, 2, 3, 1) # (B, C, H, W) > (B, H, W, C)
    return image


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
                center_crop=False,):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

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

            if self.center_crop: # TODO: Shouldn't used, consider removing
                crop = min(img.shape[0], img.shape[1])
                h, w = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2 : (h + crop) // 2, 
                            (w - crop) // 2 : (w + crop) // 2]

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
                                                max_length=self.tokenizer.model_max_length, # TODO: Ensure it's 226
                                                return_tensors="pt",).input_ids[0]
        print("TODO: Ensure it's 226", self.tokenizer.model_max_length)
        example["pixel_values"] = video_tensor

        return example


def get_language_bind_encodings(data_root, batch_size=100):
    # Define the modalities to use with LanguageBind for video
    clip_type = {'video': 'LanguageBind_Video_FT'}
    
    # Initialize the LanguageBind model with the given modalities and cache directory,
    # and move it to the GPU ("cuda")
    language_bind_model = LanguageBind(clip_type=clip_type, cache_dir='/home/joberant/NLP_2425a/baralon1/cache').to("cuda")
    
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
            gpu_status(f"Processing from video {i}")
            # Get the batch of video paths.
            batch_paths = video_paths[i: i + batch_size]
            # Process the current batch using the video modality transform and move it to GPU.
            inputs_batch = {'video': to_device(modality_transform['video'](batch_paths), "cuda")}
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


def get_dictionary_indices(args, target_video_encodings, tokenizer, dictionary_size):
    

    normalized_text_encodings = torch.load(args.path_to_encoder_embeddings)

    # calculate cosine similarities for the average video
    mean_target_video = target_video_encodings.mean(dim=0).reshape(1, -1)
    cosine_similarities = torch.cosine_similarity(mean_target_video, normalized_text_encodings).reshape(1, -1)

    if args.remove_concept_tokens:
        # === Remove concept tokens ===
        clip_type = {'video': 'LanguageBind_Video_FT'}
        language_bind_model = LanguageBind(clip_type=clip_type, cache_dir='/home/ai_center/ai_users/arielshaulov/conceptor/cache').to("cuda")
        language_bind_model.eval()
        lb_concept_inputs = tokenizer([args.concept], padding=True, return_tensors="pt").to("cuda")

        inputs = {}
        inputs['language'] = lb_concept_inputs
        lb_concept_features = language_bind_model(inputs) 

        concept_words_similarity = torch.cosine_similarity(lb_concept_features['language'], normalized_text_encodings, axis=1)
        similar_words = (np.array(concept_words_similarity.detach().cpu()) > 0.9).nonzero()[0] # TODO: Make sure it's the right threshold
        # Zero-out similar words
        for i in similar_words:
            print("removing similar word", tokenizer.decode(i))
            cosine_similarities[0, i] = 0
        language_bind_model = language_bind_model.to("cpu")
        del language_bind_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Average similarities across the videos
    mean_cosine = torch.mean(cosine_similarities, dim=0)
    _, sorted_indices = torch.sort(mean_cosine, descending=True)

    # return the indices of the words to consider in the dictionary
    return sorted_indices[:dictionary_size]


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.flatten().abs()


def main(): 
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir) # TODO: Check if used

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,)
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="balanced")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.transformer.enable_gradient_checkpointing()

    # Add the placeholder token in tokenizer
    num_added_tokens = pipe.tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {args.placeholder_token}."
                            " Please pass a different `placeholder_token` that is not already in"
                            " the tokenizer.")

    placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(args.placeholder_token)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # Freeze vae and transformer
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    pipe.text_encoder.encoder.requires_grad_(False)  # Freeze the entire encoder
    for param in pipe.text_encoder.encoder.block.parameters():
        param.requires_grad = False  # Freeze all encoder blocks

    pipe.text_encoder.encoder.final_layer_norm.requires_grad_(False)  # Freeze final layer norm (if exists)
    pipe.text_encoder.shared.requires_grad_(False)  # Freeze input token embeddings


    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate
                                * args.gradient_accumulation_steps
                                * args.train_batch_size)

    # initialize nn
    net = Net()
    net.to(torch.bfloat16)

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
                                    split="train",)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.dataloader_num_workers,)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) # TODO: Understand why it's needed
  
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(args.lr_scheduler,
                                    optimizer=optimizer,
                                    num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                                    num_training_steps=args.max_train_steps
                                    * args.gradient_accumulation_steps,)


    weight_dtype = torch.bfloat16 # Check if needed

    # Move vae and transformer to device and cast to weight_dtype
    net.to('cuda')

    # We need to recalculate our total training steps as the size of the training
    # dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (args.train_batch_size * args.gradient_accumulation_steps)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    # keep original embeddings as reference
    orig_embeds_params = (
        pipe.text_encoder
        .get_input_embeddings()
        .weight.data.clone())

    norms = [i.norm().item() for i in orig_embeds_params]
    avg_norm = np.mean(norms)
    pipe.text_encoder.get_input_embeddings().weight.requires_grad_(False)

    # get dictionary
    num_tokens = args.dictionary_size
    target_video_encodings = get_language_bind_encodings(args.train_data_dir)
    validation_video_encodings = get_language_bind_encodings(args.validation_data_dir)

    dictionary_indices = get_dictionary_indices(args, target_video_encodings, pipe.tokenizer, num_tokens).to("cuda")

    print("Saving dictionary")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dictionary_indices, f"{args.output_dir}/dictionary.pt")

    target_video_encodings.detach_().requires_grad_(False)
    validation_video_encodings.detach_().requires_grad_(False)
    dictionary = orig_embeds_params[dictionary_indices]

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    gpu_status('After loading pipe to device')

    pipe.set_progress_bar_config(disable=True)

    best_validation_score = 0
    best_alphas = None
    best_epoch = None
    best_words = None
    validation_model = LBsimilarity() 
    num_words = args.dictionary_size


    torch.cuda.empty_cache()

    for epoch in range(args.num_train_epochs):
        net.train()
        for batch_num, batch in enumerate(train_dataloader):
            gpu_status('new batch!')
            pipe.text_encoder.get_input_embeddings().weight.detach_().requires_grad_(False)

            # calculate current embeddings
            token_embeds = pipe.text_encoder.get_input_embeddings().weight
            alphas = net(dictionary) # Pass the embeddings of the encoder throw a net - each get a single number
            _, sorted_indices = torch.sort(alphas.abs(), descending=True) # TODO: Understand why abs?
            print_words = min(50, args.num_explanation_tokens)

            word_indices = sorted_indices[:num_words]
            embedding = torch.matmul(alphas[word_indices], dictionary[word_indices]) # Make an new embedding to assign to the new token
            embedding = torch.mul(embedding, 1 / embedding.norm())
            embedding = torch.mul(embedding, avg_norm)

            top_words = [pipe.tokenizer.decode(dictionary_indices[sorted_indices[i]]) for i in range(print_words)]
            print("top words: ", top_words)
            print("alphas: ", alphas[sorted_indices[:print_words]])

            token_embeds[placeholder_token_id] = embedding
            pipe.text_encoder.get_input_embeddings().weight.requires_grad_(True) # TODO: Why it's require grad? don't we need just the alpha?
            gpu_status('After Net')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            gpu_status('After Net clean')
            # Convert videos to latent space
            latents = (pipe.vae.encode(batch["pixel_values"].to("cuda").contiguous().to(dtype=weight_dtype))
                        .latent_dist.sample()
                        .detach())
            # latents = latents * vae.config.scaling_factor
            latents = latents * 0.18215 # TODO: understand why we don't use decode_latents function

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each video
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device,)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each
            # timestep (this is the forward diffusion process)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps).permute(0, 2, 1, 3, 4)
            gpu_status('After Noise')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            gpu_status('After noise clean')

            # Get the text embedding for conditioning
            encoder_hidden_states = pipe.text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
            gpu_status('After Encoder')


            # Predict the noise residual    
            model_pred = pipe.transformer(noisy_latents, 
                                            encoder_hidden_states, 
                                            timesteps).sample.permute(0, 2, 1, 3, 4) 
            gpu_status('After transformer')

            # Get the target for loss depending on the prediction type
            # TODO: Understand what is this section
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError("Unknown prediction type"
                f" {pipe.scheduler.config.prediction_type}")

            mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            top_indices = [sorted_indices[i].item() for i in range(args.num_explanation_tokens)]
            top_embedding = torch.matmul(alphas[top_indices], dictionary[top_indices])
            sparsity_loss = 1 - torch.cosine_similarity(top_embedding.reshape(1, -1), embedding.reshape(1, -1))

            # calculate final loss
            loss = mse_loss + args.sparsity_coeff * sparsity_loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.arange(len(pipe.tokenizer)) != placeholder_token_id
            with torch.no_grad():
                pipe.text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
            print(f"MSE Loss: {mse_loss}, Sparsity Loss: {sparsity_loss}")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            gpu_status("Done with batch")

            if (args.validation_prompt is not None and batch_num == 0):
                token_embeds[placeholder_token_id] = top_embedding
                print("Running validation... \n Generating"
                    f" {args.num_validation_videos} videos with prompt:"
                    f" {args.validation_prompt}.")

                # run inference
                generator = torch.Generator("cuda").manual_seed(args.validation_seed)
                probabilities = []

                with torch.no_grad():
                    videos = []
                    for i in range(args.num_validation_videos):
                        video = pipe(
                                width=720,
                                height=480,
                                prompt=args.validation_prompt,  
                                # prompt="A walking dog",
                                num_videos_per_prompt=1,
                                num_inference_steps=50,
                                num_frames=81,
                                use_dynamic_cfg=True,
                                guidance_scale=6.0,
                                generator=generator
                                ).frames[0]
                        validation_dir = f"{args.output_dir}/validation/{epoch}"
                        validation_dir = Path(validation_dir)
                        validation_dir.mkdir(exist_ok=True, parents=True)
                        video_path = f"{validation_dir}/{i}.mp4"
                        print(f"Video generated {validation_dir}/{i}.mp4")
                        export_to_video(video, video_path, fps=16)
                        
                        probability = validation_model.get_probability(video_path, target_videos=validation_video_encodings[i : i + 1])
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
                    torch.cuda.empty_cache()

                print(f"saving best alphas from validation step {best_epoch}, words = ", best_words)
                torch.save(best_alphas, f"{args.output_dir}/best_alphas.pt")


if __name__ == "__main__":
  main()
