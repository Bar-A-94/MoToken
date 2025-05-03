import PIL
from packaging import version
import torch
import torch.nn.functional as F
import glob
import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict


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
    parser.add_argument("--gradient_checkpointing",
                        action="store_true", 
                        help=("Whether or not to use gradient checkpointing to save memory at the"
                                " expense of slower backward pass."),)
    parser.add_argument("--print_gpu_status",
                        type=bool, 
                        default=False,
                        help=("Whether or not to use print gpu status"),)
    parser.add_argument("--accumulation_steps",
                        type=int, 
                        default=5,
                        help=("Whether or not to use print gpu status"),)
    parser.add_argument("--debug",
                        type=bool, 
                        default=False,
                        help=("Whether or not to use print gpu status"),)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--motion_coeff",
                        type=float,
                        default=1,
                        help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--cond_press_coeff",
                        type=float,
                        default=30,
                        help="Initial learning rate (after the potential warmup period) to use.",)
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank",)


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


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
    if args.remove_concept_tokens:
        cosine_sim[cosine_sim > 0.21] = -2

    top_sim, top_idx = torch.topk(cosine_sim, k=dictionary_size)

    # Print top matches
    if args.debug:
        for rank, (sim, idx) in enumerate(zip(top_sim, top_idx)):
            if rank < 10:
                word = tokenizer.decode([int(idx)])
                print(f"{rank}: Word: -{word}- Token id:{int(idx)} Similarity: {sim.item():.4f}")

    return top_idx