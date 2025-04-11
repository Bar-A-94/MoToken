import argparse
import torch
from transformers import T5Tokenizer
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
)
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="T5-based text encoder embedding script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--path_to_encoder_embeddings",
        type=str,
        default="./t5_cog_text_encoding.pt",
        help="Path to save the embeddings matrix."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load CogVideoX pipeline
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5B",
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Use T5 encoder and tokenizer from the pipeline
    tokenizer = pipe.tokenizer
    vocab_size = pipe.tokenizer.vocab_size
    valid_token_ids = list(range(vocab_size))
    model = pipe.text_encoder
    model.eval()

    orig_embeddings = model.get_input_embeddings().weight.clone().detach()

    imagenet_templates = [
        "a dog {}",
        "a video of a dog {}",
        "a {} dog",
    ]

    def get_embedding_for_prompt(prompt, templates):
        with torch.no_grad():
            texts = [template.format(prompt) for template in templates]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            encoder_outputs = model(**inputs).last_hidden_state  # [B, T, D]
            pooled = encoder_outputs.mean(dim=1)  # [B, D]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            mean_encoding = pooled.mean(dim=0)  # [D]
            mean_encoding = mean_encoding / mean_encoding.norm()
            return mean_encoding.float()

    top_encodings_t5 = [
        get_embedding_for_prompt(pipe.tokenizer.decode([token_id]), imagenet_templates)
        for token_id in tqdm(valid_token_ids, desc="Encoding tokens")
    ]

    top_encodings_t5 = torch.stack(top_encodings_t5, dim=0)  # [Vocab, D]
    torch.save(top_encodings_t5, args.path_to_encoder_embeddings)

if __name__ == "__main__":
    main()
