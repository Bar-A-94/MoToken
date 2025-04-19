"""A script for saving all the word embeddings."""
# pylint: disable=g-multiple-import,g-importing-member,g-bad-import-order,missing-function-docstring
import argparse

from diffusers.schedulers import LMSDiscreteScheduler
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from diffusers import LTXPipeline
from diffusers import (
	CogVideoXDPMScheduler,
	CogVideoXPipeline,
)
from tqdm import tqdm



def parse_args():
	parser = argparse.ArgumentParser(
		description="Simple example of a training script."
	)
	parser.add_argument(
		"--pretrained_model_name_or_path",
		type=str,
		default="stabilityai/stable-diffusion-2-1-base",
		help=(
			"Path to pretrained model or model identifier from"
			" huggingface.co/models."
		),
	)
	parser.add_argument(
		"--clip_model",
		type=str,
		default="openai/clip-vit-base-patch32",
		help=(
			"The CLIP model to use for the calculation of the image-text"
			" matching."
		),
	)
	parser.add_argument(
		"--path_to_encoder_embeddings",
		type=str,
		default="./clip_text_encoding.pt",
		help="Path to the saved embeddings matrix of the text encoder",
	)

	args = parser.parse_args()

	return args


def main():
	args = parse_args()
	gpu_id = 0
	device = f"cuda:{gpu_id}" if gpu_id != -1 else "cpu"

	clip_type = {
			'video': 'LanguageBind_Video_FT',}  # also LanguageBind_Video
	model = LanguageBind(clip_type=clip_type, cache_dir='/home/joberant/NLP_2425a/baralon1/cache')
	model = model.to("cuda")
	model.eval()
	pretrained_ckpt = f'lb203/LanguageBind_Image'
	tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='/home/joberant/NLP_2425a/baralon1/cache')
	print("loaded language-bind")
	pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-5B', 
												torch_dtype=torch.bfloat16, 
												device_map="balanced")
	pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
	orig_embeddings = pipe.text_encoder.get_input_embeddings().weight

	print("Lets see some orig_embeddings  !!!!!!!!!")
	print(orig_embeddings)
	print("Lets see some orig_embeddings shape !!!!!!!!!")
	print(orig_embeddings.shape)
	print("Lets see some orig_embeddings shape[0] !!!!!!!!!")
	print(orig_embeddings.shape[0])
	print("vocab_size !!!!!!!!!")
	print(pipe.tokenizer.vocab_size)


	# imagenet_templates = [
	#     "a photo of a {}",
	#     "a rendering of a {}",
	#     "a cropped photo of the {}",
	#     "the photo of a {}",
	#     "a photo of a clean {}",
	#     "a photo of a dirty {}",
	#     "a dark photo of the {}",
	#     "a photo of my {}",
	#     "a photo of the cool {}",
	#     "a close-up photo of a {}",
	#     "a bright photo of the {}",
	#     "a cropped photo of a {}",
	#     "a photo of the {}",
	#     "a good photo of the {}",
	#     "a photo of one {}",
	#     "a close-up photo of the {}",
	#     "a rendition of the {}",
	#     "a photo of the clean {}",
	#     "a rendition of a {}",
	#     "a photo of a nice {}",
	#     "a good photo of a {}",
	#     "a photo of the nice {}",
	#     "a photo of the small {}",
	#     "a photo of the weird {}",
	#     "a photo of the large {}",
	#     "a photo of a cool {}",
	#     "a photo of a small {}",
	# ]

	imagenet_templates = [
		"a dog {}",
		"a video of a dog {}",
		"a {} dog",
	]

	def get_embedding_for_prompt(prompt, templates):
		with torch.no_grad():
			texts = [template.format(prompt) for template in templates]  # format with class
			inputs = {}
			inputs['language'] = to_device(tokenizer(texts, max_length=77, padding='max_length',
											truncation=True, return_tensors='pt'), device)
			text_encodings = model(inputs)['language']
			text_encodings /= text_encodings.norm(dim=-1, keepdim=True)
			text_encodings = text_encodings.mean(dim=0)
			text_encodings /= text_encodings.norm()

		return text_encodings.float()


	top_encodings_open_clip = [get_embedding_for_prompt(pipe.tokenizer.decode([token]), imagenet_templates)
		for token in tqdm(range(orig_embeddings.shape[0]), desc="Processing Tokens") if token < pipe.tokenizer.vocab_size]
	top_encodings_open_clip = torch.stack(top_encodings_open_clip, dim=0)

	torch.save(top_encodings_open_clip, args.path_to_encoder_embeddings)


if __name__ == "__main__":
	print("start")
	main()
	print("done")
