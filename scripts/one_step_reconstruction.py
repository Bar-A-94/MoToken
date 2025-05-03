
from accelerate.logging import get_logger

from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
import torch.nn.functional as F
from tqdm.auto import tqdm
import transformers
from diffusers.utils import export_to_video, logging
import numpy as np
from diffusers.optimization import get_scheduler
from pathlib import Path
import math





# Add the project root directory to sys.path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.languagebindmodel.languagebind import LanguageBind, to_device, transform_dict
from models.model_zoo import LBsimilarity
from scripts.utils import gpu_clean_and_status, parse_args, get_language_bind_encodings, get_dictionary_indices
from scripts.dataset import ConceptDataset
from scripts.model_and_losses import Net, sorted_l1_loss, calculate_motion_loss
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch.backends.cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


logger = get_logger(__name__)

def main(): 
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppress verbose logging output 
    transformers.utils.logging.set_verbosity_error()
    logging.set_verbosity_error()

    # Load the pipe line
    weight_dtype = torch.bfloat16
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, 
                                                torch_dtype=weight_dtype, 
                                                device_map="balanced")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing() # Enables channel-wise slicing - optimize memory usage
    pipe.vae.enable_tiling() # splits the image into smaller tiles and processes each tile separately - optimize memory usage
    pipe.set_progress_bar_config(disable=True)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()


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
    net = Net().to(dtype=weight_dtype, device=device)
    # Print a sanity check
    if args.debug:
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
                                    num_warmup_steps=args.lr_warmup_steps,
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

    # Get video encoding for  val
    validation_video_encodings = get_language_bind_encodings(args.validation_data_dir, device)

    # Find the most similar tokens to the video encodings
    dictionary_indices = get_dictionary_indices(args, pipe.text_encoder, pipe.tokenizer, num_tokens, device).to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dictionary_indices, f"{args.output_dir}/dictionary.pt")

    # Make sure we don't track back gradients for the new tensors
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

    for epoch in range(args.num_train_epochs):
        net.train()
        for batch_num, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            print(f"Epoch: {epoch} Batch_num: {batch_num}")
            gpu_clean_and_status('New batch', print_gpu_status=args.print_gpu_status)
            pipe.text_encoder.get_input_embeddings().weight.detach_().requires_grad_(False)

            # Calculate current embeddings
            alphas = net(dictionary) # Pass the embeddings of the encoder throw a net - each get a single number
            alphas = torch.softmax(alphas/8, dim=0)

            _, sorted_indices = torch.sort(alphas.abs(), descending=True) # Abs to get the most important token even negative
            print_words = args.num_explanation_tokens

            # Make the embbedings - take the full dictionary and make a new embedding then apply the known norm
            embedding = torch.matmul(alphas[sorted_indices], dictionary[sorted_indices]) # Make an new embedding to assign to the new token
            embedding = torch.mul(embedding, 1 / embedding.norm())
            embedding = torch.mul(embedding, avg_norm)
            
            # Update embedding
            pipe.text_encoder.get_input_embeddings().weight[placeholder_token_id] = embedding
            concept_embed = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer(args.concept, add_special_tokens=False)["input_ids"][0]]
            concept_similarity = F.cosine_similarity(embedding, concept_embed, dim=0)

            if args.debug:
                print(f"Similarity with 'walking' embedding: {concept_similarity.item():.3f}")
                dog_embed = pipe.text_encoder.get_input_embeddings().weight[pipe.tokenizer("dog", add_special_tokens=False)["input_ids"][0]]
                dog_similarity = F.cosine_similarity(embedding, dog_embed, dim=0)
                print(f"Similarity with 'dog' embedding: {dog_similarity.item():.3f}")
                pipe.text_encoder.get_input_embeddings().weight.requires_grad_(True)

            # Print the top words for debuging
            top_words = [pipe.tokenizer.decode(dictionary_indices[sorted_indices[i]]) for i in range(print_words)]
            print(" | ".join([f"{i}: {alphas[sorted_indices[i]]} * <{top_words[i]}>" for i in range(print_words)]))            

            
            gpu_clean_and_status('After Net', print_gpu_status=args.print_gpu_status)

            # Get the text embedding for conditioning
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt([args.validation_prompt]*args.train_batch_size,
                                                                        None,
                                                                        True,
                                                                        num_videos_per_prompt=1,
                                                                        prompt_embeds=None,
                                                                        negative_prompt_embeds=None,
                                                                        max_sequence_length=226,
                                                                        device='cuda',)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            
            gpu_clean_and_status('After Encoder', print_gpu_status=args.print_gpu_status)

            # Sample a random timestep for each video
            timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, 50, 'cuda', None)
            # random_indices = torch.randint(0, len(timesteps), (1,), device=device)
            random_indices = torch.randint(0, 50, (1,), device=device)
            timesteps = timesteps[random_indices]
            
            # Convert videos to latent space
            latents = (pipe.vae.encode(batch["pixel_values"].to(device).contiguous().to(dtype=weight_dtype))
                        .latent_dist.sample().detach()).permute(0, 2, 1, 3, 4) # (B, C, F, H, W) -> (B, F, C, H, W)
            
            # latents = pipe.vae_scaling_factor_image * latents

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents) # (B, F, C, H, W)
        
            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps) # (B, F, C, H, W)

            # Create rotary embeddings
            image_rotary_emb = pipe._prepare_rotary_positional_embeddings(args.resolution[0], args.resolution[1], noisy_latents.size(1), device)

            # Prepare noisy latents for uncondtion pred
            noisy_latents_cat = torch.cat([latents] * 2) # (2 * B, F, C, H, W)
            noisy_latents_cat = pipe.scheduler.scale_model_input(noisy_latents_cat, timesteps) # added from the pipeline code
            
            # Prepare timesteps to tranformer forward
            timesteps = timesteps.expand(noisy_latents_cat.shape[0])

            gpu_clean_and_status("Before Transformer", print_gpu_status=args.print_gpu_status)

            # Predict the noise residual
            model_pred = pipe.transformer(hidden_states=noisy_latents_cat, 
                                            encoder_hidden_states=prompt_embeds, 
                                            timestep=timesteps,
                                            image_rotary_emb=image_rotary_emb)[0] # (2 * B, F, C, H, W)

            model_pred = model_pred.float() # Taken from the pipeline

            with torch.no_grad():
                # Get the text embedding for conditioning
                concept_prompt_embeds, concept_negative_prompt_embeds = pipe.encode_prompt([args.prompt]*args.train_batch_size,
                                                                            None,
                                                                            True,
                                                                            num_videos_per_prompt=1,
                                                                            prompt_embeds=None,
                                                                            negative_prompt_embeds=None,
                                                                            max_sequence_length=226,
                                                                            device='cuda',)
                concept_prompt_embeds = torch.cat([concept_negative_prompt_embeds, concept_prompt_embeds], dim=0)
                concept_model_pred = pipe.transformer(hidden_states=noisy_latents_cat, 
                                            encoder_hidden_states=concept_prompt_embeds, 
                                            timestep=timesteps,
                                            image_rotary_emb=image_rotary_emb)[0]
            
            # Handle guidance with uncoditioned
            guidance_scale = 1 + 6 * ((1 - math.cos(math.pi * ((num_inference_steps - timesteps[0].item()) / num_inference_steps) ** 5.0)) / 2)
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # (B, F, C, H, W)
            with torch.no_grad():
                concept_noise_pred_uncond, concept_noise_pred_text = concept_model_pred.chunk(2)
                concept_model_pred = noise_pred_uncond + guidance_scale * (concept_noise_pred_text - concept_noise_pred_uncond) # (B, F, C, H, W)
            cond_pred_loss = F.mse_loss(model_pred.float(), concept_model_pred.float(), reduction="mean")
            gpu_clean_and_status('After Transformer', print_gpu_status=args.print_gpu_status)

            # Get the target for loss depending on the prediction type (classic ddpm vs flow matching)
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction": # Cog using flow matching
                target = pipe.scheduler.get_velocity(latents, noise, timesteps[:args.train_batch_size])
            else:
                raise ValueError("Unknown prediction type"
                f" {pipe.scheduler.config.prediction_type}")

            # mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            ############## START CODE FOR MULTI‑SCALE MOTION LOSS (EXPLICIT) ################

            # Permute (B, F, C, H, W) to [B, F, H, W, C]
            pred_latents = model_pred.permute(0, 1, 3, 4, 2)
            orig_latents = concept_model_pred.permute(0, 1, 3, 4, 2)

            loss1 = calculate_motion_loss(pred_latents, orig_latents, 1)
            loss4 = calculate_motion_loss(pred_latents, orig_latents, 4)
            loss8 = calculate_motion_loss(pred_latents, orig_latents, 8)

            # Equal‑weight across scales
            motion_loss = (loss1 + loss4 + loss8)/3

            ############## END CODE FOR MULTI‑SCALE MOTION LOSS #############################

            # sparsity_loss = torch.norm(alphas, p=1) # l1 loss
            # sparsity_loss = sorted_l1_loss(alphas)

            # calculate final loss
            # loss = mse_loss + 0 * args.sparsity_coeff * sparsity_loss + args.motion_coeff * total_motion_loss
            # loss = 0 * pred_loss + cond_pred_loss + args.motion_coeff * motion_loss
            loss = args.cond_press_coeff * cond_pred_loss - concept_similarity + args.motion_coeff * motion_loss
            # loss = cond_pred_loss + args.motion_coeff * motion_loss

            print(f"Step: {timesteps[0].item()} Total: {loss.item():.3f} = "
                f"cond_pred_loss: {args.cond_press_coeff} * {cond_pred_loss:.3f}, "
                f"motion_loss: {args.motion_coeff} * {motion_loss:.3f}, "
                f"concept_similarity: {concept_similarity.item():.3f}, ")
            
            
            top_indices = [sorted_indices[i].item() for i in range(args.num_explanation_tokens)]
            top_embedding = torch.matmul(alphas[top_indices], dictionary[top_indices]) # Now take only the top embbedings
            top_embedding = torch.mul(top_embedding, 1 / top_embedding.norm())
            top_embedding = torch.mul(top_embedding, avg_norm)

            if loss < best_loss:
                best_train_top_embedding = top_embedding
                best_loss = loss
                print('Best currently')
            if args.debug:
                print("grad before backward:", sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None))
            loss.backward()
            if args.debug:
                print("grad after  backward:", sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None))
            if (batch_num + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()


            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.arange(len(pipe.tokenizer)) != placeholder_token_id
            with torch.no_grad():
                pipe.text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
            gpu_clean_and_status("Done with batch", print_gpu_status=args.print_gpu_status)

            # if batch_num % 5 == 0 and args.debug:
            if batch_num % 5 == 0:
                with torch.no_grad():
                    training_dir = f"{args.output_dir}/training_dir/epoch_{epoch}"
                    os.makedirs(training_dir, exist_ok=True)
                    for latent, name in [(latents, "latents"), (noisy_latents, "noisy_latents"), (model_pred, "model_pred"), (target, "target")]:
                        latent = latent.to(weight_dtype)
                        video = pipe.decode_latents(latent)
                        video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
                        video_path = f"{training_dir}/{batch_num}_{timesteps[0].item()}_{name}.mp4"
                        export_to_video(video, video_path, fps=16)

                        gpu_clean_and_status(f"Video from {name} made", print_gpu_status=args.print_gpu_status)

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
