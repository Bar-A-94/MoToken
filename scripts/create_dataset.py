from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
import torch
from pathlib import Path
from diffusers.utils import export_to_video

generator = torch.Generator("cuda").manual_seed(1024)

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX1.5-5B", 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="balanced")
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
general_man_prompts = ['A man {}', 'Video of a man {}', 'man {}', 'Good video of a man {}', 'A video of one man {}']
concept = 'painting'
for general_prompt in general_man_prompts:
    prompt = general_prompt.format(concept)
    for i in range (20):
        with torch.no_grad():
                validation_dir = f"motoken/data/A_man_{concept}/"
                validation_dir = Path(validation_dir)
                validation_dir.mkdir(exist_ok=True, parents=True)
                video = pipe(
                        width=1360,
                        height=768,
                        prompt=prompt,  
                        num_videos_per_prompt=1,
                        num_inference_steps=50,
                        num_frames=81,
                        use_dynamic_cfg=True,
                        guidance_scale=6.0,
                        generator=generator
                        )
                video = video.frames[0]

                video_path = f"{validation_dir}/{prompt}_{i}.mp4"
                print(f"Video generated {video_path}")
                export_to_video(video, video_path, fps=16)