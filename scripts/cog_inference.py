from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
import torch
from pathlib import Path
from diffusers.utils import export_to_video

generator = torch.Generator("cuda").manual_seed(1024)

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5B", 
                                                torch_dtype=torch.bfloat16, 
                                                device_map="balanced")
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
prompt = "A dog interested"
with torch.no_grad():
    validation_dir = f"motoken/output/expirements/"
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(exist_ok=True, parents=True)
    video = pipe(
            width=720,
            height=480,
            prompt=prompt,  
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=81,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=generator
            ).frames[0]
    video_path = f"{validation_dir}/{prompt}.mp4"
    print(f"Video generated {video_path}")
    export_to_video(video, video_path, fps=16)