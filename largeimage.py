import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda:1")
prompt = "a beautiful landscape photograph"
pipe.enable_vae_tiling()
device="cuda:0"
pipe.enable_sequential_cpu_offload(gpu_id=int(device.split(":")[1]))

pipe.enable_xformers_memory_efficient_attention()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20).images[0]

image.save("landscape.png")
