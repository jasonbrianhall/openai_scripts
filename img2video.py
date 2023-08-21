import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video  
from PIL import Image

prompt = input("Give a prompt: ")
imgname = input("Image Name: ")
init_image = Image.open(imgname)

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w") 
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() 

help(pipe)
#video_frames = pipe(prompt, init_image=init_image, num_inference_steps=25).frames

# Export as mp4  
video_path = export_to_video(video_frames)

# Print video filename
print(f"Video saved to: {video_path}")
