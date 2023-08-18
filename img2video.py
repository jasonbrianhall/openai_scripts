import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler  
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = input("Give a prompt: ")
imgname = input("Image Name: ")
init_image = Image.open(imgname) 


video_frames = pipe(prompt, init_image = init_image, num_inference_steps=25).frames


# Export as mp4 
video_path = export_to_video(video_frames)

# Print video filename
print(f"Video saved to: {video_path}")
