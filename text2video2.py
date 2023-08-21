import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w")

# Check for GPU with 8+ GB RAM
if torch.cuda.get_device_properties(0).total_memory > 8e9:
    print("Using GPU")
    pipe.to("cuda")
else:
    print("Using CPU")
    pipe.to("cpu")
    pipe.enable_model_cpu_offload()

# Rest of code...

prompt = input("Give a prompt: ")
seconds = input("Number of seconds: ")

num_steps = int(seconds) * 15


video_frames = pipe(prompt, num_inference_steps=num_steps).frames

video_path = export_to_video(video_frames)

print(f"Video saved to: {video_path}")
