#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline
import os, sys
import threading
import math
from time import sleep
from queue import Queue
import random
import string
import threading
import traceback

lock = threading.Lock()


def worker(device, prompt, output_file, finish_event, iterations=20, model="stabilityai/stable-diffusion-2-1"):
	savedmodel="./" + model.replace("/", "_")
	savedmodel = os.path.abspath(savedmodel)
	#help(StableDiffusionPipeline.from_pretrained)
	if not device == "cpu":
		torch.backends.cuda.matmul.allow_tf32 = True

	# Acquire the lock before accessing shared resource
	lock.acquire()  

	try:
		if not device=="cpu":
			cuda_capabilities = torch.cuda.get_device_capability(device)
			major, minor = cuda_capabilities
			if major >= 6:
				pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=savedmodel, safety_checker=None, torch_dtype=torch.float16, variant="fp16", use_safetensors=False)
			else:
				pipe = StableDiffusionPipeline.from_pretrained(savedmodel, safety_checker=None)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(savedmodel, safety_checker=None)
		
	except:
		traceback.print_exc()
		if not device=="cpu":
			cuda_capabilities = torch.cuda.get_device_capability(device)
			major, minor = cuda_capabilities
			print(major, minor)
			if major >= 6:		
				pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None, torch_dtype=torch.float16, variant="fp16")
			else:
				pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None)

		pipe.save_pretrained(savedmodel)
		pass
	# Access resource
	lock.release()


	if not device == "cpu":
		pipe.enable_sequential_cpu_offload(gpu_id=int(device.split(":")[1]))

	image = pipe(prompt, num_inference_steps=iterations).images[0]
	image.save(output_file)
	
	finish_event.set()	# Signal that the thread has finished

def main():

	args = sys.argv[1:]
	defaultmodel="stabilityai/stable-diffusion-2-1"
	model=defaultmodel
	iterations = -1
	prompt = None
	CPUEnabled = False
	numberofversions = -1
	output_base_name = None
	useOpenGL=False

	for i in range(0, len(args)):
		if args[i] == '-i':
			iterations = int(args[i+1])
		elif args[i] == '-p':  
			prompt = args[i+1]
		elif args[i] == '-c':
			CPUEnabled = True 
		elif args[i] == '-v':
			numberofversions = int(args[i+1])
		elif args[i] == '-m':
			model= args[i+1]
		elif args[i] == '-o':  
			output_base_name = args[i+1]
			print("Setting output_base_name")
		elif args[i] == '-O':  
			useOpenGL=True
			print("Setting OpenGL")
		elif args[i] == '-h':  
			print(sys.argv[0] + " -i iteraions -p prompt -c (CPU Enabled) -v numberoversions -o outputbasename -O (useopengl instead of CUDA) -m model (defaults to " + defaultmodel + ") -h (help)")
			sys.exit(1)    


	# Prompt user for input
	if prompt==None:
		prompt = input("Enter the text prompt: ")

	# Define output file base name
	if output_base_name==None:
		output_base_name = input("Enter the output base name: ")

	if numberofversions<=0:
		numberofversions = int(input("Enter the number of versions: "))

	if iterations<=0:
		iterations = int(input("Enter the number of iterations: "))

	versioncount=0

	devices = []

	for i in range(torch.cuda.device_count()):
		gpu = torch.cuda.get_device_properties(i)
		if gpu.total_memory > 8 * 1024 ** 3:  # Check if GPU has more than 8 GB of RAM
			if versioncount<numberofversions:
				versioncount+=1
				if useOpenGL==False:
					devices.append(f"cuda:{i}")
				else:
					devices.append(f"opengl:{i}")

	if CPUEnabled==True:
		num_cpus=os.cpu_count()
		num_threads = max(1, num_cpus // 16)
		if num_threads<=0:
			num_threads=1
		for x in range(0, num_threads):
			if versioncount<numberofversions:
				versioncount+=1
				devices.append("cpu")

	# Create a finish event for each device
	finish_events = {device: threading.Event() for device in devices}

	queue1 = Queue()
	queue2 = Queue()

	threads={}
	for device in devices:
		threads[device]=0


	versionNumber=0
	for device in devices:
		versionNumber+=1
		temp=threads[device]
		threads[device]+=1
		output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}.png"
		if os.path.exists(output_file):
			# Generate 8 random characters
			while os.path.exists(output_file):
				random_string = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
				output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}-{random_string}.png"
		thread = threading.Thread(name=device, target=worker, args=(device, prompt, output_file, finish_events[device], iterations, model))
		queue1.put(thread)
		thread.start()
		print("\n\nStarting version", versionNumber, "with device", device, "and prompt", prompt, "and outputbasename", output_base_name)

	while not queue1.empty():
		while not queue1.empty():
			thread=queue1.get()
			if not thread.is_alive():
				thread.join()
				if versionNumber<numberofversions:
					versionNumber+=1
					device=thread.name
					print("\n\nStarting version", versionNumber, "with device", device, "and prompt", prompt, "and outputbasename", output_base_name)
					temp=threads[device]
					threads[device]+=1
					output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}.png"
					if os.path.exists(output_file):
						while os.path.exists(output_file):
							# Generate 8 random characters
							random_string = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
							output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}-{random_string}.png"
					thread = threading.Thread(name=device, target=worker, args=(device, prompt, output_file, finish_events[device], iterations, model))
					queue2.put(thread)
					thread.start()
			else:
				queue2.put(thread)
			sleep(1)	
		while not queue2.empty():
			thread=queue2.get()
			queue1.put(thread)
			
if __name__ == '__main__':
    main()
			
			
		


