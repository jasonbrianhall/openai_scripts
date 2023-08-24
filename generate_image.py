#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline
import os
import threading
import math
from time import sleep
from queue import Queue
import random
import string

def worker(device, prompt, output_file, finish_event, iterations=20):
	if not device == "cpu":
		torch.backends.cuda.matmul.allow_tf32 = True
	pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None)
	
	if not device == "cpu":
		pipe.enable_sequential_cpu_offload(gpu_id=int(device.split(":")[1]))

	image = pipe(prompt, num_inference_steps=iterations).images[0]
	image.save(output_file)
	
	finish_event.set()	# Signal that the thread has finished

CPUENABLED=True

# Prompt user for input
prompt = input("Enter the text prompt: ")

# Define output file base name
output_base_name = input("Enter the output base name: ")

numberofversions = int(input("Enter the number of versions: "))

iterations = int(input("Enter the number of iterations: "))


devices = []

for i in range(torch.cuda.device_count()):
	gpu = torch.cuda.get_device_properties(i)
	if gpu.total_memory > 8 * 1024 ** 3:  # Check if GPU has more than 8 GB of RAM
		devices.append(f"cuda:{i}")

if CPUENABLED==True:
	num_cpus=os.cpu_count()
	num_threads = max(1, num_cpus // 16)
	if num_threads<=0:
		num_threads=1
	for x in range(0, num_threads):
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
		random_string = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
		output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}-{random_string}.png"
	thread = threading.Thread(name=device, target=worker, args=(device, prompt, output_file, finish_events[device], iterations))
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
					# Generate 8 random characters
					random_string = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(8))
					output_file = f"{output_base_name}-{device.replace(':', '')}-{temp}-{random_string}.png"
				thread = threading.Thread(name=device, target=worker, args=(device, prompt, output_file, finish_events[device], iterations))
				queue2.put(thread)
				thread.start()
		else:
			queue2.put(thread)
		sleep(1)	
	while not queue2.empty():
		thread=queue2.get()
		queue1.put(thread)
		
		
		
	


