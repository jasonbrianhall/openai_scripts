import torch
from PIL import Image	 
from tqdm import tqdm
from RealESRGAN import RealESRGAN
import argparse
import threading
import os
import subprocess
from queue import Queue
from time import sleep



# Define upscale function
def upscale_frame(i, model, total=1):
	img = Image.open(f'frames/frame{i}.png').convert('RGB')
	
	# Upscale loop
	pbar = tqdm(total=total, desc=f'Upscaling frame{i}')
	sr_img = None
	for j in range(total):
		 pbar.update(1)
		 sr_img = model.predict(img)
		 
	pbar.close()

	# Save 
	sr_img.save(f'frames/frame{i}_upscaled.png')

def main():

	# Parse args
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True, help='Input video file')
	args = parser.parse_args()

	devices = []

	CPUEnabled=True
	
	for i in range(torch.cuda.device_count()):
		gpu = torch.cuda.get_device_properties(i)
		if gpu.total_memory > 8 * 1024 ** 3:  # Check if GPU has more than 8 GB of RAM
			devices.append(f"cuda:{i}")

	if CPUEnabled==True:
		devices.append("cpu")

	# Create a finish event for each device
	finish_events = {device: threading.Event() for device in devices}

	threads={}
	for device in devices:
		threads[device]=0


	'''for device in devices:
		temp=threads[device]
		threads[device]+=1
		thread = threading.Thread(name=device, target=worker, args=(device, prompt, output_file, finish_events[device], iterations))
		queue1.put(thread)
		thread.start()
		#print("\n\nStarting version", versionNumber, "with device", device, "and prompt", prompt, "and outputbasename", output_base_name)
	'''

	'''model_cuda_0 = RealESRGAN(device='cuda:0', scale=4)	 
	model_cuda_1 = RealESRGAN(device='cuda:1', scale=4)
	model_cuda_0.load_weights('weights/RealESRGAN_x4.pth')
	model_cuda_1.load_weights('weights/RealESRGAN_x4.pth')
	model_cpu = RealESRGAN(device='cuda:0', scale=4)	 
	model_cpu.load_weights('weights/RealESRGAN_x4.pth')'''

	queue1 = Queue()
	queue2 = Queue()



	# Extract frames using ffmpeg 
	subprocess.run(['ffmpeg', '-i', args.input, '-vsync', '0', 'frames/frame%d.png'])

	# Get frame count
	count = len(os.listdir('frames'))


	i=1
	for device in devices:
		temp=threads[device]
		threads[device]+=1
		model = RealESRGAN(device=device, scale=4)
		model.load_weights('weights/RealESRGAN_x4.pth')	 	
		thread = threading.Thread(name=device, target=upscale_frame, args=(i, model))
		queue1.put(thread)
		thread.start()
		i+=1
	while not queue1.empty():
		while not queue1.empty():
			thread=queue1.get()
			if not thread.is_alive():
				thread.join()
				device=thread.name
				print("Device name is", device)
				temp=threads[device]
				threads[device]+=1
				model = RealESRGAN(device=device, scale=4)
				model.load_weights('weights/RealESRGAN_x4.pth')	 	
				if i<count:
					thread = threading.Thread(name=device, target=upscale_frame, args=(i, model))
					queue2.put(thread)
					thread.start()
					i+=1
			else:
				queue2.put(thread)
			sleep(0.1)	
		while not queue2.empty():
			thread=queue2.get()
			queue1.put(thread)


	 
	print('Done!')
	
if __name__ == '__main__':
    main()
