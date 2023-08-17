import torch

device_count = torch.cuda.device_count()
print(f"Number of available devices: {device_count}")

for i in range(device_count):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Make sure that the device ID is within the range of available device IDs
device_id = 2
if device_id >= device_count:
    print(f"Invalid device ID {device_id}.")
else:
    # Use the device ID in CUDA function call
    torch.cuda.set_device(device_id)

