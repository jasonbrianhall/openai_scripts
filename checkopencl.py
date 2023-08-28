import torch
import pyopencl as cl

platforms = cl.get_platforms()

devices = []
for platform in platforms:
    plat_devs = platform.get_devices()
    devices.extend(plat_devs)

print(devices)
