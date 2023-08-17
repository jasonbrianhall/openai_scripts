import torch, timeit

print(f"CUDA support: {torch.cuda.is_available()} (Should be \"True\")")
print(f"CUDA version: {torch.version.cuda} (Should be \"None\")")
print(f"HIP version: {torch.version.hip} (Should contain \"5.4\")")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"Current CUDA device ID: {torch.cuda.current_device()}")
print(f"Current CUDA device name: {torch.cuda.get_device_name(cuda_id)} (Should be AMD, not NVIDIA)")

def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

x = torch.randn(10000, 1024, device='cuda')

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Ran each twice to show difference before/after warmup
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
