import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    import torch.cuda
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gpu_count = torch.cuda.device_count() if cuda_available else 0
gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
gpu_capability = torch.cuda.get_device_capability(0) if cuda_available else None
gpu_memory = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
gpu_memory = gpu_memory / (1024 ** 3) if gpu_memory else None  # Convert to GB
gpu_memory = round(gpu_memory, 2) if gpu_memory else None

gpu_memory_free = torch.cuda.memory_reserved(0) / (1024 ** 3) if cuda_available else None  # Convert to GB
gpu_memory_free = round(gpu_memory_free, 2) if gpu_memory_free else None

gpu_memory_used = torch.cuda.memory_allocated(0) / (1024 ** 3) if cuda_available else None  # Convert to GB
gpu_memory_used = round(gpu_memory_used, 2) if gpu_memory_used else None

print(f"CUDA Available: {cuda_available}")
print(f"Device: {device}")
print(f"GPU Count: {gpu_count}")
print(f"GPU Name: {gpu_name}")
print(f"GPU Capability: {gpu_capability}")
print(f"GPU Memory: {gpu_memory} GB")
print(f"GPU Memory Free: {gpu_memory_free} GB")
print(f"GPU Memory Used: {gpu_memory_used} GB")
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)} bytes")
print(f"GPU Memory Summary (human-readable, detailed, with stats, with reserved, with allocated, with cached, with max, with min, with avg): {torch.cuda.memory_summary(0, 9)}")