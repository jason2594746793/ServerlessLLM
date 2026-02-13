import torch
# Check if any CUDA GPU is available
print(torch.cuda.is_available()) 

# Get the number of visible GPUs
print(torch.cuda.device_count()) 

# Get the name of a specific GPU (e.g., device 0)
print(torch.cuda.get_device_name(0)) 