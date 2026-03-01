import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))