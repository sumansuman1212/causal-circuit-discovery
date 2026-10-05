import torch

print("Pytorch version:",torch.__version__)
print("CUDA Available:",torch.cuda.is_available())
if(torch.cuda.is_available()):
   print("GPU Device:",torch.cuda.get_device_namee(0))
else:
   print("Running on CPU(GPU recomended for latter phases).")
