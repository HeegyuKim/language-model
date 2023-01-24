import transformers
import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
print(dev)
print(xm.get_xla_supported_devices())
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)