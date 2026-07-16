import torch
ckpt = torch.load('/home/linux/srcn_v2/checkpoint.pt', map_location='cpu', weights_only=True)
print("Epoch:", ckpt.get("epoch"))
print("Batch idx:", ckpt.get("batch_idx"))
w = ckpt["model"]["layer.W_raw"]
print("W_raw abs mean:", w.abs().mean().item())
print("W_raw abs max:", w.abs().max().item())
print("W_raw abs min:", w.abs().min().item())
