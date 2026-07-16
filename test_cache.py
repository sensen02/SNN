import torch, torch.nn as nn
from srcn_model import SRCNv3_1B

def test():
    device = "cuda:0"
    C, M, K = 160, 128, 8
    V_size = 1000
    B = 2
    model = SRCNv3_1B(V_size, C, M, K, 4, encoder_gain=13.0).to(device)
    input_ids = torch.randint(0, V_size, (B,), device=device)
    motor_start = model.motor_start_col
    num_motor = model.num_motor_neurons

    S = torch.zeros(B, C, M, device=device)
    V = torch.zeros(B, C, M, device=device)
    V_th = torch.full((B, C, M), 2.0, device=device)
    Ia = torch.zeros(B, C, M, device=device)
    In = torch.zeros(B, C, M, device=device)
    I_psc = torch.zeros(B, num_motor, device=device)
    tokens = torch.randint(0, V_size, (B, 4), device=device)
    target = tokens[:, 1:]
    target = torch.cat([target, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1)[:, :4]

    pooled = []
    for t in range(4):
        psc_h = []
        for st in range(4):
            ts = t * 4 + st
            S, V, V_th, Ia, In = model.forward_step(S, V, V_th, Ia, In, tokens[:, t], ts)
            Sm = S[:, motor_start:, :].reshape(B, -1)
            I_psc = (1.0 - 1.0/3.0) * I_psc + Sm
            psc_h.append(I_psc)
        pooled.append(torch.stack(psc_h, dim=0).mean(dim=0))
    win = torch.stack(pooled, dim=1)
    logits = model.vocab_head(win)
    loss = nn.CrossEntropyLoss()(logits[:, :4, :].reshape(-1, V_size), target.reshape(-1))
    loss.backward()

    print(f"Loss: {loss.item():.4f}, final SR: {S.mean().item():.4f}")
    for n, p in model.named_parameters():
        ok = p.grad is not None and p.grad.abs().max().item() > 0
        g = p.grad.abs().max().item() if p.grad is not None else 0
        print(f"  {n}: {'✓' if ok else '✗'} (max={g:.2e})")

if __name__ == "__main__":
    test()
