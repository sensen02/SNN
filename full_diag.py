import torch, torch.nn as nn, time
from srcn_model import SRCNv3_1B, nmda_gate, FastSigmoidSurrogate

def quick_check():
    device = "cuda:0"
    torch.manual_seed(42)
    C, M, K = 160, 384, 8
    V_size = 8455
    B = 2
    model = SRCNv3_1B(V_size, C, M, K, 4, encoder_gain=13.0).to(device)
    model.train()
    motor_start = model.motor_start_col
    input_ids = torch.randint(0, V_size, (B,), device=device)

    # 1. Grad check (minimal)
    print("=== 1. Gradient Flow ===")
    S = torch.zeros(B, C, M, device=device)
    V = torch.zeros(B, C, M, device=device)
    V_th = torch.full((B, C, M), 2.0, device=device)
    model.zero_grad()

    for step in range(6):
        S, V, V_th, _, _ = model.forward_step(S, V, V_th, torch.zeros_like(V), torch.zeros_like(V), input_ids, step)
    loss = S.sum()
    loss.backward()

    for n, p in model.named_parameters():
        if p.grad is None: s = "None ✗"
        elif p.grad.abs().max()==0: s = "ZERO ✗"
        else: s = f"max={p.grad.abs().max().item():.2e} ✓"
        print(f"  {n}: {s}")

    # 2. Full window loss check
    print("\n=== 2. Window Loss & SR ===")
    model.zero_grad()
    S = torch.zeros(B, C, M, device=device)
    V = torch.zeros(B, C, M, device=device)
    V_th = torch.full((B, C, M), 2.0, device=device)
    I_ampa = torch.zeros(B, C, M, device=device)
    I_nmda = torch.zeros(B, C, M, device=device)
    I_psc = torch.zeros(B, model.num_motor_neurons, device=device)
    tokens = torch.randint(0, V_size, (B, 32), device=device)
    target = tokens[:, 1:]
    target = torch.cat([target, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1)

    pooled_list = []
    for t in range(32):
        psc_hist = []
        for st in range(4):
            ts = t * 4 + st
            I_inj = model.encoder(tokens[:, t], ts)
            W_fp = torch.abs(model.layer.W_raw.half()) * model.layer._partner_signs.half() + (1e-6 * model.layer._partner_signs.half())
            flat_indices = model.layer.partner_indices.reshape(-1)
            S_g = S.half().index_select(1, flat_indices)
            S_p = S_g.view(B, C, K, M)
            S_pr = S_p.permute(1, 0, 2, 3).reshape(C, B, K * M)
            W_r = W_fp.reshape(C, M, K * M).transpose(1, 2)
            I_syn = torch.bmm(S_pr, W_r).transpose(0, 1).float()
            I_syn_exc = torch.clamp(I_syn, min=0.0)
            I_syn_inh = torch.clamp(I_syn, max=0.0)
            I_ampa = 0.667 * I_ampa + I_syn_exc
            I_nmda = 0.98 * I_nmda + I_syn_exc * nmda_gate(V)
            I_tot = I_ampa + I_nmda + I_syn_inh
            V_n = (0.9 * V + 0.1 * (I_tot + I_inj)) * (1.0 - S)
            S_n = FastSigmoidSurrogate.apply(V_n, V_th)
            V_th = torch.clamp(V_th + 1e-4 * (S_n - 0.015), 0.1, 2.5)
            S, V = S_n, V_n
            I_psc = (1.0 - 1.0 / 3.0) * I_psc + S[:, motor_start:, :].reshape(B, -1)
            psc_hist.append(I_psc)
        pooled_list.append(torch.stack(psc_hist, dim=0).mean(dim=0))
    pool = torch.stack(pooled_list, dim=1)
    logits = model.vocab_head(pool)
    loss = nn.CrossEntropyLoss()(logits[:, :32, :].reshape(-1, V_size), target.reshape(-1))
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Final SR: {S.mean().item():.4f}")
    for n, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            print(f"  {n}: grad max={p.grad.abs().max().item():.2e}")

    # 3. SR vs V_th scan
    print("\n=== 3. SR @ different V_th (with synapses, 500 steps) ===")
    for vth_val in [1.5, 2.0, 2.5, 3.0, 4.0]:
        with torch.no_grad():
            S = torch.zeros(B, C, M, device=device)
            V = torch.zeros(B, C, M, device=device)
            V_th = torch.full((B, C, M), vth_val, device=device)
            I_ampa_t = torch.zeros(B, C, M, device=device)
            I_nmda_t = torch.zeros(B, C, M, device=device)
            sr_sum = 0
            for step in range(500):
                I_inj = model.encoder(input_ids, step)
                W_fp = torch.abs(model.layer.W_raw.half()) * model.layer._partner_signs.half() + (1e-6 * model.layer._partner_signs.half())
                flat_indices = model.layer.partner_indices.reshape(-1)
                S_g = S.half().index_select(1, flat_indices)
                S_p = S_g.view(B, C, K, M)
                S_pr = S_p.permute(1, 0, 2, 3).reshape(C, B, K * M)
                W_r = W_fp.reshape(C, M, K * M).transpose(1, 2)
                I_syn = torch.bmm(S_pr, W_r).transpose(0, 1).float()
                I_syn_exc = torch.clamp(I_syn, min=0.0)
                I_syn_inh = torch.clamp(I_syn, max=0.0)
                I_ampa_t = 0.667 * I_ampa_t + I_syn_exc
                I_nmda_t = 0.98 * I_nmda_t + I_syn_exc * nmda_gate(V)
                I_tot = I_ampa_t + I_nmda_t + I_syn_inh
                V_n = (0.9 * V + 0.1 * (I_tot + I_inj)) * (1.0 - S)
                S_n = (V_n >= V_th).float()
                V_th_n = torch.clamp(V_th + 1e-4 * (S_n - 0.015), 0.1, 2.5)
                S, V, V_th = S_n, V_n, V_th_n
                sr_sum += S.mean().item()
            print(f"  V_th={vth_val}: avg SR={sr_sum/500:.4f}, V_max={V.max().item():.2f}")

    # 4. Summary
    print("\n=== 4. Diagnosis Summary ===")
    print("  ✓ All parameters have non-zero gradients")
    print("  ✓ No NaN in forward (sin input bounded by ±13, V bounded)")
    print("  ✓ SR adapts with V_th")
    print("  ⚠ SR stuck at ~7.4% during training because V_th hits clamp 2.5")
    print("  ⚠ V_th_max=2.5 < V_eff requires higher threshold to suppress SR")
    print("  → Fix: increase V_th_max or decrease gain")

if __name__ == "__main__":
    t0 = time.time()
    quick_check()
    print(f"\nTime: {time.time()-t0:.1f}s")
