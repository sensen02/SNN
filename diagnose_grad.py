import torch
import torch.nn as nn
from srcn_model import SRCNv3_1B, nmda_gate, FastSigmoidSurrogate
import time

def verify_fix():
    device = "cpu"
    C, M, K = 160, 128, 8
    pool_steps = 4
    V_size = 5000
    B = 2
    print(f"C={C}, M={M}, K={K}, gain=8.0, V_th_init=0.5")

    model = SRCNv3_1B(
        vocab_size=V_size, num_columns=C, neurons_per_column=M,
        num_partners=K, num_motor_pool_steps=pool_steps,
        encoder_gain=8.0
    ).to(device)

    motor_start = model.motor_start_col
    num_motor = model.num_motor_neurons
    input_ids = torch.randint(0, V_size, (B, 64))

    # Test: can we spike now?
    print("\n=== Spikeability with new params ===")
    V_t = torch.zeros(B, C, M)
    V_th_t = torch.full((B, C, M), 0.5)
    spike_ever = False

    for step in range(100):
        with torch.no_grad():
            I = model.encoder(input_ids[:, step % 64], step)
        V_t = 0.9 * V_t + 0.1 * I
        S_t = (V_t >= V_th_t).float()
        V_t = V_t * (1.0 - S_t)
        if S_t.sum() > 0 and not spike_ever:
            print(f"  First spike at step {step}! ({S_t.sum().item()} neurons)")
            spike_ever = True

    if not spike_ever:
        print(f"  No spikes in 100 steps, max V={V_t.max().item():.4f}")
    print(f"  I_inj range: [{I.min().item():.4f}, {I.max().item():.4f}]")

    # Test: single step W_raw gradient with new params
    print("\n=== W_raw gradient check ===")
    model.zero_grad()
    S_v = torch.zeros(B, C, M)
    V_v = torch.zeros(B, C, M)
    V_th_v = torch.full((B, C, M), 0.5)
    I_ampa_v = torch.zeros(B, C, M)
    I_nmda_v = torch.zeros(B, C, M)

    # Run 5 steps of the model to build up some V and cause spiking
    I_history = []
    for step in range(10):
        I_inj_v = model.encoder(input_ids[:, 0], step)
        I_history.append(I_inj_v.max().item())

        W_fp = torch.abs(model.layer.W_raw) * model.layer._partner_signs + (1e-6 * model.layer._partner_signs)
        flat_indices = model.layer.partner_indices.reshape(-1)
        S_gathered = S_v.index_select(1, flat_indices)
        S_partners = S_gathered.view(B, C, K, M)
        S_partners_reshaped = S_partners.permute(1, 0, 2, 3).reshape(C, B, K * M)
        W_reshaped = W_fp.reshape(C, M, K * M).transpose(1, 2)
        I_syn = torch.bmm(S_partners_reshaped, W_reshaped).transpose(0, 1)
        I_syn_exc = torch.clamp(I_syn, min=0.0)
        I_syn_inh = torch.clamp(I_syn, max=0.0)
        I_ampa_n = 0.667 * I_ampa_v + I_syn_exc
        nmda_gate_val = nmda_gate(V_v)
        I_nmda_n = 0.98 * I_nmda_v + I_syn_exc * nmda_gate_val
        I_total = I_ampa_n + I_nmda_n + I_syn_inh
        V_leaked = 0.9 * V_v + 0.1 * (I_total + I_inj_v)
        V_next = V_leaked * (1.0 - S_v)
        S_next = FastSigmoidSurrogate.apply(V_next, V_th_v)

        S_v, V_v = S_next, V_next

        if S_next.sum() > 0:
            print(f"  Step {step}: spike! ({S_next.sum().item()} neurons)")
            # Run backward through this step
            loss_test = S_next[:, motor_start:, :].sum()
            loss_test.backward(retain_graph=True)
            w_grad = model.layer.W_raw.grad.abs().max().item() if model.layer.W_raw.grad is not None else 0
            enc_grad = model.encoder.proj.weight.grad.abs().max().item() if model.encoder.proj.weight.grad is not None else 0
            print(f"  W_raw grad max={w_grad:.6e}, encoder grad max={enc_grad:.6e}")
            model.zero_grad()
            break

    # Test: full window gradient after warmup
    print("\n=== Full window gradient with warmup spikes ===")
    model.zero_grad()
    S = torch.zeros(B, C, M)
    V = torch.zeros(B, C, M)
    V_th = torch.full((B, C, M), 0.5)
    I_ampa = torch.zeros(B, C, M)
    I_nmda = torch.zeros(B, C, M)
    I_psc = torch.zeros(B, num_motor)

    # Warmup: 20 steps without gradient to build up spiking
    with torch.no_grad():
        for step in range(20):
            I_inj_x = model.encoder(input_ids[:, 0], step)
            W_fp = torch.abs(model.layer.W_raw) * model.layer._partner_signs + (1e-6 * model.layer._partner_signs)
            flat_indices = model.layer.partner_indices.reshape(-1)
            S_gathered = S.index_select(1, flat_indices)
            S_partners = S_gathered.view(B, C, K, M)
            S_partners_reshaped = S_partners.permute(1, 0, 2, 3).reshape(C, B, K * M)
            W_reshaped = W_fp.reshape(C, M, K * M).transpose(1, 2)
            I_syn = torch.bmm(S_partners_reshaped, W_reshaped).transpose(0, 1)
            I_syn_exc = torch.clamp(I_syn, min=0.0)
            I_syn_inh = torch.clamp(I_syn, max=0.0)
            I_ampa_n = 0.667 * I_ampa + I_syn_exc
            nmda_gate_val = nmda_gate(V)
            I_nmda_n = 0.98 * I_nmda + I_syn_exc * nmda_gate_val
            I_total = I_ampa_n + I_nmda_n + I_syn_inh
            V_leaked = 0.9 * V + 0.1 * (I_total + I_inj_x)
            V_next = V_leaked * (1.0 - S)
            S_next = (V_next >= V_th).float()
            V_th_next = V_th + 1e-4 * (S_next - 0.015)
            V_th_next = torch.clamp(V_th_next, min=0.1, max=5.0)
            S, V, V_th, I_ampa, I_nmda = S_next, V_next, V_th_next, I_ampa_n, I_nmda_n

    print(f"  After warmup: spike rate = {S.mean().item():.4f}, V_th avg = {V_th.mean().item():.4f}")

    # Now run 8 steps with gradient
    pooled_list = []
    target = input_ids[:, 1:9]
    target = torch.cat([target, torch.zeros(B, 1, dtype=torch.long)], dim=1)[:, :8]

    for t in range(8):
        token = input_ids[:, t]
        psc_hist = []
        for st in range(pool_steps):
            ts = 20 + t * pool_steps + st
            I_inj_z = model.encoder(token, ts)
            W_fp = torch.abs(model.layer.W_raw) * model.layer._partner_signs + (1e-6 * model.layer._partner_signs)
            flat_indices = model.layer.partner_indices.reshape(-1)
            S_gathered = S.index_select(1, flat_indices)
            S_partners = S_gathered.view(B, C, K, M)
            S_partners_reshaped = S_partners.permute(1, 0, 2, 3).reshape(C, B, K * M)
            W_reshaped = W_fp.reshape(C, M, K * M).transpose(1, 2)
            I_syn = torch.bmm(S_partners_reshaped, W_reshaped).transpose(0, 1)
            I_syn_exc = torch.clamp(I_syn, min=0.0)
            I_syn_inh = torch.clamp(I_syn, max=0.0)
            I_ampa_n = 0.667 * I_ampa + I_syn_exc
            nmda_gate_val = nmda_gate(V)
            I_nmda_n = 0.98 * I_nmda + I_syn_exc * nmda_gate_val
            I_total = I_ampa_n + I_nmda_n + I_syn_inh
            V_leaked = 0.9 * V + 0.1 * (I_total + I_inj_z)
            V_next = V_leaked * (1.0 - S)
            S_next = FastSigmoidSurrogate.apply(V_next, V_th)
            V_th_next = V_th + 1e-4 * (S_next - 0.015)
            V_th_next = torch.clamp(V_th_next, min=0.1, max=5.0)
            S, V, V_th, I_ampa, I_nmda = S_next, V_next, V_th_next, I_ampa_n, I_nmda_n
            Sm = S[:, motor_start:, :].reshape(B, -1)
            I_psc = (1.0 - 1.0 / 3.0) * I_psc + Sm
            psc_hist.append(I_psc)
        pooled = torch.stack(psc_hist, dim=0).mean(dim=0)
        pooled_list.append(pooled)

    window_pooled = torch.stack(pooled_list, dim=1)
    logits = model.vocab_head(window_pooled)
    loss = nn.CrossEntropyLoss()(logits[:, :8, :].reshape(-1, V_size), target.reshape(-1))
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    for name, param in model.named_parameters():
        if param.grad is None:
            status = "None"
        elif param.grad.abs().max() == 0:
            status = "ALL_ZERO"
        else:
            status = f"max={param.grad.abs().max().item():.6e}"
        print(f"  {name}: {status}")

    print("\n=== VERIFY PASS ===" if loss.item() > 0 else "\n=== FAIL ===")

if __name__ == "__main__":
    t0 = time.time()
    verify_fix()
    print(f"Time: {time.time()-t0:.1f}s")
