import torch
from srcn_model import SRCNv3_1B

def gpu_diag():
    device = torch.device("cuda:0")
    C, M, K = 160, 384, 8
    V_size = 8455
    B = 4
    model = SRCNv3_1B(vocab_size=V_size, num_columns=C, neurons_per_column=M,
                       num_partners=K, num_motor_pool_steps=4, encoder_gain=13.0).to(device)
    input_ids = torch.randint(0, V_size, (B, 32), device=device)

    # Check encoder output
    with torch.no_grad():
        I = model.encoder(input_ids[:, 0], 0)
        print(f"I_inj_max={I.max().item():.2f}, V_contrib={0.1*I.max().item():.2f}")
        print(f"V_ss_max ≈ {0.1*13/0.483:.3f} (calculated)")

    # Test: simulate adaptation
    S = torch.zeros(B, C, M, device=device)
    V = torch.zeros(B, C, M, device=device)
    V_th = torch.full((B, C, M), 2.0, device=device)

    print("\nStep  V_max     SR      V_th")
    for step in range(20):
        I_inj = model.encoder(input_ids[:, 0], step)
        V_leaked = 0.9 * V + 0.1 * I_inj
        V_next = V_leaked * (1.0 - S)
        S_next = (V_next >= V_th).float()
        V_th_next = V_th + 1e-4 * (S_next - 0.015)
        V_th_next = torch.clamp(V_th_next, min=0.1, max=2.5)
        sr = S_next.mean().item()
        print(f"  {step:2d}:  {V_next.max().item():.4f}  {sr:.4f}  {V_th.mean().item():.4f}")
        S, V, V_th = S_next, V_next, V_th_next

if __name__ == "__main__":
    gpu_diag()
