import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def nmda_gate(v, mg=1.0):
    return 1.0 / (1.0 + mg * torch.exp(-0.062 * v) / 3.57)

class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th):
        ctx.save_for_backward(v, v_th)
        return (v >= v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        v, v_th = ctx.saved_tensors
        diff = torch.abs(v - v_th)
        grad_v = grad_output / ((1.0 + diff) ** 2)
        return grad_v, None

class SRCNLayer(nn.Module):
    def __init__(self, num_columns=256, neurons_per_column=512, num_partners=16):
        super().__init__()
        self.C = num_columns
        self.M = neurons_per_column
        self.K = num_partners

        self.W_raw = nn.Parameter(
            torch.randn(self.C, self.M, self.K, self.M) * (0.02 / (self.K * self.M) ** 0.5)
        )

        num_excitatory = int(self.C * 200 / 256)
        num_inhibitory = self.C - num_excitatory

        partner_indices = torch.zeros(self.C, self.K, dtype=torch.long)
        partner_signs = torch.zeros(self.C, self.K)

        half_k = self.K // 2
        for c in range(self.C):
            for k_idx in range(self.K):
                partner_col = (c - half_k + k_idx) % self.C
                partner_indices[c, k_idx] = partner_col
                partner_signs[c, k_idx] = 1.0 if partner_col < num_excitatory else -1.0

        self.register_buffer("partner_indices", partner_indices)
        self.register_buffer("_partner_signs", partner_signs.view(self.C, 1, self.K, 1))

    @property
    def device(self):
        return self.W_raw.device

    def precompute_W(self):
        return torch.abs(self.W_raw) * self._partner_signs + (1e-6 * self._partner_signs)

    def forward(self, S_prev, V, V_th, I_ampa, I_nmda, I_inj,
                tau_mem=0.9, epsilon=1e-4, a_target=0.015,
                alpha_ampa=0.667, alpha_nmda=0.98, W_fp16=None):
        if W_fp16 is None:
            W_fp16 = self.precompute_W()

        batch_size = S_prev.shape[0]

        flat_indices = self.partner_indices.reshape(-1)
        S_gathered = S_prev.index_select(1, flat_indices)
        S_partners = S_gathered.view(batch_size, self.C, self.K, self.M)

        S_partners_reshaped = S_partners.permute(1, 0, 2, 3).reshape(self.C, batch_size, self.K * self.M)
        W_reshaped = W_fp16.reshape(self.C, self.M, self.K * self.M).transpose(1, 2)

        I_syn = torch.bmm(S_partners_reshaped, W_reshaped).transpose(0, 1)
        I_syn = torch.clamp(I_syn, min=-500.0, max=500.0)

        I_syn_exc = torch.clamp(I_syn, min=0.0)
        I_syn_inh = torch.clamp(I_syn, max=0.0)

        I_ampa_next = alpha_ampa * I_ampa + I_syn_exc
        nmda_gate_val = nmda_gate(V)
        I_nmda_next = alpha_nmda * I_nmda + I_syn_exc * nmda_gate_val
        I_nmda_next = torch.clamp(I_nmda_next, max=1000.0)

        I_total = I_ampa_next + I_nmda_next + I_syn_inh

        V_leaked = tau_mem * V + (1.0 - tau_mem) * (I_total + I_inj)
        V_next = V_leaked * (1.0 - S_prev)
        V_next = torch.clamp(V_next, min=-100.0, max=100.0)

        S_next = FastSigmoidSurrogate.apply(V_next, V_th)

        V_th_next = V_th + epsilon * (S_next - a_target)
        V_th_next = torch.clamp(V_th_next, min=0.1, max=5.0)

        return S_next, V_next, V_th_next, I_ampa_next, I_nmda_next

class TemporalPhaseEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_columns=256, neurons_per_column=512, gain=13.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, num_columns * neurons_per_column, bias=False)
        self.C = num_columns
        self.M = neurons_per_column
        self.gain = gain

    def forward(self, token_ids, timestep, freq=80.0, tau_mem=0.9):
        embed = self.embedding(token_ids)
        phase_offsets = self.proj(embed)
        t = timestep * 0.001
        I_inj = torch.sin(2.0 * 3.141592653589793 * freq * t + phase_offsets)
        return (I_inj * self.gain).view(-1, self.C, self.M)

class SRCNv3_1B(nn.Module):
    def __init__(self, vocab_size, num_columns=256, neurons_per_column=512,
                 num_partners=16, embed_dim=512, num_motor_pool_steps=8, encoder_gain=13.0,
                 motor_ratio=0.54):
        super().__init__()
        self.C = num_columns
        self.M = neurons_per_column
        self.num_motor_pool_steps = num_motor_pool_steps

        motor_start = int(self.C * (1.0 - motor_ratio))
        self.motor_start_col = motor_start
        self.num_motor_neurons = (self.C - motor_start) * self.M

        self.encoder = TemporalPhaseEncoder(vocab_size, embed_dim, self.C, self.M, gain=encoder_gain)
        self.layer = SRCNLayer(self.C, self.M, num_partners)
        self.vocab_head = nn.Sequential(
            nn.LayerNorm(self.num_motor_neurons),
            nn.Linear(self.num_motor_neurons, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, vocab_size, bias=True),
        )
        # Encoder readout: maps to MLP hidden dim (no class imbalance)
        self.encoder_head = nn.Linear(embed_dim, 4096, bias=False)

    @property
    def device(self):
        return self.layer.W_raw.device

    def get_projected_weights(self):
        return self.layer.get_projected_weights()

    def precompute_W(self):
        return self.layer.precompute_W()

    def forward_step(self, S_prev, V, V_th, I_ampa, I_nmda, input_token_id, timestep, W_fp16=None):
        I_inj = self.encoder(input_token_id, timestep)
        if W_fp16 is None:
            W_fp16 = self.layer.precompute_W()
        S_next, V_next, V_th_next, I_ampa_next, I_nmda_next = self.layer(
            S_prev, V, V_th, I_ampa, I_nmda, I_inj,
            0.9, 5e-5, 0.10, 0.667, 0.98, W_fp16
        )
        return S_next, V_next, V_th_next, I_ampa_next, I_nmda_next

    def forward(self, S, V, V_th, I_ampa, I_nmda, I_psc, win_tokens, t_start_tensor):
        """Run one truncated-BPTT window through the DDP-visible forward path."""
        W_fp16 = self.precompute_W()
        pooled_list, enc_list, emb_list = [], [], []
        spikes_sum = 0.0
        t_start = int(t_start_tensor.item())
        for t in range(win_tokens.shape[1]):
            token = win_tokens[:, t]
            ts_start = torch.tensor(t_start + t * self.num_motor_pool_steps,
                                    device=win_tokens.device)
            S, V, V_th, I_ampa, I_nmda, I_psc, pooled_token, I_enc, token_spikes = checkpoint(
                self.forward_token_with_psc,
                S, V, V_th, I_ampa, I_nmda, I_psc, token, ts_start, W_fp16,
                # Non-reentrant checkpointing is compatible with repeated DDP parameter use.
                use_reentrant=False,
            )
            spikes_sum = spikes_sum + token_spikes
            pooled_list.append(pooled_token)
            enc_list.append(I_enc)
            emb_list.append(self.encoder.embedding(token))

        window_pooled = torch.stack(pooled_list, dim=1)
        window_enc = torch.stack(enc_list, dim=1)
        # Recurrent path: motor → hidden (before ReLU)
        motor_input = self.vocab_head[0](window_pooled + window_enc)  # LayerNorm
        motor_hidden = self.vocab_head[1](motor_input)                # Linear1 (B,T,4096)
        # Encoder path: embedding → hidden (same 4096-dim, no class imbalance)
        enc_hidden = self.encoder_head(torch.stack(emb_list, dim=1))  # (B,T,4096)
        # Combine and classify
        combined = torch.relu(motor_hidden + enc_hidden)
        logits = self.vocab_head[3](combined)  # Linear2 → (B,T,8455)
        return S, V, V_th, I_ampa, I_nmda, I_psc, logits, spikes_sum

    def forward_token_with_psc(self, S, V, V_th, I_ampa, I_nmda, I_psc, token, t_start_tensor, W_fp16):
        psc_hist = []
        spikes_sum = 0.0
        t_start = int(t_start_tensor.item())
        I_enc = torch.zeros_like(I_psc)
        for st in range(self.num_motor_pool_steps):
            ts = t_start + st
            I_inj = self.encoder(token, ts)
            S, V, V_th, I_ampa, I_nmda = self.layer(
                S, V, V_th, I_ampa, I_nmda, I_inj,
                0.9, 5e-5, 0.10, 0.667, 0.98, W_fp16
            )
            spikes_sum = spikes_sum + S.sum()
            Sm = S[:, self.motor_start_col:, :].reshape(S.shape[0], -1)
            I_psc = (1.0 - 1.0 / 3.0) * I_psc + Sm
            I_inj_m = I_inj[:, self.motor_start_col:, :].reshape(S.shape[0], -1)
            I_enc = (1.0 - 1.0 / 3.0) * I_enc + I_inj_m * 0.1
            psc_hist.append(I_psc)
        pooled_token = torch.stack(psc_hist, dim=0).mean(dim=0)
        return S, V, V_th, I_ampa, I_nmda, I_psc, pooled_token, I_enc, spikes_sum
