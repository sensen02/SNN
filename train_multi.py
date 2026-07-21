import os, time, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import get_or_create_tokenizer, PackedChineseDataset


class FocalLoss(nn.Module):
    """Focal loss: down-weights easy negatives to fix class imbalance over large vocab"""
    def __init__(self, gamma=2.0, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             ignore_index=self.ignore_index,
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
from srcn_model import SRCNv3_1B

def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    C_default, M_default, K_default = 160, 384, 8
    C = int(os.environ.get("SRCN_C", str(C_default)))
    M = int(os.environ.get("SRCN_M", str(M_default)))
    K = int(os.environ.get("SRCN_K", str(K_default)))
    B = int(os.environ.get("SRCN_B", "64"))
    seq_len = 512
    bptt_steps = 32
    pool_steps = 4
    # Stabilized Learning Rates (Reduced by 5x to guarantee absolute stability for long-term training)
    lr = 0.00006
    lr_enc = 0.00006
    lr_w = 0.000024
    lr_w_wd = 5e-3
    lr_mlp_out = 0.00002
    lr_enc_head = 0.002
    grad_clip = 0.3
    save_interval = 1800  # 30 min

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus = os.path.join(script_dir, "annotated_corpus.jsonl")
    tokenizer = get_or_create_tokenizer(corpus, os.path.join(script_dir, "vocab_tokenizer_v3.pkl"))
    V_size = tokenizer.vocab_size

    # Load dataset using memmap for massive scale
    dataset = PackedChineseDataset(
        corpus_path='/home/linux/srcn_v2_balanced/annotated_corpus.jsonl',
        tokenizer=tokenizer,
        chunk_len=seq_len,
        cache_path='/home/linux/srcn_v2_balanced/packed_dataset_340m.pkl',
        bin_path='/data/massive_corpus/skypile_massive.bin'
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=B, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)
    if rank == 0:
        print(f"World: {world_size} GPUs | B/GPU: {B} | Eff B: {B*world_size} | bptt: {bptt_steps}")
        print(f"Dataset: {len(dataset)} chunks | Batches/epoch: {len(loader)}")

    model = SRCNv3_1B(vocab_size=V_size, num_columns=C, neurons_per_column=M, num_partners=K, num_motor_pool_steps=pool_steps).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    total = sum(p.numel() for p in model.module.parameters())
    if rank == 0:
        print(f"Params: {total:,} ({total/1e9:.3f}B) | C={C} M={M} K={K}")

    motor_start = model.module.motor_start_col
    num_motor = model.module.num_motor_neurons

    # 5 param groups: embedding(100x LR), encoder, W_raw, MLP-in, MLP-out
    emb_params = []
    enc_params = []
    w_raw_params = []
    mlp_in_params = []
    mlp_out_params = []
    enc_head_params = []
    for name, param in model.named_parameters():
        if 'encoder.embedding' in name:
            emb_params.append(param)
        elif 'W_raw' in name:
            w_raw_params.append(param)
        elif 'vocab_head.3' in name:
            mlp_out_params.append(param)
        elif 'vocab_head' in name:
            mlp_in_params.append(param)
        elif 'encoder_head' in name or 'embedding_decoder' in name:
            enc_head_params.append(param)
        else:
            enc_params.append(param)
    opt = optim.AdamW([
        {'params': emb_params, 'lr': 0.006, 'weight_decay': 1e-4},
        {'params': enc_params, 'lr': lr_enc, 'weight_decay': 1e-4},
        {'params': w_raw_params, 'lr': lr_w, 'weight_decay': lr_w_wd},
        {'params': mlp_in_params, 'lr': lr, 'weight_decay': 0.0},
        {'params': mlp_out_params, 'lr': lr_mlp_out, 'weight_decay': 0.0},
        {'params': enc_head_params, 'lr': lr_enc_head, 'weight_decay': 0.0},
    ])
    if rank == 0:
        print(f"LR: emb=0.006, enc={lr_enc}, enc_head={lr_enc_head}, mlp_in={lr}, mlp_out={lr_mlp_out}, W_raw={lr_w}(wd={lr_w_wd})")

    criterion = FocalLoss(gamma=1.0, ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    # ===== NEVER DELETE checkpoint.pt =====
    # Auto-resume on restart. Crash-safe.
    # ===== NEVER DELETE checkpoint.pt =====
    ckpt_path = os.path.join(script_dir, "checkpoint.pt")
    start_epoch = 0
    start_batch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.module.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            # Checkpoint optimizer state contains the old shortcut-dominated LRs.
            # Restore the balanced schedule explicitly after loading it.
            for group, group_lr in zip(opt.param_groups, [0.006, lr_enc, lr_w, lr, lr_mlp_out, lr_enc_head]):
                group["lr"] = group_lr
        else:
            if rank == 0:
                print("No optimizer state found in checkpoint, starting with fresh optimizer (cleared momentum).")
        start_epoch = ckpt["epoch"]
        start_batch = ckpt.get("batch_idx", 0)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch+1} batch {start_batch}")

    if rank == 0:
        print("Starting training...\n")

    torch.cuda.synchronize()
    t0 = time.time()
    last_save_time = time.time()
    total_tokens = 0
    num_epochs = 20

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch_x in enumerate(loader):
            # Skip batches already processed before checkpoint
            if batch_idx < start_batch:
                continue
            start_batch = 0  # reset after first epoch
            batch_x = batch_x.to(device)
            S = torch.zeros(B, C, M, device=device)
            V = torch.zeros(B, C, M, device=device)
            V_th = torch.full((B, C, M), 2.0, device=device)
            I_ampa = torch.zeros(B, C, M, device=device)
            I_nmda = torch.zeros(B, C, M, device=device)
            I_psc = torch.zeros(B, num_motor, device=device)
            window_loss = 0.0
            n_windows = 0
            total_spikes = 0.0
            total_steps = 0

            t_ranges = list(range(0, seq_len - 1, bptt_steps))
            opt.zero_grad(set_to_none=True)
            batch_valid = True
            for wi, t_start in enumerate(t_ranges):
                t_end = min(t_start + bptt_steps, seq_len - 1)
                win_tokens = batch_x[:, t_start:t_end]
                target = batch_x[:, t_start + 1:t_end + 1]
                num_win_tokens = win_tokens.shape[1]
                # DDP requires the forward pass itself to be inside no_sync().
                sync_context = model.no_sync() if wi < len(t_ranges) - 1 else nullcontext()
                with sync_context:
                    S, V, V_th, I_ampa, I_nmda, I_psc, logits, spikes_sum, loss_emb = model(
                        S, V, V_th, I_ampa, I_nmda, I_psc, win_tokens,
                        torch.tensor(t_start, device=device),
                    )
                    loss_vocab = criterion(logits.view(-1, V_size), target.reshape(-1))
                    loss = loss_vocab + 1.0 * loss_emb  # combine vocab + embedding reconstruction
                    if torch.isfinite(loss):
                        scaled_loss = loss * (num_win_tokens / (seq_len - 1))
                        scaled_loss.backward()
                total_spikes += spikes_sum.item()
                total_steps += num_win_tokens * pool_steps
                window_loss += loss.item()
                n_windows += 1
                if not torch.isfinite(loss) or loss.item() > 20.0:
                    batch_valid = False
                    if rank == 0:
                        if not torch.isfinite(loss):
                            print(f"  [!] NaN/Inf at E{epoch+1}B{batch_idx+1:04d} W{wi}")
                        else:
                            print(f"  [!] Loss explosion ({loss.item():.2f} > 20.0) at E{epoch+1}B{batch_idx+1:04d} W{wi}")
                    break
                S, V, V_th, I_ampa, I_nmda, I_psc = [x.detach() for x in [S, V, V_th, I_ampa, I_nmda, I_psc]]

            grad_ok = batch_valid and all(
                param.grad is None or torch.isfinite(param.grad).all()
                for param in model.parameters()
            )
            if grad_ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                if rank == 0:
                    print(f"  [!] Skipped E{epoch+1}B{batch_idx+1:04d}: non-finite loss or accumulated gradient")

            avg = window_loss / max(n_windows, 1)
            epoch_loss += avg
            n_batches += 1
            total_tokens += B * seq_len * world_size
            elapsed = time.time() - t0
            cur_mem = torch.cuda.memory_allocated() / 1e9
            max_mem = torch.cuda.max_memory_allocated() / 1e9
            if rank == 0:
                spike_rate = total_spikes / max(total_steps * B * C * M, 1) if total_steps > 0 else 0.0
                print(f"E{epoch+1}B{batch_idx+1:04d} | Loss: {avg:.6f} | Tok/s: {total_tokens/elapsed:.0f} | VRAM: {cur_mem:.2f}/{max_mem:.2f}GB | SR: {spike_rate:.4f}")

            # Periodic checkpoint
            if rank == 0 and time.time() - last_save_time > save_interval:
                epoch_ckpt_path = os.path.join(script_dir, "checkpoint.pt")
                # Save to temporary file first
                tmp_path = f"{epoch_ckpt_path}.tmp"
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx + 1,
                    'model': model.module.state_dict(),
                    'optimizer': opt.state_dict(),
                }, tmp_path)
                # Rename to final path
                os.replace(tmp_path, epoch_ckpt_path)
                last_save_time = time.time()
                print(f"  [Checkpoint saved at {int(time.time() - t0)}s] E{epoch+1}B{batch_idx+1} -> {epoch_ckpt_path}")

            # Defragment CUDA allocator every 100 batches
            if batch_idx > 0 and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg_epoch = epoch_loss / max(n_batches, 1)
        if rank == 0:
            print(f"\n=== Epoch {epoch+1} done | Avg loss: {avg_epoch:.4f} | Elapsed: {time.time()-t0:.0f}s ===\n")
            epoch_ckpt_path = os.path.join(script_dir, f"checkpoint_e{epoch+1}_final.pt")
            tmp_path = f"{epoch_ckpt_path}.tmp"
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "batch_idx": 0
            }, tmp_path)
            os.replace(tmp_path, epoch_ckpt_path)
            import shutil
            shutil.copy(epoch_ckpt_path, os.path.join(script_dir, "checkpoint.pt"))
            last_save_time = time.time()
            # Keep exactly the last 2 epoch checkpoints (epoch-1 and epoch-2)
            try:
                epoch_files = [f for f in os.listdir(script_dir) if f.startswith('checkpoint_e') and f.endswith('_final.pt')]
                epoch_files.sort(key=lambda x: os.path.getmtime(os.path.join(script_dir, x)))
                while len(epoch_files) > 2: # Keep latest 2 epoch final checkpoints
                    old_ckpt = epoch_files.pop(0)
                    os.remove(os.path.join(script_dir, old_ckpt))
            except Exception as e:
                print(f"  [!] Failed to clean up epoch checkpoints: {e}")
        start_batch = 0  # Ensure start_batch is reset if the previous epoch was completed early or skipped

    dist.destroy_process_group()
    if rank == 0:
        print(f"Done in {time.time()-t0:.0f}s | Total tokens: {total_tokens:,}")

if __name__ == "__main__":
    train()
