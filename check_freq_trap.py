"""CPU-only check: is the model collapsing to high-frequency tokens?"""
import json, pickle, collections, os, torch

torch.set_num_threads(8)
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "vocab_tokenizer_v3.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
V_size = tokenizer.vocab_size
print(f"Vocab: {V_size}")

# --- corpus char frequency (sample 8000 lines) ---
freq = collections.Counter()
with open(os.path.join(script_dir, "annotated_corpus.jsonl")) as f:
    for i, line in enumerate(f):
        if i >= 8000:
            break
        try:
            freq.update(json.loads(line)["text"])
        except Exception:
            pass
total_chars = sum(freq.values())
top20 = freq.most_common(20)
top20_share = sum(c for _, c in top20) / total_chars
top20_ids = {tokenizer.char_to_id[ch] for ch, _ in top20 if ch in tokenizer.char_to_id}
print("语料 top20 高频字:", "".join(repr(ch)[1:-1] for ch, _ in top20))
print(f"语料中 top20 字占比: {top20_share:.1%}")

# --- load model on CPU ---
from srcn_model import SRCNv3_1B
C, M, K, pool_steps = 160, 384, 8, 4
model = SRCNv3_1B(vocab_size=V_size, num_columns=C, neurons_per_column=M,
                  num_partners=K, num_motor_pool_steps=pool_steps)
ckpt = torch.load(os.path.join(script_dir, "checkpoint.pt"),
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded checkpoint: epoch {ckpt['epoch']+1}, batch {ckpt.get('batch_idx', '?')}")

# --- eval samples from corpus (skip lines used above is fine; just diverse ones) ---
texts = []
with open(os.path.join(script_dir, "annotated_corpus.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 3000 == 0:
            t = json.loads(line)["text"]
            if len(t) > 200:
                texts.append(t[:200])
        if len(texts) >= 4:
            break
B = len(texts)
seq = 129
batch = torch.full((B, seq), tokenizer.pad_id, dtype=torch.long)
for bi, t in enumerate(texts):
    ids = tokenizer.encode(t, add_bos=True, add_eos=False)[:seq]
    batch[bi, :len(ids)] = torch.tensor(ids)

S = torch.zeros(B, C, M); V = torch.zeros(B, C, M)
V_th = torch.full((B, C, M), 2.0)
Ia = torch.zeros(B, C, M); In_ = torch.zeros(B, C, M)
Ipsc = torch.zeros(B, model.num_motor_neurons)

all_logits = []
bptt = 32
with torch.no_grad():
    for t_start in range(0, seq - 1, bptt):
        t_end = min(t_start + bptt, seq - 1)
        win = batch[:, t_start:t_end]
        S, V, V_th, Ia, In_, Ipsc, logits, _, _ = model(
            S, V, V_th, Ia, In_, Ipsc, win, torch.tensor(t_start))
        all_logits.append(logits)
        print(f"  window {t_start}-{t_end} done")

logits = torch.cat(all_logits, dim=1)          # (B, seq-1, V)
targets = batch[:, 1:seq]                       # (B, seq-1)
preds = logits.argmax(dim=-1)

mask = targets != tokenizer.pad_id
p, tgt = preds[mask], targets[mask]
n = p.numel()

pred_counter = collections.Counter(p.tolist())
uniq = len(pred_counter)
top1_pred, top1_cnt = pred_counter.most_common(1)[0]
in_top20 = sum(1 for x in p.tolist() if x in top20_ids) / n
tgt_in_top20 = sum(1 for x in tgt.tolist() if x in top20_ids) / n
acc = (p == tgt).float().mean().item()

# baseline: always predict corpus most frequent char
mode_ch, _ = top20[0]
mode_id = tokenizer.char_to_id[mode_ch]
base_acc = (tgt == mode_id).float().mean().item()

probs = torch.softmax(logits[mask.unsqueeze(-1).expand_as(logits)].view(-1, V_size), dim=-1)
mean_dist = probs.mean(0)
ent_mean_dist = -(mean_dist * (mean_dist + 1e-12).log()).sum().item()
mean_ent = -(probs * (probs + 1e-12).log()).sum(-1).mean().item()

print("\n========== 高频词陷阱检查 ==========")
print(f"有效预测位置: {n}")
print(f"预测 unique 字数: {uniq}")
print(f"最常被预测的字: {tokenizer.id_to_char[top1_pred]!r} 占比 {top1_cnt/n:.1%}")
print("预测 top10:", [(tokenizer.id_to_char[i], f"{c/n:.1%}") for i, c in pred_counter.most_common(10)])
print(f"预测落在语料top20高频字的比例: {in_top20:.1%}  (目标本身该比例: {tgt_in_top20:.1%})")
print(f"top-1 accuracy: {acc:.1%}  |  永远猜'{mode_ch}'的baseline: {base_acc:.1%}")
print(f"平均预测熵: {mean_ent:.2f}  |  平均分布的熵: {ent_mean_dist:.2f} (max={torch.log(torch.tensor(float(V_size))):.2f})")
print("\n判定:")
collapse = top1_cnt / n > 0.5 or (in_top20 > 2.5 * tgt_in_top20 and uniq < 50)
if collapse:
    print("  [X] 存在高频词坍缩: 预测严重集中于少数高频字")
elif in_top20 > 1.5 * tgt_in_top20:
    print("  [!] 轻度偏向高频词 (训练早期正常, 继续观察)")
else:
    print("  [OK] 未见高频词陷阱, 预测分布与目标分布量级相当")
