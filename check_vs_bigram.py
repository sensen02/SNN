"""Is the model better than a bigram? Does context beyond last char matter?"""
import json, pickle, collections, os, torch
torch.set_num_threads(8)
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "vocab_tokenizer_v3.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
V_size = tokenizer.vocab_size

bigram = collections.defaultdict(collections.Counter)
trigram = collections.defaultdict(collections.Counter)
lines = []
with open(os.path.join(script_dir, "annotated_corpus.jsonl")) as f:
    for i, line in enumerate(f):
        if i >= 8000: break
        lines.append(json.loads(line)["text"])
for t in lines:
    for a, b in zip(t, t[1:]):
        bigram[a][b] += 1
    for a, b, c in zip(t, t[1:], t[2:]):
        trigram[(a, b)][c] += 1

texts = []
with open(os.path.join(script_dir, "annotated_corpus.jsonl")) as f:
    for i, line in enumerate(f):
        if i % 3000 == 0:
            t = json.loads(line)["text"]
            if len(t) > 200: texts.append(t[:200])
        if len(texts) >= 4: break

def ngram_acc(texts, seq=129):
    b_hit = t_hit = tot = 0
    for t in texts:
        chars = [c for c in t if c in tokenizer.char_to_id][:seq-1]
        for j in range(1, len(chars)):
            tot += 1
            cur, tgt = chars[j-1], chars[j]
            if bigram[cur] and bigram[cur].most_common(1)[0][0] == tgt: b_hit += 1
            key = (chars[j-2], cur) if j >= 2 else None
            pred_t = (trigram[key].most_common(1)[0][0] if key and trigram[key]
                      else (bigram[cur].most_common(1)[0][0] if bigram[cur] else None))
            if pred_t == tgt: t_hit += 1
    return b_hit/tot, t_hit/tot, tot

b_acc, t_acc, tot = ngram_acc(texts)
print(f"同一评测集上: bigram acc={b_acc:.1%} | trigram acc={t_acc:.1%} | 模型(上次)=36.3% | n={tot}")

# --- context sensitivity: same last chars, different prefix ---
from srcn_model import SRCNv3_1B
C, M = 160, 384
model = SRCNv3_1B(vocab_size=V_size, num_columns=C, neurons_per_column=M,
                  num_partners=8, num_motor_pool_steps=4)
ckpt = torch.load(os.path.join(script_dir, "checkpoint.pt"), map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"]); model.eval()
print(f"checkpoint: E{ckpt['epoch']+1}B{ckpt.get('batch_idx','?')}")

def next_dist(prompt):
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    S = torch.zeros(1, C, M); Vm = torch.zeros(1, C, M)
    V_th = torch.full((1, C, M), 2.0)
    Ia = torch.zeros(1, C, M); In_ = torch.zeros(1, C, M)
    Ipsc = torch.zeros(1, model.num_motor_neurons)
    with torch.no_grad():
        toks = torch.tensor([ids])
        S, Vm, V_th, Ia, In_, Ipsc, logits, _, _ = model(
            S, Vm, V_th, Ia, In_, Ipsc, toks, torch.tensor(0))
    return torch.log_softmax(logits[0, -1], dim=-1)

pairs = [
    ("小明有3个苹果", "教室里坐着3个"),      # same last 2: "3个"
    ("我们用加法计算", "他不会做减法计算"),   # same last 2: "计算"
    ("今天天气很好", "妈妈说这样很好"),       # same last 2: "很好"
    ("这是一个大的", "刚才那个小的"),         # same last 1: "的"
]
print("\n前文敏感度 (同尾字不同前缀, KL 越大越依赖前文):")
for p1, p2 in pairs:
    d1, d2 = next_dist(p1), next_dist(p2)
    kl = torch.nn.functional.kl_div(d2, d1, log_target=True, reduction="sum").item()
    top1a = tokenizer.id_to_char[d1.argmax().item()]
    top1b = tokenizer.id_to_char[d2.argmax().item()]
    print(f"  '{p1}'→{top1a!r} vs '{p2}'→{top1b!r} | KL={kl:.3f}")
