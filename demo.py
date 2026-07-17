#!/usr/bin/env python3
"""SRCN SNN 生成演示 - Loss 4.xx, 396M 脉冲神经网络"""

import torch, time
from srcn_model import SRCNv3_1B
from dataset import get_or_create_tokenizer

tokenizer = get_or_create_tokenizer("annotated_corpus.jsonl", "vocab_tokenizer_v3.pkl")
V = tokenizer.vocab_size

ckpt = torch.load("checkpoint.pt", map_location='cpu', weights_only=True)
model = SRCNv3_1B(vocab_size=V, num_columns=160, neurons_per_column=384,
                   num_partners=8, num_motor_pool_steps=4, motor_ratio=0.54)
model.load_state_dict(ckpt['model'])
model.eval()

C, M, ms, nm = 160, 384, model.motor_start_col, model.num_motor_neurons
print(f"模型: {sum(p.numel() for p in model.parameters())/1e6:.0f}M参数 | Motor: {nm:,}神经元 | 词表: {V}")
print(f"训练Loss: {ckpt.get('epoch','?')} epoch | SR ~13%")
print("=" * 60)

def generate(prompt, max_new=15, temperature=0.7, top_p=0.9, rep_penalty=1.2):
    ids = tokenizer.encode(prompt)
    S = torch.zeros(1, C, M); V_ = torch.zeros(1, C, M)
    V_th = torch.full((1, C, M), 2.0); Ia = torch.zeros(1, C, M)
    Inmda = torch.zeros(1, C, M); I_psc = torch.zeros(1, nm)
    ts = 0
    def step(token_id):
        nonlocal S, V_, V_th, Ia, Inmda, I_psc, ts
        token = torch.tensor([[token_id]], dtype=torch.long)
        S, V_, V_th, Ia, Inmda, I_psc, logits, _, _ = model(
            S, V_, V_th, Ia, Inmda, I_psc, token, torch.tensor(ts)
        )
        ts += model.num_motor_pool_steps
        return logits[0, -1]

    generated_ids = []
    with torch.no_grad():
        for token_id in ids[:-1]:  # leave EOS unconsumed, as in training targets
            logits = step(token_id)
        
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        output = prompt + tokenizer.decode([next_id])
        generated_ids.append(next_id)
        
        for _ in range(max_new - 1):
            logits = step(next_id)
            
            # Apply repetition penalty
            for gid in set(generated_ids + ids):
                logits[gid] /= rep_penalty
                
            logits = logits / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_id)
            c = tokenizer.decode([next_id])
            output += c
            if c == '' or len(output) > 300:
                break
    return output

examples = [
    # 基础语义
    ("今天天气", "天气描述"),
    ("小明和小红", "数学推理"),
    ("人工智能", "概念续写"),
    ("这本书的作者", "作品联系"),
    # 因果推理
    ("因为下雨所以", "因果逻辑"),
    ("如果明天不下雨", "条件推理"),
    # 数学题
    ("小红有5个苹果，小明有3个", "数学解题"),
    ("一个长方形长是10", "几何推理"),
    # 日常
    ("我喜欢吃", "食物偏好"),
    ("中国的首都是", "常识填空"),
    # 长句
    ("他昨天去了超市买了很多", "长上下文"),
    ("这个问题很难，但是", "转折推理"),
]

print("\n提示词 → 模型生成 (CPU纯推理 (无GPU))")
print("-" * 60)
for prompt, desc in examples:
    start = time.time()
    gen = generate(prompt, max_new=12)
    elapsed = time.time() - start
    print(f"\n[{desc}]")
    print(f"  输入: {prompt}")
    print(f"  输出: {gen}")
    print(f"  耗时: {elapsed:.1f}s")

print("\n" + "=" * 60)
print("SRCN v2 - 脉冲神经网络中文语言模型")
print("https://github.com/sensen02/SNN")
