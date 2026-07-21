import torch
import argparse
from srcn_model import SRCNv3_1B
from dataset import get_or_create_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default="今天天气很")
parser.add_argument('--max_tokens', type=int, default=50)
args = parser.parse_args()

tokenizer = get_or_create_tokenizer("annotated_corpus.jsonl", "vocab_tokenizer_v3.pkl")
V = tokenizer.vocab_size

ckpt = torch.load("checkpoint.pt", map_location='cpu', weights_only=True)
model = SRCNv3_1B(vocab_size=V, num_columns=160, neurons_per_column=384,
                   num_partners=8, num_motor_pool_steps=4, motor_ratio=0.54)
model.load_state_dict(ckpt['model'])
model.eval()

C, M, nm = 160, 384, model.num_motor_neurons
prompt = args.prompt
ids = tokenizer.encode(prompt)
S = torch.zeros(1, C, M); V_ = torch.zeros(1, C, M)
V_th = torch.full((1, C, M), 2.0); Ia = torch.zeros(1, C, M)
Inmda = torch.zeros(1, C, M); I_psc = torch.zeros(1, nm)
ts = 0

def step(token_id):
    global S, V_, V_th, Ia, Inmda, I_psc, ts
    token = torch.tensor([[token_id]], dtype=torch.long)
    S, V_, V_th, Ia, Inmda, I_psc, logits, _, _ = model(
        S, V_, V_th, Ia, Inmda, I_psc, token, torch.tensor(ts)
    )
    ts += model.num_motor_pool_steps
    return logits[0, -1]

with torch.no_grad():
    for token_id in ids[:-1]:
        logits = step(token_id)
    out = prompt
    curr_id = ids[-1]
    for _ in range(args.max_tokens):
        logits = step(curr_id)
        probs = torch.softmax(logits, dim=-1)
        curr_id = torch.argmax(probs).item()
        out += tokenizer.decode([curr_id])
    print("=== GEN RESULT ===")
    print(out)
