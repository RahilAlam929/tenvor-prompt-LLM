from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.txt"
OUT = ROOT / "checkpoints" / "tenvor-gpt.pt"

torch.manual_seed(42)
device = torch.device("cpu")

text = DATA.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
if len(data) < 20:
    raise ValueError("Training dataset bahut chhota hai.")

split = max(1, int(0.9 * len(data)))
train_data = data[:split]
val_data = data[split:] if split < len(data) else data[:split]

BATCH_SIZE = 16
BLOCK_SIZE = min(64, len(train_data) - 1)
STEPS = 500
EVAL_EVERY = 100
LR = 1e-3
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
VOCAB_SIZE = len(chars)


def get_batch(source):
    max_start = len(source) - BLOCK_SIZE - 1
    if max_start < 0:
        source = train_data
        max_start = len(source) - BLOCK_SIZE - 1
    starts = torch.randint(max_start + 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            N_EMBD, N_HEAD, dropout=DROPOUT, batch_first=True
        )
        self.norm1 = nn.LayerNorm(N_EMBD)
        self.norm2 = nn.LayerNorm(N_EMBD)
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        length = x.size(1)
        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class TenvorGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.norm = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


model = TenvorGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Device: {device}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Training characters: {len(train_data)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            x, y = get_batch(train_data)
            _, train_loss = model(x, y)
            source = val_data if len(val_data) > BLOCK_SIZE else train_data
            vx, vy = get_batch(source)
            _, val_loss = model(vx, vy)
        print(
            f"step {step:4d} | train loss {train_loss.item():.4f}"
            f" | validation loss {val_loss.item():.4f}"
        )
        model.train()

    if step == STEPS:
        break

    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "chars": chars,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
    },
    OUT,
)
print(f"\nTraining complete. Checkpoint saved: {OUT}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = model.generate(context, max_new_tokens=300)[0].tolist()
print("\n--- Sample generated text ---\n")
print(decode(sample))
