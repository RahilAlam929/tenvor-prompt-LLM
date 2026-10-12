from pathlib import Path
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

if len(text) < 200:
    raise ValueError("Training dataset bahut chhota hai.")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

BLOCK_SIZE = min(64, len(train_data) - 2, len(val_data) - 2)
BATCH_SIZE = 16
STEPS = 600
EVAL_EVERY = 100
PATIENCE = 3
LR = 1e-3
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
VOCAB_SIZE = len(chars)


def get_batch(source):
    if len(source) <= BLOCK_SIZE + 1:
        raise ValueError("Dataset split sequence length ke liye chhota hai.")

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
        attention, _ = self.attn(
            x, x, x, attn_mask=mask, need_weights=False
        )
        x = self.norm1(x + attention)
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
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
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


@torch.no_grad()
def estimate_loss(model, source, batches=10):
    model.eval()
    losses = []

    for _ in range(batches):
        x, y = get_batch(source)
        _, loss = model(x, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)


model = TenvorGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_val_loss = float("inf")
evaluations_without_improvement = 0

print(f"Device: {device}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Training characters: {len(train_data)}")
print(f"Validation characters: {len(val_data)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Sequence length: {BLOCK_SIZE}")
print("Training started...\n")

for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        train_loss = estimate_loss(model, train_data)
        val_loss = estimate_loss(model, val_data)

        print(
            f"step {step:4d} | train loss {train_loss:.4f}"
            f" | validation loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            evaluations_without_improvement = 0

            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "chars": chars,
                    "block_size": BLOCK_SIZE,
                    "n_embd": N_EMBD,
                    "n_head": N_HEAD,
                    "n_layer": N_LAYER,
                    "best_val_loss": best_val_loss,
                    "step": step,
                },
                OUT,
            )
            print("  Best checkpoint saved.")
        else:
            evaluations_without_improvement += 1

        if evaluations_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation loss improve nahi hui.")
            break

        model.train()

    if step == STEPS:
        break

    x, y = get_batch(train_data)
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

checkpoint = torch.load(OUT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"\nTraining complete.")
print(f"Best step: {checkpoint['step']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
print(f"Checkpoint: {OUT}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = model.generate(context, max_new_tokens=300)[0].tolist()
print("\n--- Sample generated text ---\n")
print("".join(itos[token] for token in sample))
