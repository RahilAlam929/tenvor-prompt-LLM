from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(42)

device = torch.device("cpu")
tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer.json")
)

train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
val_text = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")

train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)

seq_len = 64
batch_size = 4
steps = 300
eval_every = 50

if len(train_data) <= seq_len or len(val_data) <= seq_len:
    raise SystemExit(
        f"Not enough tokens. Train={len(train_data)}, Val={len(val_data)}. "
        "Add more text to the dataset before training."
    )

model = TenvorLM(
    vocab_size=tokenizer.get_vocab_size(),
    dim=64,
    heads=4,
    layers=2,
    max_length=128,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
best_val_loss = float("inf")

def get_batch(source):
    starts = torch.randint(0, len(source) - seq_len, (batch_size,))
    x = torch.stack([source[i:i + seq_len] for i in starts])
    y = torch.stack([source[i + 1:i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate():
    model.eval()
    result = {}

    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(5):
            x, y = get_batch(source)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)

    model.train()
    return result

model.train()

for step in range(1, steps + 1):
    x, y = get_batch(train_data)
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == 1 or step % eval_every == 0:
        metrics = evaluate()
        print(
            f"Step {step:3d}/{steps} | "
            f"Train loss: {metrics['train']:.4f} | "
            f"Validation loss: {metrics['val']:.4f}"
        )

        if metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.get_vocab_size(),
                    "config": {
                        "dim": 64,
                        "heads": 4,
                        "layers": 2,
                        "max_length": 128,
                    },
                    "step": step,
                    "train_loss": metrics["train"],
                    "val_loss": metrics["val"],
                },
                ROOT / "checkpoints" / "tenvor-best.pt",
            )
            print("  Saved best checkpoint")

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("Checkpoint: checkpoints/tenvor-best.pt")
