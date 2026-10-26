from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(42)

device = torch.device("cpu")
tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer-v4.json")
)

train_text = (ROOT / "data" / "train_v5.txt").read_text(encoding="utf-8")
val_text = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

if bos_id is None or eos_id is None:
    raise SystemExit("Tokenizer must contain [BOS] and [EOS] tokens.")

def encode_lines(text):
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            ids.append(bos_id)
            ids.extend(tokenizer.encode(line).ids)
            ids.append(eos_id)
    return ids

train_ids = encode_lines(train_text)
val_ids = encode_lines(val_text)

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)

seq_len = 64
batch_size = 4
steps = 600
eval_every = 25
patience = 4
min_improvement = 0.01

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
checks_without_improvement = 0

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
        total_loss = 0.0
        total_tokens = 0

        for start in range(0, len(source) - seq_len, seq_len):
            x = source[start:start + seq_len].unsqueeze(0).to(device)
            y = source[start + 1:start + seq_len + 1].unsqueeze(0).to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += y.numel()

        result[name] = total_loss / total_tokens

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

        if metrics["val"] < best_val_loss - min_improvement:
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
                ROOT / "checkpoints" / "tenvor-v6-best.pt",
            )
            checks_without_improvement = 0
            print("  Saved best checkpoint")
        else:
            checks_without_improvement += 1
            print(f"  No meaningful improvement: {checks_without_improvement}/{patience}")
            if checks_without_improvement >= patience:
                print("Early stopping: validation loss stopped improving.")
                break

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("Checkpoint: checkpoints/tenvor-v6-best.pt")
