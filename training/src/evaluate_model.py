from pathlib import Path
import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cpu")

tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer-v2.json")
)

checkpoint_path = ROOT / "checkpoints" / "tenvor-v2-best.pt"
if not checkpoint_path.exists():
    raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
config = checkpoint["config"]

model = TenvorLM(
    vocab_size=checkpoint["vocab_size"],
    dim=config["dim"],
    heads=config["heads"],
    layers=config["layers"],
    max_length=config["max_length"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_text = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")
token_ids = tokenizer.encode(val_text).ids
data = torch.tensor(token_ids, dtype=torch.long)

seq_len = 64
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for start in range(0, len(data) - seq_len, seq_len):
        x = data[start:start + seq_len].unsqueeze(0)
        y = data[start + 1:start + seq_len + 1].unsqueeze(0)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

if total_tokens == 0:
    raise SystemExit("Not enough validation tokens.")

mean_loss = total_loss / total_tokens

print("Tenvor V2 Evaluation")
print(f"Checkpoint step: {checkpoint.get('step')}")
print(f"Validation tokens evaluated: {total_tokens}")
print(f"Validation loss: {mean_loss:.4f}")
print(f"Perplexity: {math.exp(min(mean_loss, 50)):.2f}")
print("\nNote: Perplexity is a metric, not a measure of human-readable quality.")
