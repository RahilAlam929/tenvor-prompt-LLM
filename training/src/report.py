import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = ROOT / "checkpoints" / "tenvor-best.pt"
output_path = ROOT / "checkpoints" / "training-report.json"

if not checkpoint_path.exists():
    raise SystemExit("Checkpoint not found. Train Tenvor first.")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model_state = checkpoint["model_state_dict"]
parameter_count = sum(
    tensor.numel()
    for tensor in model_state.values()
)

report = {
    "model": "TenvorLM",
    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    "training_step": checkpoint.get("step"),
    "train_loss": checkpoint.get("train_loss"),
    "validation_loss": checkpoint.get("val_loss"),
    "vocabulary_size": checkpoint.get("vocab_size"),
    "model_config": checkpoint.get("config"),
    "parameter_count_in_state_dict": parameter_count,
}

output_path.write_text(json.dumps(report, indent=2) + "\n")

print("=== Tenvor Training Report ===")
print(json.dumps(report, indent=2))
print(f"\nSaved report: {output_path}")
