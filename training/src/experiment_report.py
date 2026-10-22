from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
checkpoints = ROOT / "checkpoints"
reports = []

for version in ("v2", "v3"):
    path = checkpoints / f"tenvor-{version}-best.pt"

    if not path.exists():
        print(f"{version.upper()}: checkpoint missing")
        continue

    checkpoint = torch.load(path, map_location="cpu")
    reports.append({
        "version": version.upper(),
        "checkpoint": path.name,
        "step": checkpoint.get("step"),
        "train_loss": checkpoint.get("train_loss"),
        "validation_loss": checkpoint.get("val_loss"),
        "vocabulary_size": checkpoint.get("vocab_size"),
        "config": checkpoint.get("config"),
        "size_kb": round(path.stat().st_size / 1024, 1),
    })

if not reports:
    raise SystemExit("No experiment checkpoints found.")

output = ROOT / "checkpoints" / "experiment-report.json"
output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")

print("\nTenvor experiment comparison")
for report in reports:
    print(
        f"{report['version']}: step={report['step']}, "
        f"train_loss={report['train_loss']:.4f}, "
        f"val_loss={report['validation_loss']:.4f}, "
        f"size={report['size_kb']} KB"
    )

print("\nReport saved:", output)
