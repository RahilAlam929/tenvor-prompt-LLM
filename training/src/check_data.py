from pathlib import Path
import re
import hashlib

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "train": ROOT / "data" / "train.txt",
    "validation": ROOT / "data" / "val.txt",
}

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()

for name, path in FILES.items():
    if not path.exists():
        print(f"{name}: MISSING")
        continue

    text = path.read_text(encoding="utf-8")
    words = re.findall(r"\b[\w']+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    print(f"\n{name.upper()}")
    print("Characters:", len(text))
    print("Words:", len(words))
    print("Sentences:", len(sentences))
    print("Unique words:", len(set(w.lower() for w in words)))
    print("SHA-256:", hashlib.sha256(text.encode()).hexdigest())

train_path = FILES["train"]
val_path = FILES["validation"]

if train_path.exists() and val_path.exists():
    train = normalize(train_path.read_text(encoding="utf-8"))
    val = normalize(val_path.read_text(encoding="utf-8"))

    print("\n=== DATA CHECK ===")
    print("Exact train/validation duplicate:", train == val)
    print("Validation text appears in training:", val in train if val else False)
