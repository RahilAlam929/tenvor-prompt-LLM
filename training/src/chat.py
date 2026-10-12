from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cpu")

tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer.json")
)
checkpoint = torch.load(
    ROOT / "checkpoints" / "tenvor-best.pt",
    map_location=device,
)

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

def generate(prompt, max_new_tokens=100):
    ids = tokenizer.encode(prompt).ids
    if not ids:
        return ""

    generated = torch.tensor([ids[-model.max_length:]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_length:]
            logits = model(context)
            next_logits = logits[0, -1, :] / 0.8
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).reshape(1, 1)
            generated = torch.cat([generated, next_id], dim=1)

    return tokenizer.decode(generated[0].tolist())

print("Tenvor Chat | Type 'exit' to quit")
print("Note: Experimental tiny model; responses may be nonsensical.")

while True:
    try:
        prompt = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if prompt.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    if not prompt:
        continue

    if len(prompt) < 3:
        print("Tenvor: Please enter a longer prompt (at least 3 characters).")
        continue

    print("Tenvor:", generate(prompt))
