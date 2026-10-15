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

def generate(prompt, max_new_tokens=80, temperature=0.8, top_k=20):
    ids = tokenizer.encode(prompt).ids
    if not ids:
        return ""

    generated = torch.tensor(
        [ids[-model.max_length:]], dtype=torch.long
    )
    new_tokens = []
    eos_id = tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_length:]
            logits = model(context)[0, -1, :].clone()

            for token_id in set(new_tokens[-32:]):
                logits[token_id] /= 1.2

            logits /= max(temperature, 0.1)
            k = min(top_k, logits.numel())
            threshold = torch.topk(logits, k).values[-1]
            logits[logits < threshold] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == eos_id:
                break

            new_tokens.append(next_id)
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], dtype=torch.long)],
                dim=1,
            )

    return tokenizer.decode(new_tokens).strip()

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
