from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cpu")

tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer.json")
)

checkpoint_path = ROOT / "checkpoints" / "tenvor-best.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

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

def generate(prompt, max_new_tokens=80):
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [2]

    generated = torch.tensor([ids[-128:]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_length:]
            logits = model(context)
            next_token_logits = logits[0, -1, :]

            temperature = 0.8
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).view(1, 1)

            generated = torch.cat([generated, next_id], dim=1)

    return tokenizer.decode(generated[0].tolist())

prompt = "Tenvor"
print("Prompt:", prompt)
print("Generated text:")
print(generate(prompt))
