from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import TenvorLM

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cpu")

tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer" / "tenvor-tokenizer-v2.json")
)
checkpoint_path = ROOT / "checkpoints" / "tenvor-v2-best.pt"
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

def generate(prompt, max_new_tokens=60, temperature=0.8, top_k=20):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    torch.manual_seed(42)
    prompt_ids = tokenizer.encode(prompt).ids
    generated = torch.tensor(
        [prompt_ids[-model.max_length:]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_length:]
            logits = model(context)
            next_logits = logits[0, -1, :] / max(temperature, 0.01)

            k = min(top_k, next_logits.numel())
            if k > 0:
                values, _ = torch.topk(next_logits, k)
                next_logits[next_logits < values[-1]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
            generated = torch.cat([generated, next_id], dim=1)

    continuation_ids = generated[0, len(prompt_ids):].tolist()
    return tokenizer.decode(continuation_ids)

if __name__ == "__main__":
    prompts = [
        "Artificial intelligence",
        "A software engineer",
        "A neural network",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Continuation:", repr(generate(prompt)))
