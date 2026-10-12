from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "checkpoints" / "tenvor-gpt.pt"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model import TenvorGPT


class TenvorInference:
    def __init__(self):
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"TenvorGPT checkpoint not found: {CHECKPOINT}"
            )

        self.device = torch.device("cpu")
        checkpoint = torch.load(
            CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )

        self.chars = checkpoint["chars"]
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}

        self.model = TenvorGPT(
            vocab_size=len(self.chars),
            block_size=checkpoint["block_size"],
            n_embd=checkpoint["n_embd"],
            n_head=checkpoint["n_head"],
            n_layer=checkpoint["n_layer"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.best_val_loss = checkpoint.get("best_val_loss")
        self.training_step = checkpoint.get("step")

    def generate(self, prompt, max_new_tokens=120):
        if not prompt:
            prompt = "The "

        unknown = sorted(set(char for char in prompt if char not in self.stoi))
        if unknown:
            raise ValueError(
                f"Unsupported characters in prompt: {unknown}"
            )

        start = torch.tensor(
            [[self.stoi[char] for char in prompt]],
            dtype=torch.long,
            device=self.device,
        )

        output = self.model.generate(
            start,
            count=max(1, min(int(max_new_tokens), 200)),
        )[0].tolist()

        return "".join(self.itos[token] for token in output)
