import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attention, _ = self.attn(
            x, x, x, attn_mask=mask, need_weights=False
        )
        x = self.norm1(x + attention)
        return self.norm2(x + self.ff(x))


class TenvorGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        idx = idx[:, -self.block_size:]
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        return self.head(self.norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, count=200, temperature=0.8):
        self.eval()

        for _ in range(count):
            logits = self(idx)[:, -1, :] / max(temperature, 0.01)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
