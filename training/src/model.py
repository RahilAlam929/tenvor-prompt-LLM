import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        length = x.size(1)
        mask = self.mask[:length, :length]
        output, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False
        )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TenvorLM(nn.Module):
    def __init__(self, vocab_size, dim=64, heads=4, layers=2, max_length=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_length, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.max_length = max_length

    def forward(self, token_ids):
        batch, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))
