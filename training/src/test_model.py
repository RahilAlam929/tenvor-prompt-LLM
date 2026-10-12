import torch
from tokenizers import Tokenizer
from model import TenvorLM

tokenizer = Tokenizer.from_file("tokenizer/tenvor-tokenizer.json")
model = TenvorLM(vocab_size=tokenizer.get_vocab_size())

sample = tokenizer.encode("Tenvor learns patterns")
token_ids = torch.tensor([sample.ids], dtype=torch.long)
logits = model(token_ids)

print("Model: TenvorLM")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Input shape:", tuple(token_ids.shape))
print("Output shape:", tuple(logits.shape))
print("Device:", next(model.parameters()).device)
print("Model initialization: SUCCESS")
