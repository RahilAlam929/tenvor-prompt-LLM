from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parents[1]
train_file = ROOT / "data" / "train_v2.txt"
output_file = ROOT / "tokenizer" / "tenvor-tokenizer-v2.json"

if not train_file.exists():
    raise SystemExit(f"Training data not found: {train_file}")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=512,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tokenizer.train([str(train_file)], trainer)
tokenizer.save(str(output_file))

print("Tokenizer saved:", output_file)
print("Vocabulary size:", tokenizer.get_vocab_size())

sample = "Artificial intelligence helps people solve problems."
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded.ids)

print("Sample tokens:", encoded.tokens)
print("Round-trip successful:", decoded == sample)
print("Decoded text:", repr(decoded))
