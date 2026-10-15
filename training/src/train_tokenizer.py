from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parents[1]
train_file = ROOT / "data" / "train.txt"
output_file = ROOT / "tokenizer" / "tenvor-tokenizer.json"

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
print("Sample tokens:", tokenizer.encode(
    "Artificial intelligence helps people."
).tokens)
