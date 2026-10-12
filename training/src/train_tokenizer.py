from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "train.txt"
output_path = root / "tokenizer" / "tenvor-tokenizer.json"

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=256,
    min_frequency=1,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tokenizer.train([str(data_path)], trainer)
tokenizer.save(str(output_path))

sample = "Tenvor learns patterns from text."
encoded = tokenizer.encode(sample)

print("Tokenizer saved:", output_path)
print("Vocabulary size:", tokenizer.get_vocab_size())
print("Sample:", sample)
print("Token IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))
