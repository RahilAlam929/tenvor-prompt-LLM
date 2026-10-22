import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizers import Tokenizer
from chat import generate


class TestTenvor(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        tokenizer = Tokenizer.from_file(
            str(ROOT / "tokenizer" / "tenvor-tokenizer.json")
        )
        text = "Artificial intelligence helps people."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        self.assertEqual(decoded, text)

    def test_generate_returns_string(self):
        output = generate(
            "Artificial intelligence",
            max_new_tokens=10,
            seed=42,
        )
        self.assertIsInstance(output, str)

    def test_empty_prompt(self):
        self.assertEqual(generate(""), "")

    def test_checkpoint_exists(self):
        self.assertTrue(
            (ROOT / "checkpoints" / "tenvor-best.pt").exists()
        )


if __name__ == "__main__":
    unittest.main()
