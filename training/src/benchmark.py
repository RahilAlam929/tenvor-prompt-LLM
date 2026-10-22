import time
from chat import generate

prompts = [
    "Artificial intelligence",
    "Python is used for",
    "A neural network",
    "A computer is",
    "The future of technology",
]

print("\n=== TENVOR BENCHMARK ===")

for prompt in prompts:
    start = time.perf_counter()
    output = generate(prompt, max_new_tokens=40)
    elapsed = time.perf_counter() - start

    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")
    print(f"Time: {elapsed:.2f}s")

print("\n=== BENCHMARK COMPLETE ===")
