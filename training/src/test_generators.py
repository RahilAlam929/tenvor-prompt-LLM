import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_generator(name):
    path = ROOT / "src" / f"generate_{name}.py"
    spec = importlib.util.spec_from_file_location(f"generate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

for name in ("v2", "v3"):
    module = load_generator(name)
    result = module.generate("Artificial intelligence", max_new_tokens=20)
    assert isinstance(result, str), f"{name}: output is not a string"
    assert result.strip(), f"{name}: output is empty"
    print(f"{name.upper()} generator test passed")
    print("Sample:", repr(result))
