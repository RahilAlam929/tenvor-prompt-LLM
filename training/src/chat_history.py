import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "data" / "chat_history.jsonl"


def save_message(prompt, response):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "response": response,
    }

    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(limit=20):
    if not HISTORY_FILE.exists():
        return []

    records = []
    with HISTORY_FILE.open(encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records[-max(0, limit):]


if __name__ == "__main__":
    for record in load_history():
        print(f"\nYou: {record['prompt']}")
        print(f"Tenvor: {record['response']}")
