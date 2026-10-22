import os
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "tenvor_prompt.sqlite3"
DB_PATH = Path(os.getenv("TENVOR_DB_PATH", str(DEFAULT_DB))).expanduser()


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def connection():
    db = connect()
    try:
        with db:
            yield db
    finally:
        db.close()


def init_db():
    with connection() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, created_at);
        """)


def create_conversation(title="New chat"):
    conversation_id = uuid.uuid4().hex
    timestamp = now()

    with connection() as db:
        db.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title.strip() or "New chat", timestamp, timestamp),
        )

    return get_conversation(conversation_id)


def list_conversations():
    with connection() as db:
        rows = db.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with connection() as db:
        conversation = db.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            return None

        messages = db.execute(
            """
            SELECT id, role, content, provider, model, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at, rowid
            """,
            (conversation_id,),
        ).fetchall()

    result = dict(conversation)
    result["messages"] = [dict(message) for message in messages]
    return result


def rename_conversation(conversation_id, title):
    timestamp = now()

    with connection() as db:
        cursor = db.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title.strip(), timestamp, conversation_id),
        )

    if cursor.rowcount == 0:
        return None

    return get_conversation(conversation_id)


def delete_conversation(conversation_id):
    with connection() as db:
        cursor = db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


def add_message(conversation_id, role, content, provider=None, model=None):
    message_id = uuid.uuid4().hex
    timestamp = now()

    with connection() as db:
        exists = db.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if exists is None:
            return None

        db.execute(
            """
            INSERT INTO messages
                (id, conversation_id, role, content, provider, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                provider,
                model,
                timestamp,
            ),
        )

        title = exists["title"]
        if role == "user" and title == "New chat":
            clean_title = " ".join(content.split())
            title = clean_title[:57] + ("..." if len(clean_title) > 57 else "")

        db.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, timestamp, conversation_id),
        )

    return {
        "id": message_id,
        "role": role,
        "content": content,
        "provider": provider,
        "model": model,
        "created_at": timestamp,
    }


init_db()
