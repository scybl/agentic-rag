"""带证据版本的分析缓存；只复用由同一批证据生成的回答。"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from langchain_core.documents import Document


def _normalise_question(question: str) -> str:
    return " ".join(question.casefold().split())


def evidence_identity(documents: list[Document]) -> tuple[str, list[str]]:
    """返回顺序无关的证据指纹，以及便于审计的来源 ID。"""
    records = []
    source_ids = []
    for document in documents:
        metadata = document.metadata
        source_id = str(
            metadata.get("article_id")
            or metadata.get("url")
            or metadata.get("source")
            or "unknown"
        )
        content_hash = str(metadata.get("content_hash") or "")
        if not content_hash:
            content_hash = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()
        records.append(
            {
                "source_id": source_id,
                "source_type": str(metadata.get("source_type") or "unknown"),
                "content_hash": content_hash,
            }
        )
        source_ids.append(source_id)
    encoded = json.dumps(
        sorted(records, key=lambda item: json.dumps(item, sort_keys=True)),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sorted(set(source_ids))


def build_cache_key(
    question: str,
    documents: list[Document],
    *,
    model: str,
    prompt_version: str,
) -> tuple[str, str, list[str]]:
    evidence_hash, source_ids = evidence_identity(documents)
    payload = {
        "question": _normalise_question(question),
        "evidence_hash": evidence_hash,
        "model": model,
        "prompt_version": prompt_version,
    }
    key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return key, evidence_hash, source_ids


class AnalysisCache:
    """使用 SQLite 保存有有效期、可追溯来源的分析结果。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_cache (
                cache_key TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        return connection

    def get(self, cache_key: str, *, now: int | None = None) -> str | None:
        current_time = int(time.time()) if now is None else now
        connection = self._connect()
        try:
            with connection:
                row = connection.execute(
                    "SELECT answer, expires_at FROM analysis_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                answer, expires_at = row
                if expires_at <= current_time:
                    connection.execute(
                        "DELETE FROM analysis_cache WHERE cache_key = ?", (cache_key,)
                    )
                    return None
                return str(answer)
        finally:
            connection.close()

    def put(
        self,
        cache_key: str,
        *,
        question: str,
        answer: str,
        evidence_hash: str,
        source_ids: list[str],
        model: str,
        prompt_version: str,
        ttl_seconds: int,
        now: int | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if not answer.strip():
            raise ValueError("answer must not be blank")
        created_at = int(time.time()) if now is None else now
        expires_at = created_at + ttl_seconds
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO analysis_cache (
                        cache_key, question, answer, evidence_hash, source_ids_json,
                        model, prompt_version, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        question = excluded.question,
                        answer = excluded.answer,
                        evidence_hash = excluded.evidence_hash,
                        source_ids_json = excluded.source_ids_json,
                        model = excluded.model,
                        prompt_version = excluded.prompt_version,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (
                        cache_key,
                        question,
                        answer,
                        evidence_hash,
                        json.dumps(source_ids, ensure_ascii=False),
                        model,
                        prompt_version,
                        created_at,
                        expires_at,
                    ),
                )
        finally:
            connection.close()
