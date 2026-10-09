"""严格 JSON 导出、只读 SQLite 输入以及可隔离的遥测出口。"""

import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Protocol

from .schema import Trace


def write_json(path: Path, value) -> None:
    """同目录临时文件 + 原子替换，失败时保留上次完整结果。"""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".export-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class TraceExporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class JsonExporter:
    def __init__(self, path):
        self.path = Path(path)

    def export(self, trace: Trace) -> None:
        write_json(self.path, trace.model_dump(mode="json"))


class BestEffortExporter:
    """仅用于旁路遥测；命令行显式导出使用严格出口，以非零退出暴露失败。"""

    def __init__(self, exporter: TraceExporter):
        self.exporter = exporter
        self.failures = 0
        self.last_error_type = None

    def export(self, trace: Trace) -> None:
        try:
            self.exporter.export(trace)
        except Exception as exc:
            self.failures += 1
            self.last_error_type = type(exc).__name__


def read_research(path, run_id):
    """不创建库、不迁移表、不截断事件；在同一只读快照中取运行与事件。"""
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("研究编号不存在")
        events = [{"id": r["id"], "at": r["at"], "data": json.loads(r["data"])} for r in db.execute(
            "SELECT id,at,data FROM events WHERE run_id=? ORDER BY id", (run_id,))]
        return dict(row), events
