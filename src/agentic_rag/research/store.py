"""SQLite 是任务及证据的事实源；向量库是可重建的检索投影。"""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from langchain_core.documents import Document


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def normalize(text):
    # 只清理格式，不删除数字、单位、日期或改变事实顺序。
    return "\n".join(" ".join(line.split()) for line in text.replace("\r\n", "\n").splitlines() if line.strip())


def search_tokens(text):
    """中文双字词与英文词，用于 SQLite FTS5 倒排检索。"""
    terms = re.findall(r"[a-z0-9]+", text.lower())
    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.extend(word[i:i + 2] for i in range(max(1, len(word) - 1)))
    return list(dict.fromkeys(terms))


def split_text(text, limit):
    """段落优先，无遗漏地分段；返回在规范化全文中的精确位置。"""
    if limit < 100:
        raise ValueError("reading chunk size must be >= 100")
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + limit // 2, end), text.rfind("。", start + limit // 2, end))
            if boundary >= 0:
                end = boundary + 1
        yield start, end, text[start:end]
        start = end


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
 updated REAL NOT NULL, owner TEXT, lease_until REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS articles(id TEXT PRIMARY KEY, current_version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions(
 id TEXT PRIMARY KEY, article_id TEXT NOT NULL, body TEXT NOT NULL, metadata TEXT NOT NULL,
 created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL, ordinal INTEGER NOT NULL, start INTEGER NOT NULL,
 end INTEGER NOT NULL, body TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_version ON chunks(version_id);
CREATE TABLE IF NOT EXISTS tasks(
 key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
 result TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0, owner TEXT,
 lease_until REAL NOT NULL DEFAULT 0, updated REAL NOT NULL, retryable INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS run_tasks(
 run_id TEXT NOT NULL, task_key TEXT NOT NULL, dependencies TEXT NOT NULL,
 PRIMARY KEY(run_id, task_key));
CREATE TABLE IF NOT EXISTS readings(
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL, recipe TEXT NOT NULL, status TEXT NOT NULL,
 covered INTEGER NOT NULL, total INTEGER NOT NULL, result TEXT NOT NULL, updated REAL NOT NULL);
CREATE INDEX IF NOT EXISTS reading_version ON readings(version_id,recipe);
CREATE TABLE IF NOT EXISTS vectors(
 id TEXT PRIMARY KEY, collection TEXT NOT NULL, body TEXT NOT NULL,
 metadata TEXT NOT NULL, version_id TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vector_sync(
 id TEXT NOT NULL, embedding TEXT NOT NULL, hash TEXT NOT NULL,
 error TEXT, updated REAL NOT NULL, PRIMARY KEY(id,embedding));
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, at REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id,id);
CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS reading_fts USING fts5(reading_id UNINDEXED,tokens);
CREATE TABLE IF NOT EXISTS reading_keyword_index(id TEXT PRIMARY KEY);
"""


class ResearchStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            # 加法迁移保留所有旧任务；串行检查，避免并发启动重复加列。
            db.execute("BEGIN IMMEDIATE")
            if "retryable" not in {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}:
                db.execute("ALTER TABLE tasks ADD COLUMN retryable INTEGER NOT NULL DEFAULT 1")
            # 兼容已有阅读记录；只为缺失条目补关键词索引，不调用模型。
            rows = db.execute("SELECT r.id,r.result,v.metadata FROM readings r JOIN versions v ON v.id=r.version_id WHERE r.status='complete' AND NOT EXISTS (SELECT 1 FROM reading_keyword_index f WHERE f.id=r.id)").fetchall()
            for row in rows:
                self._keywords(db, row["id"], json.loads(row["metadata"]), json.loads(row["result"]))

    @contextmanager
    def connect(self, *, immediate=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            if immediate:
                db.execute("BEGIN IMMEDIATE")
            with db:
                yield db
        finally:
            db.close()

    def start_run(self, run_id, question):
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO runs(id,question,status,created,updated) VALUES(?,?,?,?,?)",
                       (run_id, question, "running", now, now))

    def run_status(self, run_id, status):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,updated=? WHERE id=?", (status, time.time(), run_id))

    def acquire_run(self, run_id, owner, seconds):
        with self.connect(immediate=True) as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("研究记录不存在")
            if row["owner"] and row["owner"] != owner and row["lease_until"] > time.time():
                raise RuntimeError("该研究仍由另一个进程执行，请勿同时恢复")
            db.execute("UPDATE runs SET owner=?,lease_until=? WHERE id=?", (owner, time.time() + seconds, run_id))

    def release_run(self, run_id, owner):
        with self.connect() as db:
            db.execute("UPDATE runs SET owner=NULL,lease_until=0 WHERE id=? AND owner=?", (run_id, owner))

    def runs(self, limit=10):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY updated DESC LIMIT ?", (limit,))]

    def run_info(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def event(self, run_id, data):
        # 调用者显式字段优先；copy_context 让工作线程继承所属节点/执行编号。
        from ..token_usage import active_step, active_task, active_usage
        ledger = active_usage.get()
        context = {**(active_step.get() or {}), **(active_task.get() or {})}
        if ledger is not None:
            context["session_id"] = ledger.session_id
        data = {"at": time.time(), **context, **data}
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,at,data) VALUES(?,?,?)", (run_id, time.time(), encoded(data)))

    def events(self, run_id, limit=100):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit))][::-1]

    def token_events(self, run_id):
        """账本必须读取整次研究，不能沿用最近 100 条的展示窗口。"""
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT data FROM events WHERE run_id=? AND json_extract(data,'$.kind')='token_usage' ORDER BY id",
                (run_id,))]

    def register(self, document: Document, chunk_chars):
        meta = dict(document.metadata)
        identity = str(meta.get("article_id") or meta.get("url") or meta.get("source") or fingerprint(document.page_content))
        article_id = fingerprint([meta.get("source_type", "unknown"), identity])
        body = normalize(document.page_content)
        header = {k: meta.get(k, "") for k in ("title", "published_at", "content_kind", "source")}
        version = fingerprint([article_id, body, header])
        pieces = []
        with self.connect(immediate=True) as db:
            db.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?)", (version, article_id, body, encoded(meta), time.time()))
            if meta.get("memory_origin"):
                db.execute("INSERT OR IGNORE INTO articles VALUES(?,?)", (article_id, version))
            else:
                db.execute("INSERT INTO articles VALUES(?,?) ON CONFLICT(id) DO UPDATE SET current_version=excluded.current_version", (article_id, version))
            for ordinal, (start, end, text) in enumerate(split_text(body, chunk_chars)):
                chunk_id = fingerprint([version, chunk_chars, ordinal])
                chunk = dict(id=chunk_id, version_id=version, ordinal=ordinal, start=start, end=end, body=text, hash=fingerprint(text))
                pieces.append(chunk)
                db.execute("INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?,?)", tuple(chunk.values()))
                self._vector(db, "raw:" + chunk_id, "news_chunks", text,
                             {"article_id": article_id, "version_id": version, "chunk_id": chunk_id, "ordinal": ordinal,
                              "source": str(meta.get("source", "")), "published_at": str(meta.get("published_at", ""))}, version)
        return {"id": article_id, "version": version, "body": body, "metadata": meta, "header": header, "chunks": pieces}

    def document(self, version):
        with self.connect() as db:
            row = db.execute("SELECT body,metadata FROM versions WHERE id=?", (version,)).fetchone()
            if row is None:
                raise ValueError("原文版本不存在")
            return Document(page_content=row["body"], metadata=json.loads(row["metadata"]))

    def inspect_reading(self, reading_id):
        with self.connect() as db:
            row = db.execute("SELECT r.*,v.metadata FROM readings r JOIN versions v ON r.version_id=v.id WHERE r.id=?", (reading_id,)).fetchone()
            return None if row is None else {**dict(row), "result": json.loads(row["result"]), "metadata": json.loads(row["metadata"])}

    @staticmethod
    def _vector(db, item_id, collection, body, metadata, version_id):
        digest = fingerprint([body, metadata])
        db.execute("INSERT INTO vectors VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,metadata=excluded.metadata,hash=excluded.hash",
                   (item_id, collection, body, encoded(metadata), version_id, digest))

    def save_reading(self, article, recipe, results):
        reading_id = fingerprint([article["version"], recipe])
        covered = len(results)
        total = len(article["chunks"])
        status = "complete" if covered == total and total else "partial"
        claims = []
        from ..evidence import event_context, FORECAST_WORDS
        for chunk, result in results:
            for claim in result["claims"]:
                offset = chunk["body"].find(claim["quote"])
                if offset < 0:
                    raise ValueError("阅读成果引用不在原文中")
                background = event_context(article["body"], chunk["start"] + offset)
                kind = claim["kind"]
                if kind == "reported_fact" and FORECAST_WORDS.search(background + claim["quote"]):
                    kind = "attributed_forecast"
                claims.append({**claim, "kind": kind, "event_context": background,
                               "chunk_id": chunk["id"], "start": chunk["start"] + offset,
                               "end": chunk["start"] + offset + len(claim["quote"])})
        result = {"claims": claims, "limitations": [x for _, r in results for x in r.get("limitations", [])],
                  "task_keys": list(dict.fromkeys(chunk["task_key"] for chunk, _ in results if chunk.get("task_key")))}
        with self.connect(immediate=True) as db:
            previous = db.execute("SELECT * FROM readings WHERE id=?", (reading_id,)).fetchone()
            if previous and previous["covered"] >= covered:
                # 并发执行者不能用较少的阅读结果覆盖已经完成的成果。
                return {**dict(previous), **json.loads(previous["result"])}
            db.execute("INSERT INTO readings VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,covered=excluded.covered,total=excluded.total,result=excluded.result,updated=excluded.updated",
                       (reading_id, article["version"], recipe, status, covered, total, encoded(result), time.time()))
            if status == "complete":
                self._keywords(db, reading_id, article["metadata"], result)
            # 成果与待写入向量在同一事务提交；索引故障不重复调用模型。
            for chunk, _ in results:
                # 向量投影与关系库使用同一份校验后的类型和时间背景，避免召回旧的错误分类。
                text = "\n".join(f"{c['kind']}：{c['statement']}；事件背景：{c['event_context']}；原文：{c['quote']}"
                                 for c in claims if c["chunk_id"] == chunk["id"])
                if text:
                    self._vector(db, "note:" + fingerprint([reading_id, chunk["id"]]), "reading_memory", text,
                                 {"article_id": article["id"], "version_id": article["version"], "reading_id": reading_id,
                                  "recipe": recipe, "chunk_id": chunk["id"], "published_at": str(article["metadata"].get("published_at", ""))}, article["version"])
        return {"id": reading_id, "status": status, "covered": covered, "total": total, **result}

    def reading(self, version, recipe):
        with self.connect() as db:
            row = db.execute("SELECT * FROM readings WHERE version_id=? AND recipe=?", (version, recipe)).fetchone()
            return None if row is None else {**dict(row), **json.loads(row["result"])}

    @staticmethod
    def _keywords(db, reading_id, metadata, result):
        text = str(metadata.get("title", "")) + " " + encoded(result)
        db.execute("DELETE FROM reading_fts WHERE reading_id=?", (reading_id,))
        db.execute("INSERT INTO reading_fts VALUES(?,?)", (reading_id, " ".join(search_tokens(text))))
        db.execute("INSERT OR IGNORE INTO reading_keyword_index VALUES(?)", (reading_id,))

    def lexical_readings(self, query, recipe, limit=50):
        terms = search_tokens(query)[:64]
        if not terms:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        with self.connect() as db:
            rows = db.execute("SELECT f.reading_id FROM reading_fts f JOIN readings r ON r.id=f.reading_id JOIN versions v ON v.id=r.version_id JOIN articles a ON a.current_version=v.id WHERE reading_fts MATCH ? AND r.recipe=? AND r.status='complete' ORDER BY bm25(reading_fts) LIMIT ?", (match, recipe, limit))
            return [r[0] for r in rows]

    def current_readings(self, recipe, ids=None):
        if ids is not None and not ids:
            return []
        restriction = "" if ids is None else " AND r.id IN (" + ",".join("?" for _ in ids) + ")"
        with self.connect() as db:
            rows = db.execute("SELECT r.*,v.body,v.metadata,v.article_id FROM readings r JOIN versions v ON r.version_id=v.id JOIN articles a ON a.current_version=v.id WHERE r.recipe=? AND r.status='complete'" + restriction + " ORDER BY r.updated DESC", (recipe, *(ids or []))).fetchall()
            return [{**dict(r), "result": json.loads(r["result"]), "metadata": json.loads(r["metadata"])} for r in rows]

    def submit(self, run_id, key, kind, payload, dependencies, budget):
        with self.connect(immediate=True) as db:
            known = db.execute("SELECT 1 FROM run_tasks WHERE run_id=? AND task_key=?", (run_id, key)).fetchone()
            count = db.execute("SELECT count(*) FROM run_tasks WHERE run_id=?", (run_id,)).fetchone()[0] if budget is not None else 0
            if not known and budget is not None and count >= budget:
                return False
            db.execute("INSERT OR IGNORE INTO tasks(key,kind,payload,status,updated) VALUES(?,?,?,?,?)", (key, kind, encoded(payload), "pending", time.time()))
            db.execute("INSERT OR IGNORE INTO run_tasks VALUES(?,?,?)", (run_id, key, encoded(dependencies)))
            return True

    def task(self, key):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE key=?", (key,)).fetchone()
            if row is None:
                return None
            return {**dict(row), "payload": json.loads(row["payload"]), "result": json.loads(row["result"]) if row["result"] else None}

    def tasks(self, run_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT t.key,t.kind,t.status,t.attempts,t.error,rt.dependencies FROM tasks t JOIN run_tasks rt ON rt.task_key=t.key WHERE rt.run_id=? ORDER BY t.updated", (run_id,))]

    def claim(self, key, owner, lease, attempts):
        with self.connect(immediate=True) as db:
            row = db.execute("SELECT * FROM tasks WHERE key=?", (key,)).fetchone()
            if row is None or row["status"] == "complete" or not row["retryable"]:
                return False
            if row["status"] == "running" and row["lease_until"] > time.time():
                return False
            if row["attempts"] >= attempts:
                db.execute("UPDATE tasks SET status='failed',error='租约到期且尝试预算耗尽',owner=NULL,lease_until=0 WHERE key=?", (key,))
                return False
            db.execute("UPDATE tasks SET status='running',owner=?,lease_until=?,attempts=attempts+1,updated=? WHERE key=?",
                       (owner, time.time() + lease, time.time(), key))
            return True

    def renew(self, owner, lease):
        with self.connect() as db:
            db.execute("UPDATE tasks SET lease_until=? WHERE owner=? AND status='running'", (time.time() + lease, owner))

    def complete(self, key, owner, result, vector=None):
        with self.connect(immediate=True) as db:
            changed = db.execute("UPDATE tasks SET status='complete',result=?,error=NULL,lease_until=0,updated=? WHERE key=? AND owner=? AND status='running' AND lease_until>?",
                                 (encoded(result), time.time(), key, owner, time.time())).rowcount
            if not changed:
                raise RuntimeError("任务租约已失效，拒绝提交过期执行者的结果")
            if vector:
                self._vector(db, "task:" + key, "task_analysis", vector["text"], vector["metadata"], "")

    def fail(self, key, owner, error):
        with self.connect() as db:
            db.execute("UPDATE tasks SET status='failed',error=?,retryable=?,lease_until=0,updated=? WHERE key=? AND owner=? AND status='running'",
                       (str(error)[:1500], int(getattr(error, "retryable", True)), time.time(), key, owner))

    def abandon(self, owner):
        with self.connect() as db:
            db.execute("UPDATE tasks SET status='pending',owner=NULL,lease_until=0,attempts=max(0,attempts-1) WHERE owner=? AND status='running'", (owner,))

    def defer_budget(self, key, owner):
        """本轮预算不够不是材料永久失败；不消耗一次尚未完成的任务尝试。"""
        with self.connect() as db:
            db.execute("UPDATE tasks SET status='pending',owner=NULL,lease_until=0,attempts=max(0,attempts-1),retryable=1 WHERE key=? AND owner=? AND status='running'", (key, owner))

    def retry_failed(self, run_id):
        with self.connect() as db:
            return db.execute("UPDATE tasks SET status='pending',attempts=0,error=NULL,retryable=1 WHERE (status='failed' OR (status='pending' AND attempts>0)) AND key IN (SELECT task_key FROM run_tasks WHERE run_id=?)", (run_id,)).rowcount

    def cache_get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            return None if row is None else json.loads(row[0])

    def cache_put(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (key, encoded(value)))

    def pending_vectors(self, embedding, limit=100):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT v.* FROM vectors v LEFT JOIN vector_sync s ON v.id=s.id AND s.embedding=? WHERE s.id IS NULL OR s.hash!=v.hash OR s.error IS NOT NULL LIMIT ?", (embedding, limit))]

    def vector_synced(self, item, embedding, error=None):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO vector_sync VALUES(?,?,?,?,?)", (item["id"], embedding, item["hash"], error, time.time()))
