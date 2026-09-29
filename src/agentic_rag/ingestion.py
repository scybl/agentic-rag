"""本地知识摄取：增量同步文件，并可选择实时监听目录变化。"""

import argparse
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}
MANIFEST_VERSION = 1
_SYNC_LOCK = threading.RLock()


@dataclass(frozen=True)
class SyncResult:
    """一次增量同步产生的变化摘要。"""

    added_files: int = 0
    updated_files: int = 0
    deleted_files: int = 0
    unchanged_files: int = 0
    indexed_chunks: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        return self.rebuilt or any(
            (self.added_files, self.updated_files, self.deleted_files)
        )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """创建并缓存嵌入模型，避免每次检索都重新加载。"""
    model_name = settings.embedding_model
    local_model = Path(model_name)
    if not local_model.is_absolute() and (PROJECT_ROOT / local_model).exists():
        model_name = str(PROJECT_ROOT / local_model)
    return HuggingFaceEmbeddings(model_name=model_name)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _knowledge_files(base: Path) -> dict[str, Path]:
    if not base.exists():
        return {}
    return {
        path.relative_to(base).as_posix(): path
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def _load_file(path: Path, source: str, file_hash: str) -> list[Document]:
    metadata = {"source": source, "file_hash": file_hash}
    if path.suffix.lower() in TEXT_SUFFIXES:
        return [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata=metadata,
            )
        ]

    from langchain_community.document_loaders import PyPDFLoader

    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        document.metadata.update(metadata)
    return documents


def load_documents(directory: str | None = None) -> list[Document]:
    """从知识目录树中加载 .md、.txt 和 .pdf 文件。"""
    base = Path(directory or settings.knowledge_dir)
    documents: list[Document] = []
    for source, path in _knowledge_files(base).items():
        documents.extend(_load_file(path, source, _file_hash(path)))
    logger.info("Loaded %d documents from %s", len(documents), base)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """把文档切分成适合语义检索的重叠文本块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
    )


def _manifest_path() -> Path:
    safe_collection = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in settings.collection_name
    )
    return Path(settings.chroma_dir) / f"{safe_collection}.manifest.json"


def _index_signature() -> str:
    settings_used = {
        "version": MANIFEST_VERSION,
        "embedding_model": settings.embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "collection": settings.collection_name,
    }
    encoded = json.dumps(settings_used, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("索引清单无法读取，将安全地执行全量重建：%s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _collection_count(store: Chroma) -> int:
    return int(store._collection.count())


def _chunk_file(path: Path, source: str, file_hash: str) -> tuple[list[Document], list[str]]:
    chunks = split_documents(_load_file(path, source, file_hash))
    chunk_ids: list[str] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": source,
                "file_hash": file_hash,
                "chunk_index": chunk_index,
            }
        )
        raw_id = f"{source}\0{file_hash}\0{chunk_index}".encode("utf-8")
        chunk_ids.append(hashlib.sha256(raw_id).hexdigest())
    return chunks, chunk_ids


def sync_index(
    directory: str | None = None,
    *,
    rebuild: bool = False,
    manifest_path: str | Path | None = None,
) -> SyncResult:
    """将知识目录增量同步到 Chroma，只处理新增、修改和删除的文件。"""
    with _SYNC_LOCK:
        base = Path(directory or settings.knowledge_dir).resolve()
        manifest_file = Path(manifest_path) if manifest_path else _manifest_path()
        files = _knowledge_files(base)
        hashes = {source: _file_hash(path) for source, path in files.items()}
        manifest = _load_manifest(manifest_file)
        store = get_vectorstore()
        signature = _index_signature()

        manifest_files = manifest.get("files", {})
        if not isinstance(manifest_files, dict):
            manifest_files = {}

        collection_count = _collection_count(store)
        missing_manifest_for_existing_index = not manifest and collection_count > 0
        missing_index_for_existing_manifest = collection_count == 0 and bool(manifest_files)
        signature_changed = manifest.get("index_signature") not in {None, signature}
        should_rebuild = (
            rebuild
            or missing_manifest_for_existing_index
            or missing_index_for_existing_manifest
            or signature_changed
        )

        if should_rebuild:
            store.reset_collection()
            manifest_files = {}

        old_sources = set(manifest_files)
        current_sources = set(files)
        deleted_sources = old_sources - current_sources
        added_sources = current_sources - old_sources
        updated_sources = {
            source
            for source in current_sources & old_sources
            if manifest_files[source].get("file_hash") != hashes[source]
        }
        unchanged_sources = current_sources - added_sources - updated_sources

        ids_to_delete: list[str] = []
        for source in deleted_sources | updated_sources:
            ids_to_delete.extend(manifest_files[source].get("chunk_ids", []))
        if ids_to_delete:
            store.delete(ids=ids_to_delete)

        new_manifest_files = {
            source: data
            for source, data in manifest_files.items()
            if source not in deleted_sources | updated_sources
        }
        indexed_chunks = 0
        for source in sorted(added_sources | updated_sources):
            chunks, chunk_ids = _chunk_file(files[source], source, hashes[source])
            if chunks:
                store.add_documents(chunks, ids=chunk_ids)
            indexed_chunks += len(chunks)
            new_manifest_files[source] = {
                "file_hash": hashes[source],
                "chunk_ids": chunk_ids,
            }

        _save_manifest(
            manifest_file,
            {
                "version": MANIFEST_VERSION,
                "index_signature": signature,
                "files": new_manifest_files,
            },
        )

        result = SyncResult(
            added_files=len(added_sources),
            updated_files=len(updated_sources),
            deleted_files=len(deleted_sources),
            unchanged_files=len(unchanged_sources),
            indexed_chunks=indexed_chunks,
            rebuilt=should_rebuild,
        )
        logger.info(
            "Knowledge sync: added=%d updated=%d deleted=%d unchanged=%d chunks=%d",
            result.added_files,
            result.updated_files,
            result.deleted_files,
            result.unchanged_files,
            result.indexed_chunks,
        )
        return result


def ingest(directory: str | None = None, *, rebuild: bool = False) -> int:
    """兼容旧调用；返回本次新增或更新的文本块数量。"""
    return sync_index(directory, rebuild=rebuild).indexed_chunks


def ensure_index() -> None:
    """索引为空时执行一次初始同步。"""
    if _collection_count(get_vectorstore()) == 0:
        sync_index()


def get_retriever(k: int | None = None):
    return get_vectorstore().as_retriever(
        search_kwargs={"k": k or settings.retrieval_k}
    )


class KnowledgeWatcher(threading.Thread):
    """后台轮询知识目录，并把文件变化增量同步到索引。"""

    def __init__(self, directory: str | None = None, interval: float | None = None):
        super().__init__(name="knowledge-watcher", daemon=True)
        self.directory = directory
        self.interval = (
            interval if interval is not None else settings.knowledge_watch_interval
        )
        if self.interval <= 0:
            raise ValueError("监听间隔必须大于 0 秒")
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("知识目录监听已启动，检查间隔 %.1f 秒", self.interval)
        while not self._stop_event.is_set():
            try:
                result = sync_index(self.directory)
                if result.changed:
                    logger.info("检测到知识文件变化，增量索引已更新")
            except Exception:
                logger.exception("知识目录增量同步失败；监听器将在下个周期重试")
            self._stop_event.wait(self.interval)


def start_knowledge_watcher(
    directory: str | None = None,
    interval: float | None = None,
) -> KnowledgeWatcher:
    watcher = KnowledgeWatcher(directory=directory, interval=interval)
    watcher.start()
    return watcher


def _print_result(result: SyncResult) -> None:
    mode = "全量重建" if result.rebuilt else "增量同步"
    print(
        f"{mode}完成：新增 {result.added_files} 个，更新 {result.updated_files} 个，"
        f"删除 {result.deleted_files} 个，跳过 {result.unchanged_files} 个；"
        f"本次写入 {result.indexed_chunks} 个文本块。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="同步本地知识文件到 Chroma")
    parser.add_argument("--rebuild", action="store_true", help="忽略清单并全量重建索引")
    parser.add_argument("--watch", action="store_true", help="初始同步后持续监听知识目录")
    parser.add_argument(
        "--interval",
        type=float,
        default=settings.knowledge_watch_interval,
        help="监听模式的检查间隔秒数",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _print_result(sync_index(rebuild=args.rebuild))

    if not args.watch:
        return

    watcher = start_knowledge_watcher(interval=args.interval)
    print("知识目录监听中，按 Ctrl+C 停止。")
    try:
        watcher.join()
    except KeyboardInterrupt:
        watcher.stop()
        watcher.join()


if __name__ == "__main__":
    main()
