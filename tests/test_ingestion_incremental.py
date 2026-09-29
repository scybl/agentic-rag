import threading

from agentic_rag import ingestion


class _FakeCollection:
    def __init__(self, store):
        self.store = store

    def count(self):
        return len(self.store.documents)


class _FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.add_calls = []
        self.delete_calls = []
        self.reset_count = 0
        self._collection = _FakeCollection(self)

    def add_documents(self, documents, ids):
        self.add_calls.append(list(ids))
        self.documents.update(dict(zip(ids, documents)))

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        for document_id in ids:
            self.documents.pop(document_id, None)

    def reset_collection(self):
        self.reset_count += 1
        self.documents.clear()


def test_incremental_sync_adds_updates_deletes_and_skips(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    first_file = knowledge_dir / "first.md"
    first_file.write_text("第一份知识材料", encoding="utf-8")

    store = _FakeVectorStore()
    monkeypatch.setattr(ingestion, "get_vectorstore", lambda: store)

    first = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    assert first.added_files == 1
    assert first.indexed_chunks == 1
    assert store.reset_count == 0

    unchanged = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    assert not unchanged.changed
    assert unchanged.unchanged_files == 1
    assert len(store.add_calls) == 1

    second_file = knowledge_dir / "nested" / "second.txt"
    second_file.parent.mkdir()
    second_file.write_text("第二份知识材料", encoding="utf-8")
    added = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    assert added.added_files == 1
    assert added.unchanged_files == 1
    assert len(store.add_calls) == 2

    old_first_ids = set(store.add_calls[0])
    first_file.write_text("第一份知识材料已经修改", encoding="utf-8")
    updated = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    assert updated.updated_files == 1
    assert old_first_ids <= set(store.delete_calls[-1])

    second_ids = set(store.add_calls[1])
    second_file.unlink()
    deleted = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    assert deleted.deleted_files == 1
    assert second_ids <= set(store.delete_calls[-1])


def test_rebuild_resets_collection_and_reindexes_all_files(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    (knowledge_dir / "method.md").write_text("分析方法", encoding="utf-8")

    store = _FakeVectorStore()
    monkeypatch.setattr(ingestion, "get_vectorstore", lambda: store)

    ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    rebuilt = ingestion.sync_index(
        knowledge_dir,
        rebuild=True,
        manifest_path=manifest_path,
    )

    assert rebuilt.rebuilt
    assert rebuilt.added_files == 1
    assert rebuilt.indexed_chunks == 1
    assert store.reset_count == 1


def test_empty_collection_with_manifest_is_recovered(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    (knowledge_dir / "method.md").write_text("分析方法", encoding="utf-8")

    store = _FakeVectorStore()
    monkeypatch.setattr(ingestion, "get_vectorstore", lambda: store)
    ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)
    store.documents.clear()

    recovered = ingestion.sync_index(knowledge_dir, manifest_path=manifest_path)

    assert recovered.rebuilt
    assert recovered.added_files == 1
    assert store.reset_count == 1
    assert len(store.documents) == 1


def test_watcher_can_start_and_stop(monkeypatch):
    called = threading.Event()

    def fake_sync(_directory=None):
        called.set()
        return ingestion.SyncResult()

    monkeypatch.setattr(ingestion, "sync_index", fake_sync)
    watcher = ingestion.start_knowledge_watcher(interval=0.01)

    assert called.wait(timeout=1)
    watcher.stop()
    watcher.join(timeout=1)
    assert not watcher.is_alive()
