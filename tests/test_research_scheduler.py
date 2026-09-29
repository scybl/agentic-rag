"""并发上限、任务去重、依赖、重试及中断后重用。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agentic_rag.research.store import ResearchStore
from agentic_rag.research.scheduler import Scheduler, TaskSpec
from agentic_rag.research.runtime import Capacity


def test_workers_and_model_slots_are_bounded(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    gate = Capacity(2, "llm")
    active = peak = 0
    lock = threading.Lock()
    events = []

    def work(payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        with gate.slot():
            time.sleep(0.04)
        with lock:
            active -= 1
        return {"value": payload["value"]}

    specs = [TaskSpec(str(i), "reader", {"value": i}) for i in range(8)]
    results = Scheduler(db, "r", workers=4, emit=events.append).run(specs, {"reader": work})
    assert len(results) == 8
    assert 2 <= peak <= 4
    assert gate.peak == 2
    assert any(e["event"] == "waiting_llm" for e in events)


def test_two_schedulers_compute_one_task_only_once(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    calls = []
    def work(payload):
        calls.append(1)
        time.sleep(0.1)
        return {"ok": True}
    def run(run_id):
        return Scheduler(db, run_id, workers=2).run([TaskSpec("same", "reader", {})], {"reader": work})
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(run, ["r1", "r2"]))
    assert len(calls) == 1
    assert all(result["same"] == {"ok": True} for result in outputs)


def test_dependencies_do_not_deadlock_pool_and_retry_is_bounded(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    order = []
    specs = [TaskSpec("dependent", "reader", {"value": 2}, ["first"]), TaskSpec("first", "reader", {"value": 1})]
    def work(payload):
        order.append(payload["value"])
        return payload
    Scheduler(db, "r", workers=1).run(specs, {"reader": work})
    assert order == [1, 2]
    def fail(payload):
        raise ValueError("cannot parse")
    result = Scheduler(db, "r", attempts=2).run([TaskSpec("bad", "reader", {})], {"reader": fail})
    assert result["bad"] is None
    assert db.task("bad")["attempts"] == 2


def test_resume_reuses_complete_tasks_and_recovers_expired_lease(tmp_path):
    path = tmp_path / "db.sqlite"
    db = ResearchStore(path)
    specs = [TaskSpec("a", "reader", {}), TaskSpec("b", "reader", {})]
    Scheduler(db, "r").run(specs[:1], {"reader": lambda p: {"done": 1}})
    db.submit("r", "b", "reader", {}, [], 20)
    db.claim("b", "crashed", 10, 2)
    with db.connect() as conn:
        conn.execute("UPDATE tasks SET lease_until=0 WHERE key='b'")
    calls = []
    result = Scheduler(ResearchStore(path), "r").run(specs, {"reader": lambda p: calls.append(1) or {"done": 2}})
    assert len(calls) == 1
    assert result["a"] == {"done": 1} and result["b"] == {"done": 2}


def test_dependency_cycle_is_reported_not_executed(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    events = []
    specs = [TaskSpec("a", "reader", {}, ["b"]), TaskSpec("b", "reader", {}, ["a"])]
    assert Scheduler(db, "r", emit=events.append).run(specs, {"reader": lambda p: {}}) == {}
    assert sum(e["event"] == "blocked" for e in events) == 2


def test_cancellation_fences_late_worker_result(tmp_path):
    db = ResearchStore(tmp_path / "db.sqlite")
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    def work(payload):
        entered.set()
        release.wait(5)
        ended.set()
        return {"late": True}
    scheduler = Scheduler(db, "r", timeout=1)
    result = scheduler.run([TaskSpec("slow", "reader", {})], {"reader": work})
    assert entered.is_set() and "slow" not in result
    release.set()
    assert ended.wait(2)
    time.sleep(0.1)
    assert db.task("slow")["status"] != "complete"
    assert db.task("slow")["result"] is None
