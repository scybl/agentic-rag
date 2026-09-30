"""有限任务池、依赖调度、原子领取、租约续期和可恢复成果。"""

import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from contextvars import copy_context

from ..config import settings
from .runtime import task_context
from ..token_usage import active_task


@dataclass
class TaskSpec:
    key: str
    kind: str
    payload: dict
    dependencies: list[str] = field(default_factory=list)


class Scheduler:
    def __init__(self, store, run_id, *, workers=None, budget=None, emit=None, attempts=None, lease=None, timeout=None):
        self.store, self.run_id = store, run_id
        self.workers = workers or settings.agent_workers
        self.budget = budget or settings.research_max_tasks
        self.attempts = attempts or settings.task_max_attempts
        self.lease = lease or settings.task_lease_seconds
        self.timeout = timeout or settings.research_timeout
        if min(self.workers, self.budget, self.attempts, self.lease, self.timeout) < 1:
            raise ValueError("任务预算、并发、租约与超时必须为正数")
        self.callback = emit or (lambda event: None)
        self.owner = uuid.uuid4().hex
        self.cancel = threading.Event()
        self.queue = queue.SimpleQueue()

    def emit(self, kind, data=None, spec=None):
        if kind == "tool":
            # 工具事件保持统一顶层格式，终端才能展示工具、调用者和理由；任务身份在此补齐。
            event = {"run_id": self.run_id, **(data or {}), "kind": "tool"}
        else:
            event = {"kind": "research", "event": kind, "run_id": self.run_id, **(data or {})}
        if spec:
            event.update(task_id=spec.key[:12], task_key=spec.key, role=spec.kind,
                         goal=spec.payload.get("goal", ""))
        self.store.event(self.run_id, event)
        self.queue.put(event)

    def drain(self):
        while not self.queue.empty():
            self.callback(self.queue.get())

    def _execute(self, spec, handler, deadline):
        started = time.monotonic()
        while not self.cancel.is_set() and time.monotonic() < deadline:
            row = self.store.task(spec.key)
            if row["status"] == "complete":
                self.emit("reused", spec=spec)
                return row["result"]
            if row["status"] == "failed" and not row.get("retryable", True):
                self.emit("failed", {"error": row["error"], "retry_skipped": True}, spec)
                return None
            if row["attempts"] >= self.attempts and row["status"] != "running":
                self.emit("failed", {"error": row["error"] or "超过尝试上限"}, spec)
                return None
            if not self.store.claim(spec.key, self.owner, self.lease, self.attempts):
                if time.monotonic() - started > settings.task_wait_seconds:
                    self.emit("deferred", {"error": "任务由其他执行者持有或尝试预算耗尽"}, spec)
                    return None
                self.cancel.wait(0.1)
                continue
            self.emit("started", spec=spec)
            token = task_context.set({"emit": lambda kind, data: self.emit(kind, data, spec),
                                      "cancel": self.cancel, "deadline": deadline,
                                      "task_id": spec.key, "role": spec.kind,
                                      "goal": spec.payload.get("goal", "")})
            usage_token = active_task.set({"task_id": spec.key, "task_kind": spec.kind,
                                           "task_attempt": row["attempts"] + 1})
            try:
                result = handler(spec.payload)
                if self.cancel.is_set() or time.monotonic() > deadline:
                    raise TimeoutError("任务已经取消或超过本轮时间预算")
                vector = result.pop("_vector", None)
                self.store.complete(spec.key, self.owner, result, vector)
                self.emit("completed", {"elapsed": round(time.monotonic() - started, 2)}, spec)
                return result
            except Exception as exc:
                self.store.fail(spec.key, self.owner, exc)
                self.emit("attempt_failed", {"error": str(exc)[:500]}, spec)
                if getattr(exc, "retryable", True) is False:
                    self.emit("failed", {"error": str(exc)[:500], "retry_skipped": True}, spec)
                    return None
            finally:
                active_task.reset(usage_token)
                task_context.reset(token)
        return None

    def run(self, specs, handlers):
        pending = {}
        results = {}
        for spec in specs:
            if spec.kind not in handlers:
                raise ValueError(f"未注册的任务角色：{spec.kind}")
            if self.store.submit(self.run_id, spec.key, spec.kind, spec.payload, spec.dependencies, self.budget):
                pending[spec.key] = spec
                self.emit("queued", {"dependencies": [k[:12] for k in spec.dependencies]}, spec)
            else:
                results[spec.key] = None
                self.emit("budget", {"error": "本次研究任务数达到上限"}, spec)
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="research-agent")
        active = {}
        deadline = time.monotonic() + self.timeout
        try:
            while pending or active:
                if time.monotonic() > deadline:
                    self.emit("timeout", {"error": "本轮执行时间达到上限，未完成任务可恢复"})
                    break
                for key, spec in list(pending.items()):
                    if len(active) >= self.workers:
                        break
                    # 依赖在主线程检查，避免工作池全被“等待依赖”的线程占满。
                    dependencies = [self.store.task(dep) for dep in spec.dependencies]
                    if all(dep and dep["status"] == "complete" for dep in dependencies):
                        future = pool.submit(copy_context().run, self._execute, spec, handlers[spec.kind], deadline)
                        active[future] = key
                        del pending[key]
                self.store.renew(self.owner, self.lease)
                self.drain()
                if not active:
                    for spec in pending.values():
                        self.emit("blocked", {"error": "依赖未完成或形成循环，本任务暂不执行"}, spec)
                    break
                done, _ = wait(active, timeout=min(0.2, self.lease / 3), return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    results[key] = future.result()
            return results
        finally:
            self.cancel.set()
            pool.shutdown(wait=False, cancel_futures=True)
            # 失去所有权后，仍在返回中的请求不能提交成果；HTTP 自身有超时。
            self.store.abandon(self.owner)
            self.drain()
