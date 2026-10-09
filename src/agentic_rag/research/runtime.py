"""所有 Agent 共享的进程内执行限流；任务上下文不与线程身份绑定。"""

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from ..config import settings


task_context = ContextVar("research_task_context", default=None)


class Capacity:
    def __init__(self, limit, label):
        if limit < 1:
            raise ValueError(f"{label} concurrency must be positive")
        self.semaphore = threading.BoundedSemaphore(limit)
        self.limit, self.label = limit, label
        self.active = self.peak = 0
        self.lock = threading.Lock()

    @contextmanager
    def slot(self):
        context = task_context.get()
        if context:
            context["emit"]("waiting_" + self.label, {})
        acquired = False
        try:
            while not acquired:
                from ..budget import check_budget
                check_budget()
                if context and (context["cancel"].is_set() or time.monotonic() > context["deadline"]):
                    raise TimeoutError("研究任务取消或超时")
                acquired = self.semaphore.acquire(timeout=0.2)
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                active = self.active
            if context:
                context["emit"]("running_" + self.label, {"active": active, "limit": self.limit})
            yield
        finally:
            if acquired:
                with self.lock:
                    self.active -= 1
                self.semaphore.release()


@lru_cache(maxsize=1)
def llm_capacity():
    return Capacity(settings.llm_concurrency, "llm")


@lru_cache(maxsize=1)
def io_capacity():
    return Capacity(settings.io_concurrency, "io")
