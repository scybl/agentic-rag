"""一次研究共享的调度期限和模型调用预算，线程安全且恢复不计离线等待。"""

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar


active_budget = ContextVar("research_budget", default=None)


class ResearchBudgetExceeded(RuntimeError):
    retryable = False


class ResearchBudget:
    def __init__(self, *, seconds=0, calls=0, elapsed=0, used_calls=0, clock=time.monotonic):
        if min(seconds, calls, elapsed, used_calls) < 0:
            raise ValueError("研究预算不能为负数")
        self.clock, self.started = clock, clock()
        self.seconds, self.call_limit = seconds, calls
        self.elapsed, self.calls = elapsed, used_calls
        self.lock = threading.Lock()

    def remaining(self):
        return max(0, self.seconds - self.elapsed - (self.clock() - self.started)) if self.seconds else None

    def check(self):
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise ResearchBudgetExceeded("研究累计执行时间达到预算；停止派发新工作，已完成成果和检查点保留。可提高 RESEARCH_TOTAL_TIMEOUT 后恢复。")

    def model_call(self):
        with self.lock:
            self.check()
            if self.call_limit and self.calls >= self.call_limit:
                raise ResearchBudgetExceeded("研究累计模型调用达到预算；停止继续重试。可提高 RESEARCH_MAX_MODEL_CALLS 后恢复。")
            self.calls += 1


def check_budget():
    budget = active_budget.get()
    if budget:
        budget.check()


def reserve_model_call():
    budget = active_budget.get()
    if budget:
        budget.model_call()


@contextmanager
def research_budget(*, seconds=0, calls=0, elapsed=0, used_calls=0):
    budget = ResearchBudget(seconds=seconds, calls=calls, elapsed=elapsed, used_calls=used_calls)
    token = active_budget.set(budget)
    try:
        budget.check()
        yield budget
    finally:
        active_budget.reset(token)
