"""持久化运行的互斥租约；退出时释放，进程崩溃后自动到期。"""

import threading
import uuid
from contextlib import contextmanager


@contextmanager
def run_lease(store, run_id, question):
    owner = uuid.uuid4().hex
    store.start_run(run_id, question)
    store.acquire_run(run_id, owner, 45)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(10):
            store.acquire_run(run_id, owner, 45)

    thread = threading.Thread(target=heartbeat, daemon=True, name="research-run-lease")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)
        store.release_run(run_id, owner)
