"""CrossEncoder 的独立进程期限和成功结果缓存，不遗留超时推理线程。"""

import math
import json
import multiprocessing
from pathlib import Path
import time

from ..evaluation.contracts import digest


def local_predict(model_path, pairs):
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if not any(name.endswith("ForSequenceClassification") for name in config.get("architectures", [])):
        raise ValueError("重排需要已训练的序列分类模型，不能把嵌入模型加载成随机分类头")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path, device="cpu", local_files_only=True, trust_remote_code=False)
    return model.predict(pairs, show_progress_bar=False).tolist()


def _worker(connection, predictor, arguments):
    try:
        connection.send((True, predictor(*arguments)))
    except Exception as exc:
        connection.send((False, type(exc).__name__))
    finally:
        connection.close()


def bounded_predict(predictor, arguments, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("重排超时必须为有限正数")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, predictor, arguments), daemon=True)
    started = time.monotonic()
    try:
        process.start()
        sender.close()
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0 or not receiver.poll(remaining):
            raise TimeoutError("CrossEncoder 超过本次期限")
        success, result = receiver.recv()
        if not success:
            raise RuntimeError("CrossEncoder 子进程失败：" + result)
        return result
    finally:
        if process.pid is not None:
            process.join(timeout=0.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            process.close()
        receiver.close()
        sender.close()


class Reranker:
    def __init__(self, model_path, revision, *, timeout=60, predict=None):
        self.path, self.revision, self.timeout = model_path, revision, timeout
        self.predict = predict
        self.cache = {}

    def score(self, question, texts):
        key = digest([self.revision, question, texts])
        if key in self.cache:
            return list(self.cache[key]), True
        if self.predict:
            values = self.predict(question, texts)
        else:
            if not self.path or not Path(self.path).is_dir():
                raise FileNotFoundError("本地重排模型未就绪")
            values = bounded_predict(local_predict, (str(Path(self.path).resolve()),
                                                       [[question, text] for text in texts]), self.timeout)
        if len(values) != len(texts) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("重排结果必须为等长的一维有限数值列表")
        self.cache[key] = list(values)
        return list(values), False
