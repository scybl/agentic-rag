"""本地模型的精确余弦检索；实验小语料使用全量矩阵作为无 ANN 误差的基准。"""

import hashlib
from pathlib import Path

import numpy as np

from ..evaluation.contracts import digest


def model_fingerprint(path):
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("模型必须是本地目录")
    files = {}
    for file in sorted(folder.rglob("*")):
        if file.is_file() and ".git" not in file.relative_to(folder).parts and file.suffix in {
                ".json", ".safetensors", ".bin", ".txt", ".model"}:
            hasher = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            files[file.relative_to(folder).as_posix()] = hasher.hexdigest()
    if not files:
        raise ValueError("本地模型目录没有可识别的模型文件")
    return {"digest": digest(files), "files": files}


def normalize(vectors):
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("嵌入必须是有限数值二维矩阵")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("不能用零向量计算余弦检索")
    return array / norms


class DenseIndex:
    def __init__(self, documents, model_path=None, *, query_prefix="", encoder=None):
        self.ids = [d.document_id for d in documents]
        self.query_prefix = query_prefix
        if encoder is None:
            if not model_path or not Path(model_path).is_dir():
                raise ValueError("Dense 实验需要已下载的本地模型目录")
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(str(Path(model_path).resolve()), device="cpu",
                                          local_files_only=True, trust_remote_code=False)
        self.encoder = encoder
        self.vectors = normalize(self.encoder.encode([d.title + " " + d.summary for d in documents],
                                                      show_progress_bar=False))
        if len(self.vectors) != len(self.ids):
            raise ValueError("嵌入行数与材料数不一致")
        self.by_id = dict(zip(self.ids, self.vectors))

    def search(self, query, limit):
        vector = normalize(self.encoder.encode([self.query_prefix + query], show_progress_bar=False))
        if len(vector) != 1 or vector.shape[1] != self.vectors.shape[1]:
            raise ValueError("查询与语料嵌入维度不一致")
        scores = self.vectors @ vector[0]
        return sorted(zip(self.ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))[:limit]
