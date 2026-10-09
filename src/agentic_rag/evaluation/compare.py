"""配对比较运行结果，并先验证冻结输入、版本和产物完整性。"""

import json
import statistics
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, model_validator

from ..telemetry.schema import Contract, Trace
from .contracts import ExperimentConfig, digest, load_suite


class RankingItem(Contract):
    document_id: str
    rank: Annotated[StrictInt, Field(ge=1)]
    score: float
    components: dict[str, float]
    weights: dict[str, float]
    selected: bool
    rank_history: list[dict[str, JsonValue]] = Field(default_factory=list)


class Observation(Contract):
    case_id: str
    category: str
    split: Literal["dev", "test"]
    repeat: Annotated[StrictInt, Field(ge=1)]
    status: Literal["completed", "failed"]
    error_type: str | None
    elapsed_seconds: float = Field(ge=0)
    ranking: list[RankingItem]
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    trace_hash: str
    retrieval: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_ranking(self):
        if [r.rank for r in self.ranking] != list(range(1, len(self.ranking) + 1)):
            raise ValueError("排名必须从 1 连续递增")
        if len({r.document_id for r in self.ranking}) != len(self.ranking):
            raise ValueError("排名不能含重复材料")
        if (self.status == "failed") != (self.error_type is not None):
            raise ValueError("失败状态与错误类型不一致")
        if self.status == "failed" and self.ranking:
            raise ValueError("失败样本不能留下成功排名")
        return self


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的运行清单版本")
    if manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("运行未完成，不能混入成功实验比较")
    suite = load_suite(directory / "suite.json")
    if suite.fingerprint != manifest.get("suite_hash"):
        raise ValueError("材料或 qrels 与运行清单不一致")
    if (suite.suite_id, suite.version, suite.synthetic) != (
            manifest.get("suite_id"), manifest.get("suite_version"), manifest.get("synthetic")):
        raise ValueError("清单中的数据集身份与快照不一致")
    if digest([d.model_dump(mode="json") for d in suite.documents]) != manifest.get("corpus_hash"):
        raise ValueError("材料指纹不一致")
    config = ExperimentConfig.model_validate(manifest["config"])
    manifest["config"] = config.model_dump(mode="json")  # 给旧阶段一清单补协议默认值。
    raw = read_json(directory / "observations.json")
    if digest(raw) != manifest.get("observations_hash"):
        raise ValueError("观察结果指纹不一致")
    observations = [Observation.model_validate(item) for item in raw]
    if len(observations) != manifest.get("observation_count"):
        raise ValueError("观察结果数量不一致")
    cases = {c.case_id: c for c in suite.cases if config.split == "all" or c.split == config.split}
    expected = {(cid, repeat) for cid in cases for repeat in range(1, config.repeat + 1)}
    keys = {(o.case_id, o.repeat) for o in observations}
    if len(keys) != len(observations) or keys != expected:
        raise ValueError("样本缺失、重复或超出实验配置")
    for item in observations:
        case = cases[item.case_id]
        if item.category != case.category or item.split != case.split:
            raise ValueError("样本分类与快照不一致")
        if item.status == "completed":
            ranked_ids, document_ids = {r.document_id for r in item.ranking}, {d.document_id for d in suite.documents}
            if not ranked_ids <= document_ids:
                raise ValueError("排名引用了未知材料")
            if config.adapter == "current_tfidf" and ranked_ids != document_ids:
                raise ValueError("当前基线必须保留全部候选排名")
            if any(r.selected != (r.rank <= config.top_k) for r in item.ranking):
                raise ValueError("入选标记与 Top-k 不一致")
        trace_data = read_json(directory / "traces" / f"{item.trace_id}.json")
        trace = Trace.model_validate(trace_data)
        if trace.run_id != item.trace_id or digest(trace_data) != item.trace_hash:
            raise ValueError("Trace 编号或内容指纹不一致")
        root = next(span for span in trace.spans if span.parent_id is None)
        if (root.attributes.get("case_id"), root.attributes.get("repeat"), root.status) != (
                item.case_id, item.repeat, item.status):
            raise ValueError("Trace 与所属样本不一致")
        if root.attributes.get("retrieval", {}) != item.retrieval:
            raise ValueError("Trace 中的检索阶段与观察结果不一致")
        ranking_events = [event for event in trace.events if event.name == "ranking"]
        if len(ranking_events) != 1 or [RankingItem.model_validate(r) for r in ranking_events[0].attributes.get("candidates", [])] != item.ranking:
            raise ValueError("Trace 中的排名与观察结果不一致")
    failed = any(o.status == "failed" for o in observations)
    if failed != (manifest["status"] == "completed_with_errors"):
        raise ValueError("运行状态隐瞒了失败样本")
    return manifest, observations


def compare_runs(left, right, *, allow_changes=()):
    before, left_rows = load_run(left)
    after, right_rows = load_run(right)
    allowed = set(allow_changes)
    if not allowed <= (set(ExperimentConfig.model_fields) - {"repeat", "split"}) | {"implementation"}:
        raise ValueError("只能显式改变检索参数或 implementation，不能改变样本与重复次数")
    for field in ("suite_hash", "corpus_hash", "scope", "schema_version"):
        if before[field] != after[field]:
            raise ValueError(f"不可比较：{field} 不一致")
    changes = {}
    for field in set(before["config"]) | set(after["config"]):
        if before["config"].get(field) != after["config"].get(field):
            if field not in allowed:
                raise ValueError(f"未授权的实验变量变化：{field}")
            changes[field] = [before["config"].get(field), after["config"].get(field)]
    for field in ("digest", "dependencies", "python", "platform"):
        if before["implementation"][field] != after["implementation"][field]:
            if "implementation" not in allowed:
                raise ValueError(f"实现或运行环境变化：{field}；请显式声明 implementation")
            changes[f"implementation.{field}"] = [before["implementation"][field], after["implementation"][field]]
    if before.get("model_revision") != after.get("model_revision"):
        if not allowed & {"adapter", "embedding_model", "reranker_model", "implementation"}:
            raise ValueError("模型文件指纹变化，需显式声明模型或实现变量")
        changes["model_revision"] = [before.get("model_revision"), after.get("model_revision")]
    right_by_key = {(o.case_id, o.repeat): o for o in right_rows}
    if set(right_by_key) != {(o.case_id, o.repeat) for o in left_rows}:
        raise ValueError("两次实验的样本无法一一配对")
    pairs = []
    for a in left_rows:
        b = right_by_key[a.case_id, a.repeat]
        valid = a.status == b.status == "completed"
        ar, br = [r.document_id for r in a.ranking], [r.document_id for r in b.ranking]
        pairs.append({"case_id": a.case_id, "repeat": a.repeat, "category": a.category,
                      "left_status": a.status, "right_status": b.status,
                      "same_ranking": ar == br if valid else None,
                      "same_selection": [r.document_id for r in a.ranking if r.selected] ==
                                        [r.document_id for r in b.ranking if r.selected] if valid else None,
                      "latency_delta_seconds": b.elapsed_seconds - a.elapsed_seconds if valid else None})
    deltas = [p["latency_delta_seconds"] for p in pairs if p["latency_delta_seconds"] is not None]
    return {"schema_version": 1, "left": before["experiment_id"], "right": after["experiment_id"],
            "changes": changes, "pairs": pairs, "paired_count": len(pairs),
            "left_failed": sum(o.status == "failed" for o in left_rows),
            "right_failed": sum(o.status == "failed" for o in right_rows),
            "left_degraded": sum(bool(o.retrieval.get("fallbacks")) for o in left_rows),
            "right_degraded": sum(bool(o.retrieval.get("fallbacks")) for o in right_rows),
            "changed_rankings": sum(p["same_ranking"] is False for p in pairs),
            "latency_delta_median_seconds": statistics.median(deltas) if deltas else None,
            "limitations": ["只比较排序稳定性和观测耗时，不证明相关性或最终答案质量提升。",
                            "耗时包含冷启动和运行噪声，未做显著性检验。",
                            "标准检索质量请使用 score 命令；本报告保留配对运行行为。"]}
