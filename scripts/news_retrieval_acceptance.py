"""对真实冻结材料比较检索器；目标是找回标注原文，不等于全库召回或答案评分。"""

import argparse
import json
from pathlib import Path

from agentic_rag.evaluation.contracts import FinanceSuite, ExperimentConfig
from agentic_rag.evaluation.runner import run_experiment
from agentic_rag.evaluation.metrics import score_run
from agentic_rag.telemetry.exporters import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for c in snapshot["cases"]:
        # 两篇文章推理、开放预测不使用不完整 qrels 伪装检索金标准。
        if c["category"] in {"analysis", "forecast", "probability", "conflict"}:
            continue
        cases.append({"case_id": c["id"], "question": c["question"], "category": "fact", "split": "dev",
            "answerable": True, "qrels": [{"document_id": p, "relevance": 3} for p in c["sources"]],
            "reference_answer": "；".join(c["quotes"]), "key_evidence": c["quotes"],
            "forbidden_claims": c["forbidden"] or ["不得编造来源以外的数据"],
            "risk_obligations": ["保留报道的单位、事件日期、预测限定"],
            "rubric": ["找回问题标注的原文；其他未标注文档不是人工全面判定的负例"]})
    suite = FinanceSuite(suite_id="news_source_localization_v1", version="1",
        provenance="从真实新闻快照定位16道题标注的原文；单人标注开发集，其他材料未全面标注相关性；不用于宣称领域质量提升",
        synthetic=False, cases=cases, documents=[{"document_id": p, "title": d["metadata"]["title"],
            "summary": d["page_content"][:500], "body": d["page_content"],
            "published_at": d["metadata"].get("published_at"), "source": d["metadata"]["source"]}
            for p, d in snapshot["documents"].items()])
    suite_path = args.output / "suite.json"
    write_json(suite_path, suite.model_dump(mode="json"))
    reports = {}
    for adapter, mmr in [("current_tfidf", False), ("bm25", False), ("dense", False), ("hybrid", False), ("hybrid", True)]:
        label = "mmr" if mmr else adapter
        target = args.output / label
        config = ExperimentConfig(adapter=adapter, repeat=2, split="dev", mmr=mmr,
                                  embedding_model=args.embedding_model if adapter in {"dense", "hybrid"} else None)
        run_experiment(suite_path, target, config)
        reports[label] = score_run(target, (1, 4))
        write_json(args.output / f"{label}-score.json", reports[label])
        print(label, json.dumps(reports[label], ensure_ascii=False)[:1400], flush=True)
    write_json(args.output / "summary.json", reports)


if __name__ == "__main__":
    main()
