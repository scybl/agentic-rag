"""比较同源开发验收；标签改动显式记录，绝不覆盖旧运行分数。"""

import argparse
import json
from pathlib import Path

from agentic_rag.evaluation.acceptance import checks
from agentic_rag.telemetry.exporters import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(before, after, labels=None):
    before, after = Path(before), Path(after)
    old, new = read(before / "snapshot.json"), read(after / "snapshot.json")
    if old["documents"] != new["documents"]:
        raise ValueError("来源正文/元数据不同，不允许当作同源修复对照")
    old_cases = {c["id"]: c for c in old["cases"]}
    new_cases = {c["id"]: c for c in new["cases"]}
    if labels:
        proposed = {c["id"]: c for c in read(labels)["cases"]}
        if set(proposed) != set(new_cases) or any(
            proposed[k][f] != new_cases[k][f] for k in new_cases for f in ("question", "sources", "quotes", "category")):
            raise ValueError("新标签改变了问题或来源，不能重计同题结果")
        new_cases = proposed
    if set(old_cases) != set(new_cases):
        raise ValueError("问题集合不同")
    label_changes = []
    for key, case in new_cases.items():
        for field in ("question", "sources", "quotes", "category"):
            if case[field] != old_cases[key][field]:
                raise ValueError(f"问题或标注来源变化：{key}/{field}")
        if any(case[k] != old_cases[key][k] for k in ("required", "forbidden")):
            label_changes.append(key)
    manifests = [read(p / "manifest.json") for p in (before, after)]
    if any(set(m["cases"]) != set(new_cases) for m in manifests):
        raise ValueError("不是完整同题验收，不能用完整套件做分母")
    if set(manifests[0]["variants"]) != set(manifests[1]["variants"]):
        raise ValueError("比较模式不一致")
    records = []
    for key, case in new_cases.items():
        for variant in manifests[1]["variants"]:
            pair = [read(p / f"{key}-{variant}.json") for p in (before, after)]
            if any(r["case_id"] != key or r["variant"] != variant for r in pair):
                raise ValueError("记录身份与文件名不一致")
            old_recheck = checks(case, {"generation": pair[0].get("answer", ""),
                                       "evidence_context": pair[0].get("evidence_context", "")})
            new_recheck = checks(case, {"generation": pair[1].get("answer", ""),
                                       "evidence_context": pair[1].get("evidence_context", "")})
            records.append({"case_id": key, "variant": variant,
                "before_run": pair[0].get("run_id"), "after_run": pair[1].get("run_id"),
                "before_original_joint_passed": pair[0]["workflow_passed"] and pair[0]["checks"]["checks_passed"],
                "before_current_labels_joint_passed": pair[0]["workflow_passed"] and old_recheck["checks_passed"],
                "after_joint_passed": pair[1]["workflow_passed"] and pair[1]["checks"]["checks_passed"],
                "after_current_labels_joint_passed": pair[1]["workflow_passed"] and new_recheck["checks_passed"],
                "before_seconds": pair[0]["elapsed_seconds"], "after_seconds": pair[1]["elapsed_seconds"],
                "before_usage": pair[0].get("usage", {}).get("current") if pair[0].get("usage") else None,
                "after_usage": pair[1].get("usage", {}).get("current") if pair[1].get("usage") else None})
    return {"scope": "同源开发集；源码前后不同，模型真实执行；非盲测或线上SLA",
        "label_changes": label_changes, "labels_file": str(labels) if labels else "after snapshot",
        "before": read(before / "summary.json"),
        "after": read(after / "summary.json"), "records": records,
        "limits": ["规则通过不等于语义正确，必须另看答案与原文", "多个题目共享来源，不能当独立市场事件",
                   "有阅读复用与系统缓存，不是每题冷启动", "只汇总实际模型报告用量；思考属于生成分项，不重复相加"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="可选的已修正标签；只允许改变判分规则，不能改问题/来源")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("保留已有报告，请指定新文件")
    write_json(args.output, compare(args.before, args.after, args.labels))
    print(args.output)


if __name__ == "__main__":
    main()
