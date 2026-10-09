"""显式运行的本地模型会话冒烟；合成上下文，不调用新闻检索。

python scripts/conversation_smoke.py --output .chroma/conversation-smoke.json
不是自动测试默认步骤，也不把少量通过结果解释为真实指代准确率。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from agentic_rag import conversation as memory
from agentic_rag.token_usage import UsageLedger, usage_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        ("pronoun", "它的主要风险是什么？", "followup", "比亚迪", 3),
        ("ordinal", "刚才第二条具体有什么影响？", "followup", "原材料", 3),
        ("new_topic", "解释黄金与实际利率的关系", "independent", "黄金", 3),
        ("compression", "刚才第二条具体有什么影响？", "followup", "原材料", 0),
        ("new_time", "它在2025年的风险是什么？", "followup", "2025", 3),
        ("ambiguous", "它的风险呢？", "clarify", "", 3),
    ]
    report = {"time": datetime.now(timezone.utc).isoformat(), "model": memory.settings.llm_model,
              "scope": "synthetic_local_resolver_only_not_end_to_end_accuracy", "cases": []}
    ledger = UsageLedger()
    with tempfile.TemporaryDirectory(prefix="rag-conversation-smoke-") as temporary, usage_session(ledger):
        database = memory.ConversationStore(Path(temporary) / "conversations.sqlite")
        for name, question, mode, expected, recent in cases:
            session = memory.Conversation(database, recent_turns=recent)
            question_before = "分析比亚迪近期新闻" if name != "ambiguous" else "比较比亚迪和宁德时代"
            answer = "1. 比亚迪销量增长\n2. 比亚迪原材料成本下降" if name != "ambiguous" else "比亚迪与宁德时代分别面临不同风险。"
            if name == "compression":
                answer += "\n" + "这是合成的背景解释，不是实时新闻证据。\n" * 200
            database.append(session.id, expected_revision=0, run_id=name, question=question_before,
                            resolved_question=question_before, answer=answer, status="complete", resolution={})
            started = time.perf_counter()
            result = session.prepare(question)
            decision = result["conversation_resolution"]
            passed = decision["mode"] == mode and expected in result["question"]
            row = {"name": name, "passed": passed, "seconds": round(time.perf_counter() - started, 2),
                   "expected_mode": mode, "input": question, "resolved_question": result["question"], "decision": decision}
            report["cases"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report["usage"] = ledger.report()["current"]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["passed"] for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
