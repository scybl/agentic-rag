"""正式答案核验节点及后续决策测试。"""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from agentic_rag.config import settings
from agentic_rag.graph import nodes
from agentic_rag.graph.chains import AnswerAssessment


def assessment(**changes):
    values = dict(
        decision="accept",
        grounded=True, answers_question=True, needs_more_evidence=False,
        issues=[], revision_instructions="", news_queries=[], web_queries=[],
    )
    return AnswerAssessment(**{**values, **changes})


class GenerationCheckTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "question": "问题",
            "documents": [Document(page_content="证据", metadata={"source": "one"})],
            "generation": "答案",
            "retries": 0,
        }

    def test_grounded_answer_finishes(self):
        with patch.object(nodes, "get_answer_reviewer") as grader:
            grader.return_value.invoke.return_value = assessment()
            update = nodes.evaluate_generation(self.state)
        self.assertTrue(update["generation_grounded"])
        self.assertEqual(
            nodes.decide_after_generation({**self.state, **update}), "finish"
        )

    def test_blank_answer_never_passes_grounding(self):
        update = nodes.evaluate_generation({**self.state, "generation": "  "})
        self.assertFalse(update["generation_grounded"])
        self.assertIn("没有返回可用答案", update["generation_check"])

    def test_answer_context_is_balanced_and_bounded(self):
        documents = []
        for source_type in ("vectorstore", "news_api", "web_search"):
            for index in range(5):
                documents.append(
                    Document(
                        page_content=f"{source_type}-{index}-" + "证据" * 500,
                        metadata={
                            "source": f"{source_type}-{index}",
                            "source_type": source_type,
                        },
                    )
                )
        selected = nodes._select_answer_documents(documents)
        selected_types = {doc.metadata["source_type"] for doc in selected}
        context = nodes._format_docs(selected)
        self.assertLessEqual(len(selected), settings.generation_max_documents)
        self.assertEqual(
            selected_types, {"vectorstore", "news_api", "web_search"}
        )
        self.assertLessEqual(len(context), settings.generation_context_chars + 32)

    def test_ungrounded_answer_revises_then_stops_without_caching(self):
        with (patch.object(nodes, "get_answer_reviewer") as grader,
              patch.object(nodes, "AnalysisCache") as cache):
            grader.return_value.invoke.return_value = assessment(
                grounded=False, issues=["缺少依据"], revision_instructions="删除虚构事实",
            )
            retry_update = nodes.evaluate_generation(self.state)
            exhausted_update = nodes.evaluate_generation(
                {**self.state, "answer_revisions": settings.max_answer_revisions, "analysis_cache_key": "key"}
            )
        cache.assert_not_called()
        self.assertFalse(retry_update["generation_grounded"])
        self.assertEqual(
            nodes.decide_after_generation({**self.state, **retry_update}), "revise"
        )
        self.assertIn("未通过最终核验", exhausted_update["generation"])
        self.assertEqual(
            nodes.decide_after_generation(
                {**self.state, "retries": settings.max_retries, **exhausted_update}
            ),
            "finish",
        )

    def test_grounded_refusal_does_not_pass_task_completion(self):
        with (patch.object(nodes, "get_answer_reviewer") as reviewer,
              patch.object(nodes, "AnalysisCache") as cache):
            reviewer.return_value.invoke.return_value = assessment(
                answers_question=False, issues=["有事实却未作条件预测"],
                revision_instructions="依据现有供需信息写出条件情景",
            )
            result = nodes.evaluate_generation({
                **self.state, "generation": "资料没有现成的未来预测 [E1]", "task_type": "forecast",
                "analysis_cache_key": "key",
            })
        self.assertTrue(result["generation_grounded"])
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["next_action"], "revise")
        self.assertIn("条件情景", result["revision_feedback"])
        cache.assert_not_called()

    def test_missing_fact_triggers_only_new_queries(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment(
                grounded=False, needs_more_evidence=True, issues=["缺少产能基准"],
                news_queries=["生猪", "能繁母猪"], web_queries=[],
            )
            result = nodes.evaluate_generation({**self.state, "query_history": ["news_api: 生猪"]})
        self.assertEqual(result["next_action"], "supplement")
        self.assertEqual(result["pending_news_queries"], ["能繁母猪"])

    def test_refusal_guard_catches_even_erroneous_model_acceptance(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment()
            result = nodes.evaluate_generation({
                **self.state, "task_type": "forecast", "evidence_assessment": {"ready": True},
                "generation": "现有资料未提供2027年的预测结果，因此无法分析2027年的走势 [E1]。",
            })
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["next_action"], "revise")
        self.assertIn("现成未来预测", result["generation_check"])

    def test_refusal_guard_allows_real_insufficiency_and_normal_caveats(self):
        base = {**self.state, "task_type": "forecast", "evidence_assessment": {"ready": True}}
        assert not nodes._is_unnecessary_forecast_refusal({
            **base, "generation": "缺少需求数据，无法预测精确价格，但基准情景下可能震荡；若供给减少则有望回升。",
        })
        assert not nodes._is_unnecessary_forecast_refusal({
            **base, "evidence_assessment": {"ready": False},
            "generation": "没有任何数据，无法预测。",
        })

    def test_conflicting_review_flags_cannot_pass(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment(decision="revise")
            result = nodes.evaluate_generation(self.state)
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["next_action"], "revise")

    def test_accept_flag_with_written_revision_request_cannot_pass(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment(revision_instructions="应基于现有事实作条件预测")
            result = nodes.evaluate_generation(self.state)
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["next_action"], "revise")

    def test_forecast_requires_existing_evidence_citations(self):
        for answer in ("价格可能上涨", "价格可能上涨 [E99]", "事实 [E1, E99]"):
            with self.subTest(answer=answer), patch.object(nodes, "get_answer_reviewer") as reviewer:
                reviewer.return_value.invoke.return_value = assessment()
                result = nodes.evaluate_generation({**self.state, "generation": answer, "task_type": "forecast"})
            self.assertFalse(result["generation_grounded"])
            self.assertEqual(result["next_action"], "revise")

    def test_valid_grouped_citations_pass(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment()
            result = nodes.evaluate_generation({
                **self.state, "documents": self.state["documents"] * 2,
                "generation": "条件预测 [E1, E2]", "task_type": "forecast",
            })
        self.assertTrue(result["generation_grounded"])

    def test_probability_contract_overrides_erroneous_model_acceptance(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment()
            result = nodes.evaluate_generation({
                **self.state,
                "question": "预测2027年第二季度，美国降息概率",
                "original_question": "预测2027年第二季度，美国降息概率",
                "task_type": "forecast", "estimate_kind": "probability",
                "evidence_assessment": {"probability_contract": {"passed": True}},
                "generation": "2027年第二季度可能降息 [E1]。",
            })
        self.assertFalse(result["generation_complete"])
        self.assertEqual(result["next_action"], "revise")
        self.assertIn("概率", result["generation_check"])

    def test_complete_probability_answer_passes_deterministic_contract(self):
        with patch.object(nodes, "get_answer_reviewer") as reviewer:
            reviewer.return_value.invoke.return_value = assessment()
            result = nodes.evaluate_generation({
                **self.state,
                "question": "预测2027年第二季度，美国降息概率",
                "original_question": "预测2027年第二季度，美国降息概率",
                "task_type": "forecast", "estimate_kind": "probability",
                "evidence_assessment": {"probability_contract": {"passed": True}},
                "generation": "截至2026-09-30，事件为2027年第二季度降息。基于FedWatch隐含概率，估计为45% [E1]。",
            })
        self.assertTrue(result["generation_complete"])

    def test_cache_version_changes_with_actual_evidence_order(self):
        self.assertNotEqual(nodes._cache_version("[E1] 产能 [E2] 需求"),
                            nodes._cache_version("[E1] 需求 [E2] 产能"))


if __name__ == "__main__":
    unittest.main()
