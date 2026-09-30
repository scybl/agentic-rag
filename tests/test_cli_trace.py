"""终端逐步骤追踪的离线测试。"""

import io
import threading
import unittest
from contextlib import redirect_stdout

from langchain_core.documents import Document
from ollama import ResponseError

from agentic_rag.cli import (
    ConsoleInputTimeout,
    TracePrinter,
    ask,
    input_with_timeout,
    run_with_trace,
    unload_model,
    warm_up_model,
)


class FakeGraph:
    def stream(self, input_state, *, stream_mode):
        self.input_state = input_state
        self.stream_mode = stream_mode
        for event in self.updates():
            yield "updates", event
            if "route" in event:
                yield "custom", {
                    "kind": "news_request", "page": 1, "query": "生猪",
                    "start": "", "end": "", "section": "", "limit": 20,
                    "continuation": False,
                }
                yield "custom", {"kind": "news_page", "page": 1, "count": 0, "has_more": False}

    def updates(self):
        document = Document(
            page_content="证据",
            metadata={"source": "method.md", "source_type": "vectorstore"},
        )
        yield {
            "route": {
                "selected_sources": ["vectorstore", "news_api"],
                "source_queries": {"vectorstore": "财务影响", "news_api": "生猪"},
                "plan_summary": "需要联合事实和分析方法",
                "news_search_plan": {
                    "suggested_time": {"mode": "unrestricted", "start": "", "end": "", "reason": "用户未要求时间限制"},
                    "start": "", "end": "", "time_note": "未设置时间条件",
                },
                "datasource": "vectorstore+news_api",
            }
        }
        yield {
            "collect_sources": {
                "documents_by_source": {"vectorstore": [document], "news_api": []},
                "documents": [document],
                "source_errors": {},
            }
        }
        yield {"grade_documents": {"documents": [document], "documents_by_source": {"vectorstore": [document]}}}
        yield {
            "generate": {
                "generation": "答案",
                "analysis_cache_hit": False,
            }
        }
        yield {
            "evaluate_generation": {
                "generation_grounded": True,
                "generation_check": "回答能够由证据支持",
            }
        }


class ServiceUnavailableError(RuntimeError):
    status_code = 503


class FailingGraph:
    def stream(self, input_state, *, stream_mode):
        raise ServiceUnavailableError("temporary")
        yield  # pragma: no cover


class FakeWarmupClient:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise ResponseError("busy", 503)
        return {"done": True}


class CLITraceTests(unittest.TestCase):
    def test_trace_shows_each_agent_step(self):
        output = io.StringIO()
        graph = FakeGraph()
        with redirect_stdout(output):
            result = run_with_trace(graph, "问题", verbose=True)

        text = output.getvalue()
        self.assertIn("流程 1 · 规划数据源", text)
        self.assertIn("本地知识库 + 新闻 API", text)
        self.assertIn("流程 2–5 · 收集并合并证据", text)
        self.assertIn("保留：1/1 条", text)
        self.assertIn("流程 9 · 核验答案依据", text)
        self.assertEqual(result["generation"], "答案")
        self.assertEqual(graph.stream_mode, ["updates", "custom"])
        self.assertIn("查询/新闻 API：生猪", text)
        self.assertIn("模型时间选择：不限时间", text)
        self.assertIn("实际时间：不限时间（不发送 start/end）", text)
        self.assertIn("本页上限：20 条", text)
        self.assertIn("第 1 页返回：0 条", text)

    def test_verbose_trace_does_not_truncate_plan_or_hide_errors(self):
        summary = "需要核对新闻中的供需因素。" * 20
        output = io.StringIO()
        trace = TracePrinter(verbose=True)
        with redirect_stdout(output):
            trace.show("route", {"selected_sources": ["news_api"], "plan_summary": summary}, {})
            trace.show_event({"kind": "news_error", "message": "HTTP 503", "partial_count": 20})
        self.assertIn(summary, output.getvalue())
        self.assertIn("新闻 API 请求失败：HTTP 503", output.getvalue())
        self.assertIn("保留已取得的 20 条候选", output.getvalue())

    def test_suggested_search_is_not_displayed_as_executed(self):
        output = io.StringIO()
        with redirect_stdout(output):
            TracePrinter(verbose=True).show("assess_evidence", {
                "evidence_assessment": {"ready": True, "summary": "可以条件预测", "missing_factors": ["成本"]},
                "pending_news_queries": ["饲料"], "next_action": "generate",
            }, {})
        self.assertIn("建议补搜（本轮不执行）/新闻：饲料", output.getvalue())
        self.assertNotIn("待执行补搜", output.getvalue())

    def test_final_source_label_only_lists_sources_used_in_answer(self):
        output = io.StringIO()
        with redirect_stdout(output):
            ask(FakeGraph(), "问题")
        self.assertIn("[实际使用数据源：vectorstore]", output.getvalue())
        self.assertNotIn("[实际使用数据源：vectorstore+news_api]", output.getvalue())

    def test_ollama_503_is_explained_without_traceback(self):
        output = io.StringIO()
        with redirect_stdout(output):
            succeeded = ask(FailingGraph(), "问题")
        text = output.getvalue()
        self.assertFalse(succeeded)
        self.assertIn("模型服务请求返回 HTTP 503", text)
        self.assertIn("交互模式仍可继续", text)
        self.assertNotIn("Traceback", text)

    def test_warmup_retries_503_and_keeps_model_loaded(self):
        output = io.StringIO()
        client = FakeWarmupClient(failures=1)
        sleeps = []
        with redirect_stdout(output):
            succeeded = warm_up_model(client=client, sleep_fn=sleeps.append)
        self.assertTrue(succeeded)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(sleeps, [3.0])
        self.assertFalse(client.calls[-1]["think"])
        self.assertEqual(client.calls[-1]["keep_alive"], "30m")
        self.assertIn("模型已就绪", output.getvalue())

    def test_console_input_times_out_without_blocking_process_exit(self):
        release = threading.Event()

        def blocked_input(_prompt):
            release.wait(1)
            return "迟到的输入"

        try:
            with self.assertRaises(ConsoleInputTimeout):
                input_with_timeout("> ", 0.01, input_fn=blocked_input)
        finally:
            release.set()

    def test_unload_model_requests_immediate_release(self):
        output = io.StringIO()
        client = FakeWarmupClient()
        with redirect_stdout(output):
            succeeded = unload_model(client=client)
        self.assertTrue(succeeded)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["model"])
        self.assertEqual(client.calls[0]["prompt"], "")
        self.assertFalse(client.calls[0]["stream"])
        self.assertEqual(client.calls[0]["keep_alive"], 0)
        self.assertIn("已释放本地模型", output.getvalue())


if __name__ == "__main__":
    unittest.main()
