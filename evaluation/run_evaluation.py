"""让智能体处理评估数据集，并使用 RAGAS 评分。

用法：python evaluation/run_evaluation.py

程序会生成 evaluation/results.csv（逐题分数）并打印摘要。
在 CPU 上使用 3B 裁判模型时，绝对分数存在噪声；更重要的是在每次修改后
采用一致方式跟踪这些分数。如果硬件允许，可将 EVAL_MODEL 设置为更大的模型，
以获得更可靠的评估结果。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agentic_rag.config import settings
from agentic_rag.graph.build import build_graph
from agentic_rag.ingestion import ensure_index, get_embeddings
from agentic_rag.ollama_connection import ollama_client_kwargs
from agentic_rag.evaluation.answer_samples import actual_contexts

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"


def collect_samples(graph, cases: list[dict]) -> list[SingleTurnSample]:
    """让智能体处理每个评估问题，并记录其执行轨迹。"""
    samples = []
    for i, case in enumerate(cases, start=1):
        start = time.perf_counter()
        result = graph.invoke({"question": case["question"]})
        elapsed = time.perf_counter() - start
        print(f"[{i}/{len(cases)}] ({elapsed:.1f}s) {case['question']}")
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=result["generation"],
                retrieved_contexts=actual_contexts(result),
                reference=case["ground_truth"],
            )
        )
    return samples


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    print("== Running the agent over the evaluation dataset ==")
    ensure_index()
    samples = collect_samples(build_graph(), cases)

    print("\n== Scoring with RAGAS ==")
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=settings.eval_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            client_kwargs=ollama_client_kwargs(settings.ollama_base_url),
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,
        # CPU 推理会串行处理并发请求：并行裁判调用只会进入队列，最终触发
        # 默认超时并留下 NaN 分数。
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    df = result.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)

    print("\n== Average scores ==")
    metric_columns = df.select_dtypes("number").columns
    for metric in metric_columns:
        print(f"  {metric:35s} {df[metric].mean():.3f}")
    print(f"\nPer-question results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
