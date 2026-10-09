"""同一关键词、同一无日期条件的真实分页大小对照；只读后端，不用改时间掩盖超时。"""

import argparse
import time
from pathlib import Path

from agentic_rag.news_api import NewsClient
from agentic_rag.telemetry.exporters import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limits", default="2,20")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("保留旧诊断，请指定新文件")
    client = NewsClient(base_url=args.base_url)
    results = []
    for limit in map(int, args.limits.split(",")):
        row = {"query": args.query, "limit": limit, "start": "", "end": "", "order": "desc"}
        started = time.perf_counter()
        try:
            result = client.search(q=args.query, limit=limit)
            row.update(success=True, count=len(result["items"]), has_more=result.get("has_more"),
                       request_id=result.get("request_id"), article_ids=[x["article_id"] for x in result["items"]])
        except Exception as exc:
            row.update(success=False, error_type=type(exc).__name__, error=str(exc)[:600])
        row["elapsed_seconds"] = time.perf_counter() - started
        results.append(row)
        write_json(args.output, results)
        print(row, flush=True)
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
