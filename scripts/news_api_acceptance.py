"""限量的真实只读新闻接口检查，记录健康、分页、排序和正文；不修改服务器。"""

import argparse
import time
from pathlib import Path
from agentic_rag.news_api import NewsClient
from agentic_rag.telemetry.exporters import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transport-note", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("不能覆盖已有验收记录")
    client = NewsClient(base_url=args.base_url)
    result = {"started_at": time.time(), "transport": args.transport_note, "checks": [], "pages": []}
    started = time.perf_counter()
    try:
        result["health"] = client.health()
        result["checks"].append({"name": "authenticated_health", "passed": result["health"].get("ok") is True})
        first = client.search(q="黄金", limit=20)
        result["pages"].append(first)
        if first.get("has_more") and first.get("next_cursor"):
            result["pages"].append(client.search(q="黄金", limit=20, cursor=first["next_cursor"]))
        rows = [item for page in result["pages"] for item in page["items"]]
        ids = [r["article_id"] for r in rows]
        ordering = [(r.get("published_at", ""), r["article_id"]) for r in rows]
        result["checks"].extend([
            {"name": "twenty_per_page", "passed": all(len(p["items"]) == 20 for p in result["pages"])},
            {"name": "two_distinct_pages", "passed": len(result["pages"]) == 2 and len(set(ids)) == len(ids)},
            {"name": "descending_order", "passed": ordering == sorted(ordering, reverse=True)},
        ])
        body = client.article(ids[0])
        result["article"] = body
        result["checks"].append({"name": "full_article", "passed": body.get("article_id") == ids[0] and len(body.get("content", "")) > 100})
        oldest = client.search(q="黄金", limit=20, order="asc")
        result["ascending_page"] = oldest
        keys = [(r.get("published_at", ""), r["article_id"]) for r in oldest["items"]]
        result["checks"].append({"name": "ascending_order", "passed": bool(keys) and keys == sorted(keys)})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:600]}
    result["elapsed_seconds"] = time.perf_counter() - started
    result["passed"] = not result.get("error") and len(result["checks"]) == 6 and all(c["passed"] for c in result["checks"])
    write_json(args.output, result)
    print({k: result[k] for k in ("transport", "checks", "elapsed_seconds", "passed")})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
