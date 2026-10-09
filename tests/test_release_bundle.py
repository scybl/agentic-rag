"""发布文件白名单不应因源码导出或个人材料增加而泄露内容。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_bundle", Path(__file__).resolve().parents[1] / "scripts/release_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


@pytest.mark.parametrize("name", [".env", "docs/Agent面试.html", "docs/interview-assets/diagram.svg",
                                  "evaluation/runs/news_api.before.py", "rag_copy_staging/file.py",
                                  ".rag-copy-receiver.mjs", "src/agentic_rag/__pycache__/cache.pyc", "ops/private.key",
                                  "docs/炫技增强技术设计.md", "docs/实施规划.md", "docs/Roadmap.md",
                                  "docs/plans/future.md", "docs/planning/design.md", "docs/简历.docx",
                                  "docs/notes.docx", "docs/local.pdf"])
def test_private_and_transient_files_are_excluded(name):
    assert not bundle.public_file(name)


@pytest.mark.parametrize("name", ["README.md", ".env.example", "docs/status.md", "src/agentic_rag/cli.py",
                                  "ops/news_api/tls_server.py", "evaluation/suites/news_acceptance_v1/cases.json"])
def test_public_project_files_are_included(name):
    assert bundle.public_file(name)
