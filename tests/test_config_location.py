"""源码安装和wheel安装使用不同模块布局，数据必须落在用户工作区。"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("explicit", [False, True])
def test_installed_layout_resolves_workspace_and_loads_its_env(tmp_path, explicit):
    package = tmp_path / "environment/Lib/site-packages/agentic_rag"
    package.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "src/agentic_rag/config.py"
    installed = package / "config.py"
    installed.write_bytes(source.read_bytes())
    work = tmp_path / "workspace"
    work.mkdir()
    chosen = tmp_path / "explicit-home" if explicit else work
    chosen.mkdir(exist_ok=True)
    (chosen / ".env").write_text("RETRIEVAL_K=7\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"AGENTIC_RAG_HOME", "RETRIEVAL_K", "PYTHON_DOTENV_DISABLED"}}
    if explicit:
        env["AGENTIC_RAG_HOME"] = str(chosen)
    command = "import runpy,json,sys; c=runpy.run_path(sys.argv[1]); print(json.dumps([str(c['PROJECT_ROOT']),c['settings'].retrieval_k]))"
    result = subprocess.run([sys.executable, "-c", command, str(installed)], cwd=work, env=env,
                            capture_output=True, text=True, check=True, timeout=10)
    root, retrieval = json.loads(result.stdout)
    assert Path(root) == chosen
    assert retrieval == 7


def test_editable_layout_keeps_repository_root():
    from agentic_rag.config import PROJECT_ROOT
    if os.getenv("AGENTIC_RAG_HOME"):
        pytest.skip("此进程显式设置了项目根目录")
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
