"""将当前工作区的公开项目文件打包；不包含Git历史、密钥、新闻库或面试资料。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {"src", "tests", "scripts", "docs", "knowledge", "ops", ".github"}
PUBLIC_FILES = {"README.md", "pyproject.toml", "main.py", ".env.example", ".gitignore", ".gitattributes",
                "LICENSE", "LICENSE.md", "constraints-tested.txt"}


def public_file(name):
    path = Path(name)
    parts = path.parts
    if not parts or any("面试" in p or "简历" in p or "interview" in p.casefold() for p in parts):
        return False
    if parts[0] == "docs" and (
        path.name == "炫技增强技术设计.md"
        or any(p.casefold() in {"plans", "planning"} for p in parts[1:])
        or path.suffix.lower() in {".docx", ".pdf"}
        or (path.suffix.lower() == ".md" and ("规划" in path.name or "roadmap" in path.name.casefold()))
    ):
        return False
    if any(p in {"__pycache__", "node_modules", ".git", ".env", ".chroma"} for p in parts):
        return False
    if path.suffix.lower() in {".pyc", ".db", ".sqlite", ".sqlite3", ".pem", ".key"}:
        return False
    return (name in PUBLIC_FILES or parts[0] in PUBLIC_ROOTS
            or parts[:2] == ("evaluation", "suites") or name == "evaluation/run_evaluation.py"
            or name == "evaluation/dataset.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("发布包已存在，拒绝覆盖")
    # -z 避免Git对中文文件名做引号/转义处理。
    listed = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted({n for n in listed.decode("utf-8").split("\0") if n and public_file(n)})
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    if any(n and ("面试" in n or "简历" in n or "interview" in n.casefold() or n == ".env") for n in tracked):
        raise ValueError("Git仍跟踪私人文件，停止发布")
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)[1]
    manifest = {"version": version, "scope": "reviewed public working tree, not a Git history archive", "files": {}}
    contents = {}
    for name in names:
        source = ROOT / name
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            raise ValueError("发布文件越界或是符号链接：" + name)
        if not source.is_file():
            continue  # 工作区中已删除但尚未提交的文件，不从HEAD重新带回。
        raw = source.read_bytes()
        manifest["files"][name] = hashlib.sha256(raw).hexdigest()
        contents[name] = raw
    # 仅比较实际配置的新闻密钥，不将值写入错误或发布包。
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("NEWS_API_KEY", "")
    if key and len(key) >= 12 and any(key.encode() in raw for raw in contents.values()):
        raise ValueError("发布内容含真实新闻API密钥，停止；不回显密钥")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)
        archive.writestr("release-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"version": version, "public_files": len(contents),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
