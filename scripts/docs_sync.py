"""离线文档门禁：AST 事实生成、链接/定位检查、显式审阅指纹。仅用标准库。"""

import argparse
import ast
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "docs/reference.md"
LOCK = "docs/.review-state.json"


def read(root, path):
    return (root / path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def tree(root, path):
    return ast.parse(read(root, path), filename=path)


def cell(value):
    return str(value).replace("|", "&#124;").replace("\n", " ")


def code(node):
    return ast.unparse(node) if node is not None else "未指定"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)])


def config_rows(root):
    """保留默认表达式，避免误把 EVAL_MODEL 等动态回退说成固定值。"""
    result = []
    settings = next(n for n in tree(root, "src/agentic_rag/config.py").body
                    if isinstance(n, ast.ClassDef) and n.name == "Settings")
    for field in settings.body:
        if not isinstance(field, ast.AnnAssign):
            continue
        names = list(dict.fromkeys(n.args[0].value for n in ast.walk(field.value)
            if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr == "getenv")
                 or (isinstance(n.func, ast.Name) and n.func.id in {"_path", "_boolean"}))))
        result.append((" / ".join(names), field.target.id, "`" + code(field.value) + "`"))
    return result


def graph_edges(root, research):
    """仅解释拓扑声明的语法，不执行/import 业务代码；陌生分支应显式失败。"""
    fn = next(n for n in tree(root, "src/agentic_rag/graph/build.py").body
              if isinstance(n, ast.FunctionDef) and n.name == "build_graph")
    rows = []

    def literal(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name) and n.id in {"START", "END"}:
            return n.id
        if isinstance(n, ast.IfExp) and isinstance(n.test, ast.Name) and n.test.id == "research":
            return literal(n.body if research else n.orelse)
        raise ValueError("无法静态识别图连接，请更新文档提取器：" + code(n))

    def visit(statements):
        for n in statements:
            if isinstance(n, ast.If):
                if not isinstance(n.test, ast.Name) or n.test.id != "research":
                    raise ValueError("图中出现新的条件，请复核文档提取器")
                visit(n.body if research else n.orelse)
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                call = n.value
                if not isinstance(call.func, ast.Attribute):
                    continue
                if call.func.attr == "add_edge":
                    rows.append((literal(call.args[0]), "直接", literal(call.args[1])))
                elif call.func.attr == "add_conditional_edges":
                    mapping = call.args[2]
                    if not isinstance(mapping, ast.Dict):
                        raise ValueError("条件边不再是静态映射，请更新提取器")
                    for key, value in zip(mapping.keys, mapping.values):
                        rows.append((literal(call.args[0]), literal(key), literal(value)))
    visit(fn.body)
    return rows


def render_reference(root):
    sections = ["# 代码事实参考（自动生成）", "",
        "> 由 `python scripts/docs_sync.py --refresh` 从当前源码 AST 与 `.env.example` 生成；不要手改。",
        "> 不导入业务模块，不读取个人 `.env`。表达式是代码默认规则，不是本机有效配置。",
        "", "解释与操作见[文档首页](index.md)、[运行指南](operations.md)及[维护约定](documentation-maintenance.md)。",
        "", "## 配置声明", "",
        "`EVAL_MODEL` 未设置时跟随 `LLM_MODEL`；`DOCUMENTS_DIR`、`KB_DESCRIPTION` 是旧名回退。",
        "路径配置相对项目根目录解析；直接新闻配置另列。", "",
        table(["环境变量（含回退）", "Settings 字段", "源码默认表达式"], config_rows(root))]
    for path in ("src/agentic_rag/news_api.py", "src/agentic_rag/news_index.py"):
        parsed = tree(root, path)
        values = [(n.targets[0].id, "`" + code(n.value) + "`") for n in parsed.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.startswith("DEFAULT_")]
        sections += ["", f"### {path} 的新闻配置", "", table(["常量", "值"], values)]
        sections += ["", "读取的环境变量：" + "、".join("`" + s + "`" for s in sorted({
            n.args[0].value for n in ast.walk(parsed) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "getenv"
            and n.args and isinstance(n.args[0], ast.Constant)}))]
    examples = [line.split("=", 1) for line in read(root, ".env.example").splitlines()
                if line and not line.startswith("#") and "=" in line]
    sections += ["", "### 配置模板示例（不含个人配置）", "",
                 "模板显式填值会覆盖动态回退。`NEWS_API_KEY` 仅为注释占位，需本机提供。", "",
                 table(["变量", "示例值"], ((key, "`" + val + "`") for key, val in examples))]
    sections += ["", "## 命令入口与参数", "", "入口由 pyproject.toml 声明：", ""]
    project = read(root, "pyproject.toml")
    scripts = re.search(r"(?ms)^\[project.scripts\]\s*\n(.*?)(?=^\[|\Z)", project)
    if not scripts:
        raise ValueError("找不到项目命令入口")
    sections += ["```toml", scripts.group(1).strip(), "```"]
    for path in ("src/agentic_rag/cli.py", "src/agentic_rag/ingestion.py", "src/agentic_rag/news_index.py"):
        args = []
        for n in sorted(ast.walk(tree(root, path)), key=lambda n: getattr(n, "lineno", 0)):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"add_argument", "add_parser"}:
                args.append((code(n.func.value), " / ".join(code(a) for a in n.args),
                             "; ".join(k.arg + "=" + code(k.value) for k in n.keywords)))
        sections += ["", f"### `{path}`", "", table(["解析器/子命令", "参数", "声明"], args)]
    sections += ["", "## 工具公开输入（源码声明，不是运行时 JSON Schema）", "",
        "完整的运行时 Schema 请运行 `agentic-rag --list-tools`；程序注入的 RunnableConfig 不在公开参数中。",
        "声明中的校验器与共享类型同样属于契约，不能仅靠字段表判断所有规则。"]
    for path in sorted((root / "src/agentic_rag/tools").glob("*.py")):
        parsed = tree(root, path.relative_to(root))
        for n in parsed.body:
            if isinstance(n, ast.FunctionDef):
                decorators = [d for d in n.decorator_list if isinstance(d, ast.Call)
                              and isinstance(d.func, ast.Name) and d.func.id == "tool"]
                for d in decorators:
                    schema = next(k.value.id for k in d.keywords if k.arg == "args_schema")
                    cls = next(c for c in parsed.body if isinstance(c, ast.ClassDef) and c.name == schema)
                    rows = [(f.target.id, "`" + code(f.annotation) + "`", "`" + code(f.value) + "`")
                            for f in cls.body if isinstance(f, ast.AnnAssign)]
                    sections += ["", f"### `{d.args[0].value}`", "", ast.get_docstring(n) or "", "",
                        f"实现：[tools/{path.name}](../src/agentic_rag/tools/{path.name})；装饰器：`{code(d)}`", "",
                        table(["字段", "类型", "默认/约束声明"], rows)]
    sections += ["", "## 版本与切块常量", ""]
    constants = []
    for path in ("src/agentic_rag/research/service.py", "src/agentic_rag/graph/nodes.py", "src/agentic_rag/ingestion.py"):
        for n in tree(root, path).body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if name.endswith("VERSION") or name in {"CHUNK_SIZE", "CHUNK_OVERLAP"}:
                    constants.append((path, name, "`" + code(n.value) + "`"))
    sections += [table(["来源", "常量", "值"], constants)]
    for mode in (True, False):
        sections += ["", "## " + ("研究模式" if mode else "基础模式") + "精确边表", "",
                     "条件是路由函数返回值；具体预算和终止规则仍需读 nodes/service。", "",
                     table(["从", "条件", "到"], graph_edges(root, mode))]
    return "\n".join(sections) + "\n"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.links, self.refs, self.ids = [], [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag in {"a", "img"}:
            self.links.append(attrs.get("href", attrs.get("src", "")))
        if "data-source-path" in attrs:
            self.refs.append(attrs)


def source_line(root, attrs):
    path = attrs["data-source-path"]
    symbol = attrs.get("data-source-symbol")
    if symbol:
        matches = [n.lineno for n in ast.walk(tree(root, path))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol]
        if len(matches) != 1:
            raise ValueError(f"代码符号丢失或不唯一：{path}:{symbol}")
        return matches[0]
    line = int(attrs["data-source-line"])
    if not 1 <= line <= len(read(root, path).splitlines()):
        raise ValueError(f"代码行号越界：{path}:{line}")
    return line


def refresh_locations(root, text):
    def replace(match):
        fragment = match.group(0)
        attrs = Page(fragment).refs[0]
        line = source_line(root, attrs)
        fragment = re.sub(r'data-source-line="\d+"', f'data-source-line="{line}"', fragment)
        return re.sub(r"第 \d+ 行", f"第 {line} 行", fragment)
    return re.sub(r'<a\b[^>]*data-source-path="[^"]+"[^>]*>.*?</a><span class="locator">.*?</span>',
                  replace, text)


def private_document(path):
    """与 .gitignore 的面试资料规则对应；父目录命中也不参与公开文档维护。"""
    return any("面试" in part or "interview" in part.casefold() for part in Path(path).parts)


def document_paths(root):
    return sorted({root / "README.md", *(p for p in (root / "docs").rglob("*")
                  if p.suffix.lower() in {".md", ".html", ".svg"}
                  and not private_document(p.relative_to(root)))})


def inventories(root):
    sources = {root / ".env.example", root / ".gitignore", root / "pyproject.toml", root / "evaluation/dataset.json"}
    for folder in ("src", "tests", "scripts", "evaluation"):
        sources.update((root / folder).rglob("*.py"))
    for pattern in ("*.yml", "*.yaml"):
        sources.update((root / ".github").rglob(pattern))
    def hashes(paths):
        return {p.relative_to(root).as_posix(): hashlib.sha256(read(root, p).encode()).hexdigest()
                for p in sorted(paths) if p.is_file()}
    return {"sources": hashes(sources), "documents": hashes(document_paths(root))}


def validate_links(root):
    errors = []
    pages = {}
    def page_for(path):
        key = path.resolve()
        if key not in pages:
            pages[key] = Page(read(root, path))
        return pages[key]

    for path in document_paths(root):
        text = read(root, path)
        if path.suffix == ".svg":
            try:
                svg = ET.fromstring(text)
                assert svg.tag == "{http://www.w3.org/2000/svg}svg"
                width, height = float(svg.attrib["width"]), float(svg.attrib["height"])
                assert list(map(float, svg.attrib["viewBox"].split())) == [0, 0, width, height]
                for item in svg.iter():
                    if item.tag.endswith("}text"):
                        assert float(item.attrib["font-size"]) >= width / 75
                        assert 0 <= float(item.attrib["x"]) <= width
                        assert 0 <= float(item.attrib["y"]) <= height
            except (ET.ParseError, KeyError, ValueError, AssertionError) as exc:
                errors.append(f"SVG 结构/基本尺寸不合法：{path.name} {exc}")
            continue
        if path.suffix == ".html":
            page = page_for(path)
            links = page.links
            if len(page.ids) != len(set(page.ids)):
                errors.append(f"重复 HTML 锚点：{path.name}")
            for ref in page.refs:
                try:
                    if str(source_line(root, ref)) != ref["data-source-line"]:
                        errors.append(f"代码定位过时：{ref['data-source-path']}:{ref.get('data-source-symbol')}")
                except (OSError, ValueError, SyntaxError) as exc:
                    errors.append(str(exc))
        else:
            # 忽略代码块中的示范链接；本仓库使用行内 Markdown 链接。
            plain = re.sub(r"(?ms)^```.*?^```\s*$", "", text)
            links = re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", plain)
        for link in links:
            link = html.unescape(link.strip().strip("<>"))
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            dest = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if dest.is_relative_to(root.resolve()) and private_document(dest.relative_to(root.resolve())):
                errors.append(f"公开文档不能依赖本地面试资料：{path.name} → {link}")
            elif not dest.is_file() or not dest.is_relative_to(root.resolve()):
                errors.append(f"本地链接不存在或越界：{path.name} → {link}")
            elif parsed.fragment and dest.suffix == ".html":
                if unquote(parsed.fragment) not in page_for(dest).ids:
                    errors.append(f"HTML 锚点不存在：{path.name} → {link}")
    return errors


def check(root=ROOT, *, review=True):
    errors = validate_links(root)
    if not (root / REFERENCE).exists() or read(root, REFERENCE) != render_reference(root):
        errors.append("代码事实参考过时：先运行 python scripts/docs_sync.py --refresh")
    if review:
        if not (root / LOCK).exists():
            errors.append("缺少审阅记录：人工核对后执行 --acknowledge-review")
        else:
            saved = json.loads(read(root, LOCK))
            for group, actual in inventories(root).items():
                old = saved.get(group, {})
                for path in sorted(set(old) | set(actual)):
                    if old.get(path) != actual.get(path):
                        errors.append(f"未审阅变动 [{group}]：{path}")
    return errors


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="只读检查，失败退出码 1")
    mode.add_argument("--refresh", action="store_true", help="只刷新机械事实和代码定位，不确认人工审阅")
    mode.add_argument("--acknowledge-review", action="store_true", help="人工审阅后记录源码与文档指纹")
    args = parser.parse_args(argv)
    if args.refresh:
        (ROOT / REFERENCE).write_text(render_reference(ROOT), encoding="utf-8", newline="\n")
        for path in document_paths(ROOT):
            if path.suffix.lower() == ".html":
                path.write_text(refresh_locations(ROOT, read(ROOT, path)), encoding="utf-8", newline="\n")
        print("已刷新代码事实与定位；中文解释和流程图仍需人工核对，审阅记录未更新。")
        return 0
    errors = check(ROOT, review=not args.acknowledge_review)
    if errors:
        print("\n".join(errors))
        return 1
    if args.acknowledge_review:
        state = {"format": 2, **inventories(ROOT)}
        (ROOT / LOCK).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("已记录人工审阅确认。此记录不是测试或语义正确性证明。")
    else:
        print("文档检查通过：代码事实、链接、定位、SVG 结构及审阅指纹一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
