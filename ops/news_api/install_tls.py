"""经授权后在服务器本地做精确替换；默认只校验/展示摘要，--apply 才备份修改。"""

import argparse
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MARKER = "# Bounded TLS acceptance: v1"


def patched(source, helper):
    if MARKER in source:
        if "public = make_server(3001, tls)" not in source:
            raise ValueError("发现不完整的旧 TLS 修复，停止自动覆盖")
        return source
    changes = [
        ("def main():", MARKER + "\n" + helper + "\n\ndef main():"),
        ("    def make_server(port):\n        server = ThreadingHTTPServer(('0.0.0.0', port), Handler)",
         "    def make_server(port, ssl_context=None):\n"
         "        server = (ThreadingHTTPServer(('0.0.0.0', port), Handler) if ssl_context is None else\n"
         "                  BoundedTLSHTTPServer(('0.0.0.0', port), Handler, ssl_context=ssl_context))"),
        ("    public = make_server(3001)\n", ""),
        ("    public.socket = tls.wrap_socket(public.socket, server_side=True)", "    public = make_server(3001, tls)"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("服务器代码与预期形状不同，停止；未修改源文件")
        source = source.replace(old, new, 1)
    compile(source, "news_api.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--helper", type=Path, default=Path(__file__).with_name("tls_server.py"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    raw = source.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    if before != args.expected_sha256:
        raise ValueError("服务代码指纹已变化，拒绝覆盖；请重新检查")
    result = patched(raw.decode("utf-8"), args.helper.read_text(encoding="utf-8"))
    after = hashlib.sha256(result.encode()).hexdigest()
    print(f"before={before}\nafter={after}\nchanged={before != after}")
    if not args.apply or before == after:
        return
    backup = source.with_name(source.name + ".bak.tls_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    if backup.exists():
        raise FileExistsError("备份已存在，拒绝覆盖")
    shutil.copy2(source, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=source.parent,
                                         prefix=".tls-stage-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(result)
            stream.flush()
            os.fsync(stream.fileno())
        shutil.copystat(source, temporary)
        if hasattr(os, "chown"):
            original = source.stat()
            os.chown(temporary, original.st_uid, original.st_gid)
        os.replace(temporary, source)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"backup={backup}\n需只重启新闻API容器，并核对容器内指纹；未重启任何服务。")


if __name__ == "__main__":
    main()
