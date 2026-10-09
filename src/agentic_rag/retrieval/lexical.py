"""中文双字词 + 英文词的 SQLite FTS5/BM25 适配器。"""

import re
import sqlite3


def tokens(text):
    # 语料保留词频，不能复用把 terms 去重的查询帮助函数。
    result = []
    for term in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            result.extend(term[i:i + 2] for i in range(max(1, len(term) - 1)))
        else:
            result.append(term)
    return result


class LexicalIndex:
    def __init__(self, documents):
        self.db = sqlite3.connect(":memory:")
        try:
            self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(document_id UNINDEXED,title,summary)")
            self.db.executemany("INSERT INTO docs VALUES(?,?,?)", [
                (d.document_id, " ".join(tokens(d.title)), " ".join(tokens(d.summary))) for d in documents])
        except BaseException:
            self.close()
            raise

    def search(self, query, limit):
        if limit < 1:
            raise ValueError("召回上限必须为正")
        terms = list(dict.fromkeys(tokens(query)))[:64]
        if not terms:
            return []
        # MATCH 表达式仅由分词后的字母/数字/汉字构造，不执行用户 FTS 运算符。
        match = " OR ".join('"' + term + '"' for term in terms)
        return [(row[0], row[1]) for row in self.db.execute(
            "SELECT document_id,-bm25(docs,0,2,1) AS score FROM docs WHERE docs MATCH ? "
            "ORDER BY score DESC,document_id ASC LIMIT ?", (match, limit))]

    def close(self):
        self.db.close()
