"""抽取相关原文段落并去重，保留可核对证据，不生成新的事实。"""

import hashlib
import re

from langchain_core.documents import Document


FORECAST_WORDS = re.compile(r"预计|预测|预判|有望|或将|情景|假设|若.{0,40}(?:将|会|则)|展望")


def event_context(text, start):
    """保留原文最近的时间/情景标题；只摘录，不把新闻发布日期推断成事件时间。"""
    prefix = text[:start]
    headings = [line.strip() for line in prefix.splitlines() if 0 < len(line.strip()) <= 100
                and not line.startswith(("标题：", "发布时间：", "栏目："))
                and re.search(r"阶段|情景|展望|(?:20\d{2}年|\d{1,2}月).{0,15}(?:回顾|行情|走势)", line)]
    return headings[-1] if headings else "未取得独立时间背景；以引文中的事件时间为准，不以发布日期代替"


def atomic_excerpt(document, query, limit):
    """阅读成果按完整事实块裁剪，类型、时间背景与原文不能拆散。"""
    original = document.metadata.get("source_text")
    if original:
        # 阅读笔记是可复用的索引，不是原文的唯一替代；按本题从原文补回遗漏的限定。
        return whole_sentence_excerpt(original, query, limit)
    claims = document.metadata.get("reading_claims")
    if not claims:
        return excerpt(document.page_content, query, limit)
    words = set(re.findall(r"[\u4e00-\u9fff]{2}|[a-zA-Z]{3,}", query))
    claims = sorted(claims, key=lambda c: (
        c["kind"] == "reported_fact", sum(word in c["statement"] for word in words)), reverse=True)
    selected = []
    for claim in claims:
        # 引文已包含事实，不再重复一份概括；过长块留在持久化成果，禁止截成半句。
        block = f"类型={claim['kind']}；事件背景={claim.get('event_context', '未明确')}；原文={claim['quote']}"
        if len(block) + sum(len(s)+1 for s in selected) <= limit:
            selected.append(block)
    return "\n".join(selected) or "完整事实块超过上下文预算；不得凭标题或发布时间推断，需回读原文。"


def whole_sentence_excerpt(text, query, limit):
    """不改写、不截断数值/条件；过长的句子宁可明确缺失，也不把半句当完整事实。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    words = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", query):
        words.update(run[i:i+2] for i in range(len(run)-1))
    parts, blocks = [], []
    for match in re.finditer(r"[^。！？；\n]+[。！？；]?", text):
        part = match[0].strip()
        if not part or part.startswith(("标题：", "发布时间：", "栏目：")):
            continue
        background = event_context(text, match.start())
        # 原文回填同样不能丢掉上方的事件月份/情景标题；与句子作为不可拆块装箱。
        block = part if background.startswith("未取得独立时间背景") or background in part else background + "\n" + part
        parts.append(part)
        blocks.append(block)
    ranked = sorted(range(len(parts)), key=lambda i: (
        sum(word in parts[i].lower() for word in words), bool(FORECAST_WORDS.search(parts[i])), -i), reverse=True)
    selected, used = {}, 0
    for index in ranked:
        cost = len(blocks[index]) + (2 if selected else 0)
        if used + cost <= limit:
            selected[index] = blocks[index]
            used += cost
    return "\n…".join(selected[i] for i in sorted(selected)) or "完整事实块超过上下文预算；需要回读原文。"


def audit_forecast_status(answer, documents):
    """已复现的限定丢失护栏：不能把原文预计的数值写成已实现；非完整语义证明。"""
    from .quantities import quantities
    errors = []
    clean = re.sub(r"\*\*|__|`", "", answer)
    sentences = re.split(r"(?<=[。！？])|\n+", clean)
    qualified = {value for part in sentences if FORECAST_WORDS.search(part) for value in quantities(part)}
    for sentence in sentences:
        if FORECAST_WORDS.search(sentence) or re.search(r"并非|不是|不代表|未确认|尚未|不能", sentence):
            continue
        values = {(number, unit) for number, unit in quantities(sentence) if unit and unit not in {"年", "月", "日"}}
        if not values:
            continue
        ids = {int(i) for i in re.findall(r"(?<![A-Za-z0-9])E(\d+)(?!\d)", sentence)}
        linked = [doc for i, doc in enumerate(documents, 1) if i in ids] if ids else documents
        for doc in linked:
            original = doc.metadata.get("source_text", doc.page_content)
            for part in re.split(r"[。！？；\n]", original):
                if part.startswith(("标题：", "发布时间：", "栏目：")):
                    continue
                # 只检查每个预测词后紧接的首个数值，不把同句的历史对比基数算作预测。
                for marker in re.finditer(r"预计|预测|有望|计划", part):
                    predicted = quantities(part[marker.end():marker.end()+40])
                    if predicted and predicted[0] in values:
                        # 已明确限定为预计值的数字仍可用于算术核对；不能误拦正确的5倍/400%计算。
                        arithmetic = re.search(r"算术|计算|数学|比值|倍", sentence)
                        assertion = re.search(r"已经|已(?:突破|达到|实现|确认)|实际|统计结果", sentence)
                        if predicted[0] in qualified and arithmetic and not assertion:
                            continue
                        number, unit = predicted[0]
                        errors.append(f"原文中的 {number}{unit} 带有预计/计划限定，答案不能当作已确认结果；请保留来源的限定与归属。")
    return list(dict.fromkeys(errors))


def audit_growth_claims(answer):
    """仅核对同单位、明确起止数字与翻倍措辞的显著矛盾，不替代统计口径审查。"""
    from decimal import Decimal
    clean = re.sub(r"\*\*|__|`", "", answer)
    if re.search(r"不一致|不准确|并非翻倍|不等于翻倍|不能称为翻倍|矛盾|口径.{0,10}核实", clean):
        return []  # 已明确指出原文问题，不阻止先引用再纠正的回答。
    number = r"(\d+(?:\.\d+)?)\s*((?:万亿|千万|百万|亿|万|千)(?:人|元|吨|户|部)?|人|元|吨|户|部)"
    patterns = [rf"(?:从|由)\s*{number}[^。；\n]{{0,18}}?(?:至|到)\s*{number}[^。；\n]{{0,18}}?翻倍",
                rf"{number}\s*[，,]?\s*(?:较|比)[^。；，\n]{{0,24}}?{number}[^。；\n]{{0,16}}?翻倍"]
    for pattern in patterns:
        for match in re.finditer(pattern, clean):
            first, unit1, second, unit2 = match.groups()
            left, right = Decimal(first), Decimal(second)
            if unit1 != unit2 or min(left, right) <= 0:
                continue
            ratio = max(left, right) / min(left, right)
            # 接近两倍的舍入不硬拦；只拦明显不符，避免冒充完备算术/语义验证。
            if ratio < Decimal("1.9") or ratio > Decimal("2.1"):
                return [f"{first}{unit1} 与 {second}{unit2} 的比值约为{ratio:.2f}，不能直接称为翻倍；指出来源措辞与数字矛盾或删除该措辞。"]
    return []


def audit_numeric_claims(answer, documents, context, *, forecast=False):
    """阻止已知的无依据价格数字和月份错配；不是对所有自然语言事实的完备证明。"""
    errors = []
    answer = answer.replace("**", "").replace("__", "").replace("`", "")
    # 千分位与区间两端都检查，不能只检查“4000–4800美元”的4800。
    answer = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", answer)
    context = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", context)
    money = r"(?<![\d.])(\d{3,}(?:\.\d+)?)(?:\s*[-–—~至到]\s*(\d{3,}(?:\.\d+)?))?\s*(?:美元|元[/／每]?|美金)"
    attribution = r"(?:原文|机构)(?:预测|预判)|报道称|曾预计|据.{0,20}预测|(?:部分|一些|另一些)?观点(?:认为|预计|预测)|(?:投行|来源|报告).{0,12}(?:预测|预计|预判)"
    def values_in(text):
        return [value for match in re.findall(money, text) for value in match if value]
    for value in values_in(answer):
        if not re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])", context):
            errors.append(f"价格数字 {value} 未出现在实际证据上下文中，删除或补充可核对计算")
    scenario_heading = False
    for line in answer.splitlines():
        if re.match(r"^\s*#{1,6}\s", line):
            scenario_heading = bool(re.search(r"情景|预测|展望", line))
        if forecast and values_in(line) and (scenario_heading or re.search(r"目标(?:价|区间)|预测价格", line)):
            if not re.search(attribution + r"|历史|截至|现价|当前", line):
                errors.append("情景中出现未经计算的目标价格/区间；删除数值，只给基准、上行、下行的方向与触发条件。机构区间只能明确归属于来源，不能当作本研究目标。")
    for paragraph in re.split(r"\n\s*\n|[。！？]", answer):
        values = values_in(paragraph)
        if not values:
            continue
        if forecast and re.search(r"基准情景|上行情景|下行情景|目标价|目标区间|预计|预测|将.{0,15}(?:达到|维持|震荡|回落)|若|如果", paragraph):
            if not re.search(attribution, paragraph):
                errors.append("定量价格预测缺少可复核估值模型；改为方向与条件分析，不把历史或机构区间移作目标年份预测")
        months = set(re.findall(r"(?<!\d)(\d{1,2})月", paragraph))
        cited = {int(n) for n in re.findall(r"\bE(\d+)\b", paragraph)}
        linked = [d for i, d in enumerate(documents, 1) if i in cited] if cited else documents
        for match in re.finditer(money, paragraph):
            preceding_months = re.findall(r"(?<!\d)(\d{1,2})月", paragraph[:match.start()])
            local_months = {preceding_months[-1]} if preceding_months else months if len(months) == 1 else set()
            for value in filter(None, match.groups()):
                anchors = []
                for doc in linked:
                    for claim in doc.metadata.get("reading_claims", []):
                        if value in claim.get("quote", ""):
                            anchors.extend(re.findall(r"(?<!\d)(\d{1,2})月", claim.get("event_context", "") + claim.get("quote", "")))
                if local_months and anchors and not local_months.issubset(set(anchors)):
                    errors.append(f"数字 {value} 的事件月份与原文背景不一致；不能用文章发布时间替代事件时间")
    return list(dict.fromkeys(errors))


def audit_policy_claims(answer, context):
    """已复现的政策方向偷换：少加息不能直接表述为降息预期已升温。"""
    if "降息" in context or "加息" not in context:
        return []
    for sentence in re.split(r"[。！？\n]", answer):
        if re.search(r"降息预期.{0,8}(?:升温|增强|上升)|降息.{0,6}(?:已经|已发生)", sentence):
            if not re.search(r"若|如果|假设|可能|不代表|不等于|不能|未必", sentence):
                return ["证据只显示加息预期减弱，不能偷换成降息预期已升温；当前事实写为少加息，未来降息只能明确列作假设及触发条件。"]
    return []


def audit_indicator_periods(answer, context):
    """窄口径反例：季度GDP不能被并列概括成月度数据；不推断所有指标的时间。"""
    quarter = re.search(r"(第?[一二三四1-4]季度)(?:实际|名义)?\s*GDP", context, re.I)
    if not quarter:
        return []
    month = r"(?<!\d)((?:1[0-2]|[1-9]))月(?:份)?(?:的)?"
    direct = month + r"(?:实际|名义)?\s*GDP\s*(?:数据|增速|年化|终值|为|同比|环比|增长|下降)"
    grouped = month + r"(?:经济|宏观|统计|月度)?数据\s*[（(][^）)\n。]{0,60}\bGDP\b[^）)\n。]*[）)]"
    clean = re.sub(r"\*\*|__|`", "", answer)
    for sentence in re.split(r"[。！？\n]", clean):
        # 允许明确引用错误说法并纠正，不拦“8月公布的二季度GDP”等发布日期表述。
        if re.search(r"不能|不应|并非|错误|误称|不是|不等于", sentence):
            continue
        for pattern in (direct, grouped):
            for match in re.finditer(pattern, sentence, re.I):
                claimed_month = match[1]
                # 原文自身确有同月GDP数据时留给语义核验，不强行用另一季度覆盖。
                if any(m[1] == claimed_month for p in (direct, grouped) for m in re.finditer(p, context, re.I)):
                    continue
                return [f"GDP统计期在证据中为{quarter[1]}，不能并入{claimed_month}月数据；分别注明各指标统计期，区分统计期和发布日期。"]
    return []


def normalize_answer_references(answer):
    """只清理纯编号引用中的内部C项，不改正文、不删除E来源、不碰普通Markdown链接。"""
    def replace(match):
        codes = match[1]
        if not re.fullmatch(r"[EC]\d+(?:\s*[,，、]\s*[EC]\d+)*", codes):
            return match[0]
        evidence = list(dict.fromkeys(re.findall(r"\bE\d+\b", codes)))
        return "[" + ", ".join(evidence) + "]" if evidence else ""
    return re.sub(r"\[([^\]\n]+)\](?!\()", replace, answer)


def numbered_sentences(text: str) -> tuple[dict[str, tuple[int, int]], str]:
    """阅读和回读共享的句子编号；位置始终对应未改写的输入原文。"""
    sentences = {}
    for match in re.finditer(r"[^。！？；\n]+[。！？；]?", text):
        for start in range(match.start(), match.end(), 400):
            end = min(start + 400, match.end())
            sentences[f"S{len(sentences)+1}"] = (start, end)
    rendered = "\n".join(f"[{sid}] {text[start:end]}" for sid, (start, end) in sentences.items())
    return sentences, rendered


def document_key(document: Document) -> str:
    return hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()


def excerpt(text: str, query: str, limit: int) -> str:
    """按问题选取原文片段；正文尾部的相关信息也有机会保留。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    words = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", query):
        words.update(run[index:index + 2] for index in range(len(run) - 1))
    parts = []
    for paragraph in re.split(r"(?<=[。！？；])|\n+", text):
        paragraph = paragraph.strip()
        parts.extend(paragraph[index:index + 260] for index in range(0, len(paragraph), 260))
    ranked = sorted(range(len(parts)), key=lambda i: (
        sum(word in parts[i].lower() for word in words)
        + 0.2 * bool(re.search(r"\d", parts[i])) + (0.1 if i == 0 else 0)
    ), reverse=True)
    selected: dict[int, str] = {}
    remaining = limit
    for index in ranked:
        if remaining < 40:
            break
        selected[index] = parts[index][:max(0, remaining - 2)]
        remaining -= len(selected[index]) + 2
    return "\n…".join(selected[index] for index in sorted(selected))[:limit]


def deduplicate(documents: list[Document]) -> list[Document]:
    seen_content, seen_articles, seen_titles = set(), set(), set()
    unique = []
    for document in documents:
        metadata = document.metadata
        content = re.sub(r"\s+", "", document.page_content)
        article_id = metadata.get("article_id")
        title = re.split(r"[_|]| - ", str(metadata.get("title") or ""))[0]
        title = re.sub(r"\s+|[….]", "", title)
        web_title = title if metadata.get("source_type") == "web_search" and len(title) >= 18 else ""
        if content in seen_content or (article_id and article_id in seen_articles) or (web_title and web_title in seen_titles):
            continue
        seen_content.add(content)
        if article_id:
            seen_articles.add(article_id)
        if web_title:
            seen_titles.add(web_title)
        unique.append(document)
    return unique


def format_evidence(documents: list[Document], query: str, budget: int) -> str:
    if not documents:
        return "（没有可用证据）"
    headers = []
    for index, document in enumerate(documents, 1):
        metadata = document.metadata
        title = str(metadata.get("title") or metadata.get("source") or "未知来源")[:100]
        kind = metadata.get("content_kind") or metadata.get("source_type", "unknown")
        date = metadata.get("published_at") or (f"搜索日期线索（未核实）：{metadata['date_hint']}" if metadata.get("date_hint") else "日期未标注")
        screening = " | 相关性待确认" if metadata.get("relevance_decision") == "uncertain" else ""
        headers.append(f"[E{index}] {title} | 发布时间（非事件时间）={date} | {kind}{screening}\n")
    available = max(0, budget - sum(map(len, headers)) - 2 * (len(documents) - 1))
    per_document = available // len(documents)
    parts = [atomic_excerpt(document, query, per_document) for document in documents]
    missing = [i for i, part in enumerate(parts) if "完整事实块超过上下文预算" in part]
    # 短文章剩余的空间给较长事实块，不能机械均分后截掉时间/条件。
    for i in missing:
        parts[i] = ""
    remaining = available - sum(map(len, parts))
    for i in missing:
        part = atomic_excerpt(documents[i], query, remaining)
        if "完整事实块超过上下文预算" in part:
            return "完整事实块超过上下文预算；本批证据无法完整呈现，需减少批量或提高上下文额度。"
        parts[i] = part
        remaining -= len(part)
    result = "\n\n".join(header + part for header, part in zip(headers, parts))
    if len(result) > budget:
        return "完整事实块超过上下文预算；连证据标题都无法完整呈现，需减少批量或提高上下文额度。"
    return result
