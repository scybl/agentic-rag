"""质量评估只能使用生成器实际看见的片段，不能拿未传入的全文替它佐证。"""


def actual_contexts(result):
    if "evidence_context" in result:
        text = result["evidence_context"].strip()
        return [text] if text else []
    # 老记录缺少实际上下文时不回退到 documents，避免夸大证据支撑。
    raise ValueError("该记录没有保存 evidence_context，不能用于答案依据评分")
