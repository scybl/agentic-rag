"""把证据风险变成必须逐项回应的清单，而非仅凭两个布尔值放行。"""

import re


def build_concerns(concerns, documents, *, task_type, question):
    valid = {f"E{i}" for i in range(1, len(documents) + 1)}
    result = []
    # 这是核对义务，不是断言材料必然存在问题。语义判断仍由模型完成。
    if task_type == "forecast" and documents:
        checks = [
            ("time", "注明事实/预测各自所属时间；历史数据和旧机构预测只能作为历史基准，不能冒充当前或目标年份事实。"),
            ("scope", "核对地区、单位、统计对象和年份；冲突数字须解释口径或明确不可比较，不能默默择一。"),
            ("counterevidence", "明确列出支持与反对方向判断的证据及其影响；若未找到反向证据须说明检索局限，不等于反证不存在。"),
            ("baseline", "说明价格基准与缺口；缺现价/合约时不得量化涨跌空间或声称突破前高。" if "期货" in question else
             "区分历史价格基准、驱动因素和条件推断；缺少可核对基准时不编造目标价、概率和涨幅。"),
            ("mechanism", "逐条核对事实到预测对象的传导机制；假设不能补救不成立的因果关系。个体保险/套保转移风险不证明市场价格波动被平抑；现货品质分层不直接等于标准化期货合约价格分层。没有传导依据的结论须删除或明确无法判断。"),
        ]
        result.extend({"category": category, "evidence_ids": [], "detail": detail, "origin": "必核项目"}
                      for category, detail in checks)
    for concern in concerns:
        item = dict(concern)
        ids = item.get("evidence_ids", [])
        # 不让模型编造的E编号进入下游。
        item["evidence_ids"] = [eid for eid in ids if eid in valid]
        if ids and not item["evidence_ids"]:
            continue
        item["origin"] = "证据检查"
        result.append(item)
    for index, item in enumerate(result, 1):
        item["id"] = f"C{index}"
    return result


def enforce_concern_checks(review, concerns):
    """缺失、重复或未解决的风险均不能被 accept 标记覆盖。"""
    checks = review.get("concern_checks", [])
    unresolved = []
    for concern in concerns:
        matches = [c for c in checks if c.get("concern_id") == concern["id"]]
        explanation = matches[0].get("explanation", "").strip() if len(matches) == 1 else ""
        contradictory = bool(re.search(r"^(?:错误|未处理|未解决|未回应|不符合)[。:：！!]|(?:答案|回答)(?:未|没有)(?:回应|处理|说明|解决)", explanation))
        if len(matches) != 1 or not matches[0].get("addressed") or not explanation or contradictory:
            unresolved.append(f"{concern['id']}：{concern['detail']}")
    if unresolved:
        review.update(decision="revise", grounded=False)
        review.setdefault("issues", []).append("风险未逐项核验或未解决：" + "；".join(unresolved))
    return review
