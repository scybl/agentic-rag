"""保守核对摘录数字和紧邻单位；不是换算器或完整的事实验证器。"""

import re
from decimal import Decimal


_NUMBER = r"(?<![\d.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?"
_UNIT = r"(?:万亿|千万|百万|亿|万|千)?(?:美元|人民币|港元|韩元|欧元|元|吨|公斤|千克|克|人|家|部|篇|次|天|年|月|日|个百分点|个基点|百分点|基点|个|%|％)(?:\s*[/／]\s*(?:盎司|公斤|千克|克|吨|桶|人))?"
_QUANTITY = re.compile(rf"({_NUMBER})\s*({_UNIT}|万亿|千万|百万|亿|万|千)?")


def quantities(text):
    result = []
    for match in _QUANTITY.finditer(text.replace("**", "")):
        value = Decimal(match[1].replace(",", "").replace("−", "-"))
        unit = re.sub(r"\s+", "", match[2] or "").replace("％", "%").replace("／", "/")
        unit = unit.replace("千克", "公斤")
        unit = unit.replace("个百分点", "百分点").replace("个基点", "基点")
        result.append((value, unit))
    return result


def numbers_and_units_supported(statement, quote):
    """数字相同但单位/量级不同时拒绝；不把文字推理伪装成精确校验。"""
    source = quantities(quote)
    for value, unit in quantities(statement):
        if not any(value == n and (not unit or unit == u) for n, u in source):
            return False
    return True
