"""由编排器强制调用的确定性规则工具。"""

import re
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, ValidationError

from .contracts import ModelRetryPlan, RuleBundle, ToolInput, context_from


class EvidenceDescriptor(ToolInput):
    evidence_id: str = Field(pattern=r"^E\d+$")
    source_type: str = Field(default="unknown", max_length=80)
    content_kind: str = Field(default="", max_length=80)
    title: str = Field(default="", max_length=300)
    excerpt: str = Field(default="", max_length=1200)


class ProbabilityEvidenceInput(ToolInput):
    estimate_kind: Literal["none", "directional", "level", "probability"] = "none"
    question: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceDescriptor] = Field(default_factory=list, max_length=30)


class ProbabilityAnswerInput(ToolInput):
    estimate_kind: Literal["none", "directional", "level", "probability"] = "none"
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=30000)
    evidence_contract_passed: bool = False


class ModelOutputValidationInput(ToolInput):
    output_type: Literal["structured", "text"]
    schema_name: str = Field(min_length=1, max_length=200)
    payload: Any = None
    parser_error: str = Field(default="", max_length=2000)
    raw_excerpt: str = Field(default="", max_length=1000)


class ModelRetryInput(ToolInput):
    stage: str = Field(min_length=1, max_length=200)
    failure_kind: Literal[
        "schema", "empty", "truncated", "timeout", "connection",
        "server_busy", "server_error", "client_error", "unknown",
    ]
    attempt: int = Field(ge=1, le=20)
    max_attempts: int = Field(ge=1, le=20)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    current_num_predict: int = Field(ge=1)
    current_num_ctx: int = Field(ge=1)
    current_timeout_seconds: int = Field(ge=1)
    reasoning: bool = False
    output_cap: int = Field(ge=1)
    context_cap: int = Field(ge=1)
    timeout_cap: int = Field(ge=1)
    expected_output_tokens: int = Field(default=0, ge=0)
    supports_split: bool = False
    violations: list[str] = Field(default_factory=list, max_length=10)


_PROBABILITY_WORDS = re.compile(r"概率|几率|可能性|odds|probabilit|chance", re.I)
_MARKET_METHOD = re.compile(
    r"FedWatch|联邦基金期货|fed\s*funds?\s*futures?|OIS|隔夜指数掉期|"
    r"期权.{0,12}(?:隐含|概率)|隐含概率|概率分布|历史频率|统计模型|预测模型|"
    r"回归|贝叶斯|官方预报|气象模型|抽样|样本频率",
    re.I,
)


def _has_probability_value(text: str) -> bool:
    values = [float(value) for value in re.findall(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", text)]
    return any(0 <= value <= 100 for value in values) or bool(re.search(
        r"百分之[零一二三四五六七八九十百点]+", text
    ))


def is_probability_evidence_question(question: str) -> bool:
    """区分‘现有报道能否支持这个概率’与‘请预测概率’，不把礼貌的能否误判成拒答。"""
    return bool(_PROBABILITY_WORDS.search(question)
        and re.search(r"(?:这些|现有|所给|上述|所提供|这篇|这则|这份|该篇).{0,8}(?:报道|材料|证据|资料)", question)
        and re.search(r"能否|是否(?:足以|能|可以|支持)|足不足以", question)
        and not re.search(r"帮我预测|请预测|请计算|给出你的估计|能否(?:帮我|为我)?(?:预测|计算|估计|估算)", question))


def infer_estimate_kind(question: str, planned: str = "none") -> str:
    """明显的概率措辞由代码兜底，不能被规划模型降级成普通趋势预测。"""
    if is_probability_evidence_question(question):
        return "none"
    if _PROBABILITY_WORDS.search(question):
        return "probability"
    return planned


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def _validation_violations(exc: ValidationError) -> list[str]:
    reasons = []
    for error in exc.errors()[:10]:
        path = ".".join(map(str, error["loc"])) or "<root>"
        reasons.append(f"{path}: {error['msg']} ({error['type']})")
    if len(exc.errors()) > 10:
        reasons.append(f"另有 {len(exc.errors()) - 10} 项结构错误")
    return reasons


@tool("validate_model_output", args_schema=ModelOutputValidationInput,
      response_format="content_and_artifact")
def validate_model_output(output_type: str, schema_name: str, config: RunnableConfig,
                          payload: Any = None, parser_error: str = "",
                          raw_excerpt: str = "") -> tuple[str, RuleBundle]:
    """检查模型输出类型和字段结构，返回可反馈给模型的具体错误原因。"""
    violations = []
    if parser_error:
        violations.append(f"结构化解析失败：{parser_error}")
    elif output_type == "text":
        if not isinstance(payload, str):
            violations.append(f"期望非空文本，实际类型为 {type(payload).__name__}")
        elif not payload.strip():
            violations.append("期望非空文本，模型返回了空内容")
    else:
        schema = context_from(config).output_schema
        if schema is None:
            violations.append("执行器没有注入目标 Pydantic Schema")
        elif not isinstance(payload, dict):
            violations.append(f"期望 {schema_name} JSON 对象，实际类型为 {type(payload).__name__}")
        else:
            try:
                schema.model_validate(payload)
            except ValidationError as exc:
                violations.extend(_validation_violations(exc))
    checks = [_check("模型输出结构", not violations,
                     f"符合 {schema_name}" if not violations else "；".join(violations))]
    return RuleBundle(rule="model_output_schema", passed=not violations,
                      checks=checks, violations=violations).as_response()


def _round_up(value: int, step: int = 1024) -> int:
    return ((max(0, value) + step - 1) // step) * step


@tool("plan_model_retry", args_schema=ModelRetryInput,
      response_format="content_and_artifact")
def plan_model_retry(stage: str, failure_kind: str, attempt: int, max_attempts: int,
                     current_num_predict: int, current_num_ctx: int,
                     current_timeout_seconds: int, reasoning: bool,
                     output_cap: int, context_cap: int, timeout_cap: int,
                     config: RunnableConfig, input_tokens: int = 0,
                     output_tokens: int = 0, expected_output_tokens: int = 0,
                     supports_split: bool = False,
                     violations: list[str] | None = None) -> tuple[str, ModelRetryPlan]:
    """根据可观测失败和硬预算选择下一次模型参数，或明确停止/切分。"""
    violations = violations or []
    remaining = attempt < max_attempts
    next_predict = current_num_predict
    next_ctx = current_num_ctx
    next_timeout = current_timeout_seconds
    next_reasoning = reasoning
    action, retry, reason, feedback = "stop", False, "没有安全的自动调整路径", ""

    if not remaining:
        reason = f"已用完 {max_attempts} 次模型尝试，停止自动调整"
    elif failure_kind in {"schema", "empty"}:
        action, retry = "repair_output", True
        detail = "；".join(violations[:6]) or ("模型返回空内容" if failure_kind == "empty" else "输出结构不匹配")
        if attempt >= 2 and reasoning:
            next_reasoning = False
            action = "repair_output_without_reasoning"
        reason = f"资源没有耗尽，按具体输出错误重做：{detail}"
        feedback = (f"上一次 {stage} 输出未通过校验：{detail}。"
                    "请重新完成原任务，只返回目标类型；不要输出对象外说明。")
    elif failure_kind == "truncated":
        if supports_split:
            action, retry = "split_input", False
            reason = "该阅读节点支持缩小输入；切分比继续扩大长输出更可靠"
        else:
            reserve = max(1024, input_tokens // 10)
            observed = output_tokens or current_num_predict
            desired = max(current_num_predict + 1024, _round_up(int(observed * 1.25)),
                          expected_output_tokens)
            desired = min(desired, output_cap)
            required_ctx = input_tokens + desired + reserve
            if required_ctx > next_ctx and next_ctx < context_cap:
                next_ctx = min(context_cap, _round_up(required_ctx))
            available = max(0, next_ctx - input_tokens - reserve)
            candidate = min(desired, available, output_cap)
            if candidate > current_num_predict:
                next_predict = candidate
                action, retry = "increase_output_budget", True
                reason = (f"输出达到 {observed} token 上限；在上下文和节点硬上限内，"
                          f"临时把输出额度调到 {next_predict}，上下文调到 {next_ctx}")
            elif reasoning:
                next_reasoning = False
                action, retry = "retry_without_reasoning", True
                reason = "已无安全输出扩容空间，关闭思考以把额度留给最终结果"
            else:
                reason = "输出已截断，但输出额度、上下文或节点硬上限没有剩余空间"
            if retry:
                feedback = (f"上一次 {stage} 因达到 {observed} token 上限而截断，不是字段错误。"
                            "请减少过程性解释，直接返回完整最终结果，不要重复材料。")
    elif failure_kind == "timeout":
        if reasoning:
            next_reasoning = False
            action, retry = "retry_without_reasoning", True
            reason = "请求达到时间上限；先关闭思考，避免单纯延长一次失控生成"
        elif current_timeout_seconds < timeout_cap:
            next_timeout = min(timeout_cap, max(current_timeout_seconds + 30,
                                                 int(current_timeout_seconds * 1.5)))
            action, retry = "increase_timeout", True
            reason = f"无思考请求仍超时；临时把单次超时从 {current_timeout_seconds}s 调到 {next_timeout}s"
        else:
            reason = "请求超时且已达到超时硬上限"
        if retry:
            feedback = (f"上一次 {stage} 达到请求时间上限。请直接完成最终结果，"
                        "减少重复分析和过程性解释。")
    elif failure_kind in {"connection", "server_busy", "server_error"}:
        action, retry = "retry_service", True
        reason = "连接或服务端属于可恢复故障；保持任务目标和资源上限重试"
        feedback = "上一次请求未取得模型结果，请重新完成原任务。"
    elif failure_kind == "client_error":
        reason = "客户端请求错误通常不能通过原样重试恢复"
    else:
        reason = "未知错误不自动扩大资源，避免掩盖程序或配置问题"

    return ModelRetryPlan(
        action=action, retry=retry, reason=reason,
        next_num_predict=next_predict, next_num_ctx=next_ctx,
        next_timeout_seconds=next_timeout, next_reasoning=next_reasoning,
        prompt_feedback=feedback,
    ).as_response()


def _probability_queries(question: str) -> list[str]:
    years = " ".join(dict.fromkeys(re.findall(r"20\d{2}", question)))
    if re.search(r"美联储|联储|降息|加息|联邦基金", question):
        return [
            f"CME FedWatch meeting probabilities {years}".strip(),
            f"联邦基金期货 隐含概率 {years} 原始数据".strip(),
        ]
    return [f"{question} 官方概率 原始数据", f"{question} 概率模型 计算方法"]


@tool("validate_probability_evidence", args_schema=ProbabilityEvidenceInput,
      response_format="content_and_artifact")
def validate_probability_evidence(estimate_kind: str, question: str, evidence: list[dict],
                                  config: RunnableConfig) -> tuple[str, RuleBundle]:
    """检查概率估计是否取得带数值、方法和可追溯正文的直接证据。"""
    if estimate_kind != "probability":
        return RuleBundle(rule="probability_evidence", passed=True,
                          checks=[_check("适用范围", True, "不是概率估计，无需本规则")]).as_response()

    descriptors = [EvidenceDescriptor.model_validate(item) for item in evidence]
    direct = []
    snippet_only = []
    for item in descriptors:
        text = f"{item.title}\n{item.excerpt}"
        if _has_probability_value(text) and _MARKET_METHOD.search(text):
            if item.content_kind in {"search_snippet", "news_summary", ""}:
                snippet_only.append(item.evidence_id)
            else:
                direct.append(item.evidence_id)
    checks = [
        _check("直接概率证据", bool(direct),
               f"可用：{', '.join(direct)}" if direct else "没有同时包含概率数值与估算方法的正文证据"),
        _check("非搜索摘要", bool(direct),
               "搜索摘要只能用于发现来源，不能单独支撑概率" if snippet_only else "未发现仅靠搜索摘要支撑概率的情况"),
    ]
    violations = [] if direct else ["缺少可核对的直接概率数据或市场隐含概率正文，宏观新闻不能替代概率证据"]
    queries = [] if direct else _probability_queries(question)
    return RuleBundle(rule="probability_evidence", passed=bool(direct), checks=checks,
                      violations=violations, suggested_queries=queries).as_response()


@tool("validate_probability_answer", args_schema=ProbabilityAnswerInput,
      response_format="content_and_artifact")
def validate_probability_answer(estimate_kind: str, question: str, answer: str,
                                evidence_contract_passed: bool,
                                config: RunnableConfig) -> tuple[str, RuleBundle]:
    """交付前强制检查概率答案的数值、事件、时点、方法、证据与引用。"""
    if estimate_kind != "probability":
        return RuleBundle(rule="probability_answer", passed=True,
                          checks=[_check("适用范围", True, "不是概率估计，无需本规则")]).as_response()

    question_years = set(re.findall(r"20\d{2}", question))
    answer_years = set(re.findall(r"20\d{2}", answer))
    event_terms = [term for term in ("降息", "加息", "上涨", "下跌", "发生", "达到", "超过", "低于")
                   if term in question]
    checks = [
        _check("直接概率证据", evidence_contract_passed, "生成前的概率证据契约必须通过"),
        _check("概率数值或区间", _has_probability_value(answer), "答案必须给出0%到100%之间的概率或区间"),
        _check("预测时点", not question_years or bool(question_years & answer_years), "答案必须重申用户要求的预测年份"),
        _check("事件定义", not event_terms or any(term in answer for term in event_terms), "答案必须说明概率对应的事件"),
        _check("估算方法", bool(_MARKET_METHOD.search(answer)), "答案必须说明市场定价或统计估算方法"),
        _check("估值时点", bool(re.search(r"截至|估值日|数据日期|20\d{2}[-年/]\d{1,2}", answer)), "答案必须注明数据截至时间"),
        _check("证据引用", bool(re.search(r"\[E\d+(?:\s*[,，、]\s*E\d+)*\]", answer)), "关键结论必须引用证据编号"),
    ]
    violations = [item["detail"] for item in checks if not item["passed"]]
    return RuleBundle(rule="probability_answer", passed=not violations, checks=checks,
                      violations=violations).as_response()
