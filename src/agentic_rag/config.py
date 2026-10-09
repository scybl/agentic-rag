"""集中管理配置，并允许通过环境变量或 .env 文件覆盖。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """源码安装沿用仓库根；wheel安装使用工作目录，避免写入Python环境目录。"""
    explicit = os.getenv("AGENTIC_RAG_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    if (repository / "pyproject.toml").is_file() and package == repository / "src/agentic_rag":
        return repository
    return Path.cwd().resolve()


PROJECT_ROOT = _project_root()

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv 是可选依赖
    pass



def _path(env_var: str, default: str) -> str:
    """将路径配置解析为相对于项目根目录的路径。"""
    raw = os.getenv(env_var, default)
    path = Path(raw)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _boolean(env_var: str, default: bool = False) -> bool:
    """读取常见形式的布尔环境变量。"""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # 模型
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    eval_model: str = os.getenv("EVAL_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_warmup_enabled: bool = _boolean("OLLAMA_WARMUP_ENABLED", True)
    ollama_warmup_attempts: int = int(os.getenv("OLLAMA_WARMUP_ATTEMPTS", "10"))
    ollama_warmup_retry_seconds: float = float(
        os.getenv("OLLAMA_WARMUP_RETRY_SECONDS", "3")
    )
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_max_attempts: int = int(os.getenv("LLM_MAX_ATTEMPTS", "6"))
    llm_reasoning: bool = _boolean("LLM_REASONING", False)
    llm_context_window: int = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
    llm_max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    llm_adaptive_max_attempts: int = int(os.getenv("LLM_ADAPTIVE_MAX_ATTEMPTS", "3"))
    llm_adaptive_max_output_tokens: int = int(os.getenv(
        "LLM_ADAPTIVE_MAX_OUTPUT_TOKENS", str(max(llm_max_output_tokens, int(llm_max_output_tokens * 1.5)))
    ))
    llm_adaptive_max_context_window: int = int(os.getenv(
        "LLM_ADAPTIVE_MAX_CONTEXT_WINDOW", str(llm_context_window)
    ))
    llm_adaptive_max_request_timeout: int = int(os.getenv(
        "LLM_ADAPTIVE_MAX_REQUEST_TIMEOUT", str(max(120, int(os.getenv("LLM_REQUEST_TIMEOUT", "120")) * 2))
    ))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5",
    )

    # 检索
    chroma_dir: str = field(default_factory=lambda: _path("CHROMA_DIR", ".chroma"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    knowledge_dir: str = field(
        default_factory=lambda: _path(
            "KNOWLEDGE_DIR",
            os.getenv("DOCUMENTS_DIR", "knowledge"),
        )
    )
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    generation_max_documents: int = int(os.getenv("GENERATION_MAX_DOCUMENTS", "9"))
    generation_context_chars: int = int(os.getenv("GENERATION_CONTEXT_CHARS", "6000"))
    news_retrieval_k: int = int(os.getenv("NEWS_RETRIEVAL_K", "5"))
    news_vector_cache_enabled: bool = _boolean("NEWS_VECTOR_CACHE_ENABLED", True)
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    max_answer_revisions: int = int(os.getenv("MAX_ANSWER_REVISIONS", "2"))
    research_enabled: bool = _boolean("RESEARCH_ENABLED", True)
    research_db: str = field(default_factory=lambda: _path("RESEARCH_DB", ".chroma/research.sqlite3"))
    checkpoint_db: str = field(default_factory=lambda: _path("CHECKPOINT_DB", ".chroma/checkpoints.sqlite3"))
    conversation_db: str = field(default_factory=lambda: _path("CONVERSATION_DB", ".chroma/conversations.sqlite3"))
    conversation_context_bytes: int = int(os.getenv("CONVERSATION_CONTEXT_BYTES", "12000"))
    conversation_recent_turns: int = int(os.getenv("CONVERSATION_RECENT_TURNS", "3"))
    memory_vector_dir: str = field(default_factory=lambda: _path("MEMORY_VECTOR_DIR", ".chroma/memory"))
    agent_workers: int = int(os.getenv("AGENT_WORKERS", "4"))
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", "2"))
    io_concurrency: int = int(os.getenv("IO_CONCURRENCY", "4"))
    research_max_tasks: int = int(os.getenv("RESEARCH_MAX_TASKS", "20"))
    research_hard_max_tasks: int = int(os.getenv("RESEARCH_HARD_MAX_TASKS", "200"))
    research_max_rounds: int = int(os.getenv("RESEARCH_MAX_ROUNDS", "3"))
    task_max_attempts: int = int(os.getenv("TASK_MAX_ATTEMPTS", "2"))
    task_lease_seconds: int = int(os.getenv("TASK_LEASE_SECONDS", "180"))
    task_wait_seconds: int = int(os.getenv("TASK_WAIT_SECONDS", "240"))
    llm_request_timeout: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    research_timeout: int = int(os.getenv("RESEARCH_TIMEOUT", "900"))
    research_total_timeout: int = int(os.getenv("RESEARCH_TOTAL_TIMEOUT", "1800"))
    research_max_model_calls: int = int(os.getenv("RESEARCH_MAX_MODEL_CALLS", "160"))
    reading_chunk_chars: int = int(os.getenv("READING_CHUNK_CHARS", "2200"))
    reading_verbatim_max_chars: int = int(os.getenv("READING_VERBATIM_MAX_CHARS", "1200"))
    memory_recall_k: int = int(os.getenv("MEMORY_RECALL_K", "3"))
    specialist_count: int = int(os.getenv("SPECIALIST_COUNT", "3"))
    knowledge_watch_enabled: bool = _boolean("KNOWLEDGE_WATCH_ENABLED", False)
    knowledge_watch_interval: float = float(
        os.getenv("KNOWLEDGE_WATCH_INTERVAL", "2")
    )
    analysis_cache_enabled: bool = _boolean("ANALYSIS_CACHE_ENABLED", True)
    analysis_cache_path: str = field(
        default_factory=lambda: _path(
            "ANALYSIS_CACHE_PATH", ".chroma/analysis_cache.sqlite3"
        )
    )
    analysis_cache_ttl_seconds: int = int(
        os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "3600")
    )

    # 路由器提示词使用的数据源描述。旧变量名仅用于兼容已有配置。
    knowledge_description: str = os.getenv(
        "KNOWLEDGE_DESCRIPTION",
        os.getenv(
            "KB_DESCRIPTION",
            "本地财经新闻分析方法知识库，包含新闻来源核验与可信度分级、"
            "企业财务影响与重要性分析、事件研究法与市场反应评估；用于解释"
            "分析框架和方法，不包含实时新闻事实或个股投资建议。",
        ),
    )
    news_description: str = os.getenv(
        "NEWS_DESCRIPTION",
        "只读同花顺新闻库，持续更新，覆盖产经新闻、区域经济、公司新闻、"
        "国际财经、财经评论、财经要闻、宏观经济、金融市场和财经人物；"
        "支持按关键词、日期和栏目查询，适合回答近期或历史财经新闻问题，"
        "不包含通用知识教程和非新闻类公司内部资料。",
    )
    web_description: str = os.getenv(
        "WEB_DESCRIPTION",
        "DuckDuckGo 公开网络搜索，返回相关网页的标题、链接和摘要；适合查询"
        "本地文档与私有新闻库未覆盖的公开信息和非财经时效信息；结果来自公开"
        "互联网，不代表用户的私有资料，也不保证来源权威性。",
    )


settings = Settings()
