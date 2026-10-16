"""User-editable batch configuration and stable scoring identity."""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import LLMConfig, ScoringError, json_text
from .prompts import PROMPT_VERSION, EVIDENCE_PROMPT_VERSION, SCORING_PROMPT_VERSION

SCHEMA_VERSION = "1.0"
SCORE_FIELD = "verifi_action_score"
REQUEST_PROFILE = "thinking-minimal-extra-body-v2"


@dataclass
class RunConfig:
    inputs: list[str | Path] = field(default_factory=list)
    output_dir: str | Path = "outputs"
    pattern: str = "*.jsonl"
    env_file: str | Path = ".env"
    workers: int = 10
    max_attempts: int = 3  # Total trajectory attempts per invocation, not extra retries.
    retry_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 20.0
    json_retries: int = 2
    chunk_chars: int = 100000
    max_tool_chars: int = 16000
    max_new_records: int = 0  # 0 = all; useful for staged/resumable runs.
    in_place: bool = False
    backup_on_in_place: bool = True
    cleanup_on_success: bool = True
    generate_report: bool = True
    report_path: str | Path | None = None
    progress: bool = True
    print_summary: bool = True
    # None keeps the .env / process environment setting.
    request_timeout_seconds: float | None = None
    http_max_attempts: int | None = None
    max_output_tokens: int | None = None
    direct_connection: bool | None = None
    no_proxy: str | None = None

    def validate(self):
        for name in ("workers", "max_attempts", "json_retries", "chunk_chars", "max_tool_chars", "max_new_records"):
            if type(getattr(self, name)) is not int:
                raise ScoringError(f"{name} 必须为整数")
        if not self.inputs:
            raise ScoringError("请设置 inputs 或传入 --input")
        if self.workers < 1 or self.max_attempts < 1 or self.json_retries < 0:
            raise ScoringError("workers / max_attempts 须 >=1；json_retries 须 >=0")
        if self.chunk_chars < 2000 or self.max_tool_chars < 0 or self.max_new_records < 0:
            raise ScoringError("chunk_chars 须 >=2000；max_tool_chars / max_new_records 须 >=0")
        if min(self.retry_delay_seconds, self.retry_max_delay_seconds) < 0:
            raise ScoringError("重试等待时间不得为负数")
        for name in ("request_timeout_seconds", "http_max_attempts", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ScoringError(f"{name} 须 >0")
        for name in ("http_max_attempts", "max_output_tokens"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not int:
                raise ScoringError(f"{name} 必须为整数")

    def llm_overrides(self, config: LLMConfig) -> LLMConfig:
        for source, target in (("request_timeout_seconds", "timeout_seconds"),
                               ("http_max_attempts", "max_retries"),
                               ("max_output_tokens", "max_output_tokens"),
                               ("direct_connection", "direct_connection"), ("no_proxy", "no_proxy")):
            if getattr(self, source) is not None:
                setattr(config, target, getattr(self, source))
        return config


def load_config(path: Path) -> RunConfig:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("trajectory_user_settings", path)
    if spec is None or spec.loader is None:
        raise ScoringError(f"无法加载配置: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, "CONFIG", None)
    config = RunConfig(**value) if isinstance(value, dict) else value
    if not isinstance(config, RunConfig):
        raise ScoringError("配置文件须定义 CONFIG = RunConfig(...) 或 CONFIG = {...}")
    config = RunConfig(**asdict(config))
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent / p
    config.inputs = [resolve(p) for p in config.inputs]
    config.output_dir = resolve(config.output_dir)
    config.env_file = resolve(config.env_file)
    if config.report_path is not None:
        config.report_path = resolve(config.report_path)
    return config


def fingerprint(config: RunConfig, llm: LLMConfig) -> str:
    # Credentials and operational settings do not change semantic scoring identity.
    identity = {"schema": SCHEMA_VERSION, "model": llm.model_name, "endpoint": llm.url,
                "prompt": PROMPT_VERSION, "evidence": EVIDENCE_PROMPT_VERSION,
                "scoring": SCORING_PROMPT_VERSION, "profile": REQUEST_PROFILE,
                "chunk_chars": config.chunk_chars, "max_tool_chars": config.max_tool_chars,
                "max_output_tokens": llm.max_output_tokens, "json_mode": llm.json_mode}
    return hashlib.sha256(json_text(identity).encode("utf-8")).hexdigest()
