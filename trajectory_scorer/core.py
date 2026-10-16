"""Data handling, LLM transport, validation and scoring orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .prompts import (
    CATEGORIES,
    EVIDENCE_PROMPT_VERSION,
    EVIDENCE_SYSTEM,
    OBLIGATIONS_SYSTEM,
    PROMPT_VERSION,
    SCORING_PROMPT_VERSION,
    SCORING_SYSTEM,
)


class ScoringError(RuntimeError):
    """An input, transport or model response cannot be safely scored."""


def load_dotenv(path: Path) -> dict[str, str]:
    """Read a small .env file without adding a third-party dependency."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ScoringError(f".env 第 {number} 行缺少 '='")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[i + end :].strip(" \n\r\t`"):
            return value
    raise ScoringError("模型响应不是单个合法 JSON 对象")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"输入第 {line_number} 行 JSON 错误: {exc}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
                raise ScoringError(f"输入第 {line_number} 行缺少 messages 数组")
            yield line_number, value


def normalize_messages(row: dict[str, Any], max_tool_chars: int = 16000) -> list[dict[str, Any]]:
    """Keep only visible behavior; exclude metadata, tools schema and hidden reasoning."""
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(row["messages"]):
        if not isinstance(message, dict):
            raise ScoringError(f"messages[{index}] 不是对象")
        role = message.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            raise ScoringError(f"messages[{index}] 的 role 不受支持: {role!r}")
        if role == "system":
            continue
        item: dict[str, Any] = {"message_index": index, "role": role}
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json_text(content)
        if role == "tool" and max_tool_chars > 0 and len(content) > max_tool_chars:
            head_chars = max_tool_chars * 2 // 3
            tail_chars = max_tool_chars - head_chars
            item["content"] = (
                content[:head_chars]
                + f"\n[TOOL_CONTENT_TRUNCATED: omitted {len(content) - max_tool_chars} characters]\n"
                + content[-tail_chars:]
            )
            item["content_truncated"] = True
            item["original_content_chars"] = len(content)
            item["original_content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            item["content"] = content
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                fn = call.get("function", {})
                calls.append(
                    {
                        "id": call.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", ""),
                    }
                )
            item["tool_calls"] = calls
        if role == "tool" and message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def trace_digest(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json_text(messages).encode("utf-8")).hexdigest()


def _units(messages: list[dict[str, Any]], max_chars: int) -> list[str]:
    units = []
    if max_chars < 2000:
        raise ScoringError("chunk_chars 至少为 2000")
    for message in messages:
        raw = json_text(message)
        if len(raw) + 2 <= max_chars:
            units.append(raw)
            continue
        fragments = []
        pos = 0
        while pos < len(raw):
            low, high = 1, min(len(raw) - pos, max_chars)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                candidate = json_text(
                    {
                        "message_index": message["message_index"],
                        "fragment": 999999,
                        "fragment_count": 999999,
                        "serialized_message_fragment": raw[pos : pos + middle],
                    }
                )
                if len(candidate) + 2 <= max_chars:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ScoringError("chunk_chars 太小，无法容纳消息片段")
            fragments.append(raw[pos : pos + best])
            pos += best
        for part, fragment in enumerate(fragments, 1):
            units.append(
                json_text(
                    {
                        "message_index": message["message_index"],
                        "fragment": part,
                        "fragment_count": len(fragments),
                        "serialized_message_fragment": fragment,
                    }
                )
            )
    return units


def chunk_messages(
    messages: list[dict[str, Any]], max_chars: int = 100000, overlap_units: int = 2
) -> list[str]:
    """Bound chunks without silently discarding oversized tool outputs."""
    units = _units(messages, max_chars)
    chunks: list[str] = []
    start = 0
    while start < len(units):
        end = start
        size = 2
        while end < len(units) and (end == start or size + len(units[end]) + 1 <= max_chars):
            size += len(units[end]) + 1
            end += 1
        chunks.append("[" + ",".join(units[start:end]) + "]")
        if end == len(units):
            break
        start = max(start + 1, end - overlap_units)
    return chunks


def _ids(items: list[dict[str, Any]], label: str) -> set[str]:
    seen: set[str] = set()
    for item in items:
        ident = item.get("id")
        if not isinstance(ident, str) or not ident:
            raise ScoringError(f"{label} 缺少非空 id")
        if ident in seen:
            raise ScoringError(f"{label} id 重复: {ident}")
        seen.add(ident)
    return seen


def validate_obligations(value: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    deliverables = value.get("final_deliverables")
    obligations = value.get("obligations")
    if not all(isinstance(x, list) for x in (deliverables, obligations)):
        raise ScoringError("义务响应缺少 final_deliverables 或 obligations 数组")
    if not deliverables:
        raise ScoringError("至少应有一个最终交付物，包括在回复中交付的内容")
    d_ids = _ids(deliverables, "交付物")
    _ids(obligations, "义务")
    user_indices = {m["message_index"] for m in messages if m["role"] == "user"}
    for item in obligations:
        if item.get("category") not in CATEGORIES:
            raise ScoringError(f"义务类别无效: {item.get('category')}")
        if not item.get("target") or not item.get("sufficient_check"):
            raise ScoringError(f"义务 {item['id']} 缺少 target 或 sufficient_check")
        refs = item.get("source_message_indices")
        if not isinstance(refs, list) or not refs or not set(refs) <= user_indices:
            raise ScoringError(f"义务 {item['id']} 的用户消息引用无效")
        ds = item.get("deliverable_ids")
        if not isinstance(ds, list) or not ds or not set(ds) <= d_ids:
            raise ScoringError(f"义务 {item['id']} 的交付物引用无效")
    if value.get("uncovered_requirements") not in ([], None):
        raise ScoringError("仍有未覆盖的最终要求")


EVENT_TYPES = {"create", "modify", "check", "failure", "fix", "user_correction"}


def validate_events(value: dict[str, Any], obligations: dict[str, Any], allowed_indices: set[int]) -> None:
    events = value.get("events")
    if not isinstance(events, list):
        raise ScoringError("证据响应缺少 events 数组")
    valid_obligations = {o["id"] for o in obligations["obligations"]}
    for event in events:
        if event.get("type") not in EVENT_TYPES:
            raise ScoringError("证据事件 type 无效")
        refs = event.get("message_indices")
        if not isinstance(refs, list) or not refs:
            raise ScoringError(
                "每个证据事件必须包含非空 message_indices 数组（复数）；"
                "不得使用 message_index 字段替代，必须引用本段实际消息编号"
            )
        if not set(refs) <= allowed_indices:
            raise ScoringError(
                f"证据事件包含本段之外的消息编号 {sorted(set(refs) - allowed_indices)}；"
                f"本段允许编号为 {sorted(allowed_indices)}"
            )
        orefs = event.get("obligation_ids")
        if not isinstance(orefs, list) or not set(orefs) <= valid_obligations:
            raise ScoringError("证据事件引用了不存在的义务")
        if event.get("result_visibility") not in {"visible", "partial", "missing"}:
            raise ScoringError("证据事件 result_visibility 无效")


def merge_events(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged = []
    for result in results:
        for event in result["events"]:
            key = json_text(event)
            if key not in seen:
                seen.add(key)
                merged.append(event)
    merged.sort(key=lambda e: (min(e["message_indices"]), e["type"]))
    return merged


def validate_scores(value: dict[str, Any], obligations: dict[str, Any], events: list[dict[str, Any]]) -> None:
    scores = value.get("obligation_scores")
    if not isinstance(scores, list):
        raise ScoringError("评分响应缺少 obligation_scores 数组")
    expected = {o["id"] for o in obligations["obligations"]}
    actual = [s.get("obligation_id") for s in scores]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ScoringError("评分义务集合与提取的义务不一致")
    evidence_refs: dict[str, set[int]] = {ident: set() for ident in expected}
    for event in events:
        if event["type"] in {"check", "failure"}:
            for ident in event["obligation_ids"]:
                evidence_refs[ident].update(event["message_indices"])
    for score in scores:
        ident = score["obligation_id"]
        q = score.get("score")
        refs = score.get("supporting_message_indices")
        if type(q) is not int or q not in range(5):
            raise ScoringError(f"义务 {ident} 分数须为 0–4 的整数")
        if not isinstance(refs, list) or not set(refs) <= evidence_refs[ident]:
            raise ScoringError(f"义务 {ident} 引用了未经提取的校验证据")
        if q > 0 and not refs:
            raise ScoringError(f"义务 {ident} 正分却没有可见校验依据")
        if not isinstance(score.get("final_version_checked"), bool) or not score.get("reason"):
            raise ScoringError(f"义务 {ident} 缺少版本判断或评分理由")
        if score.get("observed_outcome") not in {"pass", "fail", "inconclusive", "not_observed"}:
            raise ScoringError(f"义务 {ident} 的 observed_outcome 无效")


def compute_summary(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    if not scores:
        return {
            "action_score": None,
            "attempt_rate": None,
            "observed_check_rate": None,
            "adequate_check_rate": None,
            "final_version_check_rate": None,
        }
    n = len(scores)
    return {
        "action_score": round(100 * sum(s["score"] for s in scores) / (4 * n), 1),
        "attempt_rate": round(sum(s["score"] >= 1 for s in scores) / n, 4),
        "observed_check_rate": round(sum(s["score"] >= 2 for s in scores) / n, 4),
        "adequate_check_rate": round(sum(s["score"] >= 3 for s in scores) / n, 4),
        "final_version_check_rate": round(
            sum(s["score"] >= 2 and s["final_version_checked"] for s in scores) / n, 4
        ),
    }


def compute_category_scores(
    obligations: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate action scores by obligation category, without another model call."""
    score_by_id = {item["obligation_id"]: item for item in scores}
    result = []
    for category in CATEGORIES:
        items = [item for item in obligations if item["category"] == category]
        result.append({
            "category": category,
            "obligation_count": len(items),
            "obligation_ids": [item["id"] for item in items],
            **compute_summary([score_by_id[item["id"]] for item in items]),
        })
    return result


def compact_obligations(obligations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send only fields needed to match evidence and judge checking actions."""
    fields = ("id", "deliverable_ids", "category", "target", "sufficient_check")
    return [{key: item[key] for key in fields} for item in obligations]


@dataclass
class LLMConfig:
    url: str
    api_key: str
    model_name: str
    timeout_seconds: float = 180
    max_retries: int = 3
    max_output_tokens: int = 32768
    json_mode: bool = False
    direct_connection: bool = False
    no_proxy: str = "7.156.135.202"

    @classmethod
    def from_environment(cls, env: dict[str, str]) -> "LLMConfig":
        def get(key: str) -> str:
            return os.environ.get(key, env.get(key, "")).strip()

        missing = [key for key in ("LLM_URL", "LLM_API_KEY", "LLM_MODEL_NAME") if not get(key)]
        if missing:
            raise ScoringError("请先在 .env 中填写: " + ", ".join(missing))
        return cls(
            url=get("LLM_URL"),
            api_key=get("LLM_API_KEY"),
            model_name=get("LLM_MODEL_NAME"),
            timeout_seconds=float(get("LLM_TIMEOUT_SECONDS") or 180),
            max_retries=int(get("LLM_MAX_RETRIES") or 3),
            max_output_tokens=int(get("LLM_MAX_OUTPUT_TOKENS") or 32768),
            json_mode=get("LLM_JSON_MODE").lower() in {"1", "true", "yes"},
            direct_connection=get("LLM_DIRECT_CONNECTION").lower() in {"1", "true", "yes"},
            no_proxy=get("LLM_NO_PROXY") or "7.156.135.202",
        )


class ChatCompletionsClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache_profile = "thinking-minimal-extra-body-v2"
        self.http_attempt_count = 0
        self.last_usage: dict[str, Any] | None = None

    def complete(self, system: str, user: str) -> str:
        # This endpoint is reachable directly but the system proxy returns 504.
        # Preserve any existing exclusions and add the user-specified host.
        for variable in ("no_proxy", "NO_PROXY"):
            existing = [part.strip() for part in os.environ.get(variable, "").split(",") if part.strip()]
            for host in (part.strip() for part in self.config.no_proxy.split(",") if part.strip()):
                if host not in existing:
                    existing.append(host)
            os.environ[variable] = ",".join(existing)
        url = self.config.url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "extra_body": {
                "enable_thinking": False,
                "google": {
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_level": "minimal",
                    }
                },
            },
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            url,
            data=json_text(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if self.config.direct_connection else None
        for attempt in range(self.config.max_retries):
            try:
                transport = opener.open if opener else urllib.request.urlopen
                self.http_attempt_count += 1
                with transport(request, timeout=self.config.timeout_seconds) as response:
                    body = json.load(response)
                finish_reason = body["choices"][0].get("finish_reason")
                if finish_reason == "length":
                    usage = body.get("usage", {})
                    raise ScoringError(
                        f"模型输出达到上限 {self.config.max_output_tokens} token，返回被截断；usage={usage}"
                    )
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
                if not isinstance(content, str):
                    raise ScoringError("API 响应的 message.content 不是文本")
                self.last_usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
                return content
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt + 1 == self.config.max_retries:
                    detail = exc.read(500).decode("utf-8", errors="replace")
                    raise ScoringError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 == self.config.max_retries:
                    raise ScoringError(f"API 网络错误: {exc}") from exc
            time.sleep(min(2**attempt, 15))
        raise AssertionError("unreachable")


class ScoringPipeline:
    def __init__(
        self,
        client: Any,
        model_name: str,
        cache_dir: Path,
        chunk_chars: int = 100000,
        max_tool_chars: int = 16000,
        json_retries: int = 2,
    ):
        self.client = client
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.chunk_chars = chunk_chars
        self.max_tool_chars = max_tool_chars
        self.json_retries = json_retries
        self.call_log: list[dict[str, Any]] = []

    def _call(
        self,
        cache_path: Path,
        system: str,
        payload: dict[str, Any],
        validate: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            validate(cached)
            return cached
        user = json_text(payload)
        last_error = ""
        last_response = ""
        for attempt in range(self.json_retries + 1):
            suffix = "" if not last_error else (
                f"\n上次响应不合规：{last_error}。请修复下面这份上次响应，而不是忽略错误重新猜测。"
                f"\n上次响应：{last_response[:16000]}\n请重新输出完整合法 JSON。"
            )
            raw = self.client.complete(system, user + suffix)
            call_record: dict[str, Any] = {
                "stage": cache_path.name,
                "input_chars": len(system) + len(user + suffix),
                "output_chars": len(raw),
            }
            usage = getattr(self.client, "last_usage", None)
            if isinstance(usage, dict):
                call_record["token_usage"] = usage
            self.call_log.append(call_record)
            try:
                value = parse_json_object(raw)
                validate(value)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                return value
            except ScoringError as exc:
                last_error = str(exc)
                last_response = raw
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.with_name(f"{cache_path.stem}.invalid_attempt_{attempt + 1}.txt").write_text(
                    raw, encoding="utf-8"
                )
                if attempt == self.json_retries:
                    raise ScoringError(f"模型响应连续 {attempt + 1} 次无效: {last_error}") from exc
        raise AssertionError("unreachable")

    def score_row(self, line_number: int, row: dict[str, Any]) -> dict[str, Any]:
        call_log_start = len(self.call_log)
        http_attempt_start = getattr(self.client, "http_attempt_count", None)
        messages = normalize_messages(row, self.max_tool_chars)
        digest = trace_digest(messages)
        safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.model_name)
        request_profile = getattr(self.client, "cache_profile", "")
        safe_profile = "".join(c if c.isalnum() or c in "-_" else "_" for c in request_profile)
        cache_root = self.cache_dir / (
            f"line_{line_number:06d}_{digest[:16]}_{PROMPT_VERSION}_{safe_model}_chunk{self.chunk_chars}"
            + (f"_{safe_profile}" if safe_profile else "")
        )
        users = [m for m in messages if m["role"] == "user"]
        if not users:
            raise ScoringError("轨迹没有 user 消息")
        obligations = self._call(
            cache_root / "obligations.json",
            OBLIGATIONS_SYSTEM,
            {"user_messages": users},
            lambda v: validate_obligations(v, messages),
        )
        obligation_briefs = compact_obligations(obligations["obligations"])
        chunks = chunk_messages(messages, self.chunk_chars)
        evidence_results = []
        for index, chunk in enumerate(chunks):
            allowed_indices = {unit["message_index"] for unit in json.loads(chunk)}
            result = self._call(
                cache_root / f"evidence_{EVIDENCE_PROMPT_VERSION}_{index:04d}.json",
                EVIDENCE_SYSTEM,
                {"obligations": obligation_briefs, "chunk_number": index, "trace_chunk": json.loads(chunk)},
                lambda v: validate_events(v, obligations, allowed_indices),
            )
            evidence_results.append(result)
        events = merge_events(evidence_results)
        scoring_briefs = [
            {
                **item,
                "eligible_check_message_indices": sorted({
                    index
                    for event in events
                    if event["type"] in {"check", "failure"}
                    and item["id"] in event["obligation_ids"]
                    for index in event["message_indices"]
                }),
            }
            for item in obligation_briefs
        ]
        scoring = self._call(
            cache_root / f"scores_{SCORING_PROMPT_VERSION}_{EVIDENCE_PROMPT_VERSION}.json",
            SCORING_SYSTEM,
            {"obligations": scoring_briefs, "events": events},
            lambda v: validate_scores(v, obligations, events),
        )
        summary = compute_summary(scoring["obligation_scores"])
        row_calls = self.call_log[call_log_start:]
        token_usage = {
            key: sum(call["token_usage"][key] for call in row_calls)
            if row_calls and all(
                isinstance(call.get("token_usage", {}).get(key), int) for call in row_calls
            ) else None
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        http_attempts = getattr(self.client, "http_attempt_count", None)
        return {
            "line_number": line_number,
            "trace_sha256": digest,
            "message_count": len(row["messages"]),
            "chunk_count": len(chunks),
            "truncated_tool_messages": sum(bool(m.get("content_truncated")) for m in messages),
            "prompt_version": PROMPT_VERSION,
            "evidence_prompt_version": EVIDENCE_PROMPT_VERSION,
            "scoring_prompt_version": SCORING_PROMPT_VERSION,
            "model_name": self.model_name,
            "llm_request_profile": request_profile or None,
            "final_deliverables": obligations["final_deliverables"],
            "requirement_updates": obligations.get("requirement_updates", []),
            "obligations": obligations["obligations"],
            "evidence_events": events,
            "obligation_scores": scoring["obligation_scores"],
            "category_scores": compute_category_scores(
                obligations["obligations"], scoring["obligation_scores"]
            ),
            "score_type": "verification_action_completion",
            **summary,
            "llm_call_stats": {
                "completion_calls": len(row_calls),
                "http_attempts": http_attempts - http_attempt_start
                if isinstance(http_attempts, int) and isinstance(http_attempt_start, int) else None,
                "input_chars": sum(call["input_chars"] for call in row_calls),
                "output_chars": sum(call["output_chars"] for call in row_calls),
                **token_usage,
                "calls": row_calls,
            },
            "obligation_review_flags": scoring.get("obligation_review_flags", []),
            "trajectory_evidence_limitations": scoring.get("trajectory_evidence_limitations", []),
        }
