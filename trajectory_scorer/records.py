"""Annotated JSONL records and atomic, verified file replacement."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .config import SCORE_FIELD, SCHEMA_VERSION
from .core import ScoringError, compute_category_scores, compute_summary, json_text, validate_scores


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_lines(path: Path):
    """Yield physical line number, byte offset, and row without keeping full files in RAM."""
    with path.open("rb") as stream:
        number = 0
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            number += 1
            if not raw.strip():
                yield number, offset, None
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig"))
            except (ValueError, UnicodeError) as exc:
                raise ScoringError(f"{path} 第 {number} 行不是合法 UTF-8 JSON") from exc
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ScoringError(f"{path} 第 {number} 行缺少 messages 数组")
            check_meta(row)
            yield number, offset, row


def check_meta(row):
    meta = row.get("meta_info", {})
    if not isinstance(meta, dict) or not isinstance(meta.get("unique_info", {}), dict):
        raise ScoringError("meta_info 和 meta_info.unique_info 必须为对象，不能覆盖非对象数据")


def get_score(row):
    return row.get("meta_info", {}).get("unique_info", {}).get(SCORE_FIELD)


def put_score(row, score):
    row.setdefault("meta_info", {}).setdefault("unique_info", {})[SCORE_FIELD] = score


def validate_success(value, digest=None, scoring_fingerprint=None):
    if not isinstance(value, dict) or value.get("status") != "success":
        raise ScoringError("缺少成功评分对象")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ScoringError("评分 schema_version 不兼容")
    if digest is not None and value.get("trace_sha256") != digest:
        raise ScoringError("评分与当前轨迹哈希不一致")
    if scoring_fingerprint is not None and value.get("scoring_fingerprint") != scoring_fingerprint:
        raise ScoringError("评分配置指纹不一致")
    try:
        obligations, scores, events = value["obligations"], value["obligation_scores"], value["evidence_events"]
        validate_scores({"obligation_scores": scores}, {"obligations": obligations}, events)
        for key, expected in compute_summary(scores).items():
            if value[key] != expected:
                raise ScoringError(f"评分汇总不一致: {key}")
        if value["category_scores"] != compute_category_scores(obligations, scores):
            raise ScoringError("类别分不一致")
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoringError("评分对象字段不完整或无效") from exc


def atomic_json(path: Path, value):
    atomic_lines(path, [json.dumps(value, ensure_ascii=False, indent=2) + "\n"])


def atomic_lines(path: Path, lines, before_replace=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    expected = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as stream:
            for line in lines:
                raw = line.encode("utf-8")
                stream.write(raw)
                expected.update(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if file_hash(temp) != expected.hexdigest():
            raise ScoringError(f"临时文件读回不一致: {path}")
        if before_replace is not None:
            before_replace()
        os.replace(temp, path)
        if file_hash(path) != expected.hexdigest():
            raise ScoringError(f"交付文件读回不一致: {path}")
    finally:
        if temp.exists():
            temp.unlink()


def discover(inputs, pattern="*.jsonl", exclude=None):
    found = set()
    for item in inputs:
        path = Path(item).resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = list(path.glob(pattern))
        else:
            raise ScoringError(f"输入不存在: {path}")
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file() and not (exclude and candidate.is_relative_to(exclude)):
                found.add(candidate)
    if not found:
        raise ScoringError("没有匹配的输入 JSONL 文件，请检查 inputs / pattern")
    return sorted(found, key=str)
