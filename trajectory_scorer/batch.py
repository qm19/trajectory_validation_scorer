"""Concurrent annotation, durable resume, retries, atomic writeback and cleanup."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock, Timeout
from tqdm import tqdm

from .config import RunConfig, SCHEMA_VERSION, fingerprint
from .core import (ChatCompletionsClient, LLMConfig, ScoringError, ScoringPipeline,
                   json_text, load_dotenv, normalize_messages, trace_digest)
from .records import (atomic_json, atomic_lines, discover, file_hash, get_score, put_score,
                      read_lines, validate_success)
from .state import State


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    id: str
    source: Path
    line: int
    offset: int
    digest: str
    group: str


def _cached_success(value, digest, identity):
    try:
        validate_success(value, digest, identity)
        return value
    except ScoringError:
        return None


def _usage(calls, http_attempts):
    result = {"completion_calls": len(calls), "http_attempts": http_attempts,
              "input_chars": sum(c["input_chars"] for c in calls),
              "output_chars": sum(c["output_chars"] for c in calls), "calls": calls}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [c.get("token_usage", {}).get(key) for c in calls]
        result[key] = sum(values) if values and all(isinstance(v, int) for v in values) else None
    return result


def _annotate(job, state, root, config, llm, identity, stop, events, client_factory):
    existing = state.get(job.id)
    previous = existing["value"] if existing else {}
    attempts = previous.get("attempts_total", 0)
    calls = list(previous.get("llm_call_stats", {}).get("calls", []))
    http_attempts = previous.get("llm_call_stats", {}).get("http_attempts", 0) or 0
    for local_attempt in range(config.max_attempts):
        if stop.is_set():
            return None
        attempts += 1
        running = {"attempts_total": attempts, "llm_call_stats": _usage(calls, http_attempts)}
        state.put(job.id, "running", running)
        client = client_factory(copy.copy(llm))
        pipeline = ScoringPipeline(client, llm.model_name, root / "stages" / job.id,
                                   config.chunk_chars, config.max_tool_chars, config.json_retries)
        result = None
        try:
            with job.source.open("rb") as stream:
                stream.seek(job.offset)
                row = json.loads(stream.readline().decode("utf-8-sig"))
            if trace_digest(normalize_messages(row, config.max_tool_chars)) != job.digest:
                raise ScoringError("输入轨迹在运行期间发生变化")
            result = pipeline.score_row(job.line, row)
        except Exception as exc:
            error = str(exc)[:2000]
        finally:
            calls.extend(pipeline.call_log)
            http_attempts += getattr(client, "http_attempt_count", 0)
        common = {"schema_version": SCHEMA_VERSION, "scoring_fingerprint": identity,
                  "trace_sha256": job.digest, "source_file": str(job.source),
                  "source_line_number": job.line, "source_group": job.group,
                  "attempts_total": attempts, "updated_at": now(),
                  "normalization": {"chunk_chars": config.chunk_chars, "max_tool_chars": config.max_tool_chars},
                  "llm_call_stats": _usage(calls, http_attempts)}
        if result is not None:
            final = {**result, **common, "status": "success"}
            validate_success(final, job.digest, identity)
            state.put(job.id, "success", final)
            return final
        failed = {**common, "status": "failed", "action_score": None, "error": error}
        state.put(job.id, "failed", failed)
        if local_attempt + 1 < config.max_attempts:
            events.put(("retry", job.id))
            delay = min(config.retry_delay_seconds * 2**local_attempt, config.retry_max_delay_seconds)
            if stop.wait(delay):
                return None
    return failed


def _prepare(config, state, identity):
    output_dir = Path(config.output_dir).resolve()
    sources = discover(config.inputs, config.pattern, exclude=None if config.in_place else output_dir)
    common_parent = Path(os.path.commonpath([str(p.parent) for p in sources]))
    outputs = {p: p if config.in_place else output_dir / p.relative_to(common_parent).with_name(p.stem + ".annotated.jsonl") for p in sources}
    if len(set(outputs.values())) != len(outputs):
        raise ScoringError("多个输入映射到了相同输出")
    if any(not config.in_place and path in sources for path in outputs.values()):
        raise ScoringError("输出路径与输入冲突；原地写回请使用 in_place=True")
    report = Path(config.report_path).resolve() if config.report_path else output_dir / "analysis.html"
    if (config.generate_report and (report in sources or report in outputs.values())) or output_dir / "run_summary.json" in sources:
        raise ScoringError("报告/统计路径不能覆盖输入或已打标数据")
    jobs = []
    hashes = {}
    reused = 0
    for source in sources:
        before = file_hash(source)
        saved = {}
        target = outputs[source]
        if target != source and target.exists():
            for line, _, row in read_lines(target):
                if row is not None:
                    saved[line] = get_score(row)
        for line, offset, row in read_lines(source):
            if row is None:
                continue
            digest = trace_digest(normalize_messages(row, config.max_tool_chars))
            key = hashlib.sha256(json_text([os.path.normcase(str(source)), line, digest, identity]).encode()).hexdigest()
            job = Job(key, source, line, offset, digest, source.parent.name)
            jobs.append(job)
            checkpoint = state.get(key)
            reusable = _cached_success(checkpoint["value"], digest, identity) if checkpoint and checkpoint["status"] == "success" else None
            reusable = reusable or _cached_success(get_score(row), digest, identity) or _cached_success(saved.get(line), digest, identity)
            if reusable:
                state.put(key, "success", reusable)
                reused += 1
        if file_hash(source) != before:
            raise ScoringError(f"输入在预检期间发生变化: {source}")
        hashes[source] = before
    if not jobs:
        raise ScoringError("输入文件没有轨迹")
    return jobs, outputs, hashes, reused


def _write_outputs(jobs, outputs, hashes, state, config, identity):
    index = {(j.source, j.line): j for j in jobs}
    for source, target in outputs.items():
        if file_hash(source) != hashes[source]:
            raise ScoringError(f"输入文件在打标期间被修改，已阻止覆盖: {source}")
        if config.in_place and config.backup_on_in_place:
            backup = source.with_name(source.name + ".bak")
            if not backup.exists():
                # Exclusive create prevents silently replacing a previous original backup.
                with backup.open("xb") as dest, source.open("rb") as original:
                    shutil.copyfileobj(original, dest)
        def lines():
            for number, _, row in read_lines(source):
                if row is None:
                    yield "\n"  # Preserve physical line numbers, including blank lines.
                    continue
                job = index[(source, number)]
                entry = state.get(job.id)
                if entry and entry["status"] in {"success", "failed"}:
                    value = entry["value"].copy()
                else:
                    value = {"schema_version": SCHEMA_VERSION, "status": "pending", "action_score": None,
                             "trace_sha256": job.digest, "scoring_fingerprint": identity}
                value.update(source_file=str(source), source_line_number=number, source_group=job.group)
                if value["status"] == "success":
                    value["line_number"] = number
                    validate_success(value, job.digest, identity)
                put_score(row, value)
                yield json_text(row) + "\n"
        def unchanged():
            if file_hash(source) != hashes[source]:
                raise ScoringError(f"输入在写回期间变化，已阻止替换: {source}")
        atomic_lines(target, lines(), before_replace=unchanged)
        count = 0
        for number, _, row in read_lines(target):
            if row is None:
                continue
            value = get_score(row)
            job = index[(source, number)]
            if trace_digest(normalize_messages(row, config.max_tool_chars)) != job.digest:
                raise ScoringError(f"写回后的 messages 哈希变化: {target}:{number}")
            if value["status"] == "success":
                validate_success(value, job.digest, identity)
            count += 1
        if count != sum(j.source == source for j in jobs):
            raise ScoringError(f"写回记录数不一致: {target}")


def cleanup_state(root, parent, run_id):
    root, parent = root.resolve(), parent.resolve()
    marker = root / ".owned"
    if root.parent != parent or not root.is_relative_to(parent) or root.name != run_id:
        raise ScoringError("临时目录范围不符合预期，已停止清理")
    if not marker.is_file() or marker.read_text(encoding="utf-8") != run_id:
        raise ScoringError("临时目录缺少所有权标记，已停止清理")
    shutil.rmtree(root)
    try:
        parent.rmdir()  # Only removes the empty parent; never recursively removes it.
    except OSError:
        pass


def run_batch(config: RunConfig, *, llm_config=None, client_factory=None):
    """Score records and return final summary. Originals are preserved by default."""
    from .reporting import collect, render_report, print_summary

    config.validate()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = config.llm_overrides(llm_config or LLMConfig.from_environment(load_dotenv(Path(config.env_file))))
    if llm.max_retries < 1 or llm.timeout_seconds <= 0 or llm.max_output_tokens < 1:
        raise ScoringError("HTTP 尝试次数、超时和最大输出须为正数")
    identity = fingerprint(config, llm)
    client_factory = client_factory or ChatCompletionsClient
    sources = discover(config.inputs, config.pattern, exclude=None if config.in_place else output_dir)
    run_id = hashlib.sha256(json_text([list(map(str, sources)), str(output_dir), config.in_place]).encode()).hexdigest()[:24]
    parent = output_dir / ".state"
    root = parent / run_id
    lock = FileLock(str(output_dir / ".annotation.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise ScoringError("该输出目录已有打标进程，请等待或使用不同 output_dir") from exc
    state = None
    try:
        marker = root / ".owned"
        if parent.is_symlink() or root.is_symlink():
            raise ScoringError("断点目录不能是符号链接")
        if root.exists():
            if not marker.is_file() or marker.read_text(encoding="utf-8") != run_id:
                raise ScoringError("已有断点目录缺少匹配的所有权标记")
        else:
            root.mkdir(parents=True)
            marker.write_text(run_id, encoding="utf-8")
        state = State(root)
        jobs, outputs, hashes, reused = _prepare(config, state, identity)
        pending = [j for j in jobs if not (state.get(j.id) and state.get(j.id)["status"] == "success")]
        chosen = pending[:config.max_new_records] if config.max_new_records else pending
        stop, events = threading.Event(), queue.Queue()
        successful, failed, retries, interrupted = reused, 0, 0, False
        before_calls = sum((state.get(j.id) or {}).get("value", {}).get("llm_call_stats", {}).get("completion_calls", 0) for j in jobs)
        todo = iter(chosen)
        executor = ThreadPoolExecutor(max_workers=config.workers)
        futures = {}
        bar = tqdm(total=len(jobs), initial=reused, desc="校验动作打标", unit="条", disable=not config.progress,
                   dynamic_ncols=True, mininterval=.2)
        def fill():
            while not stop.is_set() and len(futures) < config.workers:
                job = next(todo, None)
                if job is None:
                    break
                futures[executor.submit(_annotate, job, state, root, config, llm, identity, stop, events, client_factory)] = job
        try:
            fill()
            while futures:
                done, _ = wait(futures, timeout=.25, return_when=FIRST_COMPLETED)
                completed_now = 0
                for future in done:
                    job = futures.pop(future)
                    value = future.result()
                    if value:
                        successful += value["status"] == "success"
                        failed += value["status"] == "failed"
                        completed_now += 1
                        if value["status"] == "failed":
                            tqdm.write(f"失败 {job.source.name}:{job.line}: {value['error']}")
                while not events.empty():
                    events.get_nowait()
                    retries += 1
                bar.set_postfix(成功=successful, 失败=failed, 复用=reused, 重试=retries, refresh=False)
                if completed_now:
                    bar.update(completed_now)
                fill()
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            for future in futures:
                future.cancel()
            tqdm.write("正在等待已发出的请求结束并保存断点…")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            bar.close()
        _write_outputs(jobs, outputs, hashes, state, config, identity)
        stats, data = collect(list(outputs.values()))
        current_calls = sum((state.get(j.id) or {}).get("value", {}).get("llm_call_stats", {}).get("completion_calls", 0) for j in jobs) - before_calls
        stats.update({"schema_version": SCHEMA_VERSION, "created_at": now(), "workers": config.workers,
                      "reused_this_run": reused, "new_successes_this_run": stats["success"]-reused,
                      "new_completion_calls_this_run": current_calls, "interrupted": interrupted,
                      "scoring_fingerprint": identity, "output_files": list(map(str, outputs.values())),
                      "state_dir": str(root), "state_cleaned": False})
        report_path = Path(config.report_path).resolve() if config.report_path else output_dir / "analysis.html"
        if report_path in sources or report_path in outputs.values():
            raise ScoringError("HTML 报告路径不能覆盖输入或已打标数据")
        if config.generate_report:
            render_report(data, stats, report_path)
            stats["report"] = str(report_path)
        complete = stats["success"] == stats["total"] and not interrupted
        summary_path = output_dir / "run_summary.json"
        if summary_path in sources or summary_path in outputs.values():
            raise ScoringError("统计 JSON 路径不能覆盖输入或已打标数据")
        stats["summary_file"] = str(summary_path)
        # Persist final artifacts before deleting the recoverable intermediate state.
        atomic_json(summary_path, stats)
        state.close()
        state = None
        if complete and config.cleanup_on_success:
            cleanup_state(root, parent, run_id)
            stats["state_cleaned"] = True
            atomic_json(summary_path, stats)
        if config.print_summary:
            print_summary(stats)
        return stats
    finally:
        if state is not None:
            state.close()
        lock.release()
