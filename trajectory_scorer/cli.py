"""Command line interface for offline preflight, LLM scoring and reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .core import (
    ChatCompletionsClient,
    LLMConfig,
    ScoringError,
    ScoringPipeline,
    chunk_messages,
    compute_category_scores,
    iter_jsonl,
    json_text,
    load_dotenv,
    normalize_messages,
    trace_digest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 agent 轨迹中的可观察校验行为评分")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "score"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True, help="输入轨迹 JSONL")
        command.add_argument("--chunk-chars", type=int, default=100000)
        command.add_argument("--max-tool-chars", type=int, default=16000,
                             help="单条 tool.content 保留字符数；0 表示不截断")
        command.add_argument("--limit", type=int, default=0, help="最多处理多少条；0 为全部")
    score = sub.choices["score"]
    score.add_argument("--output", type=Path, required=True, help="评分结果 JSONL，可断点续跑")
    score.add_argument("--env", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    score.add_argument("--cache-dir", type=Path, default=None)
    score.add_argument("--json-retries", type=int, default=2)
    score.add_argument("--workers", type=int, default=4, help="同时评分的轨迹数；设为 1 可串行运行")
    score.add_argument("--fail-fast", action="store_true")
    score.add_argument("--direct-connection", action="store_true", help="直接连接 LLM_URL，不使用系统 HTTP 代理")
    report_parser = sub.add_parser("report", help="查看已有评分文件中一条轨迹的 C1-C9 动作分")
    report_parser.add_argument("--input", type=Path, required=True, help="评分结果 JSONL")
    report_parser.add_argument("--line", type=int, default=1, help="原始输入轨迹的行号，默认 1")
    annotate = sub.add_parser("annotate", help="正式批量打标：写回轨迹元数据、续跑、进度、自动报告")
    annotate.add_argument("--config", type=Path, help="定义 CONFIG 的 Python 配置文件")
    annotate.add_argument("--input", type=Path, action="append", help="可重复传入 JSONL 文件或目录")
    annotate.add_argument("--output-dir", type=Path)
    annotate.add_argument("--pattern", help="目录内匹配模式，默认 *.jsonl；支持 **/*.jsonl")
    annotate.add_argument("--env", type=Path)
    annotate.add_argument("--workers", type=int)
    annotate.add_argument("--max-attempts", type=int, help="每条轨迹本次运行最多尝试次数，含首次")
    annotate.add_argument("--json-retries", type=int)
    annotate.add_argument("--chunk-chars", type=int)
    annotate.add_argument("--max-tool-chars", type=int)
    annotate.add_argument("--max-new-records", type=int)
    annotate.add_argument("--in-place", action=argparse.BooleanOptionalAction, default=None)
    annotate.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    annotate.add_argument("--cleanup-on-success", action=argparse.BooleanOptionalAction, default=None)
    annotate.add_argument("--generate-report", action=argparse.BooleanOptionalAction, default=None)
    annotate.add_argument("--report", type=Path)
    analyze = sub.add_parser("analyze", help="从已打标轨迹生成统计与HTML，不调用 LLM")
    analyze.add_argument("--input", type=Path, action="append", required=True)
    analyze.add_argument("--pattern", default="*.jsonl")
    analyze.add_argument("--report", type=Path, required=True)
    return parser


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    lengths: list[int] = []
    message_counts: list[int] = []
    chunk_counts: list[int] = []
    truncated = 0
    rows = 0
    for line_number, row in iter_jsonl(args.input):
        messages = normalize_messages(row, args.max_tool_chars)
        chunks = chunk_messages(messages, args.chunk_chars)
        lengths.append(sum(len(json_text(message)) for message in messages))
        message_counts.append(len(row["messages"]))
        chunk_counts.append(len(chunks))
        truncated += sum(bool(m.get("content_truncated")) for m in messages)
        rows += 1
        if args.limit and rows >= args.limit:
            break
    if not rows:
        raise ScoringError("输入文件没有轨迹")
    return {
        "input": str(args.input.resolve()),
        "trajectory_count": rows,
        "message_count_total": sum(message_counts),
        "message_count_min_median_max": [min(message_counts), statistics.median(message_counts), max(message_counts)],
        "normalized_chars_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)],
        "chunks_total": sum(chunk_counts),
        "chunks_min_median_max": [min(chunk_counts), statistics.median(chunk_counts), max(chunk_counts)],
        "truncated_tool_messages": truncated,
        "chunk_chars": args.chunk_chars,
        "max_tool_chars": args.max_tool_chars,
        "api_called": False,
    }


def _completed(path: Path) -> dict[int, str]:
    completed: dict[int, str] = {}
    if not path.exists():
        return completed
    for line_number, row in iter_jsonl_output(path):
        source_line = row.get("line_number")
        digest = row.get("trace_sha256")
        if not isinstance(source_line, int) or not isinstance(digest, str):
            raise ScoringError(f"输出第 {line_number} 行缺少行号或轨迹哈希")
        if source_line in completed and completed[source_line] != digest:
            raise ScoringError(f"输出中输入行 {source_line} 有冲突的轨迹哈希")
        completed[source_line] = digest
    return completed


def iter_jsonl_output(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"已有输出第 {line_number} 行不是合法 JSON；请先修复输出文件") from exc
            if not isinstance(row, dict):
                raise ScoringError(f"已有输出第 {line_number} 行不是对象")
            yield line_number, row


def score(args: argparse.Namespace) -> dict[str, Any]:
    config = LLMConfig.from_environment(load_dotenv(args.env))
    if args.direct_connection:
        config.direct_connection = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output.parent / ".trajectory_score_cache"
    completed = _completed(args.output)
    errors_path = args.output.with_suffix(args.output.suffix + ".errors.jsonl")
    processed = skipped = failed = 0

    def pending_rows():
        nonlocal skipped
        selected = 0
        for line_number, row in iter_jsonl(args.input):
            digest = trace_digest(normalize_messages(row, args.max_tool_chars))
            if line_number in completed:
                if completed[line_number] != digest:
                    raise ScoringError(
                        f"输入第 {line_number} 行内容与已有输出不同；请换一个输出文件，避免结果混合"
                    )
                skipped += 1
                continue
            if args.limit and selected >= args.limit:
                break
            selected += 1
            yield line_number, row

    def score_row(line_number: int, row: dict[str, Any]) -> dict[str, Any]:
        pipeline = ScoringPipeline(
            ChatCompletionsClient(config),
            model_name=config.model_name,
            cache_dir=cache_dir,
            chunk_chars=args.chunk_chars,
            max_tool_chars=args.max_tool_chars,
            json_retries=args.json_retries,
        )
        return pipeline.score_row(line_number, row)

    with args.output.open("a", encoding="utf-8") as output, errors_path.open("a", encoding="utf-8") as errors:
        rows = iter(pending_rows())
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            ready = {}
            submitted = next_to_write = 0
            exhausted = False
            stop_scheduling = False

            def fill_workers():
                nonlocal submitted, exhausted
                while not exhausted and len(futures) < args.workers:
                    item = next(rows, None)
                    if item is None:
                        exhausted = True
                        break
                    line_number, row = item
                    future = executor.submit(score_row, line_number, row)
                    futures[future] = (submitted, line_number)
                    submitted += 1

            fill_workers()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    position, line_number = futures.pop(future)
                    try:
                        ready[position] = (line_number, future.result(), None)
                    except Exception as exc:
                        ready[position] = (line_number, None, exc)
                        if args.fail_fast:
                            stop_scheduling = True

                while next_to_write in ready:
                    line_number, result, error = ready.pop(next_to_write)
                    if error is None:
                        output.write(json_text(result) + "\n")
                        output.flush()
                        processed += 1
                        print(
                            f"line {line_number}: action_score={result['action_score']} "
                            f"obligations={len(result['obligations'])}",
                            file=sys.stderr,
                        )
                    else:
                        failed += 1
                        errors.write(json_text({"line_number": line_number, "error": str(error)}) + "\n")
                        errors.flush()
                        print(f"line {line_number}: ERROR {error}", file=sys.stderr)
                        if args.fail_fast:
                            for future in futures:
                                future.cancel()
                            raise error
                    next_to_write += 1
                if not stop_scheduling:
                    fill_workers()
    return {
        "processed": processed,
        "skipped_existing": skipped,
        "failed": failed,
        "workers": args.workers,
        "output": str(args.output.resolve()),
        "errors": str(errors_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
    }


CATEGORY_LABELS = {
    "C1": "交付与可访问性",
    "C2": "内容完整与来源一致",
    "C3": "结构与格式约束",
    "C4": "计算与规则执行",
    "C5": "视觉呈现",
    "C6": "功能与运行",
    "C7": "引用与证据",
    "C8": "外部动作状态",
    "C9": "跨交付物一致性",
}


def report(args: argparse.Namespace) -> str:
    if args.line < 1:
        raise ScoringError("--line 须为正整数")
    with args.input.open("r", encoding="utf-8-sig") as stream:
        first_line = next((line for line in stream if line.strip()), "")
    if first_line.strip() == "{":
        try:
            old_single_result = json.loads(args.input.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ScoringError("多行 JSON 评分文件内容无效") from exc
        if not isinstance(old_single_result, dict):
            raise ScoringError("多行 JSON 评分文件不是对象")
        rows = [old_single_result]
    else:
        rows = (row for _, row in iter_jsonl_output(args.input))
    matches = [row for row in rows if row.get("line_number") == args.line]
    if not matches:
        raise ScoringError(f"评分文件中没有原始输入第 {args.line} 行的结果")
    if len(matches) != 1:
        raise ScoringError(f"评分文件中原始输入第 {args.line} 行有多条结果")
    row = matches[0]
    obligations = row.get("obligations")
    scores = row.get("obligation_scores")
    if not isinstance(obligations, list) or not isinstance(scores, list):
        raise ScoringError("评分结果缺少 obligations 或 obligation_scores")
    score_by_id = {item["obligation_id"]: item for item in scores}
    if len(score_by_id) != len(scores) or set(score_by_id) != {item["id"] for item in obligations}:
        raise ScoringError("评分结果的义务 ID 与逐项评分不一致")
    categories = compute_category_scores(obligations, scores)

    lines = [f"原始输入第 {args.line} 行 | 总动作分: {row.get('action_score')} / 100"]
    for item in categories:
        label = CATEGORY_LABELS[item["category"]]
        if not item["obligation_count"]:
            lines.append(f"{item['category']} {label}: 不适用（0 项义务）")
            continue
        lines.append(
            f"{item['category']} {label}: 动作分 {item['action_score']} / 100；"
            f"义务 {item['obligation_count']} 项；"
            f"充分检查率 {item['adequate_check_rate'] * 100:.1f}%；"
            f"最终版本复验率 {item['final_version_check_rate'] * 100:.1f}%"
        )
        for obligation in (o for o in obligations if o["category"] == item["category"]):
            scored = score_by_id[obligation["id"]]
            refs = ", ".join(str(index) for index in scored.get("supporting_message_indices", [])) or "无"
            lines.append(f"  {obligation['id']} {scored['score']}/4：{obligation['target']}")
            lines.append(f"    证据消息: {refs}；理由: {scored.get('reason', '')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "annotate":
            from .batch import run_batch
            from .config import RunConfig, load_config
            config = load_config(args.config) if args.config else RunConfig()
            for source, target in (("input", "inputs"), ("output_dir", "output_dir"), ("pattern", "pattern"),
                ("env", "env_file"), ("workers", "workers"), ("max_attempts", "max_attempts"),
                ("json_retries", "json_retries"), ("chunk_chars", "chunk_chars"), ("max_tool_chars", "max_tool_chars"),
                ("max_new_records", "max_new_records"), ("in_place", "in_place"), ("progress", "progress"),
                ("cleanup_on_success", "cleanup_on_success"), ("generate_report", "generate_report"), ("report", "report_path")):
                if getattr(args, source) is not None:
                    setattr(config, target, getattr(args, source))
            result = run_batch(config)
            return 130 if result["interrupted"] else (0 if result["success"] == result["total"] else 1)
        if args.command == "analyze":
            from .records import atomic_json, discover
            from .reporting import collect, render_report, print_summary
            paths = discover(args.input, args.pattern)
            if args.report.resolve() in paths:
                raise ScoringError("报告路径不能覆盖输入")
            stats, items = collect(paths)
            render_report(items, stats, args.report)
            stats["report"] = str(args.report.resolve())
            summary_path = args.report.with_suffix(".summary.json")
            if summary_path.resolve() in paths:
                raise ScoringError("统计路径不能覆盖输入")
            atomic_json(summary_path, stats)
            print_summary(stats)
            return 0
        if args.command in {"preflight", "score"} and (args.chunk_chars < 2000 or args.max_tool_chars < 0 or args.limit < 0):
            raise ScoringError("chunk-chars 须 >= 2000；max-tool-chars 和 limit 须 >= 0")
        if args.command == "score" and args.workers < 1:
            raise ScoringError("workers 须 >= 1")
        if not args.input.is_file():
            raise ScoringError(f"输入文件不存在: {args.input}")
        if args.command == "report":
            print(report(args))
            return 0
        result = preflight(args) if args.command == "preflight" else score(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result.get("failed") else 1
    except (ScoringError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
