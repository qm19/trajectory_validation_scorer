from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from trajectory_scorer.core import (
    ChatCompletionsClient,
    compute_category_scores,
    compute_summary,
    LLMConfig,
    ScoringError,
    ScoringPipeline,
    chunk_messages,
    iter_jsonl,
    normalize_messages,
    validate_obligations,
    validate_events,
    validate_scores,
)
from trajectory_scorer.cli import build_parser, report as cli_report, score as cli_score


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "test"
    / "input"
    / "outputs-opus5-260818-115_converted_final_reflected_filtered.jsonl"
)


class FakeClient:
    """Exercises the full pipeline without pretending to perform semantic grading."""

    def complete(self, system: str, user: str) -> str:
        if "校验动作义务设计员" in system:
            return json.dumps(
                {
                    "final_deliverables": [
                        {"id": "D1", "description": "Equipment_Purchase_Schedule_2026-08.md", "source_message_indices": [1]}
                    ],
                    "requirement_updates": [
                        {"message_index": 21, "effect": "clarify", "description": "八列且顺序固定"}
                    ],
                    "obligations": [
                        {
                            "id": "O1",
                            "deliverable_ids": ["D1"],
                            "category": "C3",
                            "target": "任务板八列",
                            "sufficient_check": "读取最终文件，解析所有任务板表头和行，判读是否为指定八列；发现不符也算完成检查",
                            "source_message_indices": [1, 21],
                            "origin": "explicit",
                            "covered_requirements": ["八列及顺序"],
                        }
                    ],
                    "uncovered_requirements": [],
                },
                ensure_ascii=False,
            )
        if "校验动作证据提取员" in system:
            chunk = json.loads(user)["trace_chunk"]
            indices = {unit["message_index"] for unit in chunk}
            events = []
            if 32 in indices:
                events.append(
                    {
                        "type": "check",
                        "message_indices": [32],
                        "artifact": "D1",
                        "obligation_ids": ["O1"],
                        "method": "最终文件表格列数检查",
                        "observed_result": "所有五个表格为指定八列",
                        "result_visibility": "visible",
                        "issue_found": None,
                        "limitations": None,
                    }
                )
            return json.dumps({"events": events}, ensure_ascii=False)
        if "校验动作完成度" in system:
            return json.dumps(
                {
                    "obligation_scores": [
                        {
                            "obligation_id": "O1",
                            "score": 3,
                            "supporting_message_indices": [32],
                            "final_version_checked": True,
                            "observed_outcome": "pass",
                            "reason": "最终表格在修改后逐一核对了列数和顺序",
                            "evidence_limitations": None,
                        }
                    ],
                    "obligation_review_flags": [],
                    "trajectory_evidence_limitations": [],
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected stage")


class StructuralFakeClient:
    """Cheap schema-only responses for all 108 rows; scores have no semantic meaning."""

    def complete(self, system: str, user: str) -> str:
        if "校验动作义务设计员" in system:
            first_user = json.loads(user)["user_messages"][0]["message_index"]
            return json.dumps(
                {
                    "final_deliverables": [
                        {"id": "D1", "description": "offline smoke placeholder", "source_message_indices": [first_user]}
                    ],
                    "requirement_updates": [],
                    "obligations": [
                        {
                            "id": "O1", "deliverable_ids": ["D1"], "category": "C1",
                            "target": "offline smoke", "sufficient_check": "offline smoke",
                            "source_message_indices": [first_user], "origin": "explicit",
                            "covered_requirements": ["offline smoke"],
                        }
                    ],
                    "uncovered_requirements": [],
                },
                ensure_ascii=False,
            )
        if "校验动作证据提取员" in system:
            return '{"events":[]}'
        if "校验动作完成度" in system:
            return json.dumps(
                {
                    "obligation_scores": [
                        {
                            "obligation_id": "O1", "score": 0, "supporting_message_indices": [],
                            "final_version_checked": False, "reason": "offline smoke only",
                            "observed_outcome": "not_observed",
                            "evidence_limitations": "not semantically scored",
                        }
                    ],
                    "obligation_review_flags": [],
                    "trajectory_evidence_limitations": ["offline smoke only"],
                }
            )
        raise AssertionError("unexpected stage")


class TrajectoryScorerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(iter_jsonl(DATA_FILE)) if DATA_FILE.is_file() else []

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_all_real_trajectories_are_normalized_and_chunked_without_missing_indices(self):
        self.assertEqual(len(self.rows), 108)
        for line_number, row in self.rows:
            normalized = normalize_messages(row)
            chunks = chunk_messages(normalized)
            observed = {
                unit["message_index"]
                for chunk in chunks
                for unit in json.loads(chunk)
            }
            self.assertEqual(observed, {m["message_index"] for m in normalized}, line_number)
            self.assertTrue(all(len(chunk) <= 100000 for chunk in chunks), line_number)

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_long_tool_output_is_marked_when_truncated(self):
        row = next(row for _, row in self.rows if any(
            m.get("role") == "tool" and len(m.get("content", "")) > 16000
            for m in row["messages"]
        ))
        messages = normalize_messages(row, max_tool_chars=16000)
        truncated = [m for m in messages if m.get("content_truncated")]
        self.assertTrue(truncated)
        self.assertIn("TOOL_CONTENT_TRUNCATED", truncated[0]["content"])
        self.assertGreater(truncated[0]["original_content_chars"], 16000)

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_full_pipeline_on_real_long_trace_with_fake_client(self):
        _, row = self.rows[90]
        with tempfile.TemporaryDirectory() as temp:
            pipeline = ScoringPipeline(FakeClient(), "fake", Path(temp))
            result = pipeline.score_row(91, row)
            self.assertEqual(result["message_count"], 34)
            self.assertGreaterEqual(result["chunk_count"], 1)
            self.assertEqual(result["action_score"], 75.0)
            self.assertEqual(result["llm_call_stats"]["completion_calls"], result["chunk_count"] + 2)
            self.assertGreater(result["llm_call_stats"]["input_chars"], 0)
            self.assertEqual(result["category_scores"][2]["action_score"], 75.0)
            self.assertIsNone(result["category_scores"][0]["action_score"])
            self.assertEqual(result["obligation_scores"][0]["supporting_message_indices"], [32])
            # Cache replay exercises validated resume without another model call.
            replay = pipeline.score_row(91, row)
            self.assertEqual(replay["action_score"], 75.0)
            self.assertEqual(replay["llm_call_stats"]["completion_calls"], 0)

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_llm_payloads_use_compact_obligations_and_single_encoded_trace(self):
        class RecordingClient(FakeClient):
            def __init__(self):
                self.calls = []

            def complete(self, system, user):
                self.calls.append((system, json.loads(user)))
                return super().complete(system, user)

        _, row = self.rows[90]
        with tempfile.TemporaryDirectory() as temp:
            client = RecordingClient()
            result = ScoringPipeline(client, "fake", Path(temp)).score_row(91, row)
        self.assertEqual(len(client.calls), result["chunk_count"] + 2)
        evidence_payloads = [payload for system, payload in client.calls if "校验动作证据提取员" in system]
        self.assertTrue(evidence_payloads)
        self.assertIsInstance(evidence_payloads[0]["trace_chunk"], list)
        self.assertEqual(
            set(evidence_payloads[0]["obligations"][0]),
            {"id", "deliverable_ids", "category", "target", "sufficient_check"},
        )
        score_payload = client.calls[-1][1]
        self.assertIsInstance(score_payload["obligations"], list)
        self.assertEqual(score_payload["obligations"][0]["eligible_check_message_indices"], [32])

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_full_pipeline_accepts_all_108_real_rows_with_schema_only_fake(self):
        with tempfile.TemporaryDirectory() as temp:
            pipeline = ScoringPipeline(StructuralFakeClient(), "offline-fake", Path(temp))
            results = [pipeline.score_row(number, row) for number, row in self.rows]
        self.assertEqual(len(results), 108)
        self.assertEqual(sum(result["chunk_count"] for result in results), 215)
        self.assertTrue(all(result["action_score"] == 0.0 for result in results))

    def test_fabricated_evidence_index_is_rejected(self):
        obligations = {"obligations": [{"id": "O1"}]}
        events = [
            {"type": "check", "message_indices": [32], "obligation_ids": ["O1"]}
        ]
        bad_score = {
            "obligation_scores": [
                {
                    "obligation_id": "O1",
                    "score": 3,
                    "supporting_message_indices": [99],
                    "final_version_checked": True,
                    "observed_outcome": "pass",
                    "reason": "wrong reference",
                }
            ]
        }
        with self.assertRaises(ScoringError):
            validate_scores(bad_score, obligations, events)

    def test_evidence_schema_errors_explain_the_required_field_and_allowed_indices(self):
        obligations = {"obligations": [{"id": "O1"}]}
        event = {"type": "check", "message_index": [35, 36],
                 "obligation_ids": ["O1"], "result_visibility": "visible"}
        with self.assertRaisesRegex(ScoringError, "message_indices.*复数"):
            validate_events({"events": [event]}, obligations, {35, 36})
        event["message_indices"] = event.pop("message_index")
        validate_events({"events": [event]}, obligations, {35, 36})
        event["message_indices"] = [35, 59]
        with self.assertRaisesRegex(ScoringError, r"之外.*59.*允许.*35, 36"):
            validate_events({"events": [event]}, obligations, {35, 36})

    def test_failed_check_can_receive_full_action_credit_without_final_recheck(self):
        obligations = {"obligations": [{"id": "O1"}]}
        events = [{"type": "check", "message_indices": [5, 6], "obligation_ids": ["O1"]}]
        scored = {
            "obligation_scores": [{
                "obligation_id": "O1", "score": 3, "supporting_message_indices": [5, 6],
                "final_version_checked": False, "observed_outcome": "fail",
                "reason": "检查发现要求未满足，但检查动作完整",
            }]
        }
        validate_scores(scored, obligations, events)
        summary = compute_summary(scored["obligation_scores"])
        self.assertEqual(summary["action_score"], 75.0)
        self.assertEqual(summary["final_version_check_rate"], 0.0)

    def test_category_score_uses_only_matching_obligations_and_marks_absent_categories(self):
        obligations = [
            {"id": "O1", "category": "C1"},
            {"id": "O2", "category": "C1"},
            {"id": "O3", "category": "C3"},
        ]
        scores = [
            {"obligation_id": "O1", "score": 4, "final_version_checked": True},
            {"obligation_id": "O2", "score": 2, "final_version_checked": False},
            {"obligation_id": "O3", "score": 0, "final_version_checked": False},
        ]
        categories = compute_category_scores(obligations, scores)
        self.assertEqual(len(categories), 9)
        self.assertEqual(categories[0]["action_score"], 75.0)
        self.assertEqual(categories[0]["obligation_ids"], ["O1", "O2"])
        self.assertEqual(categories[2]["action_score"], 0.0)
        self.assertIsNone(categories[1]["action_score"])
        self.assertEqual(compute_summary(scores)["action_score"], 50.0)

    def test_report_can_read_existing_output_without_category_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output_path = Path(temp) / "old_scores.jsonl"
            output_path.write_text(json.dumps({
                "line_number": 7,
                "action_score": 75.0,
                "obligations": [{"id": "O1", "category": "C1", "target": "检查交付路径"}],
                "obligation_scores": [{
                    "obligation_id": "O1", "score": 3, "final_version_checked": False,
                    "supporting_message_indices": [12], "reason": "读取目录确认路径",
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            args = build_parser().parse_args(["report", "--input", str(output_path), "--line", "7"])
            shown = cli_report(args)
            self.assertIn("C1 交付与可访问性: 动作分 75.0 / 100", shown)
            self.assertIn("C2 内容完整与来源一致: 不适用", shown)
            self.assertIn("证据消息: 12", shown)
            pretty_path = Path(temp) / "old_pretty.jsonl"
            pretty_path.write_text(
                json.dumps(json.loads(output_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pretty_args = build_parser().parse_args(["report", "--input", str(pretty_path), "--line", "7"])
            self.assertEqual(cli_report(pretty_args), shown)

    def test_obligation_category_remains_validated_without_matrix(self):
        value = {
            "final_deliverables": [{"id": "D1", "description": "output", "source_message_indices": [1]}],
            "obligations": [{
                "id": "O1", "category": "invalid", "deliverable_ids": ["D1"],
                "target": "test", "sufficient_check": "read output", "source_message_indices": [1],
            }],
            "uncovered_requirements": [],
        }
        messages = [{"message_index": 1, "role": "user", "content": "task"}]
        with self.assertRaises(ScoringError):
            validate_obligations(value, messages)
        value["obligations"][0]["category"] = "C2"
        validate_obligations(value, messages)

    def test_chat_completions_http_payload_without_real_network(self):
        body = BytesIO(
            b'{"choices":[{"message":{"content":"{\\"events\\":[]}"}}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}'
        )
        config = LLMConfig(url="https://example.invalid/v1", api_key="test-key", model_name="test-model")
        client = ChatCompletionsClient(config)
        with patch("urllib.request.urlopen", return_value=body) as opener:
            result = client.complete("system", "user")
        self.assertEqual(result, '{"events":[]}')
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][1]["content"], "user")
        self.assertIs(payload["extra_body"]["enable_thinking"], False)
        self.assertEqual(
            payload["extra_body"]["google"]["thinking_config"],
            {"include_thoughts": True, "thinking_level": "minimal"},
        )
        self.assertNotIn("google", payload)
        self.assertEqual(client.http_attempt_count, 1)
        self.assertEqual(client.last_usage["total_tokens"], 14)
        self.assertIn("7.156.135.202", os.environ["no_proxy"].split(","))

    @unittest.skipUnless(DATA_FILE.is_file(), "可选私有108条轨迹未提供")
    def test_cli_writes_jsonl_and_resumes_without_real_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            output_path = root / "scores.jsonl"
            env_path = root / ".env"
            with input_path.open("w", encoding="utf-8") as stream:
                for _, row in self.rows[:2]:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            env_path.write_text(
                "LLM_URL=https://example.invalid/v1\nLLM_API_KEY=test-key\nLLM_MODEL_NAME=offline-fake\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["score", "--input", str(input_path), "--output", str(output_path), "--env", str(env_path)]
            )
            with patch("trajectory_scorer.cli.ChatCompletionsClient", return_value=StructuralFakeClient()):
                first = cli_score(args)
                second = cli_score(args)
            self.assertEqual(first["processed"], 2)
            self.assertEqual(second["processed"], 0)
            self.assertEqual(second["skipped_existing"], 2)
            with output_path.open(encoding="utf-8") as stream:
                self.assertEqual(sum(1 for _ in stream), 2)

    def test_cli_scores_rows_concurrently_but_writes_in_input_order(self):
        barrier = threading.Barrier(2)
        next_done = threading.Event()
        completion_order = []

        class ConcurrentFakeClient(StructuralFakeClient):
            def __init__(self):
                self.kind = ""

            def complete(self, system, user):
                if "校验动作义务设计员" in system:
                    self.kind = json.loads(user)["user_messages"][0]["content"].split()[0]
                    if self.kind in {"slow", "fast"}:
                        barrier.wait(timeout=5)
                    if self.kind == "slow" and not next_done.wait(timeout=5):
                        raise AssertionError("next row did not finish while slow row was active")
                response = super().complete(system, user)
                if "校验动作完成度" in system:
                    completion_order.append(self.kind)
                    if self.kind == "next":
                        next_done.set()
                return response

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            output_path = root / "scores.jsonl"
            env_path = root / ".env"
            rows = [
                {"messages": [{"role": "user", "content": "slow task"}]},
                {"messages": [{"role": "user", "content": "fast task"}]},
                {"messages": [{"role": "user", "content": "next task"}]},
            ]
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            env_path.write_text(
                "LLM_URL=https://example.invalid/v1\nLLM_API_KEY=test-key\nLLM_MODEL_NAME=offline-fake\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args([
                "score", "--input", str(input_path), "--output", str(output_path),
                "--env", str(env_path), "--workers", "2",
            ])
            with patch("trajectory_scorer.cli.ChatCompletionsClient", side_effect=lambda _: ConcurrentFakeClient()):
                first = cli_score(args)
                second = cli_score(args)

            self.assertEqual(first["processed"], 3)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(first["workers"], 2)
            self.assertEqual(completion_order, ["fast", "next", "slow"])
            self.assertEqual(second["skipped_existing"], 3)
            self.assertEqual(second["processed"], 0)
            self.assertEqual(
                [json.loads(line)["line_number"] for line in output_path.read_text(encoding="utf-8").splitlines()],
                [1, 2, 3],
            )

    def test_cli_parallel_failure_does_not_block_other_rows_and_can_resume(self):
        class FailingClient(StructuralFakeClient):
            def complete(self, system, user):
                if "校验动作义务设计员" in system:
                    content = json.loads(user)["user_messages"][0]["content"]
                    if content == "fail task":
                        raise ScoringError("injected failure")
                return super().complete(system, user)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            output_path = root / "scores.jsonl"
            env_path = root / ".env"
            input_path.write_text(
                "".join(json.dumps({"messages": [{"role": "user", "content": content}]}) + "\n"
                        for content in ("first task", "fail task", "third task")),
                encoding="utf-8",
            )
            env_path.write_text(
                "LLM_URL=https://example.invalid/v1\nLLM_API_KEY=test-key\nLLM_MODEL_NAME=offline-fake\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args([
                "score", "--input", str(input_path), "--output", str(output_path),
                "--env", str(env_path), "--workers", "2",
            ])
            with patch("trajectory_scorer.cli.ChatCompletionsClient", side_effect=lambda _: FailingClient()):
                first = cli_score(args)
            self.assertEqual((first["processed"], first["failed"]), (2, 1))
            self.assertEqual(
                [json.loads(line)["line_number"] for line in output_path.read_text(encoding="utf-8").splitlines()],
                [1, 3],
            )
            error_path = output_path.with_suffix(".jsonl.errors.jsonl")
            self.assertEqual(json.loads(error_path.read_text(encoding="utf-8").splitlines()[0])["line_number"], 2)

            with patch("trajectory_scorer.cli.ChatCompletionsClient", side_effect=lambda _: StructuralFakeClient()):
                second = cli_score(args)
            self.assertEqual((second["processed"], second["skipped_existing"], second["failed"]), (1, 2, 0))
            self.assertEqual(
                sorted(json.loads(line)["line_number"] for line in output_path.read_text(encoding="utf-8").splitlines()),
                [1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
