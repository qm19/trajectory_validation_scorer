from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from filelock import FileLock

from test_core import StructuralFakeClient
from trajectory_scorer.batch import run_batch, cleanup_state
from trajectory_scorer.config import RunConfig
from trajectory_scorer.core import LLMConfig, ScoringError
from trajectory_scorer.records import get_score, read_lines
from trajectory_scorer.reporting import collect, render_report


class BatchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source" / "model_a" / "traces.jsonl"
        self.source.parent.mkdir(parents=True)
        self.rows = [{"messages":[{"role":"user","content":f"task {i}"}],
                      "tools":[{"name":"source-tool"}],
                      "meta_info":{"teacher":"model_a","unique_info":{"custom":f"keep-{i}"}},
                      "extra":{"value":i}} for i in range(3)]
        self.write_source()
        self.llm = LLMConfig("https://example.invalid/v1", "test-key", "fake")
        self.config = RunConfig(inputs=[self.source],output_dir=self.root/"out",workers=2,
            retry_delay_seconds=0,progress=False,print_summary=False)

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self):
        self.source.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in self.rows),encoding="utf-8")

    def run_fake(self, factory=None):
        return run_batch(self.config,llm_config=self.llm,
                         client_factory=factory or (lambda _:StructuralFakeClient()))

    def results(self):
        return [r for _,_,r in read_lines(Path(self.config.output_dir)/"traces.annotated.jsonl") if r]

    def test_preserves_records_and_reuses_final_annotations_after_state_cleanup(self):
        original = self.source.read_bytes()
        stats = self.run_fake()
        self.assertEqual((stats["success"],stats["failed"],stats["pending"]),(3,0,0))
        self.assertTrue(stats["state_cleaned"])
        self.assertFalse((Path(self.config.output_dir)/".state").exists())
        self.assertEqual(self.source.read_bytes(),original)
        for before,after in zip(self.rows,self.results()):
            self.assertEqual(get_score(after)["status"],"success")
            self.assertEqual(get_score(after)["action_score"],0.0)
            del after["meta_info"]["unique_info"]["verifi_action_score"]
            self.assertEqual(after,before)
        self.assertTrue(Path(stats["report"]).is_file())
        again=self.run_fake(lambda _:self.fail("resume should not instantiate a client"))
        self.assertEqual(again["reused_this_run"],3)
        self.assertEqual(again["new_completion_calls_this_run"],0)
        self.assertTrue(again["state_cleaned"])

    def test_staged_run_keeps_pending_rows_and_resume_finishes_them(self):
        self.config.max_new_records=1
        first=self.run_fake()
        self.assertEqual((first["success"],first["pending"]),(1,2))
        self.assertFalse(first["state_cleaned"])
        self.assertTrue(Path(first["state_dir"]).exists())
        self.assertEqual([get_score(r)["status"] for r in self.results()],["success","pending","pending"])
        self.config.max_new_records=0
        final=self.run_fake()
        self.assertEqual(final["reused_this_run"],1)
        self.assertEqual(final["success"],3)
        self.assertTrue(final["state_cleaned"])

    def test_trajectory_retry_reuses_successful_stages(self):
        self.rows=self.rows[:1];self.write_source()
        attempts={"clients":0,"a_calls":0,"fail_once":True}
        class Flaky(StructuralFakeClient):
            def complete(self,system,user):
                if "义务设计员" in system: attempts["a_calls"]+=1
                if "证据提取员" in system and attempts["fail_once"]:
                    attempts["fail_once"]=False
                    raise ScoringError("injected transient network error")
                return super().complete(system,user)
        def factory(_):
            attempts["clients"]+=1
            return Flaky()
        stats=self.run_fake(factory)
        self.assertEqual(stats["success"],1)
        self.assertEqual(attempts["clients"],2)
        self.assertEqual(attempts["a_calls"],1)
        self.assertEqual(get_score(self.results()[0])["attempts_total"],2)

    def test_failed_records_are_null_not_zero_and_retry_on_next_invocation(self):
        self.config.max_attempts=1
        class Fail(StructuralFakeClient):
            def complete(self,system,user):
                if 'task 1' in user and "义务设计员" in system:raise ScoringError("injected failure")
                return super().complete(system,user)
        first=self.run_fake(lambda _:Fail())
        self.assertEqual((first["success"],first["failed"],first["scored_count"]),(2,1,2))
        self.assertIsNone(get_score(self.results()[1])["action_score"])
        self.assertFalse(first["state_cleaned"])
        final=self.run_fake()
        self.assertEqual((final["success"],final["reused_this_run"]),(3,2))
        self.assertEqual(get_score(self.results()[1])["attempts_total"],2)

    def test_message_or_profile_change_invalidates_resume(self):
        self.run_fake()
        self.rows[1]["messages"][0]["content"]="modified task"
        self.write_source()
        changed=self.run_fake()
        self.assertEqual(changed["reused_this_run"],2)
        self.config.chunk_chars=45000
        changed_profile=self.run_fake()
        self.assertEqual(changed_profile["reused_this_run"],0)

    def test_in_place_backup_and_embedded_resume(self):
        before=self.source.read_bytes()
        self.config.in_place=True
        self.run_fake()
        self.assertEqual(self.source.with_name(self.source.name+".bak").read_bytes(),before)
        rows=[r for _,_,r in read_lines(self.source) if r]
        self.assertTrue(all(get_score(r)["status"]=="success" for r in rows))
        again=self.run_fake(lambda _:self.fail("embedded scores should be reused"))
        self.assertEqual(again["reused_this_run"],3)
        self.assertEqual(self.source.with_name(self.source.name+".bak").read_bytes(),before)

    def test_bad_metadata_fails_before_any_api_call(self):
        self.rows[0]["meta_info"]["unique_info"]="do not overwrite me";self.write_source()
        with self.assertRaisesRegex(ScoringError,"必须为对象"):
            self.run_fake(lambda _:self.fail("preflight must not call API"))

    def test_interruption_keeps_state_and_can_resume(self):
        self.rows=self.rows[:1];self.write_source()
        started=threading.Event()
        class Slow(StructuralFakeClient):
            def complete(self,system,user):
                if "义务设计员" in system:
                    started.set();time.sleep(.05)
                return super().complete(system,user)
        def interrupt(*args,**kwargs):
            self.assertTrue(started.wait(2))
            raise KeyboardInterrupt()
        with patch("trajectory_scorer.batch.wait",side_effect=interrupt):
            partial=self.run_fake(lambda _:Slow())
        self.assertTrue(partial["interrupted"])
        self.assertFalse(partial["state_cleaned"])
        final=self.run_fake()
        self.assertEqual(final["success"],1)
        self.assertTrue(final["state_cleaned"])

    def test_cleanup_refuses_directory_without_ownership_marker(self):
        parent=self.root/"parent";path=parent/"batch";path.mkdir(parents=True)
        (path/"keep.txt").write_text("keep",encoding="utf-8")
        with self.assertRaises(ScoringError):cleanup_state(path,parent,"batch")
        self.assertTrue((path/"keep.txt").exists())

    def test_annotated_report_needs_no_llm_configuration_and_rejects_corruption(self):
        self.run_fake()
        path=Path(self.config.output_dir)/"traces.annotated.jsonl"
        stats,items=collect([path]);render_report(items,stats,self.root/"report.html")
        self.assertEqual(stats["total"],3)
        rows=self.results();get_score(rows[0])["action_score"]=99
        path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        with self.assertRaises(ScoringError):collect([path])

    def test_missing_metadata_and_blank_physical_lines_are_preserved(self):
        rows=[{"messages":[{"role":"user","content":"first"}]},
              {"messages":[{"role":"user","content":"second"}],"custom":True}]
        self.source.write_text(json.dumps(rows[0])+"\n\n"+json.dumps(rows[1])+"\n",encoding="utf-8")
        result=self.run_fake()
        self.assertEqual(result["success"],2)
        output=Path(self.config.output_dir)/"traces.annotated.jsonl"
        records=list(read_lines(output))
        self.assertIsNone(records[1][2])
        self.assertEqual(get_score(records[2][2])["source_line_number"],3)
        self.assertEqual(records[2][2]["messages"],rows[1]["messages"])
        self.assertTrue(records[2][2]["custom"])

    def test_parallel_workers_continue_when_earlier_row_is_slow(self):
        barrier=threading.Barrier(2);third_done=threading.Event();completed=[]
        class Concurrent(StructuralFakeClient):
            def __init__(self):self.kind=None
            def complete(self,system,user):
                if "义务设计员" in system:
                    self.kind=int(json.loads(user)["user_messages"][0]["content"].split()[-1])
                    if self.kind in (0,1):barrier.wait(timeout=3)
                    if self.kind==0 and not third_done.wait(timeout=3):raise AssertionError("worker not refilled")
                value=super().complete(system,user)
                if "校验动作完成度" in system:
                    completed.append(self.kind)
                    if self.kind==2:third_done.set()
                return value
        result=self.run_fake(lambda _:Concurrent())
        self.assertEqual(result["success"],3)
        self.assertEqual(completed,[1,2,0])
        self.assertEqual([get_score(r)["source_line_number"] for r in self.results()],[1,2,3])

    def test_output_directory_lock_prevents_concurrent_batch(self):
        Path(self.config.output_dir).mkdir()
        with FileLock(str(Path(self.config.output_dir)/".annotation.lock")):
            with self.assertRaisesRegex(ScoringError,"已有打标进程"):
                self.run_fake(lambda _:self.fail("locked batch must not call API"))

    def test_preexisting_unowned_state_is_not_adopted_or_deleted(self):
        from trajectory_scorer.core import json_text
        output=Path(self.config.output_dir).resolve()
        run_id=hashlib.sha256(json_text([[str(self.source.resolve())],str(output),False]).encode()).hexdigest()[:24]
        root=output/".state"/run_id;root.mkdir(parents=True)
        (root/"keep.txt").write_text("unrelated",encoding="utf-8")
        with self.assertRaisesRegex(ScoringError,"所有权"):
            self.run_fake(lambda _:self.fail("unowned state must not call API"))
        self.assertEqual((root/"keep.txt").read_text(encoding="utf-8"),"unrelated")

    def test_source_edit_during_scoring_prevents_overwrite(self):
        self.rows=self.rows[:1];self.write_source()
        source=self.source
        class Editing(StructuralFakeClient):
            def complete(self,system,user):
                if "校验动作完成度" in system:
                    source.write_text('{"messages":[{"role":"user","content":"external edit"}]}\n',encoding="utf-8")
                return super().complete(system,user)
        with self.assertRaisesRegex(ScoringError,"被修改"):
            self.run_fake(lambda _:Editing())
        self.assertIn("external edit",source.read_text(encoding="utf-8"))
        self.assertFalse((Path(self.config.output_dir)/"traces.annotated.jsonl").exists())


if __name__=="__main__":unittest.main()
