# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real disk writes, backpressure, failure recovery, and session isolation."""

from __future__ import annotations

import csv
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from humanoid_test_support import CSV_WRITER, HUMANOID


class TestBehavioralCsvWriterControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def read_rows(self, path):
        with open(path, newline="") as handle:
            return list(csv.reader(handle))

    def test_worker_preserves_order_schema_and_detaches_mutable_source(self):
        started, release = threading.Event(), threading.Event()
        thread_ids = []
        original = CSV_WRITER.BehavioralCsvWriter._append

        def blocked_append(batch):
            thread_ids.append(threading.get_ident())
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release disk worker")
            original(batch)

        path = self.root / "behavior.csv"
        records = [{"step": 1, "value": 0.5, "ignored": "extra"}, {"step": 2, "value": None}]
        fields = ["step", "value"]
        with patch.object(CSV_WRITER.BehavioralCsvWriter, "_append", staticmethod(blocked_append)):
            writer = CSV_WRITER.BehavioralCsvWriter()
            try:
                self.assertTrue(writer.enqueue(path, records, fields))
                self.assertTrue(started.wait(2))
                records[0]["step"] = 900
                records.clear()
                fields.append("late_column")
                self.assertTrue(writer.enqueue(path, [{"step": 3}], ["step", "value"]))
            finally:
                release.set()
                results = writer.close()
        self.assertEqual([result.error for result in results], [None, None])
        self.assertEqual(self.read_rows(path), [["step", "value"], ["1", "0.5"], ["2", ""], ["3", ""]])
        self.assertEqual(len(set(thread_ids)), 1)
        self.assertNotEqual(thread_ids[0], threading.get_ident())
        self.assertEqual(writer.close(), [])
        self.assertFalse(writer.enqueue(path, [{"step": 4}], ["step"]))

    def test_queue_is_bounded_and_full_queue_leaves_records_with_caller(self):
        writer = CSV_WRITER.BehavioralCsvWriter(max_pending=1)
        self.addCleanup(writer.close)
        records = [{"step": 2}]
        self.assertTrue(writer.enqueue(self.root / "one.csv", [{"step": 1}], ["step"]))
        self.assertFalse(writer.enqueue(self.root / "two.csv", records, ["step"]))
        self.assertEqual(records, [{"step": 2}])
        self.assertEqual(writer.pending_count, 1)
        writer.close()
        self.assertFalse((self.root / "two.csv").exists())

    def test_snapshot_keeps_native_scalars_and_freezes_unknown_mutable_values(self):
        mutable = [1, 2]
        batch = CSV_WRITER.CsvBatch.snapshot(
            self.root / "data.csv",
            [{"i": 1, "f": 0.5, "b": True, "n": None, "other": mutable}],
            ["i", "f", "b", "n", "other"],
        )
        mutable.append(3)
        self.assertEqual(batch.rows, ((1, 0.5, True, None, "[1, 2]"),))
        self.assertIs(type(batch.rows[0][0]), int)
        self.assertIs(type(batch.rows[0][1]), float)

    def test_audited_scalar_snapshot_owns_readonly_mappings_and_keeps_original_values(self):
        records = [{"step": 1, "value": 0.5, "flag": True, "optional": None}]
        fields = ["step", "value", "flag", "optional"]
        batch = CSV_WRITER.CsvBatch.snapshot_scalar_records(self.root / "data.csv", records, fields)
        records[0]["step"] = 99
        records.clear()
        fields.clear()
        self.assertTrue(batch.mapping_rows)
        self.assertEqual(dict(batch.rows[0]), {"step": 1, "value": 0.5, "flag": True, "optional": None})
        self.assertEqual(batch.fieldnames, ("step", "value", "flag", "optional"))
        with self.assertRaises(TypeError):
            batch.rows[0]["step"] = 2

    def test_group_defers_worker_until_all_snapshots_and_enforces_shared_capacity(self):
        writer = CSV_WRITER.BehavioralCsvWriter(max_pending=2)
        self.addCleanup(writer.close)
        path = self.root / "data.csv"
        with writer.submission_group():
            self.assertTrue(writer.enqueue_scalar_records(path, [{"step": 1, "late": "ignored"}], ["step"]))
            self.assertTrue(writer.enqueue(path, [{"step": 2}], ["step"]))
            self.assertEqual(writer.pending_count, 2)
            self.assertFalse(writer.enqueue_scalar_records(path, [{"step": 3}], ["step"]))
            # No Future has been submitted, so no worker can touch either file.
            self.assertEqual(writer._pending, [])
            self.assertFalse(path.exists())
        results = writer.close()
        self.assertTrue(all(result.error is None for result in results))
        self.assertEqual(self.read_rows(path), [["step"], ["1"], ["2"]])

    def test_group_submit_failure_retains_rejected_suffix_once_and_latches_future_submissions(self):
        writer = CSV_WRITER.BehavioralCsvWriter()
        self.addCleanup(writer.close)
        path = self.root / "data.csv"
        real_submit = writer._executor.submit
        submitted = []

        def fail_second_submit(callback, batch):
            submitted.append(batch)
            if len(submitted) == 2:
                raise RuntimeError("worker cannot accept more jobs")
            return real_submit(callback, batch)

        with patch.object(writer._executor, "submit", side_effect=fail_second_submit):
            with writer.submission_group():
                for step in (1, 2, 3):
                    self.assertTrue(writer.enqueue_scalar_records(path, [{"step": step}], ["step"]))
            self.assertEqual(writer._staged, [])
            with writer.submission_group():
                self.assertTrue(writer.enqueue_scalar_records(path, [{"step": 4}], ["step"]))
            results = writer.close()
        self.assertEqual(len(submitted), 2)
        self.assertEqual(len(results), 4)
        self.assertEqual([result.batch.rows[0]["step"] for result in results], [1, 2, 3, 4])
        self.assertIsNone(results[0].error)
        self.assertTrue(all("submission failed" in result.error for result in results[1:]))
        self.assertEqual(self.read_rows(path), [["step"], ["1"]])
        self.assertEqual(writer.close(), [])

    def test_direct_submit_failure_retains_accepted_batch_and_prevents_later_writes(self):
        writer = CSV_WRITER.BehavioralCsvWriter()
        self.addCleanup(writer.close)
        path = self.root / "data.csv"
        with patch.object(writer._executor, "submit", side_effect=RuntimeError("thread creation failed")) as submit:
            self.assertTrue(writer.enqueue(path, [{"step": 1}], ["step"]))
            self.assertTrue(writer.enqueue(path, [{"step": 2}], ["step"]))
            results = writer.close()
        self.assertEqual(submit.call_count, 1)
        self.assertEqual([result.batch.rows for result in results], [((1,),), ((2,),)])
        self.assertTrue(all("thread creation failed" in result.error for result in results))
        self.assertFalse(path.exists())

    def test_partial_append_rolls_back_and_later_same_file_batch_is_retained(self):
        path = self.root / "data.csv"
        before = b"step\r\n0\r\n"
        path.write_bytes(before)
        real_csv_writer = csv.writer

        class FailedWrite:
            def __init__(self, handle):
                self.handle = handle
                self.original = real_csv_writer(handle)

            def writerow(self, row):
                self.original.writerow(row)

            def writerows(self, _rows):
                self.handle.write("partially written row")
                raise OSError("simulated disk full")

        writer = CSV_WRITER.BehavioralCsvWriter()
        with patch.object(CSV_WRITER.csv, "writer", FailedWrite):
            self.assertTrue(writer.enqueue(path, [{"step": 1}], ["step"]))
            self.assertTrue(writer.enqueue(path, [{"step": 2}], ["step"]))
            results = writer.close()
        self.assertEqual(path.read_bytes(), before)
        self.assertIn("simulated disk full", results[0].error)
        self.assertIn("earlier batch", results[1].error)
        self.assertEqual([result.batch.rows for result in results], [((1,),), ((2,),)])

    def test_failed_destination_does_not_redirect_rows_into_other_session(self):
        writer = CSV_WRITER.BehavioralCsvWriter()
        missing = self.root / "missing_session" / "data.csv"
        next_session = self.root / "next.csv"
        writer.enqueue(missing, [{"step": 1}], ["step"])
        writer.enqueue(next_session, [{"step": 2}], ["step"])
        results = writer.close()
        self.assertIsNotNone(results[0].error)
        self.assertEqual(results[0].batch.path, missing)
        self.assertIsNone(results[1].error)
        self.assertEqual(self.read_rows(next_session), [["step"], ["2"]])

    def example(self):
        example = HUMANOID.HumanoidExample()
        example._behavioral_sessions_root = self.root
        example._start_behavioral_session()
        self.addCleanup(example._save_behavioral_data)
        return example

    def test_session_close_drains_pending_and_backpressured_streams_before_restart(self):
        example = self.example()
        first = example._behavioral_session_dir
        example._behavioral_csv_writer = CSV_WRITER.BehavioralCsvWriter(max_pending=1)
        example._behavioral_data_records.append({"step": 1})
        example._hand_tracking_records.append({"step": 1, "side": "right"})
        example._behavioral_frame_records.append({"frame_id": 0, "image_path": "frames/eye_camera/frame_000000.png"})
        example._flush_behavioral_csvs()
        self.assertEqual(example._behavioral_data_records, [])
        self.assertTrue(example._hand_tracking_records)
        # Restart itself must finish the original immutable destinations first.
        example._start_behavioral_session()
        second = example._behavioral_session_dir
        self.assertNotEqual(first, second)
        self.assertEqual(self.read_rows(first / "behavior.csv"), [["step"], ["1"]])
        self.assertEqual(self.read_rows(first / "hand_tracking.csv"), [["step", "side"], ["1", "right"]])
        self.assertEqual(len(self.read_rows(first / "frame_timestamps.csv")), 2)
        self.assertEqual(example._behavioral_csv_failed_batches, [])
        example._behavioral_data_records.append({"step": 2})
        example._save_behavioral_data()
        self.assertEqual(self.read_rows(second / "behavior.csv"), [["step"], ["2"]])
        self.assertEqual(self.read_rows(first / "behavior.csv"), [["step"], ["1"]])

    def test_failed_batches_and_unqueued_tail_survive_session_close(self):
        example = self.example()
        first = example._behavioral_session_dir
        example._behavioral_data_records.append({"step": 1})
        with patch.object(CSV_WRITER.BehavioralCsvWriter, "_append", side_effect=OSError("disk unavailable")):
            example._flush_behavioral_csvs()
            example._collect_behavioral_csv_results(close=True)
            example._behavioral_data_records.append({"step": 2})
            example._save_behavioral_data()
        failed = example._behavioral_csv_failed_batches
        self.assertEqual(dict(failed[0].batch.rows[0]), {"step": 1})
        self.assertEqual(failed[1].batch.rows, ((2,),))
        self.assertTrue(all(item.batch.path == first / "behavior.csv" for item in failed))
        self.assertFalse((first / "behavior.csv").exists())
        example._start_behavioral_session()
        second = example._behavioral_session_dir
        example._behavioral_data_records.append({"step": 3})
        example._save_behavioral_data()
        self.assertEqual(self.read_rows(second / "behavior.csv"), [["step"], ["3"]])
        self.assertEqual(len(example._behavioral_csv_failed_batches), 2)


if __name__ == "__main__":
    unittest.main()
