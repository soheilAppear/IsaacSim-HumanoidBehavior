# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, ordered CSV disk writes with no simulator access from the worker.

The caller owns submission and result collection. Each accepted batch captures
its destination, schema, and immutable scalar cells, so resetting the simulator cannot
redirect pending rows into a newer session. Only disk I/O runs on the worker.
"""

from __future__ import annotations

import csv
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

CsvScalar = str | int | float | bool | None


@dataclass(frozen=True)
class CsvBatch:
    """Immutable rows retained in full until their write result is consumed."""

    path: Path
    fieldnames: tuple[str, ...]
    rows: tuple[tuple[object, ...] | Mapping[str, CsvScalar], ...]
    mapping_rows: bool = False

    @classmethod
    def snapshot(cls, path: Path, records: list[dict], fieldnames: list[str]) -> CsvBatch:
        """Detach scalar CSV data before handing it to a different thread."""
        names = tuple(fieldnames)

        # Keep ordinary telemetry values native: csv.writer performs their text
        # formatting on the worker. Unknown objects are frozen to text here so
        # mutable data or simulator-backed values cannot cross the boundary.
        def freeze(value):
            return value if value is None or type(value) in (str, int, float, bool) else str(value)

        rows = tuple(tuple(freeze(record.get(name, "")) for name in names) for record in records)
        return cls(Path(path), names, rows)

    @classmethod
    def snapshot_scalar_records(
        cls, path: Path, records: list[dict[str, CsvScalar]], fieldnames: list[str]
    ) -> CsvBatch:
        """Copy audited telemetry dictionaries without a Python loop per cell.

        Contract: keys are strings and every value is exactly a native immutable
        ``str/int/float/bool/None``. The five Humanoid recording collectors already
        construct that schema; no Gf, NumPy, sensor, or mutable cell may be passed.
        Other callers must use ``snapshot``, which freezes unknown values. Each
        copied dictionary has a read-only view and no remaining caller reference.
        Field selection and text formatting then happen only on the worker.
        """
        return cls(Path(path), tuple(fieldnames), tuple(MappingProxyType(record.copy()) for record in records), True)


@dataclass(frozen=True)
class CsvWriteResult:
    """A successful append, or a complete failed batch ready for recovery."""

    batch: CsvBatch
    error: str | None = None


class BehavioralCsvWriter:
    """Serialize a bounded number of CSV batches on one background thread.

    ``enqueue`` and ``collect_completed`` never wait for disk. ``close`` stops
    submission and drains accepted writes; call it only during session teardown,
    outside physics callbacks. Failures include the complete immutable batch.
    Once a path fails, later accepted batches for that path are retained as
    failures too, preserving row order for explicit recovery.
    """

    def __init__(self, max_pending: int = 8):
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("CSV queue capacity must be a positive integer")
        self._max_pending = max_pending
        self._pending: list[Future] = []
        self._staged: list[CsvBatch] = []
        self._grouping = False
        self._submission_failure: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="G1BehavioralCsv")
        self._closed = False
        # Accessed only by the single worker; no simulator or caller data lives here.
        self._failed_paths: set[Path] = set()

    @property
    def pending_count(self) -> int:
        """Count accepted batches, including completed results awaiting collection."""
        return len(self._pending) + len(self._staged)

    @contextmanager
    def submission_group(self):
        """Snapshot a flush completely before its worker starts formatting rows."""
        if self._closed or self._grouping:
            raise RuntimeError("CSV submission group requires an open, ungrouped writer")
        self._grouping = True
        try:
            yield
        finally:
            self._grouping = False
            # Staging already transferred ownership from the caller. Remove the
            # staged list first so a partially successful submission can never
            # resubmit its prefix. Every batch receives exactly one result slot.
            staged, self._staged = self._staged, []
            for batch in staged:
                self._submit(batch)

    def _submit(self, batch: CsvBatch) -> None:
        """Retain submission errors exactly like disk errors, without losing rows."""
        future = None
        if self._submission_failure is None:
            try:
                future = self._executor.submit(self._write, batch)
            except Exception as error:
                # Stop submitting after a failure: otherwise later rows might
                # reach disk before a rejected earlier batch can be recovered.
                self._submission_failure = f"CSV worker submission failed: {error}"
        if future is None:
            future = Future()
            future.set_result(CsvWriteResult(batch, self._submission_failure))
        self._pending.append(future)

    def _accept(self, batch: CsvBatch) -> None:
        if self._grouping:
            self._staged.append(batch)
        else:
            self._submit(batch)

    def enqueue(self, path: Path, records: list[dict], fieldnames: list[str]) -> bool:
        """Copy and accept a batch, or leave ownership with the caller if full."""
        if self._closed or self.pending_count >= self._max_pending:
            return False
        self._accept(CsvBatch.snapshot(path, records, fieldnames))
        return True

    def enqueue_scalar_records(self, path: Path, records: list[dict[str, CsvScalar]], fieldnames: list[str]) -> bool:
        """Accept only the audited native-scalar schema described by the snapshot API."""
        if self._closed or self.pending_count >= self._max_pending:
            return False
        self._accept(CsvBatch.snapshot_scalar_records(path, records, fieldnames))
        return True

    def _write(self, batch: CsvBatch) -> CsvWriteResult:
        if batch.path in self._failed_paths:
            return CsvWriteResult(batch, "An earlier batch for this file failed; later rows retained in order")
        try:
            self._append(batch)
        except Exception as error:
            self._failed_paths.add(batch.path)
            return CsvWriteResult(batch, str(error))
        return CsvWriteResult(batch)

    @staticmethod
    def _append(batch: CsvBatch) -> None:
        """Append with rollback so a recoverable failure cannot leave partial rows."""
        previous_size = batch.path.stat().st_size if batch.path.exists() else 0
        try:
            with open(batch.path, "a", newline="") as handle:
                if batch.mapping_rows:
                    writer = csv.DictWriter(handle, fieldnames=batch.fieldnames, restval="", extrasaction="ignore")
                    if previous_size == 0:
                        writer.writeheader()
                else:
                    writer = csv.writer(handle)
                    if previous_size == 0:
                        writer.writerow(batch.fieldnames)
                writer.writerows(batch.rows)
                handle.flush()
        except Exception as error:
            try:
                if batch.path.exists():
                    with open(batch.path, "r+b") as handle:
                        handle.truncate(previous_size)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"CSV append failed: {error}; rollback failed: {rollback_error}; inspect partial file before recovery"
                ) from error
            raise

    def collect_completed(self) -> list[CsvWriteResult]:
        """Release completed queue slots in submission order without waiting."""
        results = []
        while self._pending and self._pending[0].done():
            results.append(self._pending.pop(0).result())
        return results

    def close(self) -> list[CsvWriteResult]:
        """Drain accepted jobs and stop the worker before changing session state."""
        if self._grouping:
            raise RuntimeError("Finish the CSV submission group before closing")
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True)
        return self.collect_completed()
