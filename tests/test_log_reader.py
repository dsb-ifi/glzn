import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from glzn.log import LogCollator
from glzn.log.__main__ import main
from glzn.log.reader import summarize_log


class LogReaderTests(unittest.TestCase):
    def test_summary_uses_recent_records_after_rollback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "log" / "metrics.jsonl"
            with LogCollator.local(run_id="run-a", root=directory, stdout=False) as logger:
                for epoch, iteration, step, timestamp, skipped in (
                    (0, 5, 6, 20.0, False),
                    (0, 6, 7, 30.0, False),
                    (0, 7, 8, 40.0, False),
                    (0, 5, 6, 100.0, False),
                    (0, 6, 6, 110.0, True),
                    (0, 7, 7, 120.0, False),
                ):
                    logger.log_update(
                        epoch=epoch, iteration=iteration, fullstep=step, time=timestamp,
                        update_succeeded=not skipped, step_skipped=skipped,
                        metrics={"loss/total": float("nan")} if skipped else {},
                    )
            with path.open("ab") as log:
                log.write(b'{"invalid":true}\n{"schema_version":1')

            with patch("glzn.log.reader._TAIL_BLOCK_SIZE", 128):
                summary = summarize_log(path, window=5)
                recent = summarize_log(path, window=2)

            self.assertIsNotNone(summary.latest)
            assert summary.latest is not None
            self.assertEqual(summary.latest.fullstep, 7)
            self.assertEqual(summary.records, 3)
            self.assertEqual(summary.updates_per_minute, 3.0)
            self.assertEqual(summary.amp_skips, 1)
            self.assertEqual(summary.nonfinite, 1)
            self.assertEqual(summary.invalid_lines, 1)
            self.assertEqual(recent.records, 1)
            self.assertIsNone(recent.updates_per_minute)

    def test_cli_finds_logs_and_deduplicates_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trial"
            path = root / "log" / "metrics.jsonl"
            with LogCollator.local(run_id="trial", root=root, stdout=False) as logger:
                for i in range(2):
                    logger.log_update(
                        epoch=0, iteration=i, fullstep=i + 1, time=100.0 + 10 * i,
                        update_succeeded=True, step_skipped=False,
                    )
                with redirect_stdout(io.StringIO()) as output, patch(
                    "glzn.log.__main__.time.time", return_value=120.0
                ):
                    self.assertEqual(main([directory, str(path), "--window", "2"]), 0)
            header, row = output.getvalue().splitlines()
            self.assertIn("Updates/min", header)
            self.assertEqual(row.split(maxsplit=8), [
                "trial", "0", "2", "10s", "6.0", "0", "0", "2", str(path.resolve())
            ])
