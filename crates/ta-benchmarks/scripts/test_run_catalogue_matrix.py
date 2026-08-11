from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("run_catalogue_matrix.py")
SPEC = importlib.util.spec_from_file_location("run_catalogue_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PublicationArgumentTests(unittest.TestCase):
    def arguments(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "publish": True,
            "publish_existing": False,
            "baseline": runner.DEFAULT_BASELINE,
            "optimization_evidence": runner.DEFAULT_OPTIMIZATION_EVIDENCE,
            "case": None,
            "input_length": None,
            "samples": None,
            "warmup_ms": None,
            "sample_ms": None,
            "source_archive": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_publication_accepts_defaults_and_explicit_canonical_timing(self) -> None:
        runner.validate_publication_args(self.arguments())
        runner.validate_publication_args(
            self.arguments(
                samples=runner.CANONICAL_SAMPLES,
                warmup_ms=runner.CANONICAL_WARMUP_MS,
                sample_ms=runner.CANONICAL_SAMPLE_MS,
            )
        )

    def test_publication_rejects_every_noncanonical_timing_override(self) -> None:
        for field, value in (
            ("samples", runner.CANONICAL_SAMPLES - 1),
            ("warmup_ms", runner.CANONICAL_WARMUP_MS - 1),
            ("sample_ms", runner.CANONICAL_SAMPLE_MS + 1),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                SystemExit, "noncanonical timing overrides"
            ):
                runner.validate_publication_args(self.arguments(**{field: value}))

    def test_publication_rejects_noncanonical_comparison_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            other = Path(temporary) / "other.tsv"
            for field in ("baseline", "optimization_evidence"):
                with self.subTest(field=field), self.assertRaisesRegex(
                    SystemExit, "canonical comparison inputs"
                ):
                    runner.validate_publication_args(
                        self.arguments(**{field: other})
                    )


class PublicationMatrixTests(unittest.TestCase):
    def rows(self) -> list[dict[str, str]]:
        rows = []
        for case_id in runner.CASE_IDS:
            for input_length in runner.INPUT_LENGTHS:
                for implementation, mode in runner.VARIANTS:
                    row = {
                        "case_id": case_id,
                        "input_length": str(input_length),
                        "implementation": implementation,
                        "mode": mode,
                        "indicator_family": f"family-{case_id}",
                        "indicator_definition": f"definition-{case_id}",
                        "parameters": "canonical",
                        "output_kind": "float",
                        "output_arity": "1",
                        "input_checksum": runner.INPUT_CHECKSUMS[input_length],
                        "python_version": "3.12.0",
                        "rustc": "rustc test",
                        "cpu": "test cpu",
                        "os": "test os",
                        "arch": "test arch",
                        "commit": "0123456789abcdef",
                    }
                    row.update(runner.FIXED_PUBLICATION_FIELDS)
                    rows.append(row)
        return rows

    def write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as raw:
            writer = csv.DictWriter(raw, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_publishable_requires_each_expected_matrix_cell_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.tsv"
            rows = self.rows()
            self.write_rows(raw, rows)
            runner.validate_publishable(raw)

            rows[-1] = rows[0].copy()
            self.write_rows(raw, rows)
            with self.assertRaisesRegex(RuntimeError, "incomplete case/input/mode matrix"):
                runner.validate_publishable(raw)


class SourceCacheTests(unittest.TestCase):
    def archive(self, path: Path, root_name: str) -> str:
        payload = b"complete pinned source\n"
        with tarfile.open(path, "w:gz") as bundle:
            directory = tarfile.TarInfo(root_name)
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o555
            bundle.addfile(directory)
            readme = tarfile.TarInfo(f"{root_name}/README")
            readme.size = len(payload)
            bundle.addfile(readme, io.BytesIO(payload))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_unmarked_source_tree_is_replaced_by_marked_complete_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            archive = temporary_path / "source.tar.gz"
            root_name = f"ta-lib-{runner.TALIB_VERSION}"
            digest = self.archive(archive, root_name)
            destination = temporary_path / "source"
            stale = destination / root_name
            stale.mkdir(parents=True)
            (stale / "partial").write_text("do not trust", encoding="utf-8")

            with mock.patch.object(runner, "TALIB_ARCHIVE_SHA256", digest):
                extracted = runner.extract_archive(archive, destination)

            self.assertEqual(extracted, stale)
            self.assertFalse((extracted / "partial").exists())
            self.assertEqual(
                (extracted / "README").read_text(encoding="utf-8"),
                "complete pinned source\n",
            )
            self.assertEqual(stat.S_IMODE(extracted.stat().st_mode), 0o755)
            marker = (extracted / ".fast-ta-source-pin").read_text(encoding="utf-8")
            self.assertIn(f"archive_sha256={digest}\n", marker)

            with mock.patch.object(runner, "TALIB_ARCHIVE_SHA256", digest):
                with mock.patch.object(
                    runner.tarfile,
                    "open",
                    side_effect=AssertionError("marked cache should be reused"),
                ):
                    self.assertEqual(
                        runner.extract_archive(archive, destination), extracted
                    )

    def test_failed_extraction_does_not_replace_existing_unmarked_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            archive = temporary_path / "source.tar.gz"
            digest = self.archive(archive, "wrong-root")
            destination = temporary_path / "source"
            stale = destination / f"ta-lib-{runner.TALIB_VERSION}"
            stale.mkdir(parents=True)
            sentinel = stale / "partial"
            sentinel.write_text("preserved", encoding="utf-8")

            with mock.patch.object(runner, "TALIB_ARCHIVE_SHA256", digest):
                with self.assertRaisesRegex(RuntimeError, "did not contain"):
                    runner.extract_archive(archive, destination)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")
            self.assertEqual(
                list(destination.glob(f".{stale.name}.extract-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
