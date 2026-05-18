import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bids_minimize


class BidsMinimizeTests(unittest.TestCase):
    def test_parse_bids_filename_supports_nii_gz(self):
        parsed = bids_minimize.parse_bids_filename("sub-01_task-rest_run-1_bold.nii.gz")
        self.assertIsNotNone(parsed)
        entities, suffix, extension = parsed
        self.assertEqual(entities, [("sub", "01"), ("task", "rest"), ("run", "1")])
        self.assertEqual(suffix, "bold")
        self.assertEqual(extension, ".nii.gz")

    def test_parse_bids_filename_rejects_invalid_names(self):
        self.assertIsNone(bids_minimize.parse_bids_filename("not_bids_name.nii.gz"))
        self.assertIsNone(bids_minimize.parse_bids_filename("sub-01_task-rest.nii.gz"))
        self.assertIsNone(bids_minimize.parse_bids_filename("sub-01_task-rest_run-1_bold"))

    def test_minimize_removes_non_required_entities_and_updates_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func_dir = root / "sub-01" / "func"
            func_dir.mkdir(parents=True)

            old_bold = func_dir / "sub-01_task-rest_run-1_bold.nii.gz"
            old_events = func_dir / "sub-01_task-rest_run-1_events.tsv"
            old_bold.write_bytes(b"nii")
            old_events.write_text("onset\tduration\n", encoding="utf-8")

            scans = root / "sub-01" / "sub-01_scans.tsv"
            scans.write_text(
                "filename\nfunc/sub-01_task-rest_run-1_bold.nii.gz\n",
                encoding="utf-8",
            )

            required = {"bold": {"sub", "task"}, "events": {"sub", "task"}}
            with patch.object(bids_minimize, "build_required_entities_by_suffix", return_value=required):
                operations = bids_minimize.minimize_bids_filenames(root)

            self.assertIn(
                (str(old_bold.resolve()), str((func_dir / "sub-01_task-rest_bold.nii.gz").resolve())),
                operations,
            )
            self.assertTrue((func_dir / "sub-01_task-rest_bold.nii.gz").exists())
            self.assertTrue((func_dir / "sub-01_task-rest_events.tsv").exists())

            scans_content = scans.read_text(encoding="utf-8")
            self.assertIn("func/sub-01_task-rest_bold.nii.gz", scans_content)

    def test_collision_keeps_optional_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func_dir = root / "sub-01" / "func"
            func_dir.mkdir(parents=True)

            run1 = func_dir / "sub-01_task-rest_run-1_bold.nii.gz"
            run2 = func_dir / "sub-01_task-rest_run-2_bold.nii.gz"
            run1.write_bytes(b"1")
            run2.write_bytes(b"2")

            required = {"bold": {"sub", "task"}}
            with patch.object(bids_minimize, "build_required_entities_by_suffix", return_value=required):
                operations = bids_minimize.minimize_bids_filenames(root)

            self.assertEqual(operations, [])
            self.assertTrue(run1.exists())
            self.assertTrue(run2.exists())

    def test_collision_resolution_keeps_minimal_optional_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func_dir = root / "sub-01" / "func"
            func_dir.mkdir(parents=True)

            files = [
                func_dir / "sub-01_task-rest_acq-A_run-1_bold.nii.gz",
                func_dir / "sub-01_task-rest_acq-B_run-2_bold.nii.gz",
                func_dir / "sub-01_task-rest_acq-C_run-3_bold.nii.gz",
            ]
            for index, file_path in enumerate(files):
                file_path.write_bytes(str(index).encode("utf-8"))

            required = {"bold": {"sub", "task"}}
            with patch.object(bids_minimize, "build_required_entities_by_suffix", return_value=required):
                bids_minimize.minimize_bids_filenames(root)

            self.assertTrue((func_dir / "sub-01_task-rest_acq-A_bold.nii.gz").exists())
            self.assertTrue((func_dir / "sub-01_task-rest_acq-B_bold.nii.gz").exists())
            self.assertTrue((func_dir / "sub-01_task-rest_acq-C_bold.nii.gz").exists())


if __name__ == "__main__":
    unittest.main()
