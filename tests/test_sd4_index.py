"""NIST SD4 naming, pairing and split logic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from latentfp.data.sd4_index import (
    LATENT_PREFIX,
    ROLLED_PREFIX,
    SD4File,
    index_sd4_dir,
    pair_by_subject,
    parse_sd4_filename,
    split_files,
    split_subjects,
)


class TestParseSd4Filename(unittest.TestCase):
    def test_rolled_print(self) -> None:
        self.assertEqual(parse_sd4_filename("f0001_01.png"), ("f", "0001", "01"))

    def test_latent_print(self) -> None:
        self.assertEqual(parse_sd4_filename("s0001_01.png"), ("s", "0001", "01"))

    def test_suffix_is_case_insensitive(self) -> None:
        self.assertEqual(parse_sd4_filename("F0002_03.PNG"), ("f", "0002", "03"))

    def test_directory_part_is_ignored(self) -> None:
        parsed = parse_sd4_filename("sd04/png_txt/figs_3/f1234_10.bmp")
        self.assertEqual(parsed, ("f", "1234", "10"))

    def test_three_digit_impression(self) -> None:
        self.assertEqual(parse_sd4_filename("s0007_123.png"), ("s", "0007", "123"))

    def test_unknown_suffix_still_parses(self) -> None:
        # a raw base name without any image extension
        self.assertEqual(parse_sd4_filename("f0001_01"), ("f", "0001", "01"))

    def test_rejects_non_sd4_names(self) -> None:
        for name in (
            "README.txt",
            "f001_01.png",       # three-digit subject id
            "x0001_01.png",      # unknown prefix
            "f0001-01.png",      # wrong separator
            "f0001_01_extra.png",
            "",
        ):
            with self.subTest(name=name):
                self.assertIsNone(parse_sd4_filename(name))

    def test_dot_prefixed_name_is_not_mistaken_for_a_suffix(self) -> None:
        # Path.stem would strip "_01" here; the explicit suffix list must not
        self.assertEqual(parse_sd4_filename("f0001_01"), ("f", "0001", "01"))


class TestSd4File(unittest.TestCase):
    def _make(self, name: str) -> SD4File:
        prefix, subject, impression = parse_sd4_filename(name)
        return SD4File(path=Path(name), prefix=prefix, subject=subject,
                       impression=impression)

    def test_kind_mapping(self) -> None:
        rolled = self._make("f0001_01.png")
        latent = self._make("s0001_01.png")
        self.assertEqual(rolled.kind, "rolled")
        self.assertEqual(latent.kind, "latent")
        self.assertEqual(rolled.prefix, ROLLED_PREFIX)
        self.assertEqual(latent.prefix, LATENT_PREFIX)

    def test_stem_and_pair_key(self) -> None:
        item = self._make("f0042_07.png")
        self.assertEqual(item.stem, "f0042_07")
        self.assertEqual(item.pair_key, ("0042", "07"))


class TestIndexAndPair(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for name in (
            "f0001_01.png",
            "s0001_01.png",
            "f0002_01.png",
            "s0002_01.png",
            "s0003_02.png",
            "notes.txt",       # wrong suffix
            "readme.md",       # wrong suffix
            "thumb.jpg",       # not SD4 named
        ):
            (self.root / name).write_bytes(b"\x00")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_index_skips_non_sd4_files(self) -> None:
        indexed = index_sd4_dir(self.root)
        self.assertEqual(
            [item.path.name for item in indexed],
            ["f0001_01.png", "s0001_01.png", "f0002_01.png",
             "s0002_01.png", "s0003_02.png"],
        )

    def test_index_is_sorted_and_recursive(self) -> None:
        nested = self.root / "figs_0"
        nested.mkdir()
        (nested / "s0009_01.png").write_bytes(b"\x00")
        indexed = index_sd4_dir(self.root)
        subjects = [item.subject for item in indexed]
        self.assertEqual(subjects, sorted(subjects))
        self.assertIn("0009", subjects)

    def test_index_rejects_missing_directory(self) -> None:
        with self.assertRaises(NotADirectoryError):
            index_sd4_dir(self.root / "does-not-exist")

    def test_pair_by_subject(self) -> None:
        grouped = pair_by_subject(index_sd4_dir(self.root))
        self.assertEqual(grouped["0001"]["rolled"][0].stem, "f0001_01")
        self.assertEqual(grouped["0001"]["latent"][0].stem, "s0001_01")
        # subject 0003 has a latent print but no rolled print
        self.assertEqual(grouped["0003"]["rolled"], [])
        self.assertEqual(len(grouped["0003"]["latent"]), 1)


class TestSplitSubjects(unittest.TestCase):
    SUBJECTS = [f"{index:04d}" for index in range(1, 51)]

    def test_split_is_disjoint_and_complete(self) -> None:
        train, val = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=0)
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(set(train) | set(val), set(self.SUBJECTS))

    def test_split_is_deterministic(self) -> None:
        first = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=7)
        second = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=7)
        self.assertEqual(first, second)

    def test_seed_changes_the_assignment(self) -> None:
        first = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=1)
        second = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=2)
        self.assertNotEqual(first[1], second[1])

    def test_expected_size(self) -> None:
        _, val = split_subjects(self.SUBJECTS, val_ratio=0.2, seed=0)
        self.assertEqual(len(val), 10)

    def test_zero_ratio_keeps_everything_in_train(self) -> None:
        train, val = split_subjects(self.SUBJECTS, val_ratio=0.0, seed=0)
        self.assertEqual(val, [])
        self.assertEqual(len(train), len(self.SUBJECTS))

    def test_small_pool_keeps_both_sides_non_empty(self) -> None:
        train, val = split_subjects(["0001", "0002", "0003"], val_ratio=0.9, seed=0)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 1)
        self.assertTrue(train)

    def test_empty_input(self) -> None:
        self.assertEqual(split_subjects([], val_ratio=0.2), ([], []))

    def test_duplicate_subjects_are_merged(self) -> None:
        train, val = split_subjects(
            ["0001", "0001", "0002", "0003"], val_ratio=0.25, seed=0
        )
        self.assertEqual(len(train) + len(val), 3)

    def test_invalid_ratio_raises(self) -> None:
        for ratio in (-0.1, 1.0, 1.5):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    split_subjects(self.SUBJECTS, val_ratio=ratio)


class TestSplitFiles(unittest.TestCase):
    def test_no_subject_leaks_between_splits(self) -> None:
        files = []
        for subject in range(1, 21):
            for prefix in ("f", "s"):
                name = f"{prefix}{subject:04d}_01.png"
                parsed = parse_sd4_filename(name)
                files.append(
                    SD4File(path=Path(name), prefix=parsed[0], subject=parsed[1],
                            impression=parsed[2])
                )

        train, val = split_files(files, val_ratio=0.25, seed=3)
        train_subjects = {item.subject for item in train}
        val_subjects = {item.subject for item in val}
        self.assertEqual(train_subjects & val_subjects, set())
        self.assertEqual(train_subjects | val_subjects,
                         {item.subject for item in files})
        self.assertEqual(len(train) + len(val), len(files))

    def test_both_kinds_land_on_the_same_side(self) -> None:
        files = []
        for subject in range(1, 11):
            for prefix in ("f", "s"):
                name = f"{prefix}{subject:04d}_01.png"
                parsed = parse_sd4_filename(name)
                files.append(
                    SD4File(path=Path(name), prefix=parsed[0], subject=parsed[1],
                            impression=parsed[2])
                )
        train, val = split_files(files, val_ratio=0.2, seed=0)
        for split in (train, val):
            kinds = {item.kind for item in split}
            self.assertEqual(kinds, {"rolled", "latent"})


if __name__ == "__main__":
    unittest.main()
