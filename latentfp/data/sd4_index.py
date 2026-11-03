"""NIST SD4 file naming, pairing and subject-disjoint splitting.

This module is deliberately dependency-free (standard library only) so the
dataset bookkeeping can be unit-tested and reused without PyTorch / OpenCV.

NIST Special Database 4 ships 2000 print pairs.  Each pair uses one common
four-digit subject id and a two-digit impression index:

* ``f0001_01.png`` -- the *rolled* (classifiable / tenprint) image
* ``s0001_01.png`` -- the matching *latent* (search) image

.. note::
   ``150930/FingerNet-master/README.md`` in the original workspace documents
   ``f`` as latent and ``s`` as rolled, i.e. the other way round.  The
   constants below follow the NIST convention; flip them if your own export
   of SD4 really uses the opposite mapping.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ROLLED_PREFIX",
    "LATENT_PREFIX",
    "IMAGE_SUFFIXES",
    "SD4File",
    "parse_sd4_filename",
    "index_sd4_dir",
    "pair_by_subject",
    "split_subjects",
    "split_files",
]

#: prefix of the rolled / classifiable print
ROLLED_PREFIX = "f"
#: prefix of the matching latent / search print
LATENT_PREFIX = "s"

IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

_SD4_RE = re.compile(
    r"^(?P<prefix>[fs])(?P<subject>\d{4})_(?P<impression>\d{2,3})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SD4File:
    """One indexed SD4 image."""

    path: Path
    prefix: str        # "f" or "s", lower-cased
    subject: str       # zero-padded four-digit subject id, e.g. "0001"
    impression: str    # zero-padded impression index, e.g. "01"

    @property
    def kind(self) -> str:
        """``"rolled"`` or ``"latent"``."""
        return "rolled" if self.prefix == ROLLED_PREFIX else "latent"

    @property
    def stem(self) -> str:
        """File name without the suffix, e.g. ``"f0001_01"``."""
        return self.path.stem

    @property
    def pair_key(self) -> Tuple[str, str]:
        """Key shared by a print and its own impressions, e.g. ``("0001", "01")``."""
        return (self.subject, self.impression)


def parse_sd4_filename(name: str) -> Optional[Tuple[str, str, str]]:
    """Parse an SD4 base name.

    ``name`` may carry any directory part and any known suffix; both are
    ignored.  Returns ``(prefix, subject, impression)`` or ``None`` when the
    name does not follow the ``f0001_01`` pattern.
    """
    stem = Path(str(name)).name
    # strip a known image suffix (``Path.stem`` would also eat ``0001_01``
    # style names, so do it explicitly)
    lowered = stem.lower()
    for suffix in IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = _SD4_RE.match(stem)
    if match is None:
        return None
    return (
        match.group("prefix").lower(),
        match.group("subject"),
        match.group("impression"),
    )


def index_sd4_dir(root: str | Path, recursive: bool = True) -> List[SD4File]:
    """Index every SD4-named image below ``root``, sorted by file name.

    Files that do not follow the SD4 naming scheme are silently skipped: the
    training pipeline derives parameters from the naming, so a stray file
    would be noise rather than data.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    iterator: Iterable[Path]
    iterator = root.rglob("*") if recursive else root.glob("*")

    found: List[SD4File] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = parse_sd4_filename(path.name)
        if parsed is None:
            continue
        prefix, subject, impression = parsed
        found.append(SD4File(path=path, prefix=prefix, subject=subject,
                             impression=impression))
    found.sort(key=lambda item: (item.subject, item.impression, item.prefix))
    return found


def pair_by_subject(files: Sequence[SD4File]) -> Dict[str, Dict[str, List[SD4File]]]:
    """Group ``files`` as ``{subject: {"rolled": [...], "latent": [...]}}``."""
    grouped: Dict[str, Dict[str, List[SD4File]]] = {}
    for item in files:
        bucket = grouped.setdefault(item.subject, {"rolled": [], "latent": []})
        bucket[item.kind].append(item)
    return grouped


def _stable_rng(salt: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{salt}:{seed}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def split_subjects(
    subjects: Iterable[str],
    val_ratio: float = 0.2,
    seed: int = 0,
    salt: str = "sd4",
) -> Tuple[List[str], List[str]]:
    """Split subject ids into ``(train, val)`` without shared subjects.

    Splitting by *subject* rather than by file is what keeps the validation set
    honest: two impressions of the same finger must never land on both sides.
    The shuffle is seeded from ``sha256(salt:seed)`` so the split is stable
    across processes and platforms (``random.shuffle`` on a set order is not).
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio!r}")

    unique = sorted(set(subjects))
    rng = _stable_rng(salt, seed)
    rng.shuffle(unique)

    if not unique:
        return [], []

    n_val = int(round(len(unique) * val_ratio))
    if val_ratio > 0.0:
        # always keep at least one validation subject and one training subject
        n_val = max(1, n_val)
        n_val = min(n_val, len(unique) - 1)
    else:
        n_val = 0

    val = sorted(unique[:n_val])
    train = sorted(unique[n_val:])
    return train, val


def split_files(
    files: Sequence[SD4File],
    val_ratio: float = 0.2,
    seed: int = 0,
    salt: str = "sd4",
) -> Tuple[List[SD4File], List[SD4File]]:
    """Subject-disjoint ``(train, val)`` split of ``files``."""
    train_subjects, val_subjects = split_subjects(
        (item.subject for item in files), val_ratio=val_ratio, seed=seed, salt=salt
    )
    train_set, val_set = set(train_subjects), set(val_subjects)
    train = [item for item in files if item.subject in train_set]
    val = [item for item in files if item.subject in val_set]
    return train, val
