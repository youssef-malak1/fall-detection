"""Import smoke tests.

Regression guard for a real defect: a ``.gitignore`` pattern of ``data/``
(no leading slash) matched ``src/data/`` as well, so the video reader silently
vanished from every clean clone while all 38 existing tests still passed.

Importing every module under ``src`` makes a missing source file fail loudly
here instead of surfacing as an ``ImportError`` in a script nobody runs.
"""

from __future__ import annotations

import importlib
import pkgutil

import src


def test_every_src_module_imports() -> None:
    names = sorted(m.name for m in pkgutil.walk_packages(src.__path__, prefix="src."))
    assert names, "no modules discovered under src/"

    failures: dict[str, str] = {}
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - collect all failures, report once
            failures[name] = f"{type(exc).__name__}: {exc}"

    assert not failures, f"modules that failed to import: {failures}"


def test_data_package_is_importable() -> None:
    """``src.data.video`` is imported by motion.py and two scripts."""
    from src.data import video  # noqa: F401

    assert callable(video.iter_rgb_frames)
