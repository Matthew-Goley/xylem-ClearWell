"""Shared pytest setup. Tests use synthetic data only and write only to tmp_path."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """config.get_paths() pointed at a temporary folder instead of the repo."""
    import config
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return config.get_paths(False)


@pytest.fixture
def logger(tmp_paths):
    import common
    return common.get_logger(tmp_paths, "pytest")
