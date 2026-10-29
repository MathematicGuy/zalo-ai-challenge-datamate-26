"""Shared paths. Tests read the released train split from data/train (not in git)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "train"


@pytest.fixture(scope="session")
def train_answers() -> list[dict]:
    with open(TRAIN / "answers.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
