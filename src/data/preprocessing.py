from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def split_dev_by_time(
    dev_behaviors: pd.DataFrame, validation_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    sorted_frame = dev_behaviors.sort_values(
        by=["time", "impression_id"], kind="mergesort"
    ).reset_index(drop=True)
    split_index = int(len(sorted_frame) * validation_ratio)
    validation = sorted_frame.iloc[:split_index].reset_index(drop=True)
    test = sorted_frame.iloc[split_index:].reset_index(drop=True)
    metadata = {
        "validation_ratio": validation_ratio,
        "split_index": split_index,
        "validation_start": _boundary_time(validation, "first"),
        "validation_end": _boundary_time(validation, "last"),
        "test_start": _boundary_time(test, "first"),
        "test_end": _boundary_time(test, "last"),
    }
    return validation, test, metadata


def _boundary_time(frame: pd.DataFrame, position: str) -> str | None:
    if frame.empty:
        return None
    timestamp = frame.iloc[0 if position == "first" else -1]["time"]
    return timestamp.isoformat()


def validate_news_references(
    behaviors: pd.DataFrame, known_news_ids: set[str], *, split_name: str
) -> None:
    unknown: set[str] = set()
    for row in behaviors.itertuples(index=False):
        unknown.update(news_id for news_id in row.history if news_id not in known_news_ids)
        unknown.update(
            news_id for news_id, _ in row.candidates if news_id not in known_news_ids
        )
    if unknown:
        preview = ", ".join(sorted(unknown)[:10])
        raise ValueError(f"{split_name} references unknown news IDs: {preview}")


def build_train_samples(
    behaviors: pd.DataFrame,
    known_news_ids: set[str],
    *,
    max_history_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_news_references(behaviors, known_news_ids, split_name="train")
    samples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in behaviors.itertuples(index=False):
        if not row.history:
            counters["empty_history"] += 1
            continue
        positives = [news_id for news_id, label in row.candidates if label == 1]
        negatives = [news_id for news_id, label in row.candidates if label == 0]
        if not positives:
            counters["no_positive"] += 1
            continue
        if not negatives:
            counters["no_negative_pool"] += 1
            continue

        common = {
            "impression_id": row.impression_id,
            "user_id": row.user_id,
            "time": row.time.isoformat(),
            "history": row.history[-max_history_length:],
            "negative_pool": negatives,
        }
        for positive in positives:
            samples.append({**common, "positive": positive})
    counters["input_impressions"] = len(behaviors)
    counters["output_samples"] = len(samples)
    return samples, dict(counters)


def build_evaluation_samples(
    behaviors: pd.DataFrame,
    known_news_ids: set[str],
    *,
    max_history_length: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_news_references(behaviors, known_news_ids, split_name=split_name)
    samples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in behaviors.itertuples(index=False):
        if not row.history:
            counters["empty_history"] += 1
            continue
        samples.append(
            {
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "time": row.time.isoformat(),
                "history": row.history[-max_history_length:],
                "candidates": [news_id for news_id, _ in row.candidates],
                "labels": [label for _, label in row.candidates],
            }
        )
    counters["input_impressions"] = len(behaviors)
    counters["output_samples"] = len(samples)
    return samples, dict(counters)