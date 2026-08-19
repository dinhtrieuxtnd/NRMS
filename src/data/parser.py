from __future__ import annotations

from pathlib import Path

import pandas as pd


BEHAVIOR_COLUMNS = (
    "impression_id",
    "user_id",
    "time",
    "history",
    "candidates",
)
NEWS_COLUMNS = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def _read_tsv_rows(file_path: Path, expected_columns: int) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = line.rstrip("\r\n").split("\t")
            if len(row) != expected_columns:
                raise ValueError(
                    f"{file_path}:{line_number}: expected {expected_columns} columns, "
                    f"found {len(row)}"
                )
            rows.append((line_number, row))
    return rows


def parse_candidates(value: str, *, file_path: Path, line_number: int) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for item in value.split():
        try:
            news_id, raw_label = item.rsplit("-", 1)
        except ValueError as error:
            raise ValueError(
                f"{file_path}:{line_number}: invalid candidate {item!r}"
            ) from error

        if not news_id or raw_label not in {"0", "1"}:
            raise ValueError(
                f"{file_path}:{line_number}: invalid candidate {item!r}; "
                "expected NEWS_ID-0 or NEWS_ID-1"
            )
        candidates.append((news_id, int(raw_label)))

    if not candidates:
        raise ValueError(f"{file_path}:{line_number}: impressions contain no candidates")
    return candidates


def read_news(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    records: list[dict[str, str]] = []
    for line_number, row in _read_tsv_rows(path, len(NEWS_COLUMNS)):
        record = dict(zip(NEWS_COLUMNS, row, strict=True))
        if not record["news_id"]:
            raise ValueError(f"{path}:{line_number}: news_id is empty")
        records.append(record)
    return pd.DataFrame.from_records(records, columns=NEWS_COLUMNS)


def read_behaviors(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    records: list[dict[str, object]] = []
    for line_number, row in _read_tsv_rows(path, 5):
        impression_id, user_id, raw_time, raw_history, raw_candidates = row
        if not impression_id:
            raise ValueError(f"{path}:{line_number}: impression_id is empty")
        if not user_id:
            raise ValueError(f"{path}:{line_number}: user_id is empty")
        try:
            timestamp = pd.to_datetime(raw_time, format=MIND_TIME_FORMAT, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid timestamp {raw_time!r}"
            ) from error

        records.append(
            {
                "impression_id": impression_id,
                "user_id": user_id,
                "time": timestamp,
                "history": raw_history.split(),
                "candidates": parse_candidates(
                    raw_candidates,
                    file_path=path,
                    line_number=line_number,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=BEHAVIOR_COLUMNS)