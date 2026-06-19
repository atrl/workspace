from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


class JobLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "JobLock":
        if self.path.exists():
            raise RuntimeError(f"job already locked: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(time.time()), encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path.exists():
            self.path.unlink()


def retry(func: Callable[[], T], *, attempts: int, delay_seconds: float) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as error:  # noqa: BLE001 - job retry boundary captures and retries failures.
            last_error = error
            if attempt < attempts - 1 and delay_seconds:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error
