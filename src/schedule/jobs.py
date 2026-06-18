from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleJob:
    name: str
    task: str
    cron: str
    enabled: bool = True
