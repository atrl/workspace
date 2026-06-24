from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.data import (
    QlibProviderConfig,
    QlibUnavailableError,
    audit_qlib_provider,
    build_data_update_manifest,
    write_snapshot_manifest,
)


@dataclass(frozen=True)
class QlibDataRefreshSpec:
    config: QlibProviderConfig
    instruments: list[str]
    fields: list[str]
    start: str
    end: str
    as_of: str
    output_dir: Path
    required_max_datetime: str | None = None


@dataclass(frozen=True)
class CommandDataRefresh:
    command: list[str]
    cwd: Path | None = None
    timeout_seconds: float = 900.0

    def __call__(self) -> dict[str, Any]:
        if not self.command:
            raise ValueError("data refresh command cannot be empty")
        completed = subprocess.run(
            self.command,
            cwd=self.cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        payload = {
            "operation": "command_data_refresh",
            "command": self.command,
            "cwd": self.cwd.as_posix() if self.cwd else None,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode != 0:
            raise QlibUnavailableError(
                f"blocked: qlib unavailable: data refresh command failed with returncode {completed.returncode}"
            )
        return payload


class QlibDataRefreshJob:
    def __init__(
        self,
        spec: QlibDataRefreshSpec,
        *,
        refresh_store: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.spec = spec
        self.refresh_store = refresh_store

    def run(self) -> list[dict[str, Any]]:
        refresh_result = self.refresh_store() if self.refresh_store is not None else None
        audit = audit_qlib_provider(
            self.spec.config,
            instruments=self.spec.instruments,
            fields=self.spec.fields,
            start=self.spec.start,
            end=self.spec.end,
        )
        self._validate_freshness(audit)
        manifest_path = write_snapshot_manifest(audit, self.spec.output_dir, as_of=self.spec.as_of)
        manifest = build_data_update_manifest(audit, as_of=self.spec.as_of)
        manifest["manifest_path"] = manifest_path.resolve().as_posix()
        manifest["required_max_datetime"] = self.spec.required_max_datetime
        manifest["refresh"] = refresh_result or {"operation": "none", "status": "not_configured"}
        return [manifest]

    def _validate_freshness(self, audit: dict[str, Any]) -> None:
        required = self.spec.required_max_datetime
        if required is None:
            return
        max_datetime = str(audit["max_datetime"])
        if max_datetime < required:
            raise QlibUnavailableError(
                "blocked: qlib unavailable: "
                f"stale qlib data max_datetime {max_datetime} before required_max_datetime {required}"
            )
