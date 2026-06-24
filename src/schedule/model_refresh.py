from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data import QlibProviderConfig, init_qlib_provider
from src.workflow import QlibNativeExperimentResult, QlibNativeExperimentSpec, run_qlib_native_experiment


@dataclass(frozen=True)
class WalkForwardModelRefreshSpec:
    experiment_prefix: str
    provider_uri: Path
    recorder_uri: Path
    output_root: Path
    market: str
    region: str
    instrument: str
    start: str
    end: str | None = None
    train_size: int = 120
    valid_size: int = 20
    test_size: int = 20
    step_size: int = 20
    max_folds: int | None = 1
    model_selection_metric: str = "objective_score"
    fixed_model_name: str | None = None
    promoted_from: str | None = None


def build_walk_forward_windows(spec: WalkForwardModelRefreshSpec, *, end: str) -> list[dict[str, Any]]:
    if spec.train_size <= 0 or spec.test_size <= 0 or spec.valid_size < 0 or spec.step_size <= 0:
        raise ValueError("walk-forward sizes must be positive except valid_size, which may be zero")

    config = QlibProviderConfig(provider_uri=spec.provider_uri, region=spec.region, market=spec.market)
    init_qlib_provider(config)
    from qlib.data import D

    calendar = [
        pd.Timestamp(value).date().isoformat()
        for value in D.calendar(freq=config.freq)
        if spec.start <= pd.Timestamp(value).date().isoformat() <= end
    ]
    total = spec.train_size + spec.valid_size + spec.test_size
    windows: list[dict[str, Any]] = []
    start_idx = 0
    while start_idx + total <= len(calendar):
        train_dates = calendar[start_idx : start_idx + spec.train_size]
        valid_dates = calendar[start_idx + spec.train_size : start_idx + spec.train_size + spec.valid_size]
        test_dates = calendar[start_idx + spec.train_size + spec.valid_size : start_idx + total]
        windows.append(
            {
                "fold": len(windows) + 1,
                "train": [train_dates[0], train_dates[-1]],
                "valid": [valid_dates[0], valid_dates[-1]] if valid_dates else None,
                "test": [test_dates[0], test_dates[-1]],
            }
        )
        start_idx += spec.step_size
    if spec.max_folds is not None:
        return windows[: spec.max_folds]
    return windows


class WalkForwardModelRefreshJob:
    def __init__(
        self,
        spec: WalkForwardModelRefreshSpec,
        *,
        runner: Callable[[QlibNativeExperimentSpec], QlibNativeExperimentResult] = run_qlib_native_experiment,
    ) -> None:
        self.spec = spec
        self.runner = runner

    def run(self, data_manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        end = self.spec.end or _latest_manifest_datetime(data_manifests)
        windows = build_walk_forward_windows(self.spec, end=end)
        if not windows:
            raise ValueError("walk-forward refresh has no complete qlib calendar windows")

        results: list[dict[str, Any]] = []
        for window in windows:
            output_dir = self.spec.output_root / _experiment_name(self.spec.experiment_prefix, window)
            workflow_spec = QlibNativeExperimentSpec(
                experiment=output_dir.name,
                provider_uri=self.spec.provider_uri,
                recorder_uri=self.spec.recorder_uri,
                output_dir=output_dir,
                market=self.spec.market,
                region=self.spec.region,
                instrument=self.spec.instrument,
                train=tuple(window["train"]),
                valid=tuple(window["valid"]) if window["valid"] else None,
                test=tuple(window["test"]),
                model_selection_metric=self.spec.model_selection_metric,
                fixed_model_name=self.spec.fixed_model_name,
                promoted_from=self.spec.promoted_from,
            )
            result = self.runner(workflow_spec)
            results.append(_result_manifest(workflow_spec, window, result, data_manifests))
        return results


def _latest_manifest_datetime(manifests: list[dict[str, Any]]) -> str:
    dates = [str(manifest.get("max_datetime", "")) for manifest in manifests if manifest.get("max_datetime")]
    if not dates:
        raise ValueError("walk-forward refresh requires an explicit end date or data manifest max_datetime")
    return max(dates)


def _experiment_name(prefix: str, window: dict[str, Any]) -> str:
    test_end = str(window["test"][1]).replace("-", "")
    return f"{prefix}_wf{int(window['fold']):03d}_{test_end}"


def _result_manifest(
    workflow_spec: QlibNativeExperimentSpec,
    window: dict[str, Any],
    result: QlibNativeExperimentResult,
    data_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "operation": "walk_forward_model_refresh",
        "source": "qlib_recorder",
        "experiment": workflow_spec.experiment,
        "fold": int(window["fold"]),
        "segments": {
            "train": list(workflow_spec.train),
            **({"valid": list(workflow_spec.valid)} if workflow_spec.valid else {}),
            "test": list(workflow_spec.test),
        },
        "provider_uri": workflow_spec.provider_uri.resolve().as_posix(),
        "recorder_uri": workflow_spec.recorder_uri.resolve().as_posix(),
        "recorder_id": result.recorder_id,
        "recorder_local_dir": Path(result.recorder_local_dir).as_posix(),
        "output_dir": workflow_spec.output_dir.as_posix(),
        "summary_path": (workflow_spec.output_dir / "summary.json").as_posix(),
        "model_selection_metric": workflow_spec.model_selection_metric,
        "fixed_model_name": workflow_spec.fixed_model_name,
        "promoted_from": workflow_spec.promoted_from,
        "data_manifest_count": len(data_manifests),
        "qlib_workflow_api": [
            "qlib.data.D.calendar",
            "qlib.workflow.R.start",
            "SignalRecord",
            "SigAnaRecord",
            "PortAnaRecord",
        ],
    }
