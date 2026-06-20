from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class QlibUnavailableError(RuntimeError):
    """Raised when the qlib-native path cannot be initialized safely."""


@dataclass(frozen=True)
class QlibProviderConfig:
    provider_uri: Path
    region: str = "us"
    market: str = "all"
    freq: str = "day"

    @property
    def resolved_provider_uri(self) -> Path:
        return Path(self.provider_uri).resolve()


def _require_provider(config: QlibProviderConfig) -> Path:
    provider_uri = config.resolved_provider_uri
    required = [
        provider_uri,
        provider_uri / "calendars" / f"{config.freq}.txt",
        provider_uri / "instruments" / f"{config.market}.txt",
        provider_uri / "features",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise QlibUnavailableError(f"blocked: qlib unavailable: missing provider files {missing}")
    return provider_uri


def init_qlib_provider(config: QlibProviderConfig) -> dict[str, Any]:
    provider_uri = _require_provider(config)

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri.as_posix(), region=config.region)
    calendar = D.calendar(freq=config.freq)
    instruments = D.instruments(config.market)
    if len(calendar) == 0:
        raise QlibUnavailableError("blocked: qlib unavailable: empty qlib calendar")
    return {
        "provider_uri": provider_uri.as_posix(),
        "region": config.region,
        "market": config.market,
        "freq": config.freq,
        "calendar_start": pd.Timestamp(calendar[0]).date().isoformat(),
        "calendar_end": pd.Timestamp(calendar[-1]).date().isoformat(),
        "instruments": instruments,
        "qlib_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.instruments"],
    }


def load_qlib_feature_frame(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    fields: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    init_qlib_provider(config)
    from qlib.data import D

    frame = D.features(
        instruments,
        fields,
        start_time=start,
        end_time=end,
        freq=config.freq,
    )
    frame.attrs["provider_uri"] = config.resolved_provider_uri.as_posix()
    frame.attrs["qlib_data_api"] = ["qlib.data.D.features"]
    frame.attrs["fields"] = fields
    return frame


def build_qlib_dataset(
    config: QlibProviderConfig,
    *,
    instruments: list[str],
    features: list[str],
    label: str,
    train: tuple[str, str],
    test: tuple[str, str],
):
    init_qlib_provider(config)

    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    loader = QlibDataLoader(config={"feature": features, "label": [label]}, freq=config.freq)
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=min(train[0], test[0]),
        end_time=max(train[1], test[1]),
        data_loader=loader,
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": train,
            "test": test,
        },
    )
    dataset.qlib_provider_uri = config.resolved_provider_uri.as_posix()
    dataset.qlib_dataset_api = ["DatasetH", "DataHandlerLP", "QlibDataLoader"]
    return dataset
