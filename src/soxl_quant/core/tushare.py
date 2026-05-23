from __future__ import annotations

"""Optional Tushare Pro helpers for cached external factor inputs."""

from pathlib import Path
import json
import os
from typing import Any
from urllib import request

import pandas as pd

from ..settings import DATA_DIR

TUSHARE_API_URL = "http://api.tushare.pro"
DEFAULT_US_DAILY_ADJ_FIELDS = (
    "ts_code,trade_date,close,open,high,low,pre_close,change,pct_change,"
    "vol,amount,vwap,adj_factor,turnover_ratio,free_share,total_share,"
    "free_mv,total_mv,exchange"
)


def tushare_csv_path(api_name: str, symbol: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else DATA_DIR / "raw" / "external"
    safe_symbol = symbol.upper().replace("/", "_").replace(".", "_")
    return base / f"tushare_{api_name}_{safe_symbol}.csv"


def normalize_tushare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    df = frame.copy()
    if "trade_date" not in df.columns:
        raise KeyError("Tushare frame missing trade_date")
    parsed = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.assign(date=parsed.dt.strftime("%Y-%m-%d"))
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_tushare_api(
    api_name: str,
    *,
    token: str,
    params: dict[str, Any] | None = None,
    fields: str = "",
    endpoint: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    payload = {
        "api_name": api_name,
        "token": token,
        "params": params or {},
        "fields": fields,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint or os.environ.get("TUSHARE_API_URL", TUSHARE_API_URL),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if int(body.get("code", -1)) != 0:
        raise RuntimeError(f"Tushare API error {body.get('code')}: {body.get('msg')}")
    payload_data = body.get("data") or {}
    columns = payload_data.get("fields") or []
    rows = payload_data.get("items") or []
    return pd.DataFrame(rows, columns=columns)


def fetch_us_daily_adj(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    token: str,
    output_path: Path | None = None,
    fields: str = DEFAULT_US_DAILY_ADJ_FIELDS,
) -> Path:
    frame = fetch_tushare_api(
        "us_daily_adj",
        token=token,
        params={"ts_code": symbol.upper(), "start_date": start_date, "end_date": end_date},
        fields=fields,
    )
    normalized = normalize_tushare_frame(frame)
    path = output_path or tushare_csv_path("us_daily_adj", symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False)
    return path


__all__ = [
    "DEFAULT_US_DAILY_ADJ_FIELDS",
    "TUSHARE_API_URL",
    "fetch_tushare_api",
    "fetch_us_daily_adj",
    "normalize_tushare_frame",
    "tushare_csv_path",
]
