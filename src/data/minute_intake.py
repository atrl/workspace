"""Read-only, resumable history acquisition with immutable request receipts."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def collect_futu_minute_history(
    *, context, symbol: str, start: str, end: str, output_dir: Path,
    checkpoint_dir: Path, adjustment: str, session: str,
    progress: Callable[[dict], None] | None = None,
    ktype: str = "K_1M",
) -> dict:
    """Persist pages before advancing the cursor; failed requests stay resumable.

    Checkpoints belong outside the repository. The final source response is
    preserved without filtering, interpolation or silent duplicate removal.
    Credentials and pagination cursors never enter the durable manifest.
    """
    from futu import AuType, KLType, RET_OK

    if adjustment not in {"NONE", "QFQ"} or session != "RTH" or ktype not in {"K_1M", "K_5M"}:
        raise ValueError("blocked: explicit NONE/QFQ and RTH request required")
    output_dir, checkpoint_dir = Path(output_dir), Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    request = dict(symbol=symbol.upper(), start=start, end=end, adjustment=adjustment,
                   session=session, ktype=ktype, max_count=1000)
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    destination = output_dir / f"{symbol.upper()}_{adjustment.lower()}.parquet"
    receipt_path = destination.with_suffix(".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request_sha256"] != identity or not destination.exists():
            raise ValueError("blocked: completed intake request differs")
        if hashlib.sha256(destination.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("blocked: completed source bytes changed")
        return receipt
    checkpoint = checkpoint_dir / identity
    checkpoint.mkdir(exist_ok=True)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "request_sha256": identity, "pages": [], "next_cursor": None,
        "finished": False, "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if state["request_sha256"] != identity:
        raise ValueError("blocked: checkpoint request identity differs")
    # Futu cursors are connection-scoped. Across connections resume at a whole
    # date boundary, retaining the raw page hashes and disclosing replaced rows.
    generations = state.setdefault("generations", [{"generation": 0, "start": start}])
    if state["pages"] and not state["finished"]:
        last = pd.read_parquet(checkpoint / state["pages"][-1]["path"])
        restart_date = str(last.time_key.max())[:10]
        generation = max(item["generation"] for item in generations) + 1
        generations.append({"generation": generation, "start": restart_date,
                            "reason": "connection_scoped_cursor_resume_by_whole_date"})
        state["next_cursor"] = None
        atomic_json(state, state_path)
    generation = generations[-1]["generation"]
    request_start = generations[-1]["start"]
    while not state["finished"]:
        page_key = base64.b64decode(state["next_cursor"]) if state["next_cursor"] else None
        ret, frame, next_key = context.request_history_kline(
            "US." + symbol.upper(), start=request_start, end=end, ktype=getattr(KLType, ktype),
            autype=getattr(AuType, adjustment), max_count=1000,
            page_req_key=page_key, session=session,
        )
        if ret != RET_OK:
            raise RuntimeError(f"needs_data: Futu history request failed for {symbol}: {str(frame)[:180]}")
        if frame.empty and next_key is not None:
            raise ValueError("blocked: empty page with a continuation cursor")
        if next_key is not None and next_key == page_key:
            raise ValueError("blocked: history cursor did not advance")
        number = len(state["pages"])
        path = checkpoint / f"page_{number:05d}.parquet"
        atomic_parquet(frame, path)
        state["pages"].append({"path": path.name, "generation": generation,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "rows": len(frame)})
        state["next_cursor"] = base64.b64encode(next_key).decode() if next_key is not None else None
        state["finished"] = next_key is None
        atomic_json(state, state_path)
        if progress and (number % 100 == 0 or state["finished"]):
            progress({"symbol": symbol, "adjustment": adjustment, "pages": len(state["pages"]),
                      "rows": sum(p["rows"] for p in state["pages"]), "finished": state["finished"]})
    parts = []
    for entry in state["pages"]:
        path = checkpoint / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("blocked: checkpoint page bytes changed")
        part = pd.read_parquet(path)
        part["_generation"] = entry.get("generation", 0)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    replaced_rows = 0
    for restart in generations[1:]:
        mask = result._generation.lt(restart["generation"]) & result.time_key.str[:10].ge(restart["start"])
        replaced_rows += int(mask.sum())
        result = result.loc[~mask]
    result = result.drop(columns="_generation").reset_index(drop=True)
    if result.empty or result.duplicated(["code", "time_key"]).any():
        raise ValueError("blocked: empty or duplicate historical source rows")
    if not result.code.eq("US." + symbol.upper()).all():
        raise ValueError("blocked: unexpected source instrument")
    dates = result.time_key.str[:10]
    if (dates < start).any() or (dates > end).any():
        raise ValueError("blocked: source rows outside requested dates")
    atomic_parquet(result, destination)
    receipt = {**request, "request_sha256": identity, "path": destination.name,
               "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
               "rows": len(result), "pages": len(parts), "first": result.time_key.min(),
               "last": result.time_key.max(), "fetched_at": datetime.now(timezone.utc).isoformat(),
               "started_at": state["started_at"], "read_only": True,
               "provider": "futu", "source_timezone": "America/New_York",
               "resume_generations": generations, "explicit_whole_date_replaced_rows": replaced_rows,
               "page_response_sha256": [p["sha256"] for p in state["pages"]]}
    atomic_json(receipt, receipt_path)
    return receipt


def collect_futu_bundle(
    *, requests: dict[str, str], start: str, end: str, output_dir: Path,
    checkpoint_dir: Path, host: str = "127.0.0.1", port: int = 11111,
    workers: int = 3, progress: Callable[[dict], None] | None = None,
) -> dict:
    """Collect explicit per-symbol resolutions with one connection per worker."""
    import futu
    from futu import OpenQuoteContext, RET_OK, Session, SysConfig

    if not 1 <= workers <= 4:
        raise ValueError("blocked: bounded history worker count required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = output_dir / "source_manifest.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved.get("complete"):
            if saved.get("resolutions") != requests or saved.get("start") != start or saved.get("as_of") != end:
                raise ValueError("blocked: immutable source bundle request differs")
            for entry in saved["entries"] + saved["rehab"]:
                path = output_dir / entry["path"]
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("blocked: immutable source bundle bytes changed")
            return saved
    SysConfig.set_all_thread_daemon(True)
    manifest = {"schema_version": "semis_minute_source_v1", "provider": "futu",
                "sdk_version": futu.__version__, "start": start, "as_of": end,
                "symbols": list(requests), "resolutions": requests, "adjustment": "NONE",
                "session": "RTH", "read_only": True, "entries": [], "rehab": [], "complete": False}

    def collect(symbol, ktype):
        context = OpenQuoteContext(host=host, port=port)
        try:
            receipt = collect_futu_minute_history(context=context, symbol=symbol, start=start,
                end=end, output_dir=output_dir, checkpoint_dir=checkpoint_dir,
                adjustment="NONE", session=Session.RTH, ktype=ktype, progress=progress)
            ret, rehab = context.get_rehab("US." + symbol)
            if ret != RET_OK:
                raise RuntimeError("needs_data: rehabilitation unavailable for " + symbol)
            path = output_dir / (symbol + "_rehab.parquet")
            atomic_parquet(rehab, path)
            event_receipt = {"symbol": symbol, "path": path.name, "rows": len(rehab),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fetched_at": datetime.now(timezone.utc).isoformat()}
            return receipt, event_receipt
        finally:
            context.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect, symbol, ktype): symbol for symbol, ktype in requests.items()}
        for future in as_completed(futures):
            receipt, events = future.result()
            manifest["entries"].append(receipt)
            manifest["rehab"].append(events)
            manifest["entries"].sort(key=lambda item: item["symbol"])
            manifest["rehab"].sort(key=lambda item: item["symbol"])
            atomic_json(manifest, output_dir / "source_manifest.json")
            if progress:
                progress({"symbol_complete": receipt["symbol"], "rows": receipt["rows"]})
    manifest["complete"] = len(manifest["entries"]) == len(requests)
    atomic_json(manifest, output_dir / "source_manifest.json")
    return manifest
