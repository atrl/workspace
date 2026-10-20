"""Actual Qlib lifecycle/schema regression; synthetic inputs, no campaign replay."""
import hashlib
import json

import pandas as pd
import pytest

from src.workflow import path_area_native_completion as completion


def test_native_before_writer_and_sealed_completed_stage_no_recompute(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R
    from src.backtest.ibkr_tiered import TieredFees
    from src.core.completed_checkpoints import read_completed_checkpoint
    from src.core.native_evidence import native_dataset
    from src.data import build_qlib_store_from_frame
    from src.backtest import path_area_teacher

    def forbidden(*args, **kwargs):
        raise AssertionError("native continuation recomputed an already completed stage")
    for module, names in ((completion.p2, ("_fit_new_states", "_predict_oof", "build_path_area_inputs",
                                            "clocked_smooth_targets", "run_training_shadow")),
                          (path_area_teacher, ("label_local_actions",))):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    days = pd.to_datetime(["2020-01-06", "2020-01-07"])
    stamps = pd.DatetimeIndex([day + pd.Timedelta(clock) for day in days
                              for clock in ("9h45min", "9h47min")])
    prices = pd.DataFrame(10., index=stamps, columns=["SOXL", "SOXX"])
    guarded = pd.concat([prices, pd.DataFrame(10., index=[stamps[-1] + pd.Timedelta("1min")],
                                               columns=prices.columns)])
    fields = ["open", "high", "low", "close", "execution_price", "raw_execution_price"]
    rows = [{"datetime": stamp, "instrument": symbol, **{name: value for name in fields}}
            for symbol in guarded for stamp, value in guarded[symbol].items()]
    provider = tmp_path / "provider"
    build_qlib_store_from_frame(pd.DataFrame(rows), provider_uri=provider, freq="1min", region="us",
        as_of="2020-01-07", field_map={name: name for name in fields}, source="synthetic lifecycle regression")
    tariff = {"tiers": [{"through_shares": None, "per_share": .0035}], "minimum_per_order": .35,
              "maximum_order_fraction": 1., **{key: 0. for key in ("venue_per_share",
              "venue_subdollar_fraction", "clearing_per_share", "clearing_maximum_fraction",
              "sec_sale_fraction", "taf_per_sold_share", "taf_maximum_per_trade", "cat_per_share",
              "commission_pass_through_fraction")}}
    fee = TieredFees(tariff, slippage_bps=2).quote(shares=10, raw_price=10., sell=False, stamp=stamps[1])
    ending = 1000. - fee["total"]
    points = pd.DataFrame({"stamp": stamps, "q_soxl": [0., 10., 10., 10.], "q_soxx": 0.,
        "cash": [1000., ending - 100., ending - 100., ending - 100.], "receivable": 0.,
        "nav": [1000., ending, ending, ending]})
    index = pd.DatetimeIndex([days[0]], name="trade_date")
    trace = pd.DataFrame({"eligibility_reasons": [[]], "filled_order_count": [1],
                          "total_order_fees_usd": [fee["total"]]}, index=index)
    for stage in ("source_target", "projected_target"):
        for asset, weight in zip(("soxl", "soxx", "cash"), (.1, 0., .9)):
            trace[f"{stage}_{asset}_weight"] = weight
    for asset, weight in zip(("soxl", "soxx", "cash"), (100. / ending, 0., (ending - 100.) / ending)):
        trace[f"postfill_{asset}_weight"] = weight
    labels = pd.DataFrame({"label_status": ["mature"]}, index=index)
    frame = pd.DataFrame([[1., .01]],
        index=pd.MultiIndex.from_arrays([index, ["SOXL"]], names=["datetime", "instrument"]),
        columns=pd.MultiIndex.from_tuples([("feature", "mask"), ("label", "delta")]))
    label_objects = {"labels": labels, "action_label_frame": frame,
                     "action_label_dataset": native_dataset(frame, segments={"all": (None, None)})}
    shadow = {"orders": pd.DataFrame([{"stamp": stamps[1], "symbol": "SOXL", "raw_shares": 10,
                                      "sell": False, **fee}]), "points": points,
              "replay": pd.DataFrame({"trade_date": index, "exit_date": [days[1]],
                                      "native_expected_nav": ending}),
              "corporate_receipts": pd.DataFrame(), "targets": pd.DataFrame(),
              "trace": trace, "decisions": pd.DataFrame(index=index), "audit": {"status": "passed"},
              "path_area_inputs": pd.DataFrame({"prefix_mask": [1]}, index=index), "prebranch_state": []}
    card = tmp_path / "failed"
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    with pytest.raises(RuntimeError, match="sealed synthetic failure"):
        with R.start(experiment_name="synthetic_failed", uri=str(card / "qlib_recorder")):
            failed = R.get_recorder()
            failed.save_objects(artifact_path="shadow_python", **{
                name: shadow["audit"] if name == "python_audit" else shadow[name]
                for name in completion.SHADOW_NAMES})
            failed.save_objects(artifact_path="local_labels", **label_objects)
            failed.save_objects(artifact_path="sources", identity={}, load_audit={}, H1_projection={})
            raise RuntimeError("sealed synthetic failure")
    local = card / "qlib_recorder" / str(failed.experiment_id) / failed.id
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"recorder_id": failed.id, "experiment_id": str(failed.experiment_id),
        "recorder_status": "FAILED", "run_metadata_path": (local / "meta.yaml").relative_to(card).as_posix(),
        "run_metadata_sha256": digest(local / "meta.yaml"),
        "entries": [{"path": path.relative_to(card).as_posix(), "sha256": digest(path),
                     "bytes": path.stat().st_size} for path in (local / "artifacts").rglob("*") if path.is_file()]}
    manifest_path = card / "failure_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    kwargs = dict(manifest_sha256=digest(manifest_path), recorder_id=failed.id,
                  experiment_id=str(failed.experiment_id), experiment_name="synthetic_failed",
                  artifacts=completion.CHECKPOINT_NAMES)
    cached, audit = read_completed_checkpoint(card, **kwargs)
    restored_shadow, restored_labels = completion._unpack_cached(cached)
    # Exact source text is retained even when the files are untracked. Notebook
    # identity uses the existing source-cell normalization and excludes conclusions.
    code_root = tmp_path / "code"
    code_root.mkdir()
    script = code_root / "synthetic.py"
    script.write_bytes(b"# exact source bytes\r\nvalue = 1\r\n")
    notebook = code_root / "synthetic.ipynb"
    notebook_body = {"cells": [{"id": "run", "cell_type": "code", "source": ["value = 1\n"]},
                               {"id": "path-p2-conclusion", "cell_type": "markdown", "source": ["draft"]}]}
    notebook.write_text(json.dumps(notebook_body))
    files = {script.name: digest(script), notebook.name: completion.p2._notebook_source_identity(notebook)}
    implementation = {"files": files, "sha256": completion.p2._canonical_sha(files)}
    notebook_body["cells"][-1]["source"] = ["conclusion changed without executable source changes"]
    notebook.write_text(json.dumps(notebook_body))
    snapshot = completion._verified_code_snapshot(code_root, implementation)
    assert snapshot[script.name].encode("utf-8") == script.read_bytes()
    script.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="differs from frozen implementation"):
        completion._verified_code_snapshot(code_root, implementation)
    state = {"native_account_attempts": 0, "native_accounts": 0}
    result, native, bridge = completion._native_then_record(shadow=restored_shadow, labels=restored_labels,
        provider=provider, point_prices=prices, schedule={}, tariff=tariff,
        plan={"common_training_shadow": {"slippage_bps": 2, "initial_cash_usd": 1000}},
        calendar=days, tracking=tmp_path / "completed" / "qlib_recorder", source_identity={"synthetic": True},
        state=state, persist=lambda: None, implementation_identity=implementation,
        code_snapshot=snapshot, source_audit={}, checkpoint_audit=audit)
    assert native["audit"]["status"] == "passed" and native["audit"]["orders"] == 1
    assert state["native_account_attempts"] == state["native_accounts"] == 1
    assert str(result.status) == "FINISHED" and bridge["summary"]["rows"] == 1
    assert bridge["summary"]["unknown_stages"] == []
    assert result.load_object("implementation/code") == snapshot
    assert result.load_object("implementation/identity") == implementation
    assert failed.client.get_run(failed.id).info.status == "FAILED"
    assert digest(local / "meta.yaml") == manifest["run_metadata_sha256"]
    # A writable Recorder blocks the account before a single attempt is charged.
    with R.start(experiment_name="synthetic_guard", uri=str(tmp_path / "guard")):
        with pytest.raises(ValueError, match="no active writable"):
            completion._no_writer()
    victim = local / "artifacts" / "shadow_python" / "points"
    victim.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact differs"):
        read_completed_checkpoint(card, **kwargs)
