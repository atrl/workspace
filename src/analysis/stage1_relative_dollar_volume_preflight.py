"""Fixed Qlib source and novelty screen for SOXL/SOXX traded-value proxy."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shock(value: pd.Series, lookback: int = 20) -> pd.Series:
    logged = np.log(value.where(value > 0))
    prior = logged.shift(1).rolling(lookback, min_periods=lookback).median()
    return logged - prior


def run(root: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.factor.turnover_surprise import prior_median_log_turnover_surprise

    root = Path(root).resolve()
    card = root / "research/stage1_relative_dollar_volume_source_preflight_20261023"
    if (card / "summary.json").exists():
        raise ValueError("fixed preflight already completed")
    plan = json.loads((card / "plan.json").read_text())
    assert plan["frozen"]
    source = plan["source"]
    for role in ("SOXL", "SOXX"):
        origin = root / source[role + "_card"]
        assert _sha(origin / "summary.json") == source[role + "_summary_sha256"]
        summary = json.loads((origin / "summary.json").read_text())
        raw_archive = json.loads((origin / "nas_raw_archive.json").read_text())
        assert raw_archive["status"] == "verified"
        assert raw_archive["NAS_archive_sha256"] == source[role + "_NAS_raw_archive_sha256"]
        assert raw_archive["scoped_restore_all_member_SHA256_parity"]
        assert summary["NAS_readback"]["verified"]
    parent_card = root / source["original39_card"]
    assert _sha(parent_card / "summary.json") == source["original39_summary_sha256"]
    parent_summary = json.loads((parent_card / "summary.json").read_text())
    assert parent_summary["recorder_id"] == source["original39_recorder_id"]
    assert parent_summary["nas_evidence"]["sha256"] == source["original39_NAS_archive_sha256"]
    assert _sha(root / source["Stage1_daily_provider"] / "qlib_store_manifest.json") == source["Stage1_daily_provider_manifest_sha256"]
    raw_paths = {
        "SOXL": root / "data/raw/external/stage1_futu_soxl_turnover_20261023/SOXL_NONE_day.csv",
        "SOXX": root / "data/raw/external/stage1_futu_turnover_20261023/SOXX_NONE_day.csv",
    }
    raw = {}
    source_rows = []
    ratio_medians = {}
    zero_traded_value_with_volume = {}
    for role, path in raw_paths.items():
        assert _sha(path) == source[role + "_raw_sha256"]
        frame = pd.read_csv(path)
        assert frame.code.eq("US." + role).all()
        date = pd.DatetimeIndex(pd.to_datetime(frame.time_key).dt.normalize())
        assert date.is_unique and date.is_monotonic_increasing
        close = frame.close.to_numpy(float)
        volume = frame.volume.to_numpy(float)
        proxy = close * volume
        assert np.isfinite(proxy).all() and (proxy > 0).all()
        actual = frame.turnover.to_numpy(float)
        valid_actual = np.isfinite(actual) & (actual > 0)
        ratio_medians[role] = float(np.median(proxy[valid_actual] / actual[valid_actual]))
        zero_traded_value_with_volume[role] = int(((actual <= 0) & (volume > 0)).sum())
        raw[role] = frame
        source_rows.append(pd.DataFrame({"datetime": date, "instrument": role,
                                         "dollar_volume_proxy": proxy}))
    provider = root / "data/qlib/stage1_relative_dollar_volume_20261023"
    manifest = build_qlib_store_from_frame(pd.concat(source_rows, ignore_index=True),
        provider_uri=provider, freq="day", region="us",
        field_map={"$dollar_volume_proxy": "dollar_volume_proxy"},
        as_of="2026-10-09",
        source="Futu US.SOXL and US.SOXX K_DAY AuType.NONE raw close*volume; raw NAS SHAs=" +
               source["SOXL_NAS_raw_archive_sha256"] + "," + source["SOXX_NAS_raw_archive_sha256"])
    qlib.init(provider_uri=str(provider.resolve()), region="us", kernels=1)
    source_calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time="2026-10-09", freq="day"))
    data = D.features(["SOXL", "SOXX"], ["$dollar_volume_proxy"],
                      start_time="2010-03-11", end_time="2026-10-09", freq="day")
    pair = data["$dollar_volume_proxy"].unstack("instrument").reindex(source_calendar)
    assert len(pair) == len(source_calendar) and set(pair.columns) == {"SOXL", "SOXX"}
    for role, rows in zip(("SOXL", "SOXX"), source_rows):
        expected = rows.set_index("datetime").dollar_volume_proxy.reindex(source_calendar)
        np.testing.assert_allclose(pair[role].to_numpy(float), expected.to_numpy(float),
                                   rtol=1e-6, atol=1e-2, equal_nan=True)
    shock = {role: _shock(pair[role]) for role in ("SOXL", "SOXX")}
    relative = (shock["SOXL"] - shock["SOXX"]).rename("SOXL_minus_SOXX_dollar_volume_shock20")
    qlib.init(provider_uri=str((root / source["Stage1_daily_provider"]).resolve()), region="us", kernels=1)
    parent_calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time="2026-10-09", freq="day"))
    assert source_calendar.equals(parent_calendar)
    with R.uri_context(str((parent_card / "qlib_recorder").resolve())):
        features = R.get_recorder(experiment_id=source["original39_experiment_id"],
            recorder_id=source["original39_recorder_id"]).load_object("inputs/features")
    pd.testing.assert_index_equal(features.index, parent_calendar, check_names=False)
    base = list(parent_summary["input_audit"]["original_features"])
    assert len(base) == 39 and set(base) <= set(features.columns)
    qlib.init(provider_uri=str((root / "data/qlib/stage1_futu_turnover_20261023").resolve()),
              region="us", kernels=1)
    old_rate = D.features(["SOXX"], ["$turnover_rate"], start_time="2010-03-11",
                          end_time="2026-10-09", freq="day")["$turnover_rate"].droplevel("instrument")
    old_surprise = prior_median_log_turnover_surprise(old_rate, lookback=20).reindex(parent_calendar)
    old_surprise.name = "closed_SOXX_turnover_rate_surprise20"
    joined = features[base].join(relative).join(shock["SOXX"].rename("SOXX_dollar_volume_shock20")).join(old_surprise)
    train = joined.loc["2011-01-03":"2022-12-30"]
    held = joined.loc["2023-01-03":"2026-10-09"]
    corr = train.corr(method="spearman")[relative.name]
    abs_base = float(corr.loc[base].abs().max())
    abs_soxx_proxy = float(abs(corr.loc["SOXX_dollar_volume_shock20"]))
    abs_old_rate = float(abs(corr.loc[old_surprise.name]))
    train_known = int(train[relative.name].notna().sum())
    held_known = int(held[relative.name].notna().sum())
    gates = {
        "proxy_positive_finite": bool(pair.stack().gt(0).all()),
        "actual_turnover_median_proxy_ratio_SOXL": abs(ratio_medians["SOXL"] - 1) <= .10,
        "actual_turnover_median_proxy_ratio_SOXX": abs(ratio_medians["SOXX"] - 1) <= .10,
        "train_support_95pct": train_known / len(train) >= plan["gates"]["train_2011_2022_source_known_min_fraction"],
        "held_support_95pct": held_known / len(held) >= plan["gates"]["held_2023_2026_source_known_min_fraction"],
        "not_redundant_original39": abs_base < plan["gates"]["train_max_abs_spearman_original39_below"],
        "not_redundant_SOXX_proxy": abs_soxx_proxy < plan["gates"]["train_abs_spearman_SOXX_only_proxy_shock_below"],
        "not_redundant_old_rate": abs_old_rate < plan["gates"]["train_abs_spearman_closed_SOXX_turnover_rate_surprise_below"],
    }
    decision = "conditional_source_novelty_pass_first_public_unknown" if all(gates.values()) else "closed_prefit_source_or_novelty_gate_failed"
    result = {"experiment": card.name, "decision": decision,
              "plan_sha256": _sha(card / "plan.json"),
              "source_raw_sha256": {role: source[role + "_raw_sha256"] for role in ("SOXL", "SOXX")},
              "provider_manifest_sha256": _sha(provider / "qlib_store_manifest.json"),
              "calendar_rows": len(parent_calendar), "train_rows": len(train),
              "train_known": train_known, "held_rows": len(held), "held_known": held_known,
              "actual_traded_value_proxy_ratio_median": ratio_medians,
              "actual_traded_value_zero_with_positive_volume": zero_traded_value_with_volume,
              "train_max_abs_spearman_original39": abs_base,
              "train_abs_spearman_SOXX_proxy_shock": abs_soxx_proxy,
              "train_abs_spearman_closed_SOXX_rate_surprise": abs_old_rate,
              "gates": gates, "historical_first_public_qualified": False,
              "future_return_reads": 0, "new_model_fits": 0, "new_native_accounts": 0,
              "new_market_requests": 0}
    with R.start(experiment_name=card.name, uri=str((card / "qlib_recorder").resolve())):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, provider_manifest=manifest)
        recorder.save_objects(artifact_path="analysis", relative_shock=relative,
                              SOXL_shock=shock["SOXL"], SOXX_shock=shock["SOXX"],
                              source_pair=pair, result=result)
        R.log_params(source="Futu_close_times_volume_relative_prior20", outcome_reads=0)
        R.log_metrics(train_known=train_known, held_known=held_known,
                      max_abs_spearman_original39=abs_base)
        rid, eid = recorder.id, str(recorder.experiment_id)
    result.update(recorder_id=rid, experiment_id=eid)
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    binding = json.loads((root / source["SOXL_card"] / "nas_binding.json").read_text())
    binding.pop("evidence", None)
    (card / "nas_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    archived = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                                    card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(archived["nas_path"])).split()[0]
    assert remote == archived["sha256"]
    summary = {"experiment": card.name, "decision": decision, "objective_achieved": False,
               "plan_sha256": result["plan_sha256"], "result": "result.json",
               "recorder_id": rid, "experiment_id": eid,
               "NAS_evidence": archived, "NAS_readback": {"verified": True, "sha256": remote},
               "new_model_fits": 0, "new_native_accounts": 0,
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/relative_shock",
                                                               "analysis/SOXL_shock", "analysis/SOXX_shock",
                                                               "analysis/source_pair"]},
               "limitations": ["Raw close times volume approximates traded value; source field itself is zero-coded on many historical days.",
                               "Futu historical first-public values, revisions and delivery clocks remain unqualified.",
                               "No future-return label, fitted factor, funded Stage1 account or independent forward evidence."]}
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nentrypoint: " + card.name + ".ipynb\n"
        "provider_uri: data/qlib/stage1_relative_dollar_volume_20261023\nphase: source_novelty_no_fit\n")
    (card / "conclusion.md").write_text(
        "# Stage1 SOXL/SOXX 相对成交额近似异常：无结果来源检验\n\n"
        "从已 NAS 备份的 Futu 不复权日线，用各 ETF 的收盘价×成交量近似成交额；"
        "对每只 ETF 扣除严格前20个 Qlib 交易日的对数中位数，再取 SOXL 减 SOXX。"
        "这不是供应商报告的成交额字段，也不填补其历史零值。0 新行情、结果读取、拟合和账户。\n\n"
        f"训练 {train_known}/{len(train)}、复用历史 {held_known}/{len(held)} 有值；"
        f"原39项最大绝对训练秩相关 {abs_base:.4f}，与 SOXX 单边成交额近似异常/"
        f"已关闭换手率异常分别为 {abs_soxx_proxy:.4f}/{abs_old_rate:.4f}。"
        f"八项门 {sum(gates.values())}/{len(gates)}，决定 `{decision}`。\n\n"
        "前一完整信号日的数值在次交易日09:29前可得只是研究时钟约定，历史首次发布/修订尚未认证。"
        "若门通过，只允许另冻 SOXL 相对 H5 信息检验；不得直接调仓或声称 SOXX 超额。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    return summary
