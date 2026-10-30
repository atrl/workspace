"""Matched minute-own-drawdown screen for fixed-share Stage1 path labels.

Feature construction uses the Qlib minute plane and dated parent state. The
screen trains only mature pre-2021 labels, then evaluates frozen 2021/2022
first-passage local action values with day grouping and block uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare_feature_panel(labels, frame, book, minute_bars, plan):
    columns = ["trade_date", "bar_end_at", "decision_at", "label_available_at",
               "decision_nav", "decision_cash_weight", "decision_own_drawdown",
               "parent_risk_state", "phase", "status"]
    panel = labels.loc[:, columns].copy()
    frame = frame.reset_index(drop=True)
    if len(frame) != len(book["targets"]) or not frame.trade_date.equals(book["targets"].trade_date):
        raise ValueError("blocked: dated frame/target calendar differs")
    if list(frame.columns[3:42]) != plan["base39_features"]:
        raise ValueError("blocked: frozen original39 input identity differs")
    state = frame[["trade_date", *plan["base39_features"], "soxl_weight",
                   "soxx_weight", "cash_weight", "tail_probability",
                   "alpha_probability", "feature_available_at"]].copy()
    state["opening_drawdown"] = book["targets"].known_drawdown.to_numpy(float)
    state["risk_clock"] = book["independent_replay"].risk_clock.to_numpy(float)
    quantities = []
    for day in state.trade_date:
        d = pd.Timestamp(day)
        row = []
        for minute in (47, 32):
            stamp = d + (pd.Timedelta(hours=9, minutes=minute) if minute == 47
                         else pd.Timedelta(hours=10, minutes=minute))
            pos = book["native_positions"].get(stamp)
            if pos is None:
                raise ValueError("blocked: parent decision-state position absent")
            row.extend([pos.get_stock_amount(s) if pos.check_stock(s) else 0.
                        for s in ("SOXL", "SOXX")])
        quantities.append(row)
    state[["q0_soxl", "q0_soxx", "q1_soxl", "q1_soxx"]] = np.asarray(quantities, float)
    panel = panel.merge(state, on="trade_date", how="left", validate="many_to_one")
    if panel["feature_available_at"].ge(panel["decision_at"]).any():
        raise ValueError("blocked: prior-market feature is late")
    parts = []
    for symbol in ("SOXL", "SOXX"):
        bars = minute_bars.xs(symbol, level="instrument").copy()
        dates = bars.index.normalize()
        logprice = np.log(bars["$close"].astype(float))
        out = pd.DataFrame(index=bars.index)
        for width in (5, 15):
            out[f"minute_{symbol.lower()}_ret{width}"] = logprice - logprice.groupby(dates).shift(width)
        out[f"minute_{symbol.lower()}_log1p_raw_volume"] = np.log1p(bars["$raw_volume"].astype(float))
        out[f"{symbol.lower()}_mark"] = bars["$close"].astype(float)
        out[f"{symbol.lower()}_valid"] = bars["$price_valid"].astype(float)
        parts.append(out)
    minute = parts[0].join(parts[1], how="inner")
    panel = panel.join(minute, on="bar_end_at", how="left")
    if (panel[["soxl_mark", "soxx_mark"]].isna().any().any() or
            not panel[["soxl_valid", "soxx_valid"]].eq(1).all().all()):
        raise ValueError("blocked: Qlib minute mark missing or invalid")
    late = panel.phase.eq("after_1032").to_numpy()
    qsoxl = np.where(late, panel.q1_soxl, panel.q0_soxl)
    qsoxx = np.where(late, panel.q1_soxx, panel.q0_soxx)
    panel["decision_soxl_weight"] = qsoxl * panel.soxl_mark / panel.decision_nav
    panel["decision_soxx_weight"] = qsoxx * panel.soxx_mark / panel.decision_nav
    panel["clock_fraction"] = ((panel.bar_end_at.dt.hour * 60 + panel.bar_end_at.dt.minute) - 570) / 390.
    panel["after_1032"] = late.astype(float)
    panel["own_drawdown_change"] = panel.decision_own_drawdown - panel.opening_drawdown
    cash = panel.decision_cash_weight * panel.decision_nav
    error = np.abs(qsoxl * panel.soxl_mark + qsoxx * panel.soxx_mark + cash - panel.decision_nav)
    if float(error.max()) > 1e-7:
        raise ValueError("blocked: decision weights/quantities/NAV differ")
    required = [*plan["base39_features"], *plan["state_features"],
                *plan["ordinary_minute_features"], plan["candidate_only_feature"]]
    if len(required) != len(set(required)):
        raise ValueError("blocked: duplicated feature column")
    return panel, {"rows": len(panel), "maximum_decision_nav_error_usd": float(error.max()),
                   "minute_features_available_rows": int(panel[plan["ordinary_minute_features"]].notna().all(axis=1).sum())}


def source_variation_gate(panel, plan):
    train = panel.loc[panel.trade_date.dt.year.isin(plan["train_years"]) &
                      panel.label_available_at.notna()]
    feature = plan["candidate_only_feature"]
    ranges = train.groupby("trade_date")[feature].agg(lambda x: float(x.max() - x.min()))
    days = int(ranges.gt(.001).sum())
    centered = train[["trade_date", feature, *plan["ordinary_minute_features"]]].copy()
    columns = [feature, *plan["ordinary_minute_features"]]
    centered[columns] = centered[columns] - centered.groupby("trade_date")[columns].transform("mean")
    correlations = {name: float(centered[feature].corr(centered[name], method="spearman"))
                    for name in plan["ordinary_minute_features"]}
    maximum = max(abs(value) for value in correlations.values() if np.isfinite(value))
    gate = plan["source_variation_gate"]
    passed = (days >= gate["minimum_train_days_with_intraday_own_drawdown_change_range_gt_0_1pp"]
              and maximum < gate["maximum_absolute_rank_correlation_with_single_ordinary_minute_feature"])
    return {"training_days_with_gt_0_1pp_intraday_range":days,
            "maximum_abs_within_day_ordinary_rank_correlation":maximum,
            "correlations":correlations,"passed":bool(passed)}


def _first_positive_actions(frame, soxl, soxx):
    rows = []
    data = frame[["trade_date", "bar_end_at", "soxl_value", "soxx_value",
                  "soxl_tail_damage", "soxx_tail_damage",
                  "soxl_filled_raw_shares", "soxx_filled_raw_shares"]].copy()
    data["pred_soxl"], data["pred_soxx"] = soxl, soxx
    for day, part in data.groupby("trade_date", sort=True):
        part = part.sort_values("bar_end_at")
        trigger = part[["pred_soxl", "pred_soxx"]].max(axis=1).gt(0)
        if not trigger.any():
            rows.append({"trade_date":day,"bar_end_at":pd.NaT,"arm":"HOLD",
                         "raw_shares":0,"value":0.,"tail_damage":0.})
            continue
        chosen = part.loc[trigger].iloc[0]
        arm = "SOXL" if chosen.pred_soxl > chosen.pred_soxx else "SOXX"
        lower = arm.lower()
        rows.append({"trade_date":day,"bar_end_at":chosen.bar_end_at,"arm":arm,
                     "raw_shares":int(chosen[lower + "_filled_raw_shares"]),
                     "value":float(chosen[lower + "_value"]),
                     "tail_damage":float(chosen[lower + "_tail_damage"])})
    return pd.DataFrame(rows)


def _block_lower(values, *, width, replicates, seed, quantile):
    x = np.asarray(values, float)
    n = len(x)
    if n < width:
        raise ValueError("blocked: not enough daily blocks")
    rng = np.random.default_rng(seed)
    count = int(np.ceil(n / width))
    starts = rng.integers(0, n - width + 1, size=(replicates, count))
    offsets = np.arange(width)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(replicates, -1)[:, :n]
    means = x[indices].mean(axis=1)
    return float(np.quantile(means, quantile))


def fit_information_screen(panel, labels, plan):
    full = panel.merge(labels[["trade_date", "bar_end_at", "soxl_value", "soxx_value",
                               "soxl_tail_damage", "soxx_tail_damage",
                               "soxl_filled_raw_shares", "soxx_filled_raw_shares"]],
                       on=["trade_date", "bar_end_at"], how="left", validate="one_to_one")
    end = pd.Timestamp(plan["evaluation_end"])
    eval_known = full.loc[full.trade_date.dt.year.isin(plan["evaluation_years"]) &
                          full.trade_date.le(end)]
    if not eval_known.status.eq("paired_funded").all() or eval_known.label_available_at.isna().any():
        raise ValueError("blocked: evaluation intention lacks mature fixed-share labels")
    first_eval = pd.Timestamp(eval_known.decision_at.min())
    features = [*plan["base39_features"], *plan["state_features"],
                *plan["ordinary_minute_features"]]
    candidate = [*features, plan["candidate_only_feature"]]
    metadata = ["trade_date", "bar_end_at", "label_available_at", "status",
                "soxl_tail_damage", "soxx_tail_damage",
                "soxl_filled_raw_shares", "soxx_filled_raw_shares"]
    packed = pd.concat({"feature":full[candidate],
                        "label":full[plan["targets"]],
                        "metadata":full[metadata]}, axis=1)
    packed.index = pd.MultiIndex.from_arrays(
        [full.decision_at, ["SOXL_SOXX_CASH"] * len(full)],
        names=["datetime", "instrument"])
    dataset = DatasetH(handler=DataHandlerLP.from_df(packed),
        segments={"train":(plan["start"] if "start" in plan else "2018-10-15", "2020-12-31"),
                  "eval":("2021-01-01", plan["evaluation_end"])})
    train = dataset.prepare("train", col_set=["feature", "label", "metadata"])
    evaluation = dataset.prepare("eval", col_set=["feature", "label", "metadata"])
    train = train.loc[pd.to_datetime(train[("metadata", "label_available_at")]).lt(first_eval)].copy()
    train = train.loc[train[("metadata", "status")].eq("paired_funded")].copy()
    evaluation = evaluation.loc[evaluation[("metadata", "trade_date")].le(end)].copy()
    if (not train[("metadata", "trade_date")].dt.year.isin(plan["train_years"]).all() or
            not evaluation[("metadata", "trade_date")].dt.year.isin(plan["evaluation_years"]).all()):
        raise ValueError("blocked: model clock/year source differs")
    days = pd.to_datetime(train[("metadata", "trade_date")]).to_numpy()
    counts = pd.Series(days).value_counts()
    weights = 1. / pd.Series(days).map(counts).to_numpy(float)
    models, predictions, metrics, means = {}, {}, {}, {}
    train_first = train.index.get_level_values("datetime")
    eval_days = pd.to_datetime(evaluation[("metadata", "trade_date")]).to_numpy()
    for target in plan["targets"]:
        y = train[("label", target)].to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError("blocked: training labels not mature finite")
        means[target] = float(np.average(y, weights=weights))
        eval_y = evaluation[("label", target)].to_numpy(float)
        for name, names in (("control", features), ("candidate", candidate)):
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler(), Ridge(alpha=1.0))
            model.fit(train.loc[:, pd.IndexSlice["feature", names]], y,
                      ridge__sample_weight=weights)
            pred = model.predict(evaluation.loc[:, pd.IndexSlice["feature", names]])
            models[(name, target)] = model
            predictions[(name, target)] = pred
            for year in plan["evaluation_years"]:
                mask = pd.DatetimeIndex(eval_days).year == year
                q = pd.DataFrame({"day":eval_days[mask], "se":(eval_y[mask]-pred[mask])**2,
                                  "constant_se":(eval_y[mask]-means[target])**2})
                daily = q.groupby("day").mean()
                metrics[(name, target, year)] = {"days":len(daily),
                    "day_mse":float(daily.se.mean()),
                    "train_mean_mse":float(daily.constant_se.mean())}
    zero = evaluation.loc[:, pd.IndexSlice["feature", candidate]].copy()
    zero[("feature", plan["candidate_only_feature"])] = 0.
    zero_pred = {target:models[("candidate",target)].predict(zero)
                 for target in plan["targets"]}
    names = models[("candidate", "soxl_value")].named_steps["simpleimputer"].get_feature_names_out(candidate)
    where = list(names).index(plan["candidate_only_feature"])
    relative_coefficient = float(models[("candidate", "soxl_value")].named_steps["ridge"].coef_[where] -
                                 models[("candidate", "soxx_value")].named_steps["ridge"].coef_[where])
    action_frame = pd.DataFrame({
        "trade_date":eval_days,
        "bar_end_at":pd.to_datetime(evaluation[("metadata", "bar_end_at")]).to_numpy(),
        "soxl_value":evaluation[("label", "soxl_value")].to_numpy(float),
        "soxx_value":evaluation[("label", "soxx_value")].to_numpy(float),
        "soxl_tail_damage":evaluation[("metadata", "soxl_tail_damage")].to_numpy(float),
        "soxx_tail_damage":evaluation[("metadata", "soxx_tail_damage")].to_numpy(float),
        "soxl_filled_raw_shares":evaluation[("metadata", "soxl_filled_raw_shares")].to_numpy(int),
        "soxx_filled_raw_shares":evaluation[("metadata", "soxx_filled_raw_shares")].to_numpy(int)})
    control_actions=_first_positive_actions(action_frame,predictions[("control","soxl_value")],predictions[("control","soxx_value")])
    candidate_actions=_first_positive_actions(action_frame,predictions[("candidate","soxl_value")],predictions[("candidate","soxx_value")])
    zero_actions=_first_positive_actions(action_frame,zero_pred["soxl_value"],zero_pred["soxx_value"])
    daily=control_actions.merge(candidate_actions,on="trade_date",suffixes=("_control","_candidate"),validate="one_to_one")
    daily=daily.merge(zero_actions,on="trade_date",validate="one_to_one")
    daily=daily.rename(columns={"arm":"arm_zero","bar_end_at":"bar_end_at_zero",
                                "raw_shares":"raw_shares_zero","value":"value_zero",
                                "tail_damage":"tail_damage_zero"})
    daily["candidate_minus_control"] = daily.value_candidate - daily.value_control
    daily["tail_candidate_minus_control"] = daily.tail_damage_candidate - daily.tail_damage_control
    changed = (daily.arm_candidate.ne(daily.arm_control) |
               daily.bar_end_at_candidate.ne(daily.bar_end_at_control) |
               daily.raw_shares_candidate.ne(daily.raw_shares_control))
    zero_changed = (daily.arm_candidate.ne(daily.arm_zero) |
                    daily.bar_end_at_candidate.ne(daily.bar_end_at_zero) |
                    daily.raw_shares_candidate.ne(daily.raw_shares_zero))
    # NaT != NaT is False in pandas Series.ne only after explicit fill.
    changed = (daily.arm_candidate.ne(daily.arm_control) |
               daily.bar_end_at_candidate.fillna(pd.Timestamp("1900-01-01")).ne(
                   daily.bar_end_at_control.fillna(pd.Timestamp("1900-01-01"))) |
               daily.raw_shares_candidate.ne(daily.raw_shares_control))
    zero_changed = (daily.arm_candidate.ne(daily.arm_zero) |
                    daily.bar_end_at_candidate.fillna(pd.Timestamp("1900-01-01")).ne(
                        daily.bar_end_at_zero.fillna(pd.Timestamp("1900-01-01"))) |
                    daily.raw_shares_candidate.ne(daily.raw_shares_zero))
    annual=[]
    for year, part in daily.groupby(daily.trade_date.dt.year):
        annual.append({"year":int(year),"days":len(part),
            "control_action_days":int(part.arm_control.ne("HOLD").sum()),
            "candidate_action_days":int(part.arm_candidate.ne("HOLD").sum()),
            "candidate_control_action_changed_days":int(changed.loc[part.index].sum()),
            "same_model_zero_action_changed_days":int(zero_changed.loc[part.index].sum()),
            "control_value_sum":float(part.value_control.sum()),
            "candidate_value_sum":float(part.value_candidate.sum()),
            "candidate_minus_control_value_sum":float(part.candidate_minus_control.sum()),
            "candidate_minus_control_tail_damage_sum":float(part.tail_candidate_minus_control.sum())})
    uncertainty = plan["uncertainty"]
    lower = _block_lower(daily.candidate_minus_control.to_numpy(float),
        width=uncertainty["block_sessions"],replicates=uncertainty["replicates"],
        seed=uncertainty["seed"],quantile=uncertainty["one_sided_lower_quantile"])
    train_days = int(pd.Series(days).nunique())
    return {"models":models,"predictions":predictions,"zero_predictions":zero_pred,
            "metrics":metrics,"train_means":means,"train_rows":len(train),
            "train_days":train_days,"eval_rows":len(evaluation),
            "eval_days":int(pd.Series(eval_days).nunique()),
            "relative_feature_coefficient":relative_coefficient,
            "prediction_changed_rows":int(np.sum(
                np.abs(predictions[("candidate","soxl_value")] - predictions[("control","soxl_value")]) > 1e-12)),
            "same_model_zero_score_changed_rows":int(np.sum(
                np.abs(predictions[("candidate","soxl_value")] - zero_pred["soxl_value"]) > 1e-12)),
            "daily_actions":daily,"annual_actions":annual,
            "block_lower":lower,"evaluation_day_index":pd.DatetimeIndex(eval_days)}
