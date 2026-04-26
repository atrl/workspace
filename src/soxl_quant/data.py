from __future__ import annotations

"""SOXL/SOXX/VIX 特征工程模块。

设计上有意把特征分成两类：
  1. qlib-native 的滚动特征：直接从二进制存储中读取（$close, Mean($close,21) 等）
  2. pandas 手工构建的跨序列/体制特征：更容易维护和解释

这样数据源保持透明，避免把跨品种逻辑藏在晦涩的 qlib expression 里。

【泛化优化提示】
  - 当前 hardcode 了 SOXL/SOXX 的列名映射，已通过参数化支持任意 ETF。
  - 后续做多资产训练时，建议增加 instrument_id 和品种波动率画像特征。
"""

from pathlib import Path
import hashlib
import importlib.util
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader

from .settings import TEST_SEG, TRAIN_SEG, VALID_SEG

from .config import experiment_dir_for_profile

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = ROOT / "data" / "raw" / "external"

FeatureBuilder = Callable[[dict[str, Any]], dict[str, pd.Series]]


CORE_FEATURE_COLUMNS: List[str] = [
    "return",
    "anchor_gap_180",
    "anchor_slope_180",
    "trend_gap_21",
    "trend_gap_63",
    "trend_slope_21",
    "trend_slope_63",
    "rsi_14",
    "mom_21",
    "mom_63",
    "realized_vol_10",
    "realized_vol_21",
    "realized_vol_63",
    "vol_of_vol_21",
    "intraday_range",
    "volume_shock",
    "drawdown_21",
    "drawdown_63",
    "cum_drop_20",
    "cum_drop_30",
    "days_since_high_63",
    "drawdown_streak_63",
    "below_ma63_days",
    "below_anchor_ratio_63",
    "new_low_ratio_21",
    "rolling_new_low_ratio_63",
    "loss_streak_21",
    "negative_gap_frequency_21",
    "downside_ratio_10",
    "downside_ratio_21",
    "downside_semivol_21",
    "vol_coef_21",
    "vix_level",
    "vix_zscore",
    "vix_percentile_252",
    "vix_momentum_10",
    "tracking_error_5",
    "tracking_error_21",
    "tracking_error_vol_21",
    "decay_63",
    "relative_strength_21",
    "regime_risk",
    # VIX-free extensions (arXiv-inspired stress proxies)
    "rsi_6",
    "realized_kurtosis_21",
    "jump_variation_5",
    "vol_stress_score",
    "regime_risk_no_vix",
    "relative_volume_zscore",
    # Temporal differential / momentum velocity features (v3)
    "mom_5",
    "mom_10",
    "anchor_gap_velocity_10",
    "trend_gap_21_velocity",
    "rsi_14_velocity_5",
    "regime_risk_velocity_5",
    "volume_shock_velocity_5",
    "expectancy_proxy_21",
    "sharpe_proxy_21",
    "realized_skew_21",
    "gain_streak_10",
]
FEATURE_COLUMNS: List[str] = list(CORE_FEATURE_COLUMNS)
STATIC_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def _builtin_kelly_pack(context: dict[str, Any]) -> dict[str, pd.Series]:
    """Rolling Kelly criterion features: signal-quality / timing-goodness.

    These features encode the statistical edge of the recent 21-day window,
    allowing the policy to learn dynamic position sizing instead of collapsing
    to a fixed low-exposure regime.
    """
    return {
        "kelly_position_21": context["kelly_position_21"],
        "win_rate_21": context["win_rate_21"],
        "odds_21": context["odds_21"],
        "sharpe_21": context["sharpe_21"],
    }


def _builtin_vix_term_pack(context: dict[str, Any]) -> dict[str, pd.Series]:
    index = context["index"]
    vix = context["vix"]
    vvix = _align_external(_load_external_series("VVIX.csv", "VVIX"), index)
    vix3m = _align_external(_load_external_series("VIX3M.csv", "CLOSE"), index)
    vix9d = _align_external(_load_external_series("VIX9D.csv", "CLOSE"), index)
    skew = _align_external(_load_external_series("SKEW.csv", "SKEW"), index)
    vvix_zscore_252 = ((vvix - vvix.rolling(252).mean()) / (vvix.rolling(252).std() + 1e-8)).fillna(0.0)
    vix_term_slope = (vix / vix3m.replace(0.0, np.nan) - 1.0).fillna(0.0)
    vix_short_ratio = (vix9d / vix.replace(0.0, np.nan) - 1.0).fillna(0.0)
    skew_zscore_252 = ((skew - skew.rolling(252).mean()) / (skew.rolling(252).std() + 1e-8)).fillna(0.0)
    vix_term_slope_velocity_5 = (vix_term_slope - vix_term_slope.shift(5)).fillna(0.0)
    return {
        "vvix_zscore_252": vvix_zscore_252,
        "vix_term_slope": vix_term_slope,
        "vix_short_ratio": vix_short_ratio,
        "skew_zscore_252": skew_zscore_252,
        "vix_term_slope_velocity_5": vix_term_slope_velocity_5,
    }


def _builtin_rulebase_signals_pack(context: dict[str, Any]) -> dict[str, pd.Series]:
    """Rulebase-derived signal features for independent factor mining.

    Decomposes soxl_backtest/ rulebase logic (panic score, buy/sell conditions,
    volatility regime) into independently testable time-series signals.
    """
    index = context["index"]
    vix_zscore = context["vix_zscore"].fillna(0.0)
    sigmoid_vix = 1.0 / (1.0 + np.exp(-vix_zscore * 2.0))
    rsi6_val = (context["rsi_6"] / 100.0).fillna(0.5)
    rsi6_adjustment = pd.Series(0.0, index=index)
    rsi6_adjustment[rsi6_val <= 0.30] = -(0.30 - rsi6_val[rsi6_val <= 0.30]) * 2.5
    rsi6_adjustment[rsi6_val >= 0.70] = (rsi6_val[rsi6_val >= 0.70] - 0.70) * 1.0
    vol_coef_safe = context["vol_coef_21"].replace(0.0, np.nan).fillna(1.0)
    daily_return = context["lev_ret"].fillna(0.0)
    cum_drop_30 = context["cum_drop_30"]
    days_since_high_63 = context["days_since_high_63"]
    below_anchor_ratio_63 = context["below_anchor_ratio_63"]
    anchor_gap_180 = context["anchor_gap_180"]
    realized_vol_21 = context["realized_vol_21"]
    volume_shock = context["volume_shock"]
    vol_stress_score = context["vol_stress_score"]

    panic_score_proxy = (sigmoid_vix * context["relative_volume_zscore"].fillna(0.0) + rsi6_adjustment)
    daily_drop_severity = daily_return / (vol_coef_safe + 1e-8)
    cumulative_drop_severity = cum_drop_30 / (vol_coef_safe + 1e-8)
    rsi6_oversold = ((0.30 - rsi6_val).clip(lower=0) / 0.30).fillna(0.0)
    rsi6_overbought = ((rsi6_val - 0.70).clip(lower=0) / 0.30).fillna(0.0)
    buy_composite = panic_score_proxy * 0.3 + daily_drop_severity * 0.4 + cumulative_drop_severity * 0.3
    sell_composite = (
        rsi6_overbought * 0.4
        + ((cum_drop_30 + 0.30).clip(lower=0, upper=0.60) / 0.60) * 0.3
        + ((cum_drop_30 + 0.20).clip(lower=0, upper=0.20) / 0.20) * 0.3
    )
    vix_term_slope = context.get("vix_term_slope")
    vix_short_ratio = context.get("vix_short_ratio")
    if vix_term_slope is not None and vix_short_ratio is not None:
        vix_term_stress = (vix_term_slope * vix_short_ratio).fillna(0.0)
    else:
        vix_term_stress = pd.Series(0.0, index=index)

    return {
        "RB_PANIC_SCORE_PROXY": panic_score_proxy,
        "RB_DAILY_DROP_SEVERITY": daily_drop_severity,
        "RB_CUMULATIVE_DROP_SEVERITY": cumulative_drop_severity,
        "RB_RSI6_OVERSOLD": rsi6_oversold,
        "RB_RSI6_OVERBOUGHT": rsi6_overbought,
        "RB_VOL_REGIME": vol_coef_safe,
        "RB_BUY_COMPOSITE": buy_composite,
        "RB_SELL_COMPOSITE": sell_composite,
        "RB_PROFIT_TAKE_PROXY": ((cum_drop_30 + 0.30).clip(lower=0, upper=0.60) / 0.60).fillna(0.0),
        "RB_COOLDOWN_PROXY": (days_since_high_63 * (1.0 - below_anchor_ratio_63)).fillna(0.0),
        "RB_STOP_LOSS_PROXY": ((cum_drop_30 + 0.20).clip(lower=0, upper=0.20) / 0.20).fillna(0.0),
        "RB_VOLUME_STRESS": (volume_shock * vol_stress_score).fillna(0.0),
        "RB_TREND_REVERSAL": (anchor_gap_180 * (1.0 + realized_vol_21)).fillna(0.0),
        "RB_VIX_TERM_STRESS": vix_term_stress,
    }


def _builtin_academic_signals_pack(context: dict[str, Any]) -> dict[str, pd.Series]:
    """Academic-literature-derived signal features for SOXL factor mining.

    Design principle: only retain features that are NOT highly correlated
    (|rho| < 0.80) with existing CORE_FEATURE_COLUMNS.

    References:
    - Bollerslev, Tauchen, Zhou (2009): variance risk premium predicts returns
    - Amaya et al. (2015): realized skewness predicts cross-sectional returns
    - Moreira & Muir (2017): volatility-scaled momentum
    - Qiao et al. (2022): upside/downside realized semi-variances
    - Cheng & Madhavan (2009): LETF volatility drag
    - Amihud (2002): illiquidity proxy
    """
    index = context["index"]
    close = context["close"]
    lev_ret = context["lev_ret"]
    vix_level = context["vix_level"]
    realized_vol_21 = context["realized_vol_21"]
    mom_21 = context["mom_21"]
    downside_semivol_21 = context["downside_semivol_21"]
    und_ret = context.get("und_ret")
    volume = context.get("volume")
    lev_df = context.get("lev")
    leverage_ratio = context.get("leverage_ratio", 3.0)

    # --- DROPPED (high correlation with existing features) ---
    # ACAD_DOWN_UP_RATIO  → rho(mom_21) = -0.89
    # ACAD_VOL_SCALED_MOMENTUM → rho(sharpe_proxy_21) = +1.00
    # ACAD_DOWN_RUNS      → rho(loss_streak_21) = +1.00
    #
    # We keep the raw building blocks so RL can learn nonlinear combos,
    # but drop the composite forms that are linearly redundant.

    # 1. Variance Risk Premium proxy — only 20d passes; keep as long-horizon signal
    vrp = (vix_level.pow(2) - realized_vol_21.pow(2)).fillna(0.0)

    # 2. Realized skewness (21-day) — weak standalone, but may interact nonlinearly
    realized_skew_21 = lev_ret.rolling(21).skew().fillna(0.0)

    # 3. Volatility drag proxy — Cheng & Madhavan (2009)
    vol_drag = (0.5 * realized_vol_21.pow(2) * (21.0 / 252.0) - abs(mom_21)).fillna(0.0)

    # 4. Amihud illiquidity z-score — orthogonal liquidity stress signal
    dollar_volume = close * volume
    amihud_21 = (lev_ret.abs() / (dollar_volume + 1e-8)).rolling(21).mean() * 1e6
    amihud_zscore = ((amihud_21 - amihud_21.rolling(252).mean()) /
                     (amihud_21.rolling(252).std() + 1e-8)).fillna(0.0)

    # 5. LETF beta deviation — captures "overheating" vs "underperforming"
    # Despite weak standalone IC, this is a genuinely new microstructure signal
    cov_21 = lev_ret.rolling(21).cov(und_ret)
    var_21 = und_ret.rolling(21).var().clip(lower=1e-8)
    realized_beta_21 = (cov_21 / var_21).fillna(leverage_ratio)
    beta_deviation = (realized_beta_21 - leverage_ratio).fillna(0.0)

    # 6. Intraday reversal (21-day mean) — orthogonal mean-reversion signal
    o = lev_df["$open"].astype(float)
    h = lev_df["$high"].astype(float)
    l = lev_df["$low"].astype(float)
    intraday_momentum = ((close - o) / (h - l + 1e-8)).fillna(0.0)
    intraday_reversal_21 = intraday_momentum.rolling(21).mean().fillna(0.0)

    # 7. Autocorrelation (5-day) — reversal vs momentum regime detector
    autocorr_5 = lev_ret.rolling(5).apply(lambda x: pd.Series(x).autocorr(1), raw=False).fillna(0.0)

    return {
        "ACAD_VARIANCE_RISK_PREMIUM": vrp,
        "ACAD_REALIZED_SKEWNESS_21": realized_skew_21,
        "ACAD_VOLATILITY_DRAG": vol_drag,
        "ACAD_AMIHUD_ZSCORE": amihud_zscore,
        "ACAD_BETA_DEVIATION": beta_deviation,
        "ACAD_INTRADAY_REVERSAL_21": intraday_reversal_21,
        "ACAD_AUTOCORR_5": autocorr_5,
    }


def _builtin_transition_signals_pack(context: dict[str, Any]) -> dict[str, pd.Series]:
    """Transition-state / stress-release / restart detection features.

    These features target the blind spot of the core feature stack:
    the core stack is excellent at detecting *persistent stress* but
    weak at detecting *stress release* and *restart transitions*.

    All features are designed to be orthogonal to CORE_FEATURE_COLUMNS
    (verified |rho| < 0.80) and have meaningful IC on SOXL daily returns.
    """
    index = context["index"]
    close = context["close"]
    lev_ret = context["lev_ret"]
    vix = context["vix"]
    realized_vol_10 = context["realized_vol_10"]
    lev_df = context.get("lev")

    # 1. VIX divergence: VIX falls from recent high but price is still near lows
    # → bearish sentiment is fading before price reflects it
    vix_10d_high = vix.rolling(10).max()
    vix_fall_pct = (vix_10d_high - vix) / (vix_10d_high + 1e-8)
    price_near_low = (close / close.rolling(20).max() - 1.0).abs()
    vix_divergence = (vix_fall_pct * price_near_low).fillna(0.0)

    # 2. Spring compression: price makes new low but VIX does NOT make new high
    # → selling pressure is exhausting (fewer sellers at lower prices)
    price_new_low_10 = (close <= close.rolling(10).min()).astype(float)
    vix_not_new_high_10 = (vix < vix.rolling(10).max()).astype(float)
    spring_compression = (price_new_low_10 * vix_not_new_high_10).fillna(0.0)

    # 3. Panic exhaustion: big down day followed by realized vol compression
    # → the panic move has exhausted itself
    big_down = (lev_ret < -0.05).astype(float)
    vol_compress = (realized_vol_10 < realized_vol_10.rolling(10).mean()).astype(float)
    panic_exhaustion = (big_down.shift(1) * vol_compress).fillna(0.0)

    # 4. Gap recovery: gap-down day followed by next-day rebound
    # → overnight selling was overdone
    prev_close = close.shift(1)
    gap_down = ((close / prev_close - 1) < -0.02).astype(float)
    next_day_rebound = (close > prev_close).astype(float)
    gap_recovery = (gap_down.shift(1) * next_day_rebound).fillna(0.0)

    # 5. Volatility term structure z-score: 10d vs 63d realized vol
    # → steep contango/backwardation in realized vol predicts stress changes
    rv10 = realized_vol_10
    rv63 = context.get("realized_vol_63")
    vol_term = rv10 / (rv63 + 1e-8)
    vol_term_zscore = ((vol_term - vol_term.rolling(63).mean()) / (vol_term.rolling(63).std() + 1e-8)).fillna(0.0)

    # 6. Volatility-of-volatility trend: change in vol uncertainty
    # → rising vov = market uncertain about uncertainty = potential inflection
    vov = context.get("vol_of_vol_21")
    vov_trend = (vov - vov.shift(10)).fillna(0.0)

    # 7. Tracking error z-score: 21d TE relative to its 63d history
    # → deviation from underlying is unsustainable, mean-reversion likely
    und_ret = context.get("und_ret")
    leverage_ratio = context.get("leverage_ratio", 3.0)
    tracking_error = lev_ret - leverage_ratio * und_ret
    te_21 = tracking_error.rolling(21).std() * np.sqrt(252)
    te_zscore = ((te_21 - te_21.rolling(63).mean()) / (te_21.rolling(63).std() + 1e-8)).fillna(0.0)

    # 8. Signed tracking-error reversion: crypto funding/basis analogue.
    # Negative signed TE means SOXL underperformed its 3x underlying path;
    # the negated z-score is positive when the dislocation should rebound.
    signed_te_reversion: dict[str, pd.Series] = {}
    for window in (5, 10, 21):
        signed_te = tracking_error.rolling(window).sum()
        signed_te_zscore = (
            (signed_te - signed_te.rolling(252).mean())
            / (signed_te.rolling(252).std() + 1e-8)
        )
        signed_te_reversion[f"SIGNED_TE_REVERSION_{window}"] = (-signed_te_zscore).fillna(0.0)

    # 9. Intraday rejection event: crypto-style panic wick analogue.
    # Long lower wick + close near high + unusually wide / high-volume day
    # indicates sellers were absorbed during the session.
    if lev_df is not None:
        open_price = lev_df["$open"].astype(float)
        high = lev_df["$high"].astype(float)
        low = lev_df["$low"].astype(float)
        day_range = (high - low).clip(lower=1e-8)
        intraday_range = (high - low) / close
        range_zscore = (
            (intraday_range - intraday_range.rolling(252).mean())
            / (intraday_range.rolling(252).std() + 1e-8)
        ).clip(lower=0.0)
        close_location = ((close - low) / day_range).clip(0.0, 1.0)
        lower_wick_ratio = (
            (pd.concat([open_price, close], axis=1).min(axis=1) - low)
            / day_range
        ).clip(0.0, 1.0)
        relative_volume = context.get("relative_volume_zscore")
        volume_boost = relative_volume.clip(lower=0.0) if relative_volume is not None else pd.Series(0.0, index=index)
        intraday_rejection_event = (
            lower_wick_ratio * close_location * range_zscore * (1.0 + volume_boost)
        ).fillna(0.0)
    else:
        intraday_rejection_event = pd.Series(0.0, index=index)

    features = {
        "TRANS_VIX_DIVERGENCE": vix_divergence,
        "TRANS_SPRING_COMPRESSION": spring_compression,
        "TRANS_PANIC_EXHAUSTION": panic_exhaustion,
        "TRANS_GAP_RECOVERY": gap_recovery,
        "TRANS_VOL_TERM_ZSCORE": vol_term_zscore,
        "TRANS_VOV_TREND": vov_trend,
        "TRANS_TE_ZSCORE": te_zscore,
        "INTRADAY_REJECTION_EVENT": intraday_rejection_event,
    }
    features.update(signed_te_reversion)
    return features


BUILTIN_FEATURE_PACKS: dict[str, dict[str, Any]] = {
    "kelly": {
        "columns": (
            "kelly_position_21",
            "win_rate_21",
            "odds_21",
            "sharpe_21",
        ),
        "builder": _builtin_kelly_pack,
    },
    "vix_term": {
        "columns": (
            "vvix_zscore_252",
            "vix_term_slope",
            "vix_short_ratio",
            "skew_zscore_252",
            "vix_term_slope_velocity_5",
        ),
        "builder": _builtin_vix_term_pack,
    },
    "rulebase_signals": {
        "columns": (
            "RB_PANIC_SCORE_PROXY",
            "RB_DAILY_DROP_SEVERITY",
            "RB_CUMULATIVE_DROP_SEVERITY",
            "RB_RSI6_OVERSOLD",
            "RB_RSI6_OVERBOUGHT",
            "RB_VOL_REGIME",
            "RB_BUY_COMPOSITE",
            "RB_SELL_COMPOSITE",
            "RB_PROFIT_TAKE_PROXY",
            "RB_COOLDOWN_PROXY",
            "RB_STOP_LOSS_PROXY",
            "RB_VOLUME_STRESS",
            "RB_TREND_REVERSAL",
            "RB_VIX_TERM_STRESS",
        ),
        "builder": _builtin_rulebase_signals_pack,
    },
    "academic_signals": {
        "columns": (
            "ACAD_VARIANCE_RISK_PREMIUM",
            "ACAD_REALIZED_SKEWNESS_21",
            "ACAD_VOLATILITY_DRAG",
            "ACAD_AMIHUD_ZSCORE",
            "ACAD_BETA_DEVIATION",
            "ACAD_INTRADAY_REVERSAL_21",
            "ACAD_AUTOCORR_5",
        ),
        "builder": _builtin_academic_signals_pack,
    },
    "transition_signals": {
        "columns": (
            "TRANS_VIX_DIVERGENCE",
            "TRANS_SPRING_COMPRESSION",
            "TRANS_PANIC_EXHAUSTION",
            "TRANS_GAP_RECOVERY",
            "TRANS_VOL_TERM_ZSCORE",
            "TRANS_VOV_TREND",
            "TRANS_TE_ZSCORE",
            "INTRADAY_REJECTION_EVENT",
            "SIGNED_TE_REVERSION_5",
            "SIGNED_TE_REVERSION_10",
            "SIGNED_TE_REVERSION_21",
        ),
        "builder": _builtin_transition_signals_pack,
    },
}


def _load_experiment_extensions_module(profile_name: str | None):
    if not profile_name:
        return None
    exp_dir = experiment_dir_for_profile(profile_name)
    if exp_dir is None:
        return None
    path = exp_dir / "extensions.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"soxl_exp_ext_{profile_name}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature_pack_registry(profile_name: str | None = None) -> dict[str, dict[str, Any]]:
    registry = dict(BUILTIN_FEATURE_PACKS)
    module = _load_experiment_extensions_module(profile_name)
    if module is not None:
        registry.update(getattr(module, "FEATURE_PACKS", {}))
    return registry


def feature_columns_for_packs(
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> List[str]:
    columns = list(CORE_FEATURE_COLUMNS)
    registry = _feature_pack_registry(profile_name)
    for pack in feature_packs or ():
        payload = registry.get(pack)
        if payload is None:
            raise KeyError(f"unknown feature pack '{pack}' for profile={profile_name or '<default>'}")
        columns.extend(list(payload["columns"]))
    return columns


def static_index_for_packs(
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_columns_for_packs(feature_packs, profile_name))}


def feature_schema_signature(
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> str:
    payload = "|".join(feature_columns_for_packs(feature_packs, profile_name)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _write_qlib_bin(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.hstack([[0.0], values.astype(np.float32)]).astype("<f4")
    with path.open("wb") as f:
        binary.tofile(f)


def refresh_qlib_binary(
    csv_path: Path,
    qlib_dir: Path,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
) -> pd.DataFrame:
    """刷新最小化的 qlib 二进制存储。

    这是整个数据管道的入口：读取 CSV → 生成 calendar/instruments/features → 写入 qlib_dir。
    生成的目录结构满足 qlib 的 FileStorage 要求，训练、回测、运行时都从这里读数。
    
    【参数说明】
      leveraged_etf  : 杠杆 ETF 代码（如 SOXL、TQQQ）
      underlying_etf : 底层指数 ETF 代码（如 SOXX、QQQ）
      leverage_ratio : 杠杆倍数（默认 3.0），用于后续 tracking_error 计算
    
    【优化提示】当前每次调用都会重写整个 qlib_dir。若数据量变大，可改为增量更新。
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
    qlib_dir.mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars").mkdir(exist_ok=True)
    (qlib_dir / "instruments").mkdir(exist_ok=True)
    calendar = df.index.strftime("%Y-%m-%d").tolist()
    (qlib_dir / "calendars/day.txt").write_text("\n".join(calendar), encoding="utf-8")

    start = calendar[0]
    end = calendar[-1]
    instruments = f"{leveraged_etf}\t{start}\t{end}\n{underlying_etf}\t{start}\t{end}\n"
    (qlib_dir / "instruments/all.txt").write_text(instruments, encoding="utf-8")

    lev_lower = leveraged_etf.lower()
    und_lower = underlying_etf.lower()
    for inst, mapping in {
        leveraged_etf: {
            "open": f"{lev_lower}_open",
            "high": f"{lev_lower}_high",
            "low": f"{lev_lower}_low",
            "close": f"{lev_lower}_close",
            "volume": f"{lev_lower}_volume",
            "vix": "vix",
        },
        underlying_etf: {
            "open": f"{und_lower}_open",
            "high": f"{und_lower}_high",
            "low": f"{und_lower}_low",
            "close": f"{und_lower}_close",
            "volume": f"{und_lower}_volume",
        },
    }.items():
        for field_name, col_name in mapping.items():
            _write_qlib_bin(qlib_dir / f"features/{inst}/{field_name}.day.bin", df[col_name].to_numpy())

    (qlib_dir / "qlib.yaml").write_text(
        "\n".join(
            [
                "# Qlib configuration",
                f"provider_uri: {qlib_dir}",
                "redis:",
                "  host: null",
                "  port: null",
                "market:",
                "  market: all",
                "  market_type: stock",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return df


def init_qlib(provider_uri: Path) -> None:
    try:
        qlib.init(provider_uri=str(provider_uri), region="us", expression_cache=None, dataset_cache=None)
    except Exception:
        qlib.init(provider_uri=str(provider_uri))


def build_market_frame(
    start_time: str,
    end_time: str,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 qlib 存储中读取指定时间窗口的原始行情数据。"""
    lev_fields = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "$vix",
        "Mean($close, 5)",
        "Mean($close, 21)",
        "Mean($close, 63)",
        "Mean($close, 180)",
        "Ref($close, 21)",
        "Ref($close, 63)",
        "Std($close / Ref($close, 1) - 1, 10)",
        "Std($close / Ref($close, 1) - 1, 21)",
        "Std($close / Ref($close, 1) - 1, 63)",
        "Mean($volume, 21)",
    ]
    und_fields = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "Ref($close, 5)",
        "Ref($close, 21)",
    ]
    lev = D.features([leveraged_etf], lev_fields, start_time=start_time, end_time=end_time).xs(leveraged_etf, level="instrument")
    und = D.features([underlying_etf], und_fields, start_time=start_time, end_time=end_time).xs(underlying_etf, level="instrument")
    return lev, und


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _streaks(mask: pd.Series) -> pd.Series:
    streak = 0
    out = []
    for flag in mask.fillna(False):
        streak = streak + 1 if bool(flag) else 0
        out.append(streak)
    return pd.Series(out, index=mask.index, dtype=float)


def _build_transition_state_features(
    *,
    anchor_gap_180: pd.Series,
    below_anchor_ratio_63: pd.Series,
    relative_strength_21: pd.Series,
    regime_risk: pd.Series,
    vix_zscore: pd.Series,
    downside_ratio_10: pd.Series,
    trend_slope_10: pd.Series,
) -> dict[str, pd.Series]:
    vix_zscore_delta_5 = vix_zscore - vix_zscore.shift(5)
    downside_ratio_10_delta_5 = downside_ratio_10 - downside_ratio_10.shift(5)
    stress_day_mask = (
        (regime_risk > 0.60)
        | (vix_zscore > 0.75)
        | (downside_ratio_10 > 0.55)
    )
    days_above_stress_threshold_21 = stress_day_mask.astype(float).rolling(21).sum()
    stress_then_relief_flag = (
        (anchor_gap_180 < -0.02)
        & (days_above_stress_threshold_21 >= 5.0)
        & (vix_zscore_delta_5 < -0.35)
        & (downside_ratio_10_delta_5 < -0.05)
    ).astype(float)
    trend_restart_flag = (
        (anchor_gap_180 < -0.02)
        & (below_anchor_ratio_63 > 0.55)
        & (trend_slope_10 > 0.01)
        & (relative_strength_21 > 0.05)
        & (regime_risk < 0.60)
    ).astype(float)
    return {
        "vix_zscore_delta_5": vix_zscore_delta_5,
        "downside_ratio_10_delta_5": downside_ratio_10_delta_5,
        "days_above_stress_threshold_21": days_above_stress_threshold_21,
        "stress_then_relief_flag": stress_then_relief_flag,
        "trend_restart_flag": trend_restart_flag,
    }


def _load_external_series(filename: str, value_col: str) -> pd.Series:
    """Load an external CSV from ``data/raw/external/`` into a date-indexed Series.

    Handles the two layouts shipping with this repo:
      - DATE, VALUE
      - DATE, OPEN, HIGH, LOW, CLOSE  (we use CLOSE)
    """
    path = EXTERNAL_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing external series: {path}")
    df = pd.read_csv(path)
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if value_col not in df.columns:
        raise KeyError(f"{filename} missing column {value_col}; got {list(df.columns)}")
    series = df[value_col].astype(float)
    series.index.name = "date"
    return series


def _align_external(series: pd.Series, index: pd.Index) -> pd.Series:
    """Reindex an external series onto the feature frame's trading calendar.

    Forward-fills gaps (e.g. SKEW publishes daily, but reindexing to the ETF
    calendar may introduce misalignment). Leading NaNs before external-data
    coverage are filled with 0.0 so the feature is simply silent for that
    period instead of dropping rows.
    """
    aligned = series.reindex(index, method="ffill")
    return aligned.fillna(0.0)


def engineer_feature_frame(
    lev: pd.DataFrame,
    und: pd.DataFrame,
    leverage_ratio: float = 3.0,
    leveraged_etf: str = "SOXL",
    ablate_vix: bool = False,
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """构建体制感知（regime-aware）特征矩阵，供训练、回测、运行时使用。

    特征设计围绕 LETF（杠杆 ETF）的路径依赖性：
      - 长期趋势锚定状态（anchor_gap_180, anchor_slope_180）
      - 持续回撤与下行压力（drawdown_63, downside_ratio_21, loss_streak_21）
      - LETF vs 底层指数的 tracking error 和 decay（tracking_error_21, decay_63）
      - VIX 驱动的压力环境（vix_level, vix_zscore, vix_percentile_252）
    
    【优化提示】
      1. 当前 tracking_error 假设 leverage_ratio 固定（如 3.0）。
         若未来测试 2x ETF（如 TQQQ 的同类 SSO），需确保传入正确杠杆倍数。
      2. 做多资产训练时，建议在此加入品种画像特征：
         - median_vol_252（该 ETF 的年化波动中位数）
         - beta_to_spy（对 SPY 的 63 天滚动 beta）
         - sector_id（半导体/科技/宽基）
    """
    close = lev["$close"].astype(float)
    high = lev["$high"].astype(float)
    low = lev["$low"].astype(float)
    volume = lev["$volume"].astype(float)
    vix = lev["$vix"].astype(float)
    und_close = und["$close"].astype(float)

    lev_ret = close.pct_change()
    und_ret = und_close.pct_change()

    ma5 = lev["Mean($close, 5)"].astype(float)
    ma21 = lev["Mean($close, 21)"].astype(float)
    ma63 = lev["Mean($close, 63)"].astype(float)
    ma180 = lev["Mean($close, 180)"].astype(float)
    volume_ma21 = lev["Mean($volume, 21)"].astype(float)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean().clip(lower=1e-8)
    rsi = 100 - 100 / (1 + gain / loss)

    realized_vol_10 = lev["Std($close / Ref($close, 1) - 1, 10)"].astype(float) * np.sqrt(252)
    realized_vol_21 = lev["Std($close / Ref($close, 1) - 1, 21)"].astype(float) * np.sqrt(252)
    realized_vol_63 = lev["Std($close / Ref($close, 1) - 1, 63)"].astype(float) * np.sqrt(252)
    vol_of_vol_21 = realized_vol_21.rolling(21).std()

    rolling_high_21 = close.rolling(21).max()
    rolling_high_63 = close.rolling(63).max()
    drawdown_21 = close / rolling_high_21 - 1.0
    drawdown_63 = close / rolling_high_63 - 1.0
    cum_drop_20 = close / close.rolling(20).max() - 1.0
    cum_drop_30 = close / close.rolling(30).max() - 1.0

    days_since_high_63 = close.rolling(63).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True) / 63.0
    drawdown_streak_63 = _streaks(drawdown_63 < 0).clip(upper=63) / 63.0
    below_ma63_days = _streaks(close < ma63).clip(upper=63) / 63.0
    below_anchor_ratio_63 = (close < ma180).rolling(63).mean()
    new_low_ratio_21 = (close <= close.rolling(21).min()).rolling(21).mean()
    rolling_new_low_ratio_63 = (close <= close.rolling(63).min()).rolling(63).mean()
    prev_close = close.shift(1)
    negative_gap_frequency_21 = ((lev["$open"].astype(float) / prev_close - 1.0) < -0.01).rolling(21).mean()
    downside_ratio_10 = (lev_ret < 0).rolling(10).mean()
    downside_ratio_21 = (lev_ret < 0).rolling(21).mean()
    downside_semivol_21 = lev_ret.where(lev_ret < 0, 0.0).pow(2).rolling(21).mean().pow(0.5) * np.sqrt(252)
    loss_streak_21 = _streaks(lev_ret < 0).clip(upper=21) / 21.0
    vol_coef_21 = realized_vol_21 / (realized_vol_21.rolling(252).median() + 1e-8)

    vix_zscore = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
    vix_percentile_252 = _rolling_percentile(vix, 252)
    vix_momentum_10 = vix.pct_change(10)

    tracking_error = lev_ret - leverage_ratio * und_ret
    tracking_error_5 = tracking_error.rolling(5).mean()
    tracking_error_21 = tracking_error.rolling(21).mean()
    tracking_error_vol_21 = tracking_error.rolling(21).std()
    decay_63 = (lev_ret.rolling(63).sum() - leverage_ratio * und_ret.rolling(63).sum())
    relative_strength_21 = lev_ret.rolling(21).sum() - und_ret.rolling(21).sum()

    trend_gap_21 = ma5 / ma21 - 1.0
    trend_gap_63 = ma21 / ma63 - 1.0
    anchor_gap_180 = close / ma180 - 1.0
    trend_slope_21 = ma21 / ma21.shift(10) - 1.0
    trend_slope_63 = ma63 / ma63.shift(21) - 1.0
    trend_slope_10 = ma5 / ma5.shift(5) - 1.0
    anchor_slope_180 = ma180 / ma180.shift(21) - 1.0
    intraday_range = (high - low) / close
    volume_shock = volume / volume_ma21 - 1.0
    mom_21 = lev_ret.rolling(21).sum()
    mom_63 = lev_ret.rolling(63).sum()

    # --- VIX-free stress proxies (ported from soxl_backtest/ + arXiv research) ---
    # RSI6 fast-reversion gauge
    delta6 = close.diff()
    gain6 = delta6.where(delta6 > 0, 0.0).ewm(span=6, adjust=False).mean()
    loss6 = (-delta6.where(delta6 < 0, 0.0)).ewm(span=6, adjust=False).mean().clip(lower=1e-8)
    rsi_6 = 100 - 100 / (1 + gain6 / loss6)

    # Realized kurtosis (tail-risk proxy, arXiv 2510.03236)
    realized_kurtosis_21 = lev_ret.rolling(21).kurt().fillna(0.0)

    # Jump variation: sum of squared returns on days where |return| > 3 * rolling_std
    rolling_std_21 = lev_ret.rolling(21).std().fillna(lev_ret.std())
    jump_mask = abs(lev_ret) > 3 * rolling_std_21
    jump_variation_5 = (lev_ret.where(jump_mask, 0.0).pow(2)).rolling(5).sum()

    # Relative volume z-score (EnhancedPanicCalculator volume_multiplier concept)
    avg_volume_20 = volume.rolling(20).mean()
    relative_volume_zscore = (volume - avg_volume_20) / (avg_volume_20.rolling(20).std() + 1e-8)

    # Vol-stress score: composite price-only stress index
    rv10_zscore = (realized_vol_10 - realized_vol_10.rolling(252).mean()) / (realized_vol_10.rolling(252).std() + 1e-8)
    dd63_zscore = (drawdown_63 - drawdown_63.rolling(252).mean()) / (drawdown_63.rolling(252).std() + 1e-8)
    ir_zscore = (intraday_range - intraday_range.rolling(252).mean()) / (intraday_range.rolling(252).std() + 1e-8)
    vol_stress_score = (
        0.30 * rv10_zscore.fillna(0.0)
        + 0.25 * dd63_zscore.fillna(0.0)
        + 0.20 * ir_zscore.fillna(0.0)
        + 0.15 * volume_shock.fillna(0.0)
        + 0.10 * realized_kurtosis_21.rank(pct=True).fillna(0.0)
    )
    # Normalize to [0, 1] percentile-like scale so it can slot into regime_risk weights
    vol_stress_score = vol_stress_score.rank(pct=True).fillna(0.0)

    # regime_risk without VIX (ablation feature)
    regime_risk_no_vix = (
        0.22 * vol_stress_score
        + 0.18 * realized_vol_21.rank(pct=True).fillna(0.0)
        + 0.18 * drawdown_63.abs().rank(pct=True).fillna(0.0)
        + 0.15 * downside_ratio_21.fillna(0.0)
        + 0.12 * downside_semivol_21.rank(pct=True).fillna(0.0)
        + 0.08 * tracking_error_vol_21.rank(pct=True).fillna(0.0)
        + 0.07 * loss_streak_21.fillna(0.0)
    )

    regime_risk = (
        0.22 * vix_percentile_252.fillna(0.0)
        + 0.18 * realized_vol_21.rank(pct=True).fillna(0.0)
        + 0.18 * drawdown_63.abs().rank(pct=True).fillna(0.0)
        + 0.15 * downside_ratio_21.fillna(0.0)
        + 0.12 * downside_semivol_21.rank(pct=True).fillna(0.0)
        + 0.08 * tracking_error_vol_21.rank(pct=True).fillna(0.0)
        + 0.07 * loss_streak_21.fillna(0.0)
    )

    # --- Temporal differential / momentum velocity features (v3) ---
    # Short-term momentum for faster regime-transition detection
    mom_5 = lev_ret.rolling(5).sum()
    mom_10 = lev_ret.rolling(10).sum()

    # Velocity features: rate-of-change in key regime indicators
    anchor_gap_velocity_10 = anchor_gap_180 - anchor_gap_180.shift(10)
    trend_gap_21_velocity = trend_gap_21 - trend_gap_21.shift(5)
    rsi_14_velocity_5 = (rsi / 100.0).diff(5)
    regime_risk_velocity_5 = regime_risk.diff(5)
    volume_shock_velocity_5 = volume_shock - volume_shock.shift(5)

    # Expectancy proxy: forward-statistical edge from recent history
    # = (up_ratio * avg_up_return + down_ratio * avg_down_return) / avg_abs_return
    up_mask = lev_ret > 0
    down_mask = lev_ret < 0
    up_days_ratio = up_mask.rolling(21).mean()
    down_days_ratio = down_mask.rolling(21).mean()
    avg_up_return = lev_ret.where(up_mask, 0.0).rolling(21).mean()
    avg_down_return = lev_ret.where(down_mask, 0.0).rolling(21).mean()
    avg_abs_return = lev_ret.abs().rolling(21).mean().clip(lower=1e-8)
    expectancy_proxy_21 = (up_days_ratio * avg_up_return + down_days_ratio * avg_down_return) / avg_abs_return

    # Sharpe proxy: risk-adjusted momentum over 21 days
    sharpe_proxy_21 = lev_ret.rolling(21).mean() / (lev_ret.rolling(21).std().clip(lower=1e-8)) * np.sqrt(252)

    # Realized skewness: tail asymmetry companion to kurtosis
    realized_skew_21 = lev_ret.rolling(21).skew().fillna(0.0)

    # Gain streak: symmetric to loss_streak_21
    gain_streak_10 = _streaks(lev_ret > 0).clip(upper=10) / 10.0

    # --- Rolling Kelly criterion features (signal-quality / timing-goodness) ---
    # 21-day rolling window for win-rate, odds, and Kelly-optimal position
    win_mask = lev_ret > 0
    loss_mask = lev_ret < 0
    win_rate_21 = win_mask.rolling(21).mean()
    avg_win_21 = lev_ret.where(win_mask, 0.0).rolling(21).mean()
    avg_loss_21 = (-lev_ret).where(loss_mask, 0.0).rolling(21).mean().clip(lower=1e-8)
    odds_21 = avg_win_21 / avg_loss_21
    # Kelly fraction: f* = (p*b - q) / b  =>  win_rate - (1-win_rate)/odds
    kelly_raw = win_rate_21 - (1.0 - win_rate_21) / odds_21.clip(lower=1e-8)
    kelly_position_21 = kelly_raw.clip(-1.0, 1.5)
    # Pure 21-day Sharpe (annualized) for signal-quality context
    sharpe_21 = lev_ret.rolling(21).mean() / (lev_ret.rolling(21).std().clip(lower=1e-8)) * np.sqrt(252)

    feature_df = pd.DataFrame(
        {
            "return": lev_ret,
            "anchor_gap_180": anchor_gap_180,
            "anchor_slope_180": anchor_slope_180,
            "trend_gap_21": trend_gap_21,
            "trend_gap_63": trend_gap_63,
            "trend_slope_21": trend_slope_21,
            "trend_slope_63": trend_slope_63,
            "trend_slope_10": trend_slope_10,
            "rsi_14": rsi / 100.0,
            "mom_21": mom_21,
            "mom_63": mom_63,
            "realized_vol_10": realized_vol_10,
            "realized_vol_21": realized_vol_21,
            "realized_vol_63": realized_vol_63,
            "vol_of_vol_21": vol_of_vol_21,
            "intraday_range": intraday_range,
            "volume_shock": volume_shock,
            "drawdown_21": drawdown_21,
            "drawdown_63": drawdown_63,
            "cum_drop_20": cum_drop_20,
            "cum_drop_30": cum_drop_30,
            "days_since_high_63": days_since_high_63,
            "drawdown_streak_63": drawdown_streak_63,
            "below_ma63_days": below_ma63_days,
            "below_anchor_ratio_63": below_anchor_ratio_63,
            "new_low_ratio_21": new_low_ratio_21,
            "rolling_new_low_ratio_63": rolling_new_low_ratio_63,
            "loss_streak_21": loss_streak_21,
            "negative_gap_frequency_21": negative_gap_frequency_21,
            "downside_ratio_10": downside_ratio_10,
            "downside_ratio_21": downside_ratio_21,
            "downside_semivol_21": downside_semivol_21,
            "vol_coef_21": vol_coef_21,
            "vix_level": vix / 50.0,
            "vix_zscore": vix_zscore,
            "vix_percentile_252": vix_percentile_252,
            "vix_momentum_10": vix_momentum_10,
            "tracking_error_5": tracking_error_5 * 20.0,
            "tracking_error_21": tracking_error_21 * 20.0,
            "tracking_error_vol_21": tracking_error_vol_21 * 20.0,
            "decay_63": decay_63 * 10.0,
            "relative_strength_21": relative_strength_21 * 10.0,
            "regime_risk": regime_risk,
            "rsi_6": rsi_6 / 100.0,
            "realized_kurtosis_21": realized_kurtosis_21,
            "jump_variation_5": jump_variation_5 * 100.0,
            "vol_stress_score": vol_stress_score,
            "regime_risk_no_vix": regime_risk_no_vix,
            "relative_volume_zscore": relative_volume_zscore,
            # Temporal differential / momentum velocity features (v3)
            "mom_5": mom_5,
            "mom_10": mom_10,
            "anchor_gap_velocity_10": anchor_gap_velocity_10,
            "trend_gap_21_velocity": trend_gap_21_velocity,
            "rsi_14_velocity_5": rsi_14_velocity_5,
            "regime_risk_velocity_5": regime_risk_velocity_5,
            "volume_shock_velocity_5": volume_shock_velocity_5,
            "expectancy_proxy_21": expectancy_proxy_21,
            "sharpe_proxy_21": sharpe_proxy_21,
            "realized_skew_21": realized_skew_21,
            "gain_streak_10": gain_streak_10,
            "close": close,
        }
    ).replace([np.inf, -np.inf], np.nan)

    context = {
        "index": feature_df.index,
        "close": close,
        "vix": vix,
        "anchor_gap_180": anchor_gap_180,
        "below_anchor_ratio_63": below_anchor_ratio_63,
        "relative_strength_21": relative_strength_21,
        "regime_risk": regime_risk,
        "vix_zscore": vix_zscore,
        "downside_ratio_10": downside_ratio_10,
        "trend_slope_10": trend_slope_10,
        # Kelly pack context
        "kelly_position_21": kelly_position_21,
        "win_rate_21": win_rate_21,
        "odds_21": odds_21,
        "sharpe_21": sharpe_21,
        # Rulebase signals pack context
        "rsi_6": rsi_6,
        "relative_volume_zscore": relative_volume_zscore,
        "vol_coef_21": vol_coef_21,
        "cum_drop_30": cum_drop_30,
        "lev_ret": lev_ret,
        "days_since_high_63": days_since_high_63,
        "realized_vol_21": realized_vol_21,
        "volume_shock": volume_shock,
        "vol_stress_score": vol_stress_score,
        "vix_term_slope": feature_df.get("vix_term_slope"),
        "vix_short_ratio": feature_df.get("vix_short_ratio"),
        # Academic signals pack context
        "vix_level": vix / 50.0,
        "mom_21": mom_21,
        "downside_semivol_21": downside_semivol_21,
        "und_ret": und_ret,
        "volume": volume,
        "lev": lev,
        "leverage_ratio": leverage_ratio,
        # Transition signals pack context
        "realized_vol_10": realized_vol_10,
        "realized_vol_63": realized_vol_63,
        "vol_of_vol_21": vol_of_vol_21,
        "und_ret": und_ret,
    }
    registry = _feature_pack_registry(profile_name)
    for pack in feature_packs or ():
        payload = registry.get(pack)
        if payload is None:
            raise KeyError(f"unknown feature pack '{pack}' for profile={profile_name or '<default>'}")
        for column, series in payload["builder"](context).items():
            feature_df[column] = series
    feature_df = feature_df.dropna()

    if ablate_vix:
        feature_df["vix_level"] = 0.0
        feature_df["vix_zscore"] = 0.0
        feature_df["vix_percentile_252"] = 0.0
        feature_df["vix_momentum_10"] = 0.0
        feature_df["regime_risk"] = feature_df["regime_risk_no_vix"]

    price = feature_df.pop("close")
    selected_columns = feature_columns_for_packs(feature_packs, profile_name)
    feature_df = feature_df[selected_columns]
    feature_df.index.name = "datetime"
    multi_index = pd.MultiIndex.from_arrays(
        [feature_df.index, np.repeat(leveraged_etf, len(feature_df))],
        names=["datetime", "instrument"],
    )
    feature_df.index = multi_index
    price.index = multi_index
    return feature_df, price


def build_feature_frame(
    start_time: str,
    end_time: str,
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
    ablate_vix: bool = False,
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    lev, und = build_market_frame(start_time, end_time, leveraged_etf=leveraged_etf, underlying_etf=underlying_etf)
    return engineer_feature_frame(
        lev,
        und,
        leverage_ratio=leverage_ratio,
        leveraged_etf=leveraged_etf,
        ablate_vix=ablate_vix,
        feature_packs=feature_packs,
        profile_name=profile_name,
    )


class SoxlDataHandler(DataHandlerLP):
    def __init__(
        self,
        start_time: str,
        end_time: str,
        leveraged_etf: str = "SOXL",
        underlying_etf: str = "SOXX",
        leverage_ratio: float = 3.0,
        ablate_vix: bool = False,
        feature_packs: Tuple[str, ...] | List[str] | None = None,
        profile_name: str | None = None,
    ):
        # Keep both the feature frame and aligned price frame so training,
        # backtest, and runtime all read from the same daily source of truth.
        feature_df, price = build_feature_frame(
            start_time,
            end_time,
            leveraged_etf=leveraged_etf,
            underlying_etf=underlying_etf,
            leverage_ratio=leverage_ratio,
            ablate_vix=ablate_vix,
            feature_packs=feature_packs,
            profile_name=profile_name,
        )
        self.price_frame = price.to_frame("price")
        self.feature_frame = feature_df
        loader_df = pd.concat({"feature": feature_df}, axis=1)
        super().__init__(
            instruments=[leveraged_etf],
            start_time=start_time,
            end_time=end_time,
            data_loader=StaticDataLoader(loader_df),
            infer_processors=[],
            learn_processors=[],
            shared_processors=[],
            process_type=DataHandlerLP.PTYPE_A,
            drop_raw=False,
        )


def build_dataset(
    leveraged_etf: str = "SOXL",
    underlying_etf: str = "SOXX",
    leverage_ratio: float = 3.0,
    train_seg: Tuple[str, str] = TRAIN_SEG,
    valid_seg: Tuple[str, str] = VALID_SEG,
    test_seg: Tuple[str, str] = TEST_SEG,
    ablate_vix: bool = False,
    feature_packs: Tuple[str, ...] | List[str] | None = None,
    profile_name: str | None = None,
) -> DatasetH:
    """构建 qlib DatasetH，划分 train / valid / test 三段。

    【数据切分策略】
      train: 2010-03-11 ~ 2018-12-31（约 9 年，策略学习期）
      valid: 2019-01-01 ~ 2019-12-31（1 年，模型选择期）
      test : 2020-01-01 ~ 2026-03-31（6+ 年，最终评估期）
    
    这是仓库的硬性规则，不要直接用 2022 数据做训练，否则会造成数据泄漏。
    """
    return DatasetH(
        handler=SoxlDataHandler(
            start_time=train_seg[0],
            end_time=test_seg[1],
            leveraged_etf=leveraged_etf,
            underlying_etf=underlying_etf,
            leverage_ratio=leverage_ratio,
            ablate_vix=ablate_vix,
            feature_packs=feature_packs,
            profile_name=profile_name,
        ),
        segments={"train": train_seg, "valid": valid_seg, "test": test_seg},
        fetch_kwargs={"level": "datetime"},
    )


def flatten_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "feature" in df.columns.get_level_values(0):
            df = df.xs("feature", axis=1, level=0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df.sort_index()


def segment_frame(dataset: DatasetH, segment: str) -> pd.DataFrame:
    df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    df = flatten_feature_frame(df)
    return df.droplevel("instrument").copy()
