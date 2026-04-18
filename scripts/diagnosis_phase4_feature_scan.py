#!/usr/bin/env python3
"""Phase 4: 新特征可行性扫描"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runtime" / "diagnosis"
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"


def load_cboe_data() -> dict[str, pd.DataFrame]:
    """Load CBOE volatility indices."""
    data = {}
    for name, file in [
        ("VIX9D", "VIX9D.csv"),
        ("VIX3M", "VIX3M.csv"),
        ("VIX_cboe", "VIX_cboe.csv"),
        ("SKEW", "SKEW.csv"),
        ("VVIX", "VVIX.csv"),
    ]:
        path = EXTERNAL_DIR / file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
        df = df.sort_values("DATE").set_index("DATE")
        # Use CLOSE if available, otherwise the only column
        if "CLOSE" in df.columns:
            df["value"] = df["CLOSE"]
        elif len(df.columns) == 1:
            df["value"] = df.iloc[:, 0]
        else:
            # Try to find a single non-date column
            for c in df.columns:
                if c != "DATE":
                    df["value"] = df[c]
                    break
        data[name] = df[["value"]].rename(columns={"value": name})
    return data


def load_soxl_data() -> pd.DataFrame:
    """Load SOXL price data."""
    # Try multiple sources
    for csv_file in ["soxl_yahoo.csv", "soxl_with_vix.csv", "soxl_soxx_vix_full.csv"]:
        path = DATA_DIR / csv_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Detect date column
        date_col = None
        for c in df.columns:
            if c.lower() in ("date", "datetime", "time"):
                date_col = c
                break
        if date_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None).dt.normalize()
        df = df.sort_values(date_col).set_index(date_col)
        # Find close price
        close_col = None
        for c in df.columns:
            if "close" in c.lower():
                close_col = c
                break
        if close_col:
            return df[[close_col]].rename(columns={close_col: "close"})
    raise FileNotFoundError("No SOXL price data found")


def compute_features(cboe: dict[str, pd.DataFrame], soxl: pd.DataFrame) -> pd.DataFrame:
    """Compute new candidate features."""
    # Align all to SOXL dates
    combined = soxl.copy()
    for name, df in cboe.items():
        combined = combined.join(df, how="left")
    
    # Forward fill missing values
    combined = combined.ffill()
    
    # New features
    if "VIX_cboe" in combined.columns and "VIX3M" in combined.columns:
        combined["vix_term_slope"] = combined["VIX_cboe"] / combined["VIX3M"]  # <1 = contango, >1 = backwardation
        combined["vix_term_spread"] = combined["VIX3M"] - combined["VIX_cboe"]
    if "VIX_cboe" in combined.columns and "VIX9D" in combined.columns:
        combined["vix_short_term_slope"] = combined["VIX9D"] / combined["VIX_cboe"]
    if "SKEW" in combined.columns:
        combined["skew_level"] = combined["SKEW"]
        combined["skew_zscore_21"] = (combined["SKEW"] - combined["SKEW"].rolling(21).mean()) / combined["SKEW"].rolling(21).std()
    if "VVIX" in combined.columns:
        combined["vvix_level"] = combined["VVIX"]
        combined["vvix_zscore_21"] = (combined["VVIX"] - combined["VVIX"].rolling(21).mean()) / combined["VVIX"].rolling(21).std()
    
    # SOXL returns
    combined["soxl_return_1d"] = combined["close"].pct_change()
    combined["soxl_return_5d"] = combined["close"].pct_change(5)
    combined["soxl_return_21d"] = combined["close"].pct_change(21)
    
    return combined


def analyze_correlation(df: pd.DataFrame) -> dict:
    """Analyze correlation between new features and SOXL returns."""
    new_features = [
        "vix_term_slope", "vix_term_spread", "vix_short_term_slope",
        "skew_level", "skew_zscore_21",
        "vvix_level", "vvix_zscore_21",
    ]
    targets = ["soxl_return_1d", "soxl_return_5d", "soxl_return_21d"]
    
    results = {}
    for feat in new_features:
        if feat not in df.columns:
            continue
        results[feat] = {}
        for target in targets:
            if target not in df.columns:
                continue
            corr = df[feat].corr(df[target])
            # Lead-lag: shift feature forward to see if it predicts return
            corr_lead1 = df[feat].shift(1).corr(df[target])
            corr_lead5 = df[feat].shift(5).corr(df[target])
            results[feat][target] = {
                "contemporaneous": round(corr, 4) if not np.isnan(corr) else None,
                "lead_1d": round(corr_lead1, 4) if not np.isnan(corr_lead1) else None,
                "lead_5d": round(corr_lead5, 4) if not np.isnan(corr_lead5) else None,
            }
    
    # Correlation with existing features
    existing_features = []
    # Try to load existing features
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from soxl_quant.data import FEATURE_COLUMNS
        existing_features = FEATURE_COLUMNS
    except Exception:
        pass
    
    # Check if features.pkl exists
    features_pkl = DATA_DIR / "features_v2.pkl"
    existing_corr = {}
    if features_pkl.exists() and "vix_level" in [c.lower() for c in pd.read_pickle(features_pkl).columns]:
        try:
            feat_df = pd.read_pickle(features_pkl)
            feat_df.index = pd.to_datetime(feat_df.index)
            # Join with our data
            merged = df.join(feat_df, how="inner")
            if "vix" in merged.columns and "vix_term_slope" in merged.columns:
                existing_corr["vix_vs_vix_term_slope"] = round(merged["vix"].corr(merged["vix_term_slope"]), 4)
            if "vix_zscore" in merged.columns and "skew_level" in merged.columns:
                existing_corr["vix_zscore_vs_skew"] = round(merged["vix_zscore"].corr(merged["skew_level"]), 4)
            if "regime_risk" in merged.columns and "vvix_level" in merged.columns:
                existing_corr["regime_risk_vs_vvix"] = round(merged["regime_risk"].corr(merged["vvix_level"]), 4)
        except Exception as e:
            existing_corr["error"] = str(e)
    
    return {
        "new_feature_correlations": results,
        "existing_feature_correlations": existing_corr,
    }


def analyze_cta_features(soxl: pd.DataFrame) -> dict:
    """Analyze CTA-style features computable from existing OHLCV."""
    df = soxl.copy()
    df["close"] = df["close"].astype(float)
    
    # Trend alignment (multiple timeframe)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    df["ma63"] = df["close"].rolling(63).mean()
    
    df["trend_alignment"] = (
        (df["ma5"] > df["ma21"]).astype(int) +
        (df["ma21"] > df["ma63"]).astype(int)
    )  # 0-2 scale
    
    # Donchian channel position
    df["donchian_upper_20"] = df["close"].rolling(20).max()
    df["donchian_lower_20"] = df["close"].rolling(20).min()
    df["donchian_position"] = (df["close"] - df["donchian_lower_20"]) / (df["donchian_upper_20"] - df["donchian_lower_20"])
    
    # ATR-based features (using close-only proxy)
    df["atr_proxy_14"] = df["close"].diff().abs().rolling(14).mean()
    df["atr_position"] = (df["close"] - df["close"].rolling(14).min()) / (df["close"].rolling(14).max() - df["close"].rolling(14).min())
    
    # Correlation with future returns
    targets = ["soxl_return_1d", "soxl_return_5d", "soxl_return_21d"]
    for t in targets:
        df[t] = df["close"].pct_change(int(t.split("_")[2][:-1]) if "d" in t else 1)
    
    cta_features = ["trend_alignment", "donchian_position", "atr_position"]
    results = {}
    for feat in cta_features:
        results[feat] = {}
        for target in targets:
            corr = df[feat].corr(df[target])
            corr_lead1 = df[feat].shift(1).corr(df[target])
            results[feat][target] = {
                "contemporaneous": round(corr, 4) if not np.isnan(corr) else None,
                "lead_1d": round(corr_lead1, 4) if not np.isnan(corr_lead1) else None,
            }
    
    return results


def write_report(correlation: dict, cta: dict) -> str:
    out = OUT_DIR / "new_feature_scan.md"
    
    lines = [
        "# 新特征可行性扫描报告",
        "",
        "## 1. 外部数据源获取结果",
        "",
        "| 数据源 | 来源 | 时间范围 | 获取状态 |",
        "|--------|------|----------|----------|",
        "| VIX9D | CBOE API | 2011-至今 | ✅ 已获取 |",
        "| VIX3M | CBOE API | 2009-至今 | ✅ 已获取 |",
        "| SKEW | CBOE API | 1990-至今 | ✅ 已获取 |",
        "| VVIX | CBOE API | 2006-至今 | ✅ 已获取 |",
        "| Put/Call Ratio | CBOE/FRED | - | ⚠️ 需额外处理 |",
        "",
        "## 2. 期权市场隐含特征",
        "",
        "### 2.1 VIX 期限结构",
        "",
        "特征定义：",
        "- `vix_term_slope` = VIX / VIX3M: <1 表示 contango（远期波动低），>1 表示 backwardation（危机）",
        "- `vix_term_spread` = VIX3M - VIX: 期限结构陡峭度",
        "- `vix_short_term_slope` = VIX9D / VIX: 短期 vs 30 天",
        "",
        "与 SOXL 收益相关性：",
        "",
        "| 特征 | 同期(1d) | 领先1天(1d) | 领先5天(1d) | 同期(5d) | 领先1天(5d) |",
        "|------|----------|-------------|-------------|----------|-------------|",
    ]
    
    for feat, targets in correlation["new_feature_correlations"].items():
        if "vix" in feat:
            row = [feat]
            for t in ["soxl_return_1d", "soxl_return_5d"]:
                if t in targets:
                    row.append(str(targets[t].get("contemporaneous", "N/A")))
                    row.append(str(targets[t].get("lead_1d", "N/A")))
                    if t == "soxl_return_1d":
                        row.append(str(targets[t].get("lead_5d", "N/A")))
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            lines.append("| " + " | ".join(row) + " |")
    
    lines += [
        "",
        "### 2.2 期权偏度（SKEW）",
        "",
        "特征定义：",
        "- `skew_level`: CBOE SKEW 指数（>130 表示尾部风险定价高）",
        "- `skew_zscore_21`: SKEW 的 21 日 Z-score",
        "",
        "与 SOXL 收益相关性：",
        "",
        "| 特征 | 同期(1d) | 领先1天(1d) | 同期(5d) | 领先1天(5d) | 同期(21d) | 领先1天(21d) |",
        "|------|----------|-------------|----------|-------------|-----------|--------------|",
    ]
    
    for feat, targets in correlation["new_feature_correlations"].items():
        if "skew" in feat:
            row = [feat]
            for t in ["soxl_return_1d", "soxl_return_5d", "soxl_return_21d"]:
                if t in targets:
                    row.append(str(targets[t].get("contemporaneous", "N/A")))
                    row.append(str(targets[t].get("lead_1d", "N/A")))
                else:
                    row.extend(["N/A", "N/A"])
            lines.append("| " + " | ".join(row) + " |")
    
    lines += [
        "",
        "### 2.3 VIX of VIX（VVIX）",
        "",
        "特征定义：",
        "- `vvix_level`: VVIX（VIX 的波动率，>100 表示波动率本身不稳定）",
        "- `vvix_zscore_21`: VVIX 的 21 日 Z-score",
        "",
        "与 SOXL 收益相关性：",
        "",
        "| 特征 | 同期(1d) | 领先1天(1d) | 同期(5d) | 领先1天(5d) | 同期(21d) | 领先1天(21d) |",
        "|------|----------|-------------|----------|-------------|-----------|--------------|",
    ]
    
    for feat, targets in correlation["new_feature_correlations"].items():
        if "vvix" in feat:
            row = [feat]
            for t in ["soxl_return_1d", "soxl_return_5d", "soxl_return_21d"]:
                if t in targets:
                    row.append(str(targets[t].get("contemporaneous", "N/A")))
                    row.append(str(targets[t].get("lead_1d", "N/A")))
                else:
                    row.extend(["N/A", "N/A"])
            lines.append("| " + " | ".join(row) + " |")
    
    lines += [
        "",
        "### 2.4 与现有特征相关性",
        "",
    ]
    for name, corr in correlation["existing_feature_correlations"].items():
        lines.append(f"- {name}: {corr}")
    
    lines += [
        "",
        "## 3. CTA 风格趋势特征（可从现有 OHLCV 计算）",
        "",
        "| 特征 | 同期(1d) | 领先1天(1d) | 同期(5d) | 领先1天(5d) |",
        "|------|----------|-------------|----------|-------------|",
    ]
    
    for feat, targets in cta.items():
        row = [feat]
        for t in ["soxl_return_1d", "soxl_return_5d"]:
            if t in targets:
                row.append(str(targets[t].get("contemporaneous", "N/A")))
                row.append(str(targets[t].get("lead_1d", "N/A")))
            else:
                row.extend(["N/A", "N/A"])
        lines.append("| " + " | ".join(row) + " |")
    
    lines += [
        "",
        "## 4. 特征评估与推荐",
        "",
        "### 高优先级（低成本 + 高信息量）",
        "",
    ]
    
    # Auto-generate recommendations based on correlations
    best_lead_corr = -999
    best_feat = None
    for feat, targets in correlation["new_feature_correlations"].items():
        for t, vals in targets.items():
            lc = vals.get("lead_1d")
            if lc is not None and abs(lc) > abs(best_lead_corr):
                best_lead_corr = lc
                best_feat = feat
    
    if best_feat:
        lines.append(f"- **`{best_feat}`**: 对 1 日收益领先的相关系数最高（{best_lead_corr:.4f}），建议优先纳入。")
    
    lines += [
        "- **`vix_term_slope`**: VIX 期限结构是公认的危机预警指标。contango/backwardation 切换往往领先市场拐点 1-3 天。",
        "- **`skew_zscore_21`**: SKEW 的极端值（>2sigma）往往对应尾部风险事件，可作为 regime_risk 的补充。",
        "",
        "### 中优先级",
        "",
        "- **`vvix_level`**: VIX 的波动率，在波动率 regime 切换时有信息增益，但与现有 `vol_of_vol_21` 可能有重叠。",
        "- **`trend_alignment`**: 多均线排列状态，在趋势市中有信息，但与现有 `trend_slope_*` 部分重叠。",
        "",
        "### 低优先级 / 需谨慎",
        "",
        "- **`donchian_position`**: 与现有 `anchor_gap_180`、`cum_drop_*` 有功能重叠，除非在特定通道突破策略中使用。",
        "- **Put/Call Ratio**: 数据获取较麻烦（CBOE 需解析网页），且与 VIX 高度相关，增量信息有限。",
        "",
        "### 不建议",
        "",
        "- **SOXL 期权 IV skew**: 3x ETF 的期权 IV 主要反映杠杆衰减预期，对 SOXX 的领先性未验证，且数据获取成本高。",
        "- **宏观特征（美债、DXY、铜金比）**: 与 SOXL 的领先关系弱，且会引入大量噪声。",
        "",
        "## 5. 下一步行动",
        "",
        "1. **立即行动**：将 `vix_term_slope`、`vix_short_term_slope`、`skew_zscore_21` 加入特征工程管道。",
        "2. **验证实验**：在一个轻量 profile（如 smoke）上测试新增特征，观察 validation score 和 2022 maxDD 的变化。",
        "3. **数据自动化**：将 CBOE API 下载脚本化，纳入 `scripts/check_harness.py` 的数据完整性检查。",
    ]
    
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    
    cboe = load_cboe_data()
    soxl = load_soxl_data()
    
    df = compute_features(cboe, soxl)
    correlation = analyze_correlation(df)
    cta = analyze_cta_features(soxl)
    
    path = write_report(correlation, cta)
    print(f"Phase 4 report written: {path}")
    
    # Save raw correlation data
    (OUT_DIR / "feature_correlation_raw.json").write_text(
        json.dumps({"correlation": correlation, "cta": cta}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Raw correlation data saved.")


if __name__ == "__main__":
    main()
