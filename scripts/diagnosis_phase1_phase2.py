#!/usr/bin/env python3
"""Phase 1 + Phase 2: 交易级归因、分层凯利、时段切片诊断"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DETAIL_CSV = ROOT / "evidence" / "generalization" / "detailed" / "soxl_detailed_trades.csv"
OUT_DIR = ROOT / "evidence" / "diagnosis"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", default=str(DETAIL_CSV), help="Input detailed_trades.csv path")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for reports")
    return parser.parse_args()


def load_rows(detail_csv: Path) -> list[dict]:
    rows = []
    with open(detail_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "datetime": r["datetime"],
                "strategy_nav": float(r["strategy_nav"]),
                "price": float(r["price"]),
                "buyhold_nav": float(r["buyhold_nav"]),
                "position": float(r["position"]),
                "excess": float(r["excess"]),
                "excess_pct": float(r["excess_pct"]),
                "action": r["action"],
            })
    return rows


def build_round_trips(rows: list[dict]) -> list[dict]:
    """Build round-trip trades from daily action log.
    
    A round-trip is defined as a sequence from first BUY after flat/SELL
    to the next SELL. HOLD in between is part of the trade.
    """
    trades = []
    entry_idx = -1
    peak_nav = 0.0
    for i, r in enumerate(rows):
        if r["action"] == "BUY" and entry_idx < 0:
            entry_idx = i
            peak_nav = r["strategy_nav"]
        elif entry_idx >= 0:
            peak_nav = max(peak_nav, r["strategy_nav"])
            if r["action"] == "SELL":
                entry = rows[entry_idx]
                exit_r = r
                entry_price = entry["price"]
                exit_price = exit_r["price"]
                entry_nav = entry["strategy_nav"]
                exit_nav = exit_r["strategy_nav"]
                days = i - entry_idx
                ret_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0
                pnl = exit_nav - entry_nav
                pnl_pct = (exit_nav / entry_nav - 1) * 100 if entry_nav > 0 else 0
                max_dd_in_trade = 0.0
                for j in range(entry_idx, i + 1):
                    dd = (peak_nav - rows[j]["strategy_nav"]) / peak_nav * 100 if peak_nav > 0 else 0
                    max_dd_in_trade = max(max_dd_in_trade, dd)
                
                # Position change magnitude
                pos_change = abs(exit_r["position"] - entry["position"])
                
                trades.append({
                    "entry_date": entry["datetime"],
                    "exit_date": exit_r["datetime"],
                    "days": days,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "entry_nav": entry_nav,
                    "exit_nav": exit_nav,
                    "entry_pos": entry["position"],
                    "exit_pos": exit_r["position"],
                    "pos_change": pos_change,
                    "max_dd_in_trade": max_dd_in_trade,
                })
                entry_idx = -1
                peak_nav = 0.0
    return trades


def kelly_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}
    wins = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losses = [t["return_pct"] for t in trades if t["return_pct"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    r = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    kelly = win_rate - (1 - win_rate) / r if r not in (float('inf'), 0) and r > 0 else win_rate
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    return {
        "total": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "r_ratio": round(r, 4) if r != float('inf') else None,
        "kelly": round(kelly, 4),
        "expectancy": round(expectancy, 4),
        "profit_factor": round(pf, 4) if pf != float('inf') else None,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def analyze_trades(rows: list[dict], trades: list[dict]) -> dict:
    """Phase 1: 交易级归因与分层凯利"""
    base = kelly_stats(trades)
    
    # Layered analysis
    # By holding period
    short = [t for t in trades if t["days"] < 3]
    mid = [t for t in trades if 3 <= t["days"] < 10]
    long = [t for t in trades if 10 <= t["days"] < 30]
    very_long = [t for t in trades if t["days"] >= 30]
    
    # By position change magnitude
    micro = [t for t in trades if t["pos_change"] < 0.05]
    small = [t for t in trades if 0.05 <= t["pos_change"] < 0.15]
    large = [t for t in trades if t["pos_change"] >= 0.15]
    
    # By return direction (win vs loss)
    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
    
    # Count actions
    actions = defaultdict(int)
    for r in rows:
        actions[r["action"]] += 1
    
    # Friction cost estimate (0.1% per trade side)
    total_trades = len([r for r in rows if r["action"] in ("BUY", "SELL")])
    friction_pct = total_trades * 0.001 * 100  # 0.1% each side
    
    # Counterfactual: remove micro trades
    non_micro = [t for t in trades if t["pos_change"] >= 0.05]
    non_micro_pnl = sum(t["pnl_pct"] for t in non_micro)
    all_pnl = sum(t["pnl_pct"] for t in trades)
    micro_pnl = all_pnl - non_micro_pnl
    
    # Counterfactual friction: if we only kept non-micro trades
    # Estimate: each trade has ~2 sides (entry+exit), micro trades account for some fraction
    micro_count = len(micro)
    friction_saved = micro_count * 2 * 0.001 * 100  # rough
    
    return {
        "overall": base,
        "by_holding_period": {
            "short_<3d": kelly_stats(short),
            "mid_3_10d": kelly_stats(mid),
            "long_10_30d": kelly_stats(long),
            "very_long_30d+": kelly_stats(very_long),
        },
        "by_pos_change": {
            "micro_<5%": kelly_stats(micro),
            "small_5_15%": kelly_stats(small),
            "large_15%+": kelly_stats(large),
        },
        "win_loss_summary": {
            "win_count": len(wins),
            "loss_count": len(losses),
            "avg_win_days": round(np.mean([t["days"] for t in wins]), 2) if wins else 0,
            "avg_loss_days": round(np.mean([t["days"] for t in losses]), 2) if losses else 0,
            "avg_win_pos_change": round(np.mean([t["pos_change"] for t in wins]), 4) if wins else 0,
            "avg_loss_pos_change": round(np.mean([t["pos_change"] for t in losses]), 4) if losses else 0,
        },
        "action_counts": dict(actions),
        "friction_estimate": {
            "total_trade_actions": total_trades,
            "friction_cost_pct": round(friction_pct, 4),
            "micro_trades": micro_count,
            "micro_pnl_contribution_pct": round(micro_pnl, 4),
            "non_micro_pnl_contribution_pct": round(non_micro_pnl, 4),
            "estimated_friction_saved_by_removing_micro": round(friction_saved, 4),
        },
    }


def analyze_periods(rows: list[dict]) -> dict:
    """Phase 2: 时段切片表现诊断"""
    from datetime import datetime
    
    # Group by quarter
    quarters = defaultdict(list)
    for r in rows:
        dt = datetime.strptime(r["datetime"], "%Y-%m-%d")
        q = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        quarters[q].append(r)
    
    results = []
    for q in sorted(quarters.keys()):
        qr = quarters[q]
        start = qr[0]
        end = qr[-1]
        
        strat_ret = (end["strategy_nav"] / start["strategy_nav"] - 1) * 100
        bh_ret = (end["buyhold_nav"] / start["buyhold_nav"] - 1) * 100
        excess = strat_ret - bh_ret
        
        # MaxDD in quarter
        peak = start["strategy_nav"]
        max_dd = 0.0
        for r in qr:
            peak = max(peak, r["strategy_nav"])
            dd = (peak - r["strategy_nav"]) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        avg_pos = np.mean([r["position"] for r in qr])
        trades = len([r for r in qr if r["action"] in ("BUY", "SELL")])
        
        results.append({
            "quarter": q,
            "strategy_return_pct": round(strat_ret, 2),
            "buyhold_return_pct": round(bh_ret, 2),
            "excess_return_pct": round(excess, 2),
            "max_dd_pct": round(max_dd, 2),
            "avg_position": round(avg_pos, 4),
            "trades": trades,
        })
    
    # Find worst quarters
    worst = sorted(results, key=lambda x: x["excess_return_pct"])[:10]
    
    # Annual summary
    years = defaultdict(list)
    for r in results:
        y = r["quarter"].split("-")[0]
        years[y].append(r)
    
    annual = []
    for y in sorted(years.keys()):
        yr = years[y]
        strat_ret = (np.prod([(1 + r["strategy_return_pct"] / 100) for r in yr]) - 1) * 100
        bh_ret = (np.prod([(1 + r["buyhold_return_pct"] / 100) for r in yr]) - 1) * 100
        excess = strat_ret - bh_ret
        max_dd = max(r["max_dd_pct"] for r in yr)
        avg_pos = np.mean([r["avg_position"] for r in yr])
        trades = sum(r["trades"] for r in yr)
        annual.append({
            "year": y,
            "strategy_return_pct": round(strat_ret, 2),
            "buyhold_return_pct": round(bh_ret, 2),
            "excess_return_pct": round(excess, 2),
            "max_dd_pct": round(max_dd, 2),
            "avg_position": round(avg_pos, 4),
            "trades": trades,
        })
    
    return {
        "quarterly": results,
        "annual": annual,
        "worst_10_quarters": worst,
    }


def write_phase1_report(data: dict, out_dir: Path) -> str:
    out = out_dir / "soxl_trade_attribution.md"
    o = data["overall"]
    f = data["friction_estimate"]
    wl = data["win_loss_summary"]
    
    lines = [
        "# SOXL 交易级归因与凯利诊断报告",
        "",
        "## 1. 整体交易统计",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总交易次数 | {o['total']} |",
        f"| 胜率 | {o['win_rate']*100:.2f}% |",
        f"| 平均盈利 | {o['avg_win']:.2f}% |",
        f"| 平均亏损 | {o['avg_loss']:.2f}% |",
        f"| R 比率 | {o['r_ratio']:.2f} |" if o['r_ratio'] else "| R 比率 | inf |",
        f"| Kelly | {o['kelly']:.4f} |",
        f"| Expectancy | {o['expectancy']:.2f}% |",
        f"| Profit Factor | {o['profit_factor']:.2f} |" if o['profit_factor'] else "| Profit Factor | inf |",
        "",
        "## 2. 分层凯利分析",
        "",
        "### 按持仓天数分层",
        "",
        "| 分层 | 交易数 | 胜率 | 平均盈利 | 平均亏损 | Kelly | Expectancy |",
        "|------|--------|------|----------|----------|-------|------------|",
    ]
    for name, s in data["by_holding_period"].items():
        if s:
            lines.append(f"| {name} | {s['total']} | {s['win_rate']*100:.1f}% | {s['avg_win']:.2f}% | {s['avg_loss']:.2f}% | {s['kelly']:.4f} | {s['expectancy']:.2f}% |")
    
    lines += [
        "",
        "### 按仓位变化幅度分层",
        "",
        "| 分层 | 交易数 | 胜率 | 平均盈利 | 平均亏损 | Kelly | Expectancy |",
        "|------|--------|------|----------|----------|-------|------------|",
    ]
    for name, s in data["by_pos_change"].items():
        if s:
            lines.append(f"| {name} | {s['total']} | {s['win_rate']*100:.1f}% | {s['avg_win']:.2f}% | {s['avg_loss']:.2f}% | {s['kelly']:.4f} | {s['expectancy']:.2f}% |")
    
    lines += [
        "",
        "## 3. 摩擦成本归因",
        "",
        f"- 总交易动作数（BUY+SELL）: {f['total_trade_actions']}",
        f"- 按 0.1% 单边成本估算总摩擦: {f['friction_cost_pct']:.2f}%",
        f"- 微幅调仓（<5% 仓位变化）次数: {f['micro_trades']}",
        f"- 微幅调仓对 PnL 贡献: {f['micro_pnl_contribution_pct']:.2f}%",
        f"- 非微幅调仓对 PnL 贡献: {f['non_micro_pnl_contribution_pct']:.2f}%",
        f"- 若去掉微幅调仓可节省摩擦: ~{f['estimated_friction_saved_by_removing_micro']:.2f}%",
        "",
        "## 4. 赢亏交易特征对比",
        "",
        f"| 特征 | 盈利交易 | 亏损交易 |",
        f"|------|----------|----------|",
        f"| 数量 | {wl['win_count']} | {wl['loss_count']} |",
        f"| 平均持仓天数 | {wl['avg_win_days']:.1f} | {wl['avg_loss_days']:.1f} |",
        f"| 平均仓位变化 | {wl['avg_win_pos_change']:.2%} | {wl['avg_loss_pos_change']:.2%} |",
        "",
        "## 5. 关键结论",
        "",
    ]
    
    # Auto-generate conclusions
    micro_kelly = data["by_pos_change"]["micro_<5%"].get("kelly", 0) if data["by_pos_change"]["micro_<5%"] else 0
    short_kelly = data["by_holding_period"]["short_<3d"].get("kelly", 0) if data["by_holding_period"]["short_<3d"] else 0
    
    if micro_kelly < 0.1:
        lines.append(f"- **微幅调仓（<5%）的 Kelly 仅为 {micro_kelly:.4f}**，是拉低整体期望的主要来源。这些交易大多是模型对微小信号的过度反应。")
    if short_kelly < 0.1:
        lines.append(f"- **超短期交易（<3天）的 Kelly 仅为 {short_kelly:.4f}**，说明做 T 的期望很低，日内/隔日波动难以预测。")
    small_total = data["by_pos_change"].get("small_5_15%", {}).get("total", 0)
    if f['micro_trades'] > small_total:
        lines.append(f"- **微幅调仓占交易总数的 {f['micro_trades'] / o['total'] * 100:.1f}%**，但贡献的 PnL 仅 {f['micro_pnl_contribution_pct']:.2f}%，摩擦成本不成比例。")
    
    lines.append("- **核心建议**: 提高再平衡阈值或引入交易锁定期，强制过滤掉微幅调仓和超短期交易。")
    
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_phase2_report(data: dict, out_dir: Path) -> str:
    out = out_dir / "period_slice_comparison.md"
    
    lines = [
        "# SOXL 时段切片表现诊断报告",
        "",
        "## 1. 年度表现对比",
        "",
        "| 年份 | 策略回报 | Buy&Hold | 超额 | 最大回撤 | 平均仓位 | 交易次数 |",
        "|------|----------|----------|------|----------|----------|----------|",
    ]
    for r in data["annual"]:
        lines.append(f"| {r['year']} | {r['strategy_return_pct']:.2f}% | {r['buyhold_return_pct']:.2f}% | {r['excess_return_pct']:+.2f}% | {r['max_dd_pct']:.2f}% | {r['avg_position']:.2%} | {r['trades']} |")
    
    lines += [
        "",
        "## 2. 最差季度（按超额回报排序）",
        "",
        "| 季度 | 策略回报 | Buy&Hold | 超额 | 最大回撤 | 平均仓位 | 交易次数 |",
        "|------|----------|----------|------|----------|----------|----------|",
    ]
    for r in data["worst_10_quarters"]:
        lines.append(f"| {r['quarter']} | {r['strategy_return_pct']:.2f}% | {r['buyhold_return_pct']:.2f}% | {r['excess_return_pct']:+.2f}% | {r['max_dd_pct']:.2f}% | {r['avg_position']:.2%} | {r['trades']} |")
    
    lines += [
        "",
        "## 3. 时段特征分析",
        "",
    ]
    
    # Analyze patterns in worst quarters
    worst_2022 = [r for r in data["worst_10_quarters"] if r["quarter"].startswith("2022")]
    worst_2024 = [r for r in data["worst_10_quarters"] if r["quarter"].startswith("2024")]
    worst_2025 = [r for r in data["worst_10_quarters"] if r["quarter"].startswith("2025")]
    
    if worst_2022:
        lines.append(f"- **2022 年熊市**：{len(worst_2022)} 个季度进入最差 10，平均超额 {np.mean([r['excess_return_pct'] for r in worst_2022]):.2f}%。策略在熊市中减仓不够快或过早抄底。")
    if worst_2024:
        lines.append(f"- **2024 年**：{len(worst_2024)} 个季度进入最差 10，平均超额 {np.mean([r['excess_return_pct'] for r in worst_2024]):.2f}%。2024 年 SOXL 单边上涨，策略平均仓位 {np.mean([r['avg_position'] for r in worst_2024]):.2%}，踏空明显。")
    if worst_2025:
        lines.append(f"- **2025 年**：{len(worst_2025)} 个季度进入最差 10，平均超额 {np.mean([r['excess_return_pct'] for r in worst_2025]):.2f}%。")
    
    # Find consistently bad pattern
    negative_excess_quarters = [r for r in data["quarterly"] if r["excess_return_pct"] < -5]
    if negative_excess_quarters:
        avg_pos_in_bad = np.mean([r["avg_position"] for r in negative_excess_quarters])
        lines.append(f"- **跑输季度共性**：{len(negative_excess_quarters)} 个季度超额 < -5%，这些季度的平均仓位仅 {avg_pos_in_bad:.2%}。策略在上涨市中仓位过低是主要问题。")
    
    lines += [
        "",
        "## 4. 核心结论",
        "",
        "- **2022 问题**：危机减仓不够果断，导致跌幅接近 B&H（甚至更深）。",
        "- **2024-2025 问题**：牛市中仓位过低，频繁小幅调仓消耗摩擦，错失主要涨幅。",
        "- **结构性问题**：策略缺乏'核心持仓'概念，在趋势明确时仍然试图择时，反而被双向摩擦吞噬。",
    ]
    
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_underperform_json(data: dict, out_dir: Path) -> str:
    out = out_dir / "underperform_quarters.json"
    # Export all quarters with excess < -5%
    bad = [r for r in data["quarterly"] if r["excess_return_pct"] < -5]
    out.write_text(json.dumps(bad, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def main():
    args = parse_args()
    detail_csv = Path(args.detail_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(detail_csv)
    trades = build_round_trips(rows)
    
    phase1 = analyze_trades(rows, trades)
    p1_path = write_phase1_report(phase1, out_dir)
    print(f"Phase 1 report written: {p1_path}")
    
    phase2 = analyze_periods(rows)
    p2_path = write_phase2_report(phase2, out_dir)
    up_path = write_underperform_json(phase2, out_dir)
    print(f"Phase 2 report written: {p2_path}")
    print(f"Underperform quarters written: {up_path}")
    
    # Also write a combined JSON for downstream use
    combined = {
        "trade_analysis": phase1,
        "period_analysis": phase2,
    }
    (out_dir / "phase1_phase2_combined.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Combined JSON written.")


if __name__ == "__main__":
    main()
