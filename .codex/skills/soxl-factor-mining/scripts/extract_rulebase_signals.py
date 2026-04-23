"""Extract rulebase strategy logic as candidate SOXL features.

Reads soxl_backtest/ buy/sell/panic logic and translates it into
quantifiable time-series expressions that can be tested via IC/backtest.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/extract_rulebase_signals.py \
      --output docs/factor_research/rulebase_hypotheses.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

# Rulebase logic → feature hypothesis mapping
# These are *declarative* descriptions; the actual pandas expressions
# are generated in src/soxl_quant/data.py by _build_rulebase_signal_features().
RULEBASE_HYPOTHESES = [
    {
        "name": "RB_PANIC_SCORE_PROXY",
        "expression": "sigmoid_vix_zscore * relative_volume_zscore + rsi6_adjustment",
        "category": "sentiment",
        "source": "rulebase_panic_calculator",
        "market_logic": "VIX驱动的恐慌情绪(sigmoid映射) × 成交量放大器 + RSI6超卖调节",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "high",
        "risks": "VIX数据缺失时退化为vol_stress_score",
    },
    {
        "name": "RB_DAILY_DROP_SEVERITY",
        "expression": "daily_return / (vol_coef_21 + 1e-8)",
        "category": "drawdown",
        "source": "rulebase_buy_strategy",
        "market_logic": "日跌幅经波动率标准化，极端值暗示反转",
        "expected_direction": "negative",
        "expected_horizon": "3d",
        "priority": "high",
        "risks": "高波动期可能连续下跌",
    },
    {
        "name": "RB_CUMULATIVE_DROP_SEVERITY",
        "expression": "cum_drop_30 / (vol_coef_21 + 1e-8)",
        "category": "drawdown",
        "source": "rulebase_buy_strategy",
        "market_logic": "30日累积跌幅经波动率标准化，深度累积后反弹概率高",
        "expected_direction": "negative",
        "expected_horizon": "10d",
        "priority": "high",
        "risks": "熊市中可能持续累积下跌",
    },
    {
        "name": "RB_RSI6_OVERSOLD",
        "expression": "(30 - rsi_6 * 100).clip(lower=0) / 30.0",
        "category": "momentum",
        "source": "rulebase_panic_calculator",
        "market_logic": "RSI6超卖程度量化，0-1区间，越接近1超卖越严重",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "medium",
        "risks": "暴跌中RSI可能长期钝化",
    },
    {
        "name": "RB_RSI6_OVERBOUGHT",
        "expression": "(rsi_6 * 100 - 70).clip(lower=0) / 30.0",
        "category": "momentum",
        "source": "rulebase_panic_calculator",
        "market_logic": "RSI6超买程度量化，0-1区间，越接近1超买越严重",
        "expected_direction": "positive",
        "expected_horizon": "5d",
        "priority": "medium",
        "risks": "强趋势中RSI可能长期高位",
    },
    {
        "name": "RB_VOL_REGIME",
        "expression": "vol_coef_21",
        "category": "volatility",
        "source": "rulebase_execution",
        "market_logic": "波动率系数 = HV_20 / 历史中位数，>1高波动 <1低波动",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "medium",
        "risks": "高波动可能持续",
    },
    {
        "name": "RB_BUY_COMPOSITE",
        "expression": "panic_score_proxy * 0.3 + daily_drop_severity * 0.4 + cum_drop_severity * 0.3",
        "category": "composite",
        "source": "rulebase_buy_strategy",
        "market_logic": "rulebase买入条件的综合信号：恐慌 + 日跌 + 累积跌",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "high",
        "risks": "复合信号可能过度拟合历史买入点",
    },
    {
        "name": "RB_SELL_COMPOSITE",
        "expression": "rsi6_overbought * 0.4 + profit_position_proxy * 0.3 + stop_loss_proxy * 0.3",
        "category": "composite",
        "source": "rulebase_sell_strategy",
        "market_logic": "rulebase卖出条件的综合信号：超买 + 止盈 + 止损",
        "expected_direction": "positive",
        "expected_horizon": "3d",
        "priority": "high",
        "risks": "需要持仓状态信息，非纯价格特征",
    },
    {
        "name": "RB_PROFIT_TAKE_PROXY",
        "expression": "(cum_drop_30 + 0.30).clip(lower=0, upper=0.60) / 0.60",
        "category": "drawdown",
        "source": "rulebase_sell_strategy",
        "market_logic": "从深度回撤中恢复的程度，接近1暗示获利了结",
        "expected_direction": "positive",
        "expected_horizon": "3d",
        "priority": "medium",
        "risks": "仅反映价格恢复，非真实持仓盈亏",
    },
    {
        "name": "RB_COOLDOWN_PROXY",
        "expression": "days_since_high_63 * (1 - below_anchor_ratio_63)",
        "category": "timing",
        "source": "rulebase_buy_strategy",
        "market_logic": "距离高点天数 × 脱离底部程度，高值暗示冷却期结束",
        "expected_direction": "negative",
        "expected_horizon": "10d",
        "priority": "low",
        "risks": "间接信号，噪声较大",
    },
    {
        "name": "RB_STOP_LOSS_PROXY",
        "expression": "(cum_drop_30 + 0.20).clip(lower=0, upper=0.20) / 0.20",
        "category": "drawdown",
        "source": "rulebase_sell_strategy",
        "market_logic": "接近-20%止损线的程度，高值暗示止损风险",
        "expected_direction": "positive",
        "expected_horizon": "1d",
        "priority": "medium",
        "risks": "止损后可能立即反弹",
    },
    {
        "name": "RB_VOLUME_STRESS",
        "expression": "volume_shock * vol_stress_score",
        "category": "liquidity",
        "source": "rulebase_enhanced_panic",
        "market_logic": "成交量异动 × 波动压力得分，高值暗示情绪极端",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "medium",
        "risks": "高成交量可能伴随趋势延续",
    },
    {
        "name": "RB_TREND_REVERSAL",
        "expression": "anchor_gap_180 * (1 + realized_vol_21)",
        "category": "trend",
        "source": "rulebase_buy_strategy",
        "market_logic": "长期均线偏离 × 波动率放大，极端负值暗示反转",
        "expected_direction": "negative",
        "expected_horizon": "10d",
        "priority": "medium",
        "risks": "趋势延续时间不确定",
    },
    {
        "name": "RB_VIX_TERM_STRESS",
        "expression": "vix_term_slope * vix_short_ratio",
        "category": "sentiment",
        "source": "rulebase_panic_calculator",
        "market_logic": "VIX期限结构斜率 × 短期VIX比率，异常结构暗示压力",
        "expected_direction": "negative",
        "expected_horizon": "5d",
        "priority": "medium",
        "risks": "需要VIX3M/VIX9D数据",
    },
]


def generate_hypotheses(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "auto",
        "count": len(RULEBASE_HYPOTHESES),
        "hypotheses": RULEBASE_HYPOTHESES,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, allow_unicode=True, sort_keys=False)
    print(f"Wrote {len(RULEBASE_HYPOTHESES)} hypotheses to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract rulebase signals as feature hypotheses")
    parser.add_argument("--output", default=str(ROOT / "docs" / "factor_research" / "rulebase_hypotheses.yaml"))
    args = parser.parse_args()
    generate_hypotheses(Path(args.output))


if __name__ == "__main__":
    main()
