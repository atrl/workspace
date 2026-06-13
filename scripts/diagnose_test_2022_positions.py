#!/usr/bin/env python3
"""Diagnose why test_2022 has high MaxDD despite low avg_pos."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import json
from src.soxl_quant.policy.positioning import rule_based_position
from src.soxl_quant.policy.observation import make_observation, feature_columns_for_packs, EXTRA_OBS_DIM, N_ASSETS
from src.soxl_quant.data import build_dataset, init_qlib
from src.soxl_quant.settings import QLIB_DATA_PATH
from src.soxl_quant.config import load_profile
from src.soxl_quant.policy.environment import SoxlSimulator, EpisodeConfig
from src.soxl_quant.policy.policy_builder import build_policy, actor_action


def run_diagnosis():
    profile = load_profile('balanced_rule_residual_imitation_v2')
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(feature_packs=tuple(profile.feature.packs))
    df = dataset.handler.fetch(col_set='feature')
    soxl_df = df.xs('SOXL', level='instrument')

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    
    # Get test_2022 period
    test_2022 = soxl_df[(soxl_df.index >= '2022-01-01') & (soxl_df.index <= '2022-12-31')].copy()
    
    # Create a mini episode config for the test period
    full_df = soxl_df.reset_index()
    test_start_idx = full_df[full_df['datetime'] >= '2022-01-01'].index[0]
    test_end_idx = full_df[full_df['datetime'] <= '2022-12-31'].index[-1]
    
    # Save test period to temp pickle
    temp_pkl = '/tmp/test_2022_diag.pkl'
    test_2022.to_pickle(temp_pkl)
    
    initial = EpisodeConfig(
        pkl_path=temp_pkl,
        start_idx=0,
        end_idx=len(test_2022),
        profile_name=profile.name,
        feature_packs=tuple(profile.feature.packs),
        lookback=30,
        cash=100000.0,
        asset_idx=0,
        observation_window=profile.model.observation_window,
        decision_frequency=1,
        nday_window=5,
    )
    
    # Build policy and action interpreter
    obs_dim = len(feature_columns) * profile.model.observation_window + EXTRA_OBS_DIM + N_ASSETS
    policy_result = build_policy(obs_dim, profile)
    policy = policy_result[0] if isinstance(policy_result, tuple) else policy_result
    bundle_path = 'runtime/model/card/balanced_rule_residual_imitation_v2/soxl_balanced_rule_residual_imitation_v2_actor.pt'
    bundle = torch.load(bundle_path, map_location='cpu')
    policy.actor.load_state_dict(bundle['state_dict'])
    policy.actor.eval()
    
    from src.soxl_quant.app.selection import build_action_interpreter
    action_interpreter = build_action_interpreter(profile)

    sim = SoxlSimulator(initial)

    # Run simulation step by step
    records = []
    while not sim.done():
        state = sim.get_state()
        obs = state['obs']
        
        # Get model action
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            actor_out = policy.actor(obs_tensor)
            if isinstance(actor_out, tuple):
                dist = actor_out[0]
            else:
                dist = actor_out
            if hasattr(dist, 'logits'):
                action = int(dist.logits.argmax(dim=-1).item())
            elif hasattr(dist, 'probs'):
                action = int(dist.probs.argmax(dim=-1).item())
            else:
                logits = actor_out if not isinstance(actor_out, tuple) else actor_out[0]
                if isinstance(logits, torch.Tensor):
                    action = int(logits.argmax(dim=-1).item())
                else:
                    action = 2
        
        # Use action interpreter to convert discrete action to position
        model_pos = action_interpreter.interpret(state, action)
        rule_pos = state['rule_pos']
        
        records.append({
            'date': test_2022.index[sim.idx] if sim.idx < len(test_2022) else None,
            'model_pos': model_pos,
            'rule_pos': rule_pos,
            'market_return': state.get('market_return', 0.0),
            'step_return': state.get('step_return', 0.0),
            'drawdown': state.get('drawdown', 0.0),
            'episode_max_dd': state.get('episode_max_dd', 0.0),
            'regime_risk': state.get('regime_risk', 0.0),
            'turnover': state.get('turnover', 0.0),
            'action': action,
        })
        
        sim.step(model_pos)

    df = pd.DataFrame(records)
    df = df.dropna(subset=['date'])
    
    # Compute cumulative returns
    df['model_cumret'] = (1 + df['step_return']).cumprod()
    df['model_peak'] = df['model_cumret'].cummax()
    df['model_dd'] = (df['model_cumret'] - df['model_peak']) / df['model_peak']
    
    # Find max drawdown period
    max_dd_idx = df['model_dd'].idxmin()
    peak_idx = df['model_cumret'][:max_dd_idx+1].idxmax()
    
    print("=" * 70)
    print("TEST 2022 DIAGNOSIS — v2 Model")
    print("=" * 70)
    print(f"\nOverall: return={(df['model_cumret'].iloc[-1]-1)*100:.1f}% "
          f"maxdd={abs(df['model_dd'].min())*100:.1f}% "
          f"avg_pos={df['model_pos'].mean():.2f}")
    print(f"         sharpe={df['step_return'].mean()/df['step_return'].std()*np.sqrt(252):.2f} "
          f"trades={(df['turnover'] > 0.01).sum()}")
    
    print(f"\nMax DD Period:")
    print(f"  Start: {df.loc[peak_idx, 'date']} (peak cumret={df.loc[peak_idx, 'model_cumret']:.4f})")
    print(f"  End:   {df.loc[max_dd_idx, 'date']} (trough cumret={df.loc[max_dd_idx, 'model_cumret']:.4f})")
    print(f"  Depth: {abs(df.loc[max_dd_idx, 'model_dd'])*100:.1f}%")
    print(f"  Duration: {max_dd_idx - peak_idx} days")
    
    dd_df = df.iloc[peak_idx:max_dd_idx+1].copy()
    print(f"\n--- During Drawdown Period ---")
    print(f"  Model avg pos: {dd_df['model_pos'].mean():.2f}")
    print(f"  Rulebase avg pos: {dd_df['rule_pos'].mean():.2f}")
    print(f"  Model pos range: [{dd_df['model_pos'].min():.2f}, {dd_df['model_pos'].max():.2f}]")
    print(f"  Days with pos > 0.5: {(dd_df['model_pos'] > 0.5).sum()}")
    print(f"  Days with pos < 0.3: {(dd_df['model_pos'] < 0.3).sum()}")
    
    # Worst days
    print(f"\n--- Worst 5 Days (by step_return) ---")
    worst = dd_df.nsmallest(5, 'step_return')[['date', 'model_pos', 'rule_pos', 'market_return', 'step_return', 'drawdown', 'action']]
    for _, row in worst.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: model_pos={row['model_pos']:.2f} "
              f"rule_pos={row['rule_pos']:.2f} market={row['market_return']*100:.2f}% "
              f"step_ret={row['step_return']*100:.2f}% dd={row['drawdown']*100:.1f}% action={int(row['action'])}")
    
    # Days where model was high but market crashed
    print(f"\n--- High Position + Market Crash Days ---")
    crash_days = df[(df['model_pos'] > 0.5) & (df['market_return'] < -0.03)]
    print(f"Count: {len(crash_days)}")
    for _, row in crash_days.head(10).iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: model_pos={row['model_pos']:.2f} "
              f"market={row['market_return']*100:.2f}% step_ret={row['step_return']*100:.2f}%")
    
    # Position trajectory around worst days
    print(f"\n--- Position Trajectory Around Max DD ---")
    for idx in range(max(0, max_dd_idx-3), min(len(df), max_dd_idx+4)):
        row = df.iloc[idx]
        marker = " <-- MAX DD" if idx == max_dd_idx else ""
        print(f"  {row['date'].strftime('%Y-%m-%d')}: pos={row['model_pos']:.2f} -> {df.iloc[idx+1]['model_pos']:.2f} "
              f"(action={int(row['action'])}) market={row['market_return']*100:.2f}% "
              f"dd={row['drawdown']*100:.1f}%{marker}")
    
    # Full year position distribution
    print(f"\n--- Full Year Position Distribution ---")
    print(f"Model:  p10={df['model_pos'].quantile(0.1):.2f} p25={df['model_pos'].quantile(0.25):.2f} "
          f"p50={df['model_pos'].median():.2f} p75={df['model_pos'].quantile(0.75):.2f} p90={df['model_pos'].quantile(0.9):.2f}")
    print(f"Rule:   p10={df['rule_pos'].quantile(0.1):.2f} p25={df['rule_pos'].quantile(0.25):.2f} "
          f"p50={df['rule_pos'].median():.2f} p75={df['rule_pos'].quantile(0.75):.2f} p90={df['rule_pos'].quantile(0.9):.2f}")
    
    # Feature values during max DD
    print(f"\n--- Key Feature Values During Max DD Period ---")
    dd_dates = dd_df['date'].tolist()
    for col in ['regime_risk', 'anchor_gap_180', 'cum_drop_30', 'realized_vol_10', 'vix_zscore']:
        if col in test_2022.columns:
            vals = test_2022.loc[test_2022.index.isin(dd_dates), col].values
            if len(vals) > 0:
                print(f"  {col}: min={np.min(vals):.3f} max={np.max(vals):.3f} avg={np.mean(vals):.3f}")
    
    # Save detailed record for further analysis
    df.to_csv('/tmp/test_2022_v2_diagnosis.csv', index=False)
    print(f"\nDetailed records saved to: /tmp/test_2022_v2_diagnosis.csv")


if __name__ == "__main__":
    run_diagnosis()
