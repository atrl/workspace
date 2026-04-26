# Counterfactual Ranking: balanced_position_cap_counterfactual

- profile: `balanced`
- model: `artifacts/models/soxl_balanced_actor.pt`
- variants: `10`

## Ranking
1. `bear_floor35_regime065`: gate_passes=1 | 2022 `-37.84% / Sharpe -1.013 / MaxDD 46.05% / avg_pos 0.381` | proxy `Sharpe 0.750 / return 375.42% / maxDD 61.99%` | 2023-2024 `60.77% / Sharpe 0.798` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=pass
2. `bear_floor35_regime070`: gate_passes=1 | 2022 `-38.02% / Sharpe -1.022 / MaxDD 46.20% / avg_pos 0.380` | proxy `Sharpe 0.753 / return 380.06% / maxDD 62.54%` | 2023-2024 `61.83% / Sharpe 0.805` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=pass
3. `bear_floor25_regime060`: gate_passes=0 | 2022 `-35.95% / Sharpe -1.020 / MaxDD 43.55% / avg_pos 0.354` | proxy `Sharpe 0.686 / return 284.15% / maxDD 60.08%` | 2023-2024 `56.08% / Sharpe 0.767` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
4. `bear_floor25_regime055`: gate_passes=0 | 2022 `-36.33% / Sharpe -1.039 / MaxDD 43.90% / avg_pos 0.353` | proxy `Sharpe 0.677 / return 272.88% / maxDD 60.47%` | 2023-2024 `62.52% / Sharpe 0.818` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
5. `bull70_bear30_regime060`: gate_passes=0 | 2022 `-36.74% / Sharpe -1.012 / MaxDD 44.67% / avg_pos 0.367` | proxy `Sharpe 0.713 / return 319.51% / maxDD 62.40%` | 2023-2024 `59.22% / Sharpe 0.790` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
6. `bear_floor30_regime060`: gate_passes=0 | 2022 `-36.74% / Sharpe -1.012 / MaxDD 44.67% / avg_pos 0.367` | proxy `Sharpe 0.704 / return 308.13% / maxDD 62.50%` | 2023-2024 `59.22% / Sharpe 0.790` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
7. `bear_floor30_regime065`: gate_passes=0 | 2022 `-37.36% / Sharpe -1.030 / MaxDD 45.20% / avg_pos 0.369` | proxy `Sharpe 0.698 / return 303.38% / maxDD 62.82%` | 2023-2024 `57.52% / Sharpe 0.777` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
8. `base_bundle`: gate_passes=0 | 2022 `-37.36% / Sharpe -1.030 / MaxDD 45.20% / avg_pos 0.369` | proxy `Sharpe 0.698 / return 303.38% / maxDD 62.82%` | 2023-2024 `57.52% / Sharpe 0.777` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
9. `bull70_bear35_regime060`: gate_passes=0 | 2022 `-37.53% / Sharpe -1.004 / MaxDD 45.79% / avg_pos 0.380` | proxy `Sharpe 0.737 / return 354.33% / maxDD 61.53%` | 2023-2024 `63.47% / Sharpe 0.817` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail
10. `bear_floor35_regime060`: gate_passes=0 | 2022 `-37.53% / Sharpe -1.004 / MaxDD 45.79% / avg_pos 0.380` | proxy `Sharpe 0.737 / return 354.33% / maxDD 61.52%` | 2023-2024 `63.47% / Sharpe 0.817` | 2022_maxdd=fail, 2022_avg_pos=fail, proxy_sharpe=fail

## Top Candidates
- `bear_floor35_regime065` with gate_passes=1 and `2022 max_dd=46.05%`, `2022 avg_pos=0.381`, `proxy sharpe=0.750`
- `bear_floor35_regime070` with gate_passes=1 and `2022 max_dd=46.20%`, `2022 avg_pos=0.380`, `proxy sharpe=0.753`
- `bear_floor25_regime060` with gate_passes=0 and `2022 max_dd=43.55%`, `2022 avg_pos=0.354`, `proxy sharpe=0.686`
