# Failure Modes

- intraday_gap_reversal: passes_both_count=0; best_return=291.76% with MaxDD=-73.88%; best_drawdown_gate=none.
- overnight_momentum: passes_both_count=0; best_return=539007.35% with MaxDD=-97.26%; best_drawdown_gate=none.
- tug_of_war_close_to_close: passes_both_count=0; best_return=21052.25% with MaxDD=-98.08%; best_drawdown_gate=60.62% at MaxDD -13.42%.
- next_action: do not repeat pure overnight/intraday OHLC session splitting unless a new independent signal source is added.
