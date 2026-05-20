# Dashboard Static Default v1

Date: 2026-05-21

Purpose:

- Validate that the Vercel/static dashboard opens on the simulated
  paper-trading view when `paper_trading` is exported.

Evidence:

- `dashboard_data/index.json`: `latest_version=paper_trading`.
- `scripts/check_dashboard_static_export.py`:
  `missing=0`, `latest=paper_trading`, `paper_trading_rows=4`,
  `paper_trading_end=2026-05-20`.
- Headless Chrome runtime read:
  `readyState=complete`, `selectValue=paper_trading`,
  `selectedText="paper_trading (SOXL)"`, `optionCount=26`.
- Screenshot:
  `runtime/production/card/dashboard_static_default_v1/dashboard_static_default_2026-05-21.png`.

Decision:

- Keep paper trading as the static dashboard first-load default while it is
  present in the export.
- Keep offline post-train and SOXX snapshots selectable, but do not let their
  newer file timestamps override the production paper view.
