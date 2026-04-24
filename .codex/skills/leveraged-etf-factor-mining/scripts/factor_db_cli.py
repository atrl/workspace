"""SOXL factor library CLI — SQLite-backed feature registry.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py list [--status STATUS] [--category CAT] [--source SOURCE]
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py show FEATURE_NAME
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py upsert FEATURE_NAME --expression "..." --category ... --source ... --status ...
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py ic FEATURE_NAME [--horizon 5]
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py backtest FEATURE_NAME [--horizon 5]
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/factor_db_cli.py summary
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DB_PATH = ROOT / "data" / "soxl_factor_library.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS features (
    name TEXT PRIMARY KEY,
    expression TEXT,
    category TEXT,
    source TEXT,
    status TEXT DEFAULT 'Candidate',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS feature_ic_tests (
    feature_name TEXT,
    horizon_days INTEGER,
    test_start TEXT,
    test_end TEXT,
    n_days INTEGER,
    ic_mean REAL,
    ic_std REAL,
    ic_t REAL,
    ic_positive_ratio REAL,
    PRIMARY KEY (feature_name, horizon_days)
);

CREATE TABLE IF NOT EXISTS feature_backtests (
    feature_name TEXT,
    horizon_days INTEGER,
    threshold_long REAL,
    threshold_short REAL,
    hold_days INTEGER,
    signal_accuracy REAL,
    profit_factor REAL,
    avg_return REAL,
    max_drawdown REAL,
    sharpe REAL,
    win_rate REAL,
    num_signals INTEGER,
    PRIMARY KEY (feature_name, horizon_days, threshold_long, hold_days)
);

CREATE TABLE IF NOT EXISTS feature_stability (
    feature_name TEXT,
    window_days INTEGER,
    stability_score REAL,
    ic_recent_vs_full REAL,
    max_dd_days INTEGER,
    PRIMARY KEY (feature_name, window_days)
);
"""


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA_SQL)
    return conn


def cmd_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    sql = "SELECT name, category, source, status FROM features WHERE 1=1"
    params = []
    if args.status:
        sql += " AND status = ?"
        params.append(args.status)
    if args.category:
        sql += " AND category = ?"
        params.append(args.category)
    if args.source:
        sql += " AND source = ?"
        params.append(args.source)
    sql += " ORDER BY name"
    rows = conn.execute(sql, params).fetchall()
    print(f"{'Name':<35} {'Category':<18} {'Source':<14} {'Status':<10}")
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<35} {r[1] or '':<18} {r[2] or '':<14} {r[3] or '':<10}")
    print(f"\n({len(rows)} features)")


def cmd_show(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = conn.execute("SELECT * FROM features WHERE name = ?", (args.name,)).fetchone()
    if not row:
        print(f"Feature '{args.name}' not found.")
        return
    cols = [d[0] for d in conn.execute("PRAGMA table_info(features)").fetchall()]
    print(f"{'='*60}")
    print(f"Feature: {args.name}")
    print(f"{'='*60}")
    for k, v in zip(cols, row):
        if v is not None and v != "":
            print(f"  {k:20s}: {v}")

    # IC tests
    ic_rows = conn.execute(
        "SELECT horizon_days, ic_mean, ic_t, ic_positive_ratio FROM feature_ic_tests WHERE feature_name = ? ORDER BY horizon_days",
        (args.name,),
    ).fetchall()
    if ic_rows:
        print("\n--- IC Tests ---")
        for r in ic_rows:
            print(f"  {r[0]:>2}d  IC={r[1]:+.4f}  t={r[2]:+.2f}  pos_ratio={r[3]:.2%}")

    # Backtests
    bt_rows = conn.execute(
        "SELECT horizon_days, threshold_long, hold_days, signal_accuracy, profit_factor, sharpe, win_rate, num_signals FROM feature_backtests WHERE feature_name = ? ORDER BY horizon_days",
        (args.name,),
    ).fetchall()
    if bt_rows:
        print("\n--- Backtests ---")
        for r in bt_rows:
            print(f"  {r[0]:>2}d thr={r[1]:+.3f} hold={r[2]:>2}d  acc={r[3]:.2%} PF={r[4]:.2f} SR={r[5]:.2f} WR={r[6]:.2%} n={r[7]}")


def cmd_upsert(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    conn.execute(
        """INSERT INTO features (name, expression, category, source, status, notes)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
             expression=excluded.expression,
             category=excluded.category,
             source=excluded.source,
             status=excluded.status,
             notes=excluded.notes""",
        (args.name, args.expression, args.category, args.source, args.status, args.notes),
    )
    conn.commit()
    print(f"Upserted: {args.name} ({args.status})")


def cmd_summary(conn: sqlite3.Connection, _args: argparse.Namespace) -> None:
    total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    by_status = conn.execute("SELECT status, COUNT(*) FROM features GROUP BY status").fetchall()
    by_source = conn.execute("SELECT source, COUNT(*) FROM features GROUP BY source").fetchall()
    ic_count = conn.execute("SELECT COUNT(DISTINCT feature_name) FROM feature_ic_tests").fetchone()[0]
    bt_count = conn.execute("SELECT COUNT(DISTINCT feature_name) FROM feature_backtests").fetchone()[0]
    print(f"Total features: {total}")
    print("By status:", {s: c for s, c in by_status})
    print("By source:", {s: c for s, c in by_source})
    print(f"With IC tests: {ic_count}")
    print(f"With backtests: {bt_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SOXL Factor Library CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List features")
    p_list.add_argument("--status", help="Filter by status")
    p_list.add_argument("--category", help="Filter by category")
    p_list.add_argument("--source", help="Filter by source")

    p_show = sub.add_parser("show", help="Show feature details")
    p_show.add_argument("name", help="Feature name")

    p_upsert = sub.add_parser("upsert", help="Upsert a feature")
    p_upsert.add_argument("name", help="Feature name")
    p_upsert.add_argument("--expression", required=True, help="Feature expression or formula")
    p_upsert.add_argument("--category", default="misc", help="Feature category")
    p_upsert.add_argument("--source", default="manual", help="Source")
    p_upsert.add_argument("--status", default="Candidate", help="Status")
    p_upsert.add_argument("--notes", default="", help="Notes")

    p_summary = sub.add_parser("summary", help="Show summary")

    args = parser.parse_args()
    conn = _ensure_db()
    try:
        if args.command == "list":
            cmd_list(conn, args)
        elif args.command == "show":
            cmd_show(conn, args)
        elif args.command == "upsert":
            cmd_upsert(conn, args)
        elif args.command == "summary":
            cmd_summary(conn, args)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
