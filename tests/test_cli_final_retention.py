from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.soxl_quant.app import cli as cli_module
from src.soxl_quant.config import load_profile


def test_cli_walk_forward_updates_final_model_retention_decision(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "profile": "balanced",
                "promotion": {"eligible": False, "walk_forward_evaluated": False},
                "final_model_retention_decision": {
                    "status": "blocked",
                    "reason": "walk-forward promotion gate not evaluated",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli_module,
        "parse_args",
        lambda: argparse.Namespace(
            profiles=["balanced"],
            smoke=False,
            skip_data_refresh=True,
            walk_forward=True,
            multi_asset=False,
            moe=False,
            ablate_vix=False,
            seed=7,
            worker_run_profile="",
        ),
    )
    monkeypatch.setattr(cli_module, "bootstrap_runtime", lambda _args: (None, object(), object(), object(), object(), None))
    monkeypatch.setattr(cli_module, "load_profile", lambda _name: load_profile("balanced"))
    monkeypatch.setattr(
        cli_module,
        "run_profile",
        lambda **_kwargs: {
            "summary_path": str(summary_path),
            "model_path": "runtime/model/card/balanced/soxl_balanced_actor.pt",
            "slices": {},
            "test_proxy": {},
        },
    )
    monkeypatch.setattr(
        cli_module,
        "run_walk_forward_isolated",
        lambda **_kwargs: {"wf_2023_2024": {"return": -1.0, "sharpe": -0.1}},
    )
    monkeypatch.setattr(
        cli_module,
        "build_promotion_report",
        lambda *_args, **_kwargs: {
            "eligible": False,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {
                "eligible": False,
                "checks": {"wf_2023_2024_return": {"passed": False}},
            },
            "walk_forward_evaluated": True,
        },
    )
    monkeypatch.setattr(cli_module, "print_profile_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "persist_and_print_results", lambda *_args, **_kwargs: None)

    cli_module.main()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["promotion"]["walk_forward_evaluated"] is True
    assert payload["final_model_retention_decision"]["status"] == "blocked"
    assert payload["final_model_retention_decision"]["final_model_retained"] is False
    assert payload["final_model_retention_decision"]["reason"] == "walk-forward promotion gate failed"
