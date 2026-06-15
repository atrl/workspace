from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.soxl_quant.app import train as train_module
from src.soxl_quant.config import CheckpointRetentionConfig, load_profile


class _DummyActor:
    output_dim = 11

    def state_dict(self):
        return {"weight": 1}

    def load_state_dict(self, _state):
        return None


def test_run_profile_persists_summary_after_main_backtest_corrections():
    profile = load_profile("balanced")
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=dates)
    price_series = pd.Series([100.0, 101.0], index=dates)
    captured: dict[str, object] = {}
    original = {
        "observation_dim_for_profile": train_module.observation_dim_for_profile,
        "feature_columns_for_packs": train_module.feature_columns_for_packs,
        "build_initial_states": train_module.build_initial_states,
        "build_policy": train_module.build_policy,
        "train_with_selection": train_module.train_with_selection,
        "persist_actor_bundle": train_module.persist_actor_bundle,
        "persist_retained_checkpoint_bundles": train_module.persist_retained_checkpoint_bundles,
        "slice_metrics": train_module.slice_metrics,
        "evaluate_split": train_module.evaluate_split,
        "backtest_profile": train_module.backtest_profile,
        "build_promotion_report": train_module.build_promotion_report,
        "persist_summary_bundle": train_module.persist_summary_bundle,
        "export_review_daily_csv": train_module.export_review_daily_csv,
    }

    def fake_backtest_profile(*_args, **kwargs):
        if kwargs.get("test_seg") == ("2022-01-01", "2022-12-31"):
            return {"return": -8.0, "sharpe": -0.2, "max_dd": 12.0}
        return {
            "return": 123.0,
            "sharpe": 1.23,
            "max_dd": 45.0,
            "bnh_return": 200.0,
            "relative_outperformance": -77.0,
        }

    def fake_persist_summary_bundle(**kwargs):
        captured["test_proxy"] = dict(kwargs["test_proxy"])
        captured["slice_result"] = dict(kwargs["slice_result"])
        captured["promotion_report"] = dict(kwargs["promotion_report"])
        return Path("runtime/model/card/balanced/soxl_balanced_summary.json")

    try:
        train_module.observation_dim_for_profile = lambda _profile: 3
        train_module.feature_columns_for_packs = lambda _packs, _name: []
        train_module.build_initial_states = lambda *_args, **_kwargs: [object()]
        train_module.build_policy = lambda *_args, **_kwargs: (object(), _DummyActor())
        train_module.train_with_selection = lambda **_kwargs: {
            "best_metrics": {"return": 1.0, "sharpe": 0.5, "max_dd": 2.0},
            "best_score": 3.0,
            "history": [{"iter": 0, "score": 3.0}],
            "retained_checkpoints": [],
        }
        train_module.persist_actor_bundle = lambda *_args, **_kwargs: Path("actor.pt")
        train_module.persist_retained_checkpoint_bundles = lambda **_kwargs: []
        train_module.slice_metrics = lambda *_args, **_kwargs: {
            "test_2022": {"return": -9.0, "sharpe": -0.3, "max_dd": 13.0}
        }
        train_module.evaluate_split = lambda *_args, **_kwargs: {
            "return": 10.0,
            "sharpe": 0.1,
            "max_dd": 9.0,
        }
        train_module.backtest_profile = fake_backtest_profile
        train_module.build_promotion_report = lambda _profile, _slices, test_proxy, **_kwargs: {
            "proxy_return": test_proxy["return"]
        }
        train_module.persist_summary_bundle = fake_persist_summary_bundle
        train_module.export_review_daily_csv = lambda *_args, **_kwargs: None

        result = train_module.run_profile(
            profile=profile,
            train_df=frame,
            valid_df=frame,
            test_df=frame,
            price_series=price_series,
            smoke=False,
            seed=1,
        )
    finally:
        for name, value in original.items():
            setattr(train_module, name, value)

    assert result["test_proxy"]["return"] == 123.0
    assert captured["test_proxy"]["return"] == 123.0
    assert captured["test_proxy"]["relative_return"] == -77.0
    assert captured["slice_result"]["test_2022"]["max_dd"] == 12.0
    assert captured["promotion_report"]["proxy_return"] == 123.0


def test_run_profile_records_checkpoint_retention_gate_block_reason():
    profile = replace(
        load_profile("balanced"),
        checkpoint_retention=CheckpointRetentionConfig(enabled=True, keep_stage=True, keep_top_n=1),
    )
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=dates)
    price_series = pd.Series([100.0, 101.0], index=dates)
    captured: dict[str, object] = {}
    original = {
        "observation_dim_for_profile": train_module.observation_dim_for_profile,
        "feature_columns_for_packs": train_module.feature_columns_for_packs,
        "build_initial_states": train_module.build_initial_states,
        "build_policy": train_module.build_policy,
        "train_with_selection": train_module.train_with_selection,
        "persist_actor_bundle": train_module.persist_actor_bundle,
        "persist_retained_checkpoint_bundles": train_module.persist_retained_checkpoint_bundles,
        "slice_metrics": train_module.slice_metrics,
        "evaluate_split": train_module.evaluate_split,
        "backtest_profile": train_module.backtest_profile,
        "build_promotion_report": train_module.build_promotion_report,
        "persist_summary_bundle": train_module.persist_summary_bundle,
        "export_review_daily_csv": train_module.export_review_daily_csv,
    }

    def fake_persist_retained_checkpoint_bundles(**kwargs):
        captured["retention_gate_evidence"] = kwargs.get("retention_gate_evidence")
        return []

    def fake_persist_summary_bundle(**kwargs):
        captured["checkpoint_retention_decision"] = kwargs.get("checkpoint_retention_decision")
        captured["retained_checkpoints"] = kwargs["retained_checkpoints"]
        return Path("runtime/model/card/balanced/soxl_balanced_summary.json")

    try:
        train_module.observation_dim_for_profile = lambda _profile: 3
        train_module.feature_columns_for_packs = lambda _packs, _name: []
        train_module.build_initial_states = lambda *_args, **_kwargs: [object()]
        train_module.build_policy = lambda *_args, **_kwargs: (object(), _DummyActor())
        train_module.train_with_selection = lambda **_kwargs: {
            "best_metrics": {"return": 1.0, "sharpe": 0.5, "max_dd": 2.0},
            "best_score": 3.0,
            "history": [{"iter": 0, "score": 3.0}],
            "retained_checkpoints": [
                {
                    "iter": 20,
                    "score": 3.0,
                    "metrics": {"return": 1.0},
                    "state_dict": {"weight": 2.0},
                    "reasons": ["top_n"],
                }
            ],
        }
        train_module.persist_actor_bundle = lambda *_args, **_kwargs: Path("actor.pt")
        train_module.persist_retained_checkpoint_bundles = fake_persist_retained_checkpoint_bundles
        train_module.slice_metrics = lambda *_args, **_kwargs: {
            "test_2022": {"return": -9.0, "sharpe": -0.3, "max_dd": 13.0}
        }
        train_module.evaluate_split = lambda *_args, **_kwargs: {
            "return": 10.0,
            "sharpe": 0.1,
            "max_dd": 9.0,
        }
        train_module.backtest_profile = lambda *_args, **_kwargs: {
            "return": 12.0,
            "sharpe": 0.3,
            "max_dd": 8.0,
        }
        train_module.build_promotion_report = lambda _profile, _slices, _test_proxy, **_kwargs: {
            "eligible": False,
            "walk_forward_evaluated": False,
        }
        train_module.persist_summary_bundle = fake_persist_summary_bundle
        train_module.export_review_daily_csv = lambda *_args, **_kwargs: None

        train_module.run_profile(
            profile=profile,
            train_df=frame,
            valid_df=frame,
            test_df=frame,
            price_series=price_series,
            smoke=False,
            seed=1,
        )
    finally:
        for name, value in original.items():
            setattr(train_module, name, value)

    assert captured["retention_gate_evidence"] is None
    assert captured["retained_checkpoints"] == []
    assert captured["checkpoint_retention_decision"] == {
        "profile": "balanced",
        "reason": "strict retention gate evidence missing",
        "status": "blocked",
    }


if __name__ == "__main__":
    test_run_profile_persists_summary_after_main_backtest_corrections()
    test_run_profile_records_checkpoint_retention_gate_block_reason()
    print("test_train_summary_persistence: OK")
