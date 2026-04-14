from __future__ import annotations

"""SOXL qlib-native 包的训练与评估入口。

本模块负责模型选择、实验输出和晋升门控（promotion gate）。
它不应包含 broker 相关的执行逻辑；实盘/模拟盘执行在 runtime.py 中。

【训练流程概览】
  1. build_dataset() 构建 train / valid / test 三段数据
  2. build_initial_states() 把长序列切成多个 episode
  3. qlib_train() 调用 tianshou PPO 进行训练
  4. evaluate_split() 在 valid 集上评估，保存最佳 checkpoint
  5. backtest_profile() 在 test 集上跑官方 qlib 回测
  6. 检查 promotion_gate，决定是否允许模型晋升到 semi-auto / live

【优化提示】
  - 当前 walk_forward 是硬编码的 3 个窗口（wf_2020, wf_2022, wf_2023_2024）。
  - 做多资产训练时，需要在此扩展 initial_states 的生成逻辑，支持从多只 ETF 采样。
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
from qlib.data import D
from qlib.rl.trainer.api import train as qlib_train

from .backtest import backtest_profile, save_results
from .compat import patch_gym_compat
from .data import FEATURE_COLUMNS, build_dataset, refresh_qlib_binary, init_qlib, segment_frame
from .rl import ContinuousReward, PositionAction, SoxlSimulator, SoxlState, build_initial_states, build_policy, evaluate_split
from .settings import (
    CSV_PATH,
    EXTRA_OBS_DIM,
    MODEL_DIR,
    PromotionGateConfig,
    QLIB_DATA_PATH,
    RESULT_PNG,
    RESULT_TXT,
    ROOT,
    TEST_SEG,
    load_profile,
    model_artifact_path,
    summary_artifact_path,
)

warnings.filterwarnings("ignore")

WalkForwardWindow = Tuple[str, Tuple[str, str], Tuple[str, str], Tuple[str, str]]

WALK_FORWARD_WINDOWS: Tuple[WalkForwardWindow, ...] = (
    # 格式：(窗口名, train, valid, test)
    ("wf_2020", ("2010-03-11", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31")),
    ("wf_2022", ("2010-03-11", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
    ("wf_2023_2024", ("2010-03-11", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
)

WORKER_RESULT_DIR = ROOT / "runtime" / "soxl" / "wf_workers"


def validation_score(metrics: Dict[str, float]) -> float:
    """模型选择用的单一评分函数。

    公式：return + sharpe * 20 - max_dd * 0.12
    Sharpe 的权重很高（×20），说明我们更看重风险调整后收益；
    回撤惩罚相对温和（×0.12），避免模型过度保守导致收益过低。
    
    【优化提示】如果后续发现模型在 valid 上收益高但 test 上回撤大，
    可提高 max_dd 的惩罚系数，或改用 Sortino 替代 Sharpe。
    """


def slice_frame(df, seg: Tuple[str, str]):
    return df.loc[seg[0] : seg[1]].copy()


def describe_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"[{name}] return={metrics['return']:.1f}% "
        f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}%"
    )


def compact_result(result: Dict[str, object]) -> Dict[str, object]:
    """Drop heavyweight objects before persisting cross-process worker output."""

    return {
        "return": float(result["return"]),
        "sharpe": float(result["sharpe"]),
        "max_dd": float(result["max_dd"]),
        "trades": int(result["trades"]),
        "valid": result.get("valid", {}),
        "slices": result.get("slices", {}),
        "test_proxy": result.get("test_proxy", {}),
        "promotion": result.get("promotion", {}),
        "model_path": result.get("model_path", ""),
        "summary_path": result.get("summary_path", ""),
    }


def fixed_promotion_report(slices: Dict[str, Dict[str, float]], test_proxy: Dict[str, float], gate: PromotionGateConfig) -> Dict[str, object]:
    """Evaluate the fixed-split hard gate for model promotion."""

    test_2022 = slices.get("test_2022", {})
    checks = {
        "test_2022_max_dd": {
            "passed": bool(test_2022) and float(test_2022.get("max_dd", float("inf"))) <= gate.max_test_2022_drawdown,
            "actual": float(test_2022.get("max_dd", float("inf"))),
            "limit": float(gate.max_test_2022_drawdown),
        },
        "test_2022_avg_pos": {
            "passed": bool(test_2022) and float(test_2022.get("avg_pos", float("inf"))) <= gate.max_test_2022_avg_pos,
            "actual": float(test_2022.get("avg_pos", float("inf"))),
            "limit": float(gate.max_test_2022_avg_pos),
        },
        "test_proxy_sharpe": {
            "passed": float(test_proxy.get("sharpe", float("-inf"))) >= gate.min_test_proxy_sharpe,
            "actual": float(test_proxy.get("sharpe", float("-inf"))),
            "limit": float(gate.min_test_proxy_sharpe),
        },
    }
    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}


def walk_forward_promotion_report(walk_forward_results: Dict[str, Dict[str, object]], gate: PromotionGateConfig) -> Dict[str, object]:
    """Evaluate walk-forward gates used before semi-auto/live usage."""

    wf_2023_2024 = walk_forward_results.get("wf_2023_2024", {})
    checks = {
        "wf_2023_2024_return": {
            "passed": bool(wf_2023_2024) and float(wf_2023_2024.get("return", float("-inf"))) >= gate.min_wf_2023_2024_return,
            "actual": float(wf_2023_2024.get("return", float("-inf"))),
            "limit": float(gate.min_wf_2023_2024_return),
        },
        "wf_2023_2024_sharpe": {
            "passed": bool(wf_2023_2024) and float(wf_2023_2024.get("sharpe", float("-inf"))) >= gate.min_wf_2023_2024_sharpe,
            "actual": float(wf_2023_2024.get("sharpe", float("-inf"))),
            "limit": float(gate.min_wf_2023_2024_sharpe),
        },
    }
    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}


def build_promotion_report(
    profile,
    slices: Dict[str, Dict[str, float]],
    test_proxy: Dict[str, float],
    walk_forward_results: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Combine fixed-split and walk-forward gates into one promotion verdict."""

    fixed_report = fixed_promotion_report(slices, test_proxy, profile.promotion_gate)
    if walk_forward_results:
        walk_forward_report = walk_forward_promotion_report(walk_forward_results, profile.promotion_gate)
        walk_forward_evaluated = True
    else:
        walk_forward_report = {"eligible": False, "checks": {}, "status": "missing"}
        walk_forward_evaluated = False
    eligible = fixed_report["eligible"] and walk_forward_report["eligible"] if walk_forward_evaluated else False
    return {
        "eligible": eligible,
        "fixed_split": fixed_report,
        "walk_forward": walk_forward_report,
        "walk_forward_evaluated": walk_forward_evaluated,
    }


def persist_promotion_report(summary_path: Path, promotion_report: Dict[str, object]) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["promotion"] = promotion_report
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def slice_metrics(actor, df, slices: Dict[str, Tuple[str, str]], profile) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for name, seg in slices.items():
        segment = slice_frame(df, seg)
        if len(segment) <= 120:
            continue
        metrics[name] = evaluate_split(actor, segment, position_cap_config=profile.position_cap)
    return metrics


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_with_selection(policy, actor, initial_states, obs_dim: int, profile, valid_df, smoke: bool) -> Dict[str, object]:
    """Train the actor and keep the best checkpoint by validation score.

    The balanced non-smoke path still runs as one contiguous qlib trainer
    session because that path is already expensive. Shorter/staged profiles
    checkpoint every small block so we can stop on plateau instead of blindly
    trusting the final iteration.
    """
    if profile.name == "balanced" and not smoke:
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=PositionAction(position_cap_config=profile.position_cap),
            initial_states=initial_states,
            policy=policy,
            reward=ContinuousReward(profile),
            vessel_kwargs={
                "episode_per_iter": profile.episode_per_iter,
                "update_kwargs": {"batch_size": profile.update_batch_size, "repeat": profile.update_repeat},
            },
            trainer_kwargs={"finite_env_type": "dummy", "concurrency": 1, "max_iters": profile.max_iters},
        )
        final_metrics = evaluate_split(actor, valid_df, position_cap_config=profile.position_cap)
        final_score = validation_score(final_metrics)
        history = [
            {"iter": 0, "score": final_score, **final_metrics},
            {"iter": profile.max_iters, "score": final_score, **final_metrics},
        ]
        print(
            f"[valid:{profile.name}@{profile.max_iters}] return={final_metrics['return']:.1f}% "
            f"sharpe={final_metrics['sharpe']:.2f} maxdd={final_metrics['max_dd']:.1f}% score={final_score:.2f}"
        )
        return {"best_metrics": final_metrics, "best_score": final_score, "history": history}

    stage_iters = profile.max_iters if smoke else min(20, profile.max_iters)
    completed = 0
    best_state = copy.deepcopy(actor.state_dict())
    best_metrics = evaluate_split(actor, valid_df, position_cap_config=profile.position_cap)
    best_score = validation_score(best_metrics)
    history = [{"iter": 0, "score": best_score, **best_metrics}]

    while completed < profile.max_iters:
        current_iters = min(stage_iters, profile.max_iters - completed)
        qlib_train(
            simulator_fn=SoxlSimulator,
            state_interpreter=SoxlState(obs_dim),
            action_interpreter=PositionAction(position_cap_config=profile.position_cap),
            initial_states=initial_states,
            policy=policy,
            reward=ContinuousReward(profile),
            vessel_kwargs={
                "episode_per_iter": profile.episode_per_iter,
                "update_kwargs": {"batch_size": profile.update_batch_size, "repeat": profile.update_repeat},
            },
            trainer_kwargs={"finite_env_type": "dummy", "concurrency": 1, "max_iters": current_iters},
        )
        completed += current_iters
        metrics = evaluate_split(actor, valid_df, position_cap_config=profile.position_cap)
        score = validation_score(metrics)
        history.append({"iter": completed, "score": score, **metrics})
        print(
            f"[valid:{profile.name}@{completed}] return={metrics['return']:.1f}% "
            f"sharpe={metrics['sharpe']:.2f} maxdd={metrics['max_dd']:.1f}% score={score:.2f}"
        )
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(actor.state_dict())

    actor.load_state_dict(best_state)
    return {"best_metrics": best_metrics, "best_score": best_score, "history": history}


def run_profile(
    profile,
    train_df,
    valid_df,
    test_df,
    price_series,
    smoke: bool,
    seed: int,
    artifact_tag: str = "",
) -> Dict[str, object]:
    """Execute one full train -> select -> persist -> backtest cycle."""
    obs_dim = len(FEATURE_COLUMNS) + EXTRA_OBS_DIM
    pkl_path = ROOT / f"data/soxl_profile_{profile.name}.pkl"
    train_df.to_pickle(pkl_path)
    initial_states = build_initial_states(pkl_path, train_df, profile, smoke)
    if not initial_states:
        raise RuntimeError(f"no initial states generated for profile={profile.name}")

    set_global_seed(seed)
    print(
        f"\n[train:{profile.name}] seed={seed} episodes={len(initial_states)} "
        f"iters={profile.max_iters} epi/iter={profile.episode_per_iter}"
    )

    policy, actor = build_policy(obs_dim, profile)
    selection = train_with_selection(
        policy=policy,
        actor=actor,
        initial_states=initial_states,
        obs_dim=obs_dim,
        profile=profile,
        valid_df=valid_df,
        smoke=smoke,
    )
    valid_metrics = selection["best_metrics"]
    print(
        f"[valid:{profile.name}] return={valid_metrics['return']:.1f}% "
        f"sharpe={valid_metrics['sharpe']:.2f} maxdd={valid_metrics['max_dd']:.1f}% "
        f"best_score={selection['best_score']:.2f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_artifact_path(profile.name, artifact_tag)
    summary_path = summary_artifact_path(profile.name, artifact_tag)
    torch.save(
        {
            "profile": profile.name,
            "seed": seed,
            "hidden_sizes": list(profile.hidden_sizes),
            "obs_dim": obs_dim,
            "n_actions": int(getattr(actor, "output_dim", 0) or 11),
            "position_cap": {
                "start_drawdown": profile.position_cap.start_drawdown,
                "full_drawdown": profile.position_cap.full_drawdown,
                "min_cap": profile.position_cap.min_cap,
                "curve_power": profile.position_cap.curve_power,
            },
            "state_dict": actor.state_dict(),
        },
        model_path,
    )
    slice_result = slice_metrics(
        actor,
        test_df,
        {
            "test_2020": ("2020-01-01", "2020-12-31"),
            "test_2022": ("2022-01-01", "2022-12-31"),
            "test_2023_2024": ("2023-01-01", "2024-12-31"),
        },
        profile,
    )
    test_proxy = evaluate_split(actor, test_df, position_cap_config=profile.position_cap)
    promotion_report = build_promotion_report(profile, slice_result, test_proxy)
    summary_path.write_text(
        json.dumps(
            {
                "profile": profile.name,
                "seed": seed,
                "position_cap": {
                    "start_drawdown": profile.position_cap.start_drawdown,
                    "full_drawdown": profile.position_cap.full_drawdown,
                    "min_cap": profile.position_cap.min_cap,
                    "curve_power": profile.position_cap.curve_power,
                },
                "model_path": str(model_path),
                "valid": valid_metrics,
                "best_score": selection["best_score"],
                "selection_history": selection["history"],
                "slices": slice_result,
                "test_proxy": test_proxy,
                "promotion": promotion_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    bt_result = backtest_profile(actor, test_df, price_series, position_cap_config=profile.position_cap)
    bt_result["valid"] = valid_metrics
    bt_result["selection_history"] = selection["history"]
    bt_result["model_path"] = str(model_path)
    bt_result["summary_path"] = str(summary_path)
    bt_result["slices"] = slice_result
    bt_result["test_proxy"] = test_proxy
    bt_result["promotion"] = promotion_report
    return bt_result


def run_walk_forward(profile, full_df, price_frame, smoke: bool, seed: int) -> Dict[str, Dict[str, object]]:
    """Run expanding-window walk-forward slices used by the non-leak harness."""
    results: Dict[str, Dict[str, object]] = {}
    for name, train_seg, valid_seg, test_seg in WALK_FORWARD_WINDOWS:
        wf_profile = profile
        train_slice = slice_frame(full_df, train_seg)
        valid_slice = slice_frame(full_df, valid_seg)
        test_slice = slice_frame(full_df, test_seg)
        price_slice = price_frame.loc[test_seg[0] : test_seg[1]]
        if min(len(train_slice), len(valid_slice), len(test_slice)) <= 120:
            continue
        print(
            f"\n[walk-forward:{profile.name}:{name}] "
            f"train={train_seg[0]}->{train_seg[1]} valid={valid_seg[0]}->{valid_seg[1]} test={test_seg[0]}->{test_seg[1]}"
        )
        result = run_profile(
            profile=wf_profile,
            train_df=train_slice,
            valid_df=valid_slice,
            test_df=test_slice,
            price_series=price_slice,
            smoke=smoke,
            seed=seed,
            artifact_tag=name,
        )
        results[name] = result
        print(describe_metrics(f"walk-forward:{profile.name}:{name}", result))
        if smoke:
            break
    return results


def run_walk_forward_isolated(profile, smoke: bool, seed: int, skip_data_refresh: bool) -> Dict[str, Dict[str, object]]:
    """Run each walk-forward window in a disposable subprocess.

    qlib occasionally dies during Python 3.14 shutdown after already computing
    the metrics we need. Running each window in a worker process lets us write a
    compact result to disk and then terminate the worker immediately with
    `os._exit(0)`, avoiding teardown noise from poisoning the parent process.
    """

    results: Dict[str, Dict[str, object]] = {}
    WORKER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name, train_seg, valid_seg, test_seg in WALK_FORWARD_WINDOWS:
        print(
            f"\n[walk-forward:{profile.name}:{name}:isolated] "
            f"train={train_seg[0]}->{train_seg[1]} valid={valid_seg[0]}->{valid_seg[1]} test={test_seg[0]}->{test_seg[1]}"
        )
        worker_output = WORKER_RESULT_DIR / f"{profile.name}-{name}-seed{seed}.json"
        if worker_output.exists():
            worker_output.unlink()
        cmd = [
            sys.executable,
            str(ROOT / "soxl_qlib_strategy_v2.py"),
            "--profiles",
            profile.name,
            "--seed",
            str(seed),
            "--worker-run-profile",
            profile.name,
            "--worker-window-name",
            name,
            "--worker-train-start",
            train_seg[0],
            "--worker-train-end",
            train_seg[1],
            "--worker-valid-start",
            valid_seg[0],
            "--worker-valid-end",
            valid_seg[1],
            "--worker-test-start",
            test_seg[0],
            "--worker-test-end",
            test_seg[1],
            "--worker-output",
            str(worker_output),
        ]
        if smoke:
            cmd.append("--smoke")
        if skip_data_refresh:
            cmd.append("--skip-data-refresh")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if not worker_output.exists():
            raise RuntimeError(f"walk-forward worker failed before writing result: {name} exit={proc.returncode}")
        worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))
        results[name] = worker_payload
        print(describe_metrics(f"walk-forward:{profile.name}:{name}", worker_payload))
        if smoke:
            break
    return results


def run_worker_profile(args: argparse.Namespace) -> None:
    """Run one train/valid/test slice and persist a compact JSON result."""

    patch_gym_compat()
    set_global_seed(args.seed)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    full_df = pd.concat(
        [
            segment_frame(dataset, "train"),
            segment_frame(dataset, "valid"),
            segment_frame(dataset, "test"),
        ]
    ).sort_index()
    price_frame = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()
    profile = load_profile(args.worker_run_profile)
    if args.smoke or args.worker_run_profile == "smoke":
        profile = load_profile("smoke")
    result = run_profile(
        profile=profile,
        train_df=slice_frame(full_df, (args.worker_train_start, args.worker_train_end)),
        valid_df=slice_frame(full_df, (args.worker_valid_start, args.worker_valid_end)),
        test_df=slice_frame(full_df, (args.worker_test_start, args.worker_test_end)),
        price_series=price_frame.loc[args.worker_test_start : args.worker_test_end],
        smoke=args.smoke,
        seed=args.seed,
        artifact_tag=args.worker_window_name,
    )
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact_result(result), indent=2), encoding="utf-8")
    # qlib cleanup under Python 3.14 is unreliable; the parent process only
    # needs the persisted JSON, so terminate this worker immediately.
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOXL qlib-native PPO strategy")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["balanced", "aggressive"],
        choices=["balanced", "aggressive", "smoke"],
        help="profiles to train and backtest",
    )
    parser.add_argument("--smoke", action="store_true", help="force short smoke run")
    parser.add_argument("--skip-data-refresh", action="store_true", help="reuse existing qlib binary files")
    parser.add_argument("--walk-forward", action="store_true", help="run expanding-window walk-forward evaluation")
    parser.add_argument("--seed", type=int, default=20260411, help="global random seed for Python, NumPy, and Torch")
    parser.add_argument("--worker-run-profile", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-window-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-train-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-train-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-valid-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-valid-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-test-start", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-test-end", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_run_profile:
        run_worker_profile(args)
        return
    patch_gym_compat()
    set_global_seed(args.seed)

    print("=" * 72)
    print("SOXL Strategy v2 - qlib native package")
    print("=" * 72)

    if not args.skip_data_refresh:
        csv_df = refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
        print(f"[data] qlib binary refreshed: {len(csv_df)} rows, end={csv_df.index[-1].date()}")
    else:
        print("[data] skip binary refresh")

    init_qlib(QLIB_DATA_PATH)
    probe = D.features(["SOXL", "SOXX"], ["$close"], start_time="2026-03-30", end_time="2026-04-02")
    print(f"[qlib] instruments ok: {sorted(probe.index.get_level_values('instrument').unique().tolist())}")

    dataset = build_dataset()
    train_df = segment_frame(dataset, "train")
    valid_df = segment_frame(dataset, "valid")
    test_df = segment_frame(dataset, "test")
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

    print(f"[dataset] train={len(train_df)} valid={len(valid_df)} test={len(test_df)} features={len(FEATURE_COLUMNS)}")

    results: Dict[str, Dict[str, object]] = {}
    walk_forward_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for name in args.profiles:
        profile = load_profile(name if name != "smoke" else "balanced")
        if args.smoke or name == "smoke":
            profile = load_profile("smoke")
        bt_result = run_profile(
            profile=profile,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            price_series=price_series,
            smoke=args.smoke or name == "smoke",
            seed=args.seed,
        )
        results[profile.name] = bt_result
        print(
            f"[test:{profile.name}] return={bt_result['return']:.1f}% "
            f"sharpe={bt_result['sharpe']:.2f} maxdd={bt_result['max_dd']:.1f}% trades={bt_result['trades']} "
            f"model={Path(bt_result['model_path']).name}"
        )
        if args.walk_forward:
            walk_forward_results[profile.name] = run_walk_forward_isolated(
                profile=profile,
                smoke=args.smoke or name == "smoke",
                seed=args.seed,
                skip_data_refresh=True,
            )
            promotion_report = build_promotion_report(
                profile,
                bt_result["slices"],
                bt_result["test_proxy"],
                walk_forward_results=walk_forward_results[profile.name],
            )
            bt_result["promotion"] = promotion_report
            persist_promotion_report(Path(bt_result["summary_path"]), promotion_report)

    save_results(results)
    print(f"\n[done] chart={RESULT_PNG.name} summary={RESULT_TXT.name}")
    for name, result in results.items():
        print(
            f"  - {name}: {result['return']:.1f}% | Sharpe {result['sharpe']:.2f} | "
            f"MaxDD {result['max_dd']:.1f}% | trades {result['trades']}"
        )
        for wf_name, wf_result in walk_forward_results.get(name, {}).items():
            print(
                f"    {wf_name}: {wf_result['return']:.1f}% | Sharpe {wf_result['sharpe']:.2f} | "
                f"MaxDD {wf_result['max_dd']:.1f}% | trades {wf_result['trades']}"
            )
        promotion = result.get("promotion", {})
        if promotion:
            status = "eligible" if promotion.get("eligible") else "blocked"
            print(f"    promotion: {status}")
