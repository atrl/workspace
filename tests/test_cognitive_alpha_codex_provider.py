from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


def _evolution_frame(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    close = 30.0 + np.linspace(0.0, 5.0, rows) + np.sin(np.arange(rows) / 5.0)
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 1_000,
        }
    )
    frame["label_return_5d"] = pd.Series(close).pct_change(5).shift(-5).ffill().fillna(0.0)
    return frame


def _fake_codex_bin(tmp_path):
    fake = tmp_path / "fake_codex.py"
    fake.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib, re, sys",
                "args = sys.argv[1:]",
                "if args[:2] == ['doctor', '--json']:",
                "    print(json.dumps({'overallStatus': 'ok', 'codexVersion': 'unit', 'checks': {'auth.credentials': {'details': {'stored auth mode': 'chatgpt', 'stored API key': 'false', 'stored ChatGPT tokens': 'true'}}}}))",
                "    raise SystemExit(0)",
                "prompt = sys.stdin.read()",
                "record_path = pathlib.Path(os.environ['FAKE_CODEX_RECORD'])",
                "record_path.write_text(json.dumps({'args': args, 'prompt': prompt, 'openai_api_key': os.environ.get('OPENAI_API_KEY')}), encoding='utf-8')",
                "match = re.search(r'\"request_index\":\\s*(\\d+)', prompt)",
                "idx = int(match.group(1)) if match else 0",
                "out = pathlib.Path(args[args.index('--output-last-message') + 1])",
                "out.write_text(json.dumps({'candidates': [{",
                "    'name': f'codex_exec_alpha_{idx:03d}',",
                "    'level': 'market_structure_cycle',",
                "    'agent': 'AgentMarketCycle',",
                "    'theme': 'codex exec generated lagged trend alpha',",
                "    'required_columns': ['close'],",
                "    'lookback': 10,",
                "    'code': 'close.pct_change(5).shift(1)',",
                "    'economic_rationale': 'Codex CLI emitted a lagged trend persistence factor.'",
                "}]}), encoding='utf-8')",
                "print('fake codex ok')",
            ]
        ),
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | 0o111)
    return fake


def test_codex_exec_completion_invokes_codex_without_openai_api_key(tmp_path, monkeypatch):
    from src.workflow import codex_exec_completion

    fake = _fake_codex_bin(tmp_path)
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_CODEX_RECORD", record.as_posix())
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    result = codex_exec_completion(
        {
            "paper_reference": "arxiv:2511.18850",
            "request_index": 3,
            "target_candidate_count": 1,
            "level": "market_structure_cycle",
            "agent": "AgentMarketCycle",
            "available_columns": ["close", "volume"],
        },
        codex_bin=fake.as_posix(),
        model="unit-codex",
        cwd=tmp_path,
        timeout_seconds=5,
    )

    captured = json.loads(record.read_text(encoding="utf-8"))
    assert captured["args"][0] == "exec"
    assert "--output-schema" in captured["args"]
    assert "--output-last-message" in captured["args"]
    assert "--ask-for-approval" in captured["args"]
    assert "never" in captured["args"]
    assert captured["openai_api_key"] is None
    assert "arxiv:2511.18850" in captured["prompt"]
    assert "Return only JSON" in captured["prompt"]
    assert result["candidates"][0]["name"] == "codex_exec_alpha_003"


def test_codex_exec_candidate_provider_backs_cognitive_alpha_evolution(tmp_path, monkeypatch):
    from src.workflow import (
        CognitiveAlphaEvolutionSpec,
        build_cognitive_alpha_codex_exec_candidate_provider,
        run_cognitive_alpha_evolution,
    )

    fake = _fake_codex_bin(tmp_path)
    monkeypatch.setenv("FAKE_CODEX_RECORD", (tmp_path / "record.json").as_posix())
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    provider = build_cognitive_alpha_codex_exec_candidate_provider(
        codex_bin=fake.as_posix(),
        provider_id="unit_codex_exec",
        model="unit-codex",
        cwd=tmp_path,
        timeout_seconds=5,
    )
    spec = CognitiveAlphaEvolutionSpec(
        run_id="codex_exec_provider",
        profile="soxl_smoke",
        runtime_dir=tmp_path,
        generations=1,
        initial_pool_size=2,
        parent_pool_size=1,
        children_per_generation=2,
        qualified_percentile=0.50,
        elite_percentile=0.75,
        fallback_top_n=1,
    )

    result = run_cognitive_alpha_evolution(
        _evolution_frame(rows=120),
        spec=spec,
        label_col="label_return_5d",
        candidate_provider=provider,
    )

    provider_summary = result.summary["candidate_provider"]
    assert provider_summary["provider_id"] == "unit_codex_exec"
    assert provider_summary["provider_kind"] == "codex_exec_code"
    assert provider_summary["openai_api_key_required"] is False
    assert provider_summary["codex_direct_compatible"] is True
    assert provider_summary["codex_execution_mode"] == "codex_exec_chatgpt_auth_no_openai_api_key"
