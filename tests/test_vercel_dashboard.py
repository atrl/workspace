from __future__ import annotations

import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_serves_dashboard_as_static_root():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert {"source": "/", "destination": "/dashboard.html"} in config["rewrites"]
    header_sources = {entry["source"]: entry for entry in config["headers"]}
    assert "/" in header_sources
    assert "/dashboard.html" in header_sources
    assert "/dashboard_data/:path*" in header_sources

    no_cache_header = {
        "key": "Cache-Control",
        "value": "public, max-age=0, must-revalidate",
    }
    assert no_cache_header in header_sources["/"]["headers"]
    assert no_cache_header in header_sources["/dashboard.html"]["headers"]
    assert no_cache_header in header_sources["/dashboard_data/:path*"]["headers"]


def test_dashboard_static_export_is_trackable():
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    active_ignore_lines = {
        line.strip()
        for line in ignore_lines
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "/dashboard.html" not in active_ignore_lines
    assert "/dashboard_data/" not in active_ignore_lines
    assert (ROOT / "dashboard.html").exists()
    assert (ROOT / "dashboard_data" / "index.json").exists()
    assert (ROOT / "dashboard_data" / "model_versions.json").exists()
    assert (ROOT / "dashboard_data" / "model_replay_freshness.json").exists()
    assert (ROOT / "dashboard_data" / "research_backtests.json").exists()
    assert (ROOT / "dashboard_data" / "production_status.json").exists()
    assert (ROOT / "dashboard_data" / "production_goal_gates.json").exists()


def test_dashboard_page_loads_production_status_export():
    html = (ROOT / "dashboard.html").read_text(encoding="utf-8")

    assert 'id="productionStatusCard"' in html
    assert "dashboard_data/production_status.json" in html
    assert "dashboard_data/production_goal_gates.json" in html
    assert "renderProductionStatus" in html
    assert 'id="productionGateOverall"' in html


def test_dashboard_page_loads_research_backtest_export():
    html = (ROOT / "dashboard.html").read_text(encoding="utf-8")

    assert 'id="researchBacktests"' in html
    assert "dashboard_data/research_backtests.json" in html
    assert "renderResearchBacktests" in html


def test_production_goal_gates_parser_extracts_overall_blockers():
    from scripts.export_production_goal_gates import parse_goal_gates

    payload = parse_goal_gates(
        "\n".join(
            [
                "=== SOXL Scheduler Health Check ===",
                "--- Production Goal Gates ---",
                "  data: ok last_date=2026-05-20 expected=2026-05-20",
                "  Vercel remote: blocked status=blocked_no_credentials auth=blocked_no_credentials",
                "  post-close composer target: pending latest_success_marker lacks schedule_slot=postclose",
                "  ordinary paper proof: pending selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=duplicate_skipped",
                "  overall: pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof",
            ]
        ),
        checked_at="2026-05-21T10:20:00+00:00",
    )

    assert payload["checked_at"] == "2026-05-21T10:20:00+00:00"
    assert payload["overall"] == {
        "status": "pending",
        "issues": ["external_vercel_setup", "scheduled_postclose_target", "scheduled_paper_proof"],
        "detail": "external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof",
    }
    assert payload["gates"]["vercel_remote"]["status"] == "blocked"
    assert payload["gates"]["post_close_composer_target"]["status"] == "pending"


def test_dashboard_static_export_integrity_check_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_dashboard_static_export.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing=0" in result.stdout
    assert "latest=paper_trading" in result.stdout
    assert "paper_trading" in result.stdout
    assert "production_status=" in result.stdout
    assert "goal_gates=pending" in result.stdout
    assert "model_replays=" in result.stdout
    assert "dashboard_v2=" in result.stdout
    assert "paper_latest_signal=" in result.stdout
    assert "research_backtests=" in result.stdout


def test_dashboard_static_export_rejects_stale_paper_latest_signal():
    from scripts.check_dashboard_static_export import check_static_export

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "dashboard_data"
        paper_dir = data_dir / "versioned" / "paper_trading"
        paper_dir.mkdir(parents=True)
        (root / "dashboard.html").write_text("<html></html>", encoding="utf-8")
        (root / "vercel.json").write_text(
            json.dumps(
                {
                    "rewrites": [{"source": "/", "destination": "/dashboard.html"}],
                    "headers": [
                        {
                            "source": source,
                            "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}],
                        }
                        for source in ("/", "/dashboard.html", "/dashboard_data/:path*")
                    ],
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "index.json").write_text(
            json.dumps(
                {
                    "versions": ["paper_trading"],
                    "latest_version": "paper_trading",
                    "version_summaries": {
                        "paper_trading": {
                            "etfs": ["SOXL"],
                            "windows": {"SOXL": {"rows": 1, "end": "2026-05-20"}},
                            "meta": {"source_kind": "paper_trading"},
                        }
                    },
                    "etfs": ["SOXL"],
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "model_versions.json").write_text("[]", encoding="utf-8")
        (data_dir / "model_replay_freshness.json").write_text(
            json.dumps(
                {
                    "counts": {"current": 1, "refreshed": 0, "failed": 0, "skipped": 0},
                    "profiles": [{"name": "balanced_rule_residual_v22"}],
                    "safe_backtest_end": "2026-05-21",
                    "latest_data_date": "2026-05-22",
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "production_status.json").write_text(
            json.dumps(
                {
                    "status": "ready_for_deploy",
                    "auth_status": "token",
                    "project_status": "env_configured",
                    "static_export": {"latest": "paper_trading"},
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "research_backtests.json").write_text(
            json.dumps([{"experiment": "x", "decision": "y", "summary_path": "z"}]),
            encoding="utf-8",
        )
        (paper_dir / "SOXL.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "source_kind": "paper_trading",
                        "latest_signal_date": "2026-05-21",
                    },
                    "rows": [
                        {
                            "datetime": "2026-05-20",
                            "strategy_nav": 1.0,
                            "buyhold_nav": 1.0,
                            "price": 190.56,
                            "position": 0.35,
                            "action": "BUY",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        errors, info = check_static_export(root)

    assert info["paper_latest_signal"] == "2026-05-21"
    assert any("paper_trading latest signal is stale" in error for error in errors)


def test_dashboard_static_export_rejects_sensitive_paper_review_fields():
    from scripts.check_dashboard_static_export import check_static_export

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "dashboard_data"
        paper_dir = data_dir / "versioned" / "paper_trading"
        paper_dir.mkdir(parents=True)
        (root / "dashboard.html").write_text("<html></html>", encoding="utf-8")
        (root / "vercel.json").write_text(
            json.dumps(
                {
                    "rewrites": [{"source": "/", "destination": "/dashboard.html"}],
                    "headers": [
                        {
                            "source": source,
                            "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}],
                        }
                        for source in ("/", "/dashboard.html", "/dashboard_data/:path*")
                    ],
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "index.json").write_text(
            json.dumps(
                {
                    "versions": ["paper_trading"],
                    "latest_version": "paper_trading",
                    "version_summaries": {
                        "paper_trading": {
                            "etfs": ["SOXL"],
                            "windows": {"SOXL": {"rows": 1, "end": "2026-05-22"}},
                            "meta": {"source_kind": "paper_trading"},
                        }
                    },
                    "etfs": ["SOXL"],
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "model_versions.json").write_text("[]", encoding="utf-8")
        (data_dir / "model_replay_freshness.json").write_text(
            json.dumps({"counts": {}, "profiles": [], "safe_backtest_end": "2026-05-22", "latest_data_date": "2026-05-22"}),
            encoding="utf-8",
        )
        (data_dir / "production_status.json").write_text(
            json.dumps(
                {
                    "status": "ready_for_deploy",
                    "auth_status": "token",
                    "project_status": "env_configured",
                    "static_export": {"latest": "paper_trading"},
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "research_backtests.json").write_text(
            json.dumps([{"experiment": "x", "decision": "y", "summary_path": "z"}]),
            encoding="utf-8",
        )
        (paper_dir / "SOXL.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "source_kind": "paper_trading",
                        "latest_public_review": {
                            "entry_status": "filled",
                            "account_value": 12345.67,
                        },
                    },
                    "rows": [
                        {
                            "datetime": "2026-05-22",
                            "strategy_nav": 12345.67,
                            "buyhold_nav": 12345.67,
                            "price": 190.56,
                            "position": 0.35,
                            "action": "BUY",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        errors, _info = check_static_export(root)

    assert any("sensitive paper_trading field" in error for error in errors)
    assert any("paper_trading rows must use normalized nav" in error for error in errors)


def test_vercel_deploy_command_is_non_interactive_and_token_safe():
    from scripts.deploy_vercel_dashboard import build_deploy_command, sanitize_command

    command = build_deploy_command(Path("/repo"), token="secret-token", prod=True)

    assert command[:3] == ["vercel", "deploy", "/repo"]
    assert "--yes" in command
    assert "--prod" in command
    assert "--no-color" in command
    assert "--token" in command
    assert "secret-token" in command
    assert "secret-token" not in " ".join(sanitize_command(command))
    assert "<redacted>" in sanitize_command(command)


def test_vercel_auth_status_blocks_without_token_or_login():
    from scripts.deploy_vercel_dashboard import auth_status

    status = auth_status(
        env={},
        whoami_returncode=1,
        whoami_stdout="",
        whoami_stderr="No credentials \u256d\n  Update available   \n   ",
    )

    assert status["status"] == "blocked_no_credentials"
    assert "VERCEL_TOKEN" in status["detail"]
    assert "\u256d" not in status["detail"]
    assert all(not line.endswith(" ") for line in status["detail"].splitlines())


def test_vercel_project_link_status_reports_missing_link_without_ids():
    from scripts.deploy_vercel_dashboard import project_link_status

    with TemporaryDirectory() as tmpdir:
        status = project_link_status(Path(tmpdir), env={})

    assert status["status"] == "missing_project_link"
    assert "vercel link --yes --team" in status["detail"]


def test_vercel_project_link_status_accepts_local_project_json():
    from scripts.deploy_vercel_dashboard import project_link_status

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        vercel_dir = root / ".vercel"
        vercel_dir.mkdir()
        (vercel_dir / "project.json").write_text(
            json.dumps({"projectId": "project_secretish", "orgId": "team_secretish"}),
            encoding="utf-8",
        )

        status = project_link_status(root, env={})

    assert status["status"] == "linked"
    assert "project_secretish" not in status["detail"]
    assert "team_secretish" not in status["detail"]


def test_vercel_project_link_status_accepts_project_env_ids():
    from scripts.deploy_vercel_dashboard import project_link_status

    with TemporaryDirectory() as tmpdir:
        status = project_link_status(
            Path(tmpdir),
            env={"VERCEL_PROJECT_ID": "project_secretish", "VERCEL_ORG_ID": "team_secretish"},
        )

    assert status["status"] == "env_configured"
    assert "project_secretish" not in status["detail"]
    assert "team_secretish" not in status["detail"]


def test_vercel_extracts_deployment_url_from_cli_output():
    from scripts.deploy_vercel_dashboard import extract_deployment_url

    output = "Queued\nhttps://soxl-dashboard-example.vercel.app\nInspect: https://vercel.com/example"

    assert extract_deployment_url(output) == "https://soxl-dashboard-example.vercel.app"


def test_vercel_check_reads_launch_env_file_for_readiness_without_leaking():
    from scripts import deploy_vercel_dashboard

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "repo"
        root.mkdir()
        output_dir = Path(tmpdir) / "evidence"
        env_file = Path(tmpdir) / "soxl.env"
        env_file.write_text(
            "\n".join(
                [
                    "export VERCEL_TOKEN='token_secretish'",
                    "export VERCEL_PROJECT_ID='project_secretish'",
                    "export VERCEL_ORG_ID='team_secretish'",
                ]
            ),
            encoding="utf-8",
        )

        original_check_static_export = deploy_vercel_dashboard.check_static_export
        original_run_whoami = deploy_vercel_dashboard._run_whoami
        original_argv = list(deploy_vercel_dashboard.sys.argv)
        try:
            deploy_vercel_dashboard.check_static_export = lambda _root: (
                [],
                {
                    "latest": "paper_trading",
                    "paper_trading_end": "2026-05-20",
                },
            )
            deploy_vercel_dashboard._run_whoami = lambda: subprocess.CompletedProcess(
                ["vercel", "whoami"], 1, "", "No credentials"
            )
            deploy_vercel_dashboard.sys.argv = [
                "deploy_vercel_dashboard.py",
                "--root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--env-file",
                str(env_file),
            ]

            returncode = deploy_vercel_dashboard.main()
        finally:
            deploy_vercel_dashboard.check_static_export = original_check_static_export
            deploy_vercel_dashboard._run_whoami = original_run_whoami
            deploy_vercel_dashboard.sys.argv = original_argv

        evidence_text = (output_dir / "vercel_deployment_check.json").read_text(encoding="utf-8")
        notes_text = (output_dir / "notes.md").read_text(encoding="utf-8")
        payload = json.loads(evidence_text)

    assert returncode == 0
    assert payload["status"] == "ready_for_deploy"
    assert payload["auth"]["status"] == "token"
    assert payload["project"]["status"] == "env_configured"
    assert "- env_file: `present`" in notes_text
    assert "- env_file_applied_keys: `VERCEL_ORG_ID,VERCEL_PROJECT_ID,VERCEL_TOKEN`" in notes_text
    assert "token_secretish" not in evidence_text
    assert "project_secretish" not in evidence_text
    assert "team_secretish" not in evidence_text
    assert "token_secretish" not in notes_text
    assert "project_secretish" not in notes_text
    assert "team_secretish" not in notes_text


def test_vercel_check_exports_dashboard_production_status_without_leaking():
    from scripts import deploy_vercel_dashboard

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "repo"
        root.mkdir()
        output_dir = Path(tmpdir) / "evidence"
        env_file = Path(tmpdir) / "soxl.env"
        env_file.write_text(
            "\n".join(
                [
                    "export VERCEL_TOKEN='token_secretish'",
                    "export VERCEL_PROJECT_ID='project_secretish'",
                    "export VERCEL_ORG_ID='team_secretish'",
                ]
            ),
            encoding="utf-8",
        )

        original_check_static_export = deploy_vercel_dashboard.check_static_export
        original_run_whoami = deploy_vercel_dashboard._run_whoami
        original_argv = list(deploy_vercel_dashboard.sys.argv)
        try:
            deploy_vercel_dashboard.check_static_export = lambda _root: (
                [],
                {
                    "generated_at": "2026-05-21 12:01:45",
                    "versions": 35,
                    "missing": 0,
                    "model_versions": 94,
                    "latest": "paper_trading",
                    "paper_trading_rows": 4,
                    "paper_trading_end": "2026-05-20",
                },
            )
            deploy_vercel_dashboard._run_whoami = lambda: subprocess.CompletedProcess(
                ["vercel", "whoami"], 1, "", "No credentials"
            )
            deploy_vercel_dashboard.sys.argv = [
                "deploy_vercel_dashboard.py",
                "--root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--env-file",
                str(env_file),
            ]

            returncode = deploy_vercel_dashboard.main()
        finally:
            deploy_vercel_dashboard.check_static_export = original_check_static_export
            deploy_vercel_dashboard._run_whoami = original_run_whoami
            deploy_vercel_dashboard.sys.argv = original_argv

        status_text = (root / "dashboard_data" / "production_status.json").read_text(encoding="utf-8")
        status = json.loads(status_text)

    assert returncode == 0
    assert status["status"] == "ready_for_deploy"
    assert status["auth_status"] == "token"
    assert status["project_status"] == "env_configured"
    assert status["deployment_url"] is None
    assert status["static_export"]["latest"] == "paper_trading"
    assert status["static_export"]["paper_trading_end"] == "2026-05-20"
    assert status["env_file"]["present"] is True
    assert status["env_file"]["applied_keys"] == ["VERCEL_ORG_ID", "VERCEL_PROJECT_ID", "VERCEL_TOKEN"]
    assert "token_secretish" not in status_text
    assert "project_secretish" not in status_text
    assert "team_secretish" not in status_text


def test_vercel_env_file_ignores_placeholder_values():
    from scripts.deploy_vercel_dashboard import read_env_file

    with TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / "soxl.env"
        env_file.write_text(
            "\n".join(
                [
                    "export VERCEL_TOKEN='REPLACE_ME'",
                    "export VERCEL_PROJECT_ID='replace-me'",
                    "export VERCEL_ORG_ID=''",
                ]
            ),
            encoding="utf-8",
        )

        values = read_env_file(env_file)

    assert values == {}
