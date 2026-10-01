import json
from contextlib import contextmanager

import pytest

from src.workflow import stage1_regime_release as release


def _source(root, **changes):
    source = {
        "promotion": {"native_reconciliation_pass": True},
        "native_qlib_audit": {"status": "passed", "maximum_session_return_error": 1e-10, "tolerance": 1e-7},
        "baseline_reproduction_pass": True,
        "source_sha256": {},
        "selected": {"historical_all_pass": False},
    }
    source.update(changes)
    card = root / "research" / release.RESEARCH
    card.mkdir(parents=True)
    (card / "summary.json").write_text(json.dumps(source))
    return source


def _forbid_qlib_start(monkeypatch):
    import qlib

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid release reached Qlib initialization or recorder loading")

    monkeypatch.setattr(qlib, "init", forbidden)


def test_release_cannot_bypass_unverified_native_execution(monkeypatch, tmp_path):
    _source(tmp_path, promotion={"native_reconciliation_pass": False}, selected={"historical_all_pass": True})
    _forbid_qlib_start(monkeypatch)
    with pytest.raises(ValueError, match="native"):
        release.seal_paper_release(root=tmp_path)
    assert not (tmp_path / "production").exists()


@pytest.mark.parametrize("audit", [
    {"status": "failed", "maximum_session_return_error": 0., "tolerance": 1e-7},
    {"status": "passed", "maximum_session_return_error": 1e-3, "tolerance": 1e-7},
    {"status": "passed", "maximum_session_return_error": float("nan"), "tolerance": 1e-7},
])
def test_release_rejects_native_flag_that_disagrees_with_evidence(monkeypatch, tmp_path, audit):
    _source(tmp_path, native_qlib_audit=audit)
    _forbid_qlib_start(monkeypatch)
    with pytest.raises(ValueError, match="native"):
        release.seal_paper_release(root=tmp_path)


def test_release_requires_reproduced_baseline_even_for_observation(monkeypatch, tmp_path):
    _source(tmp_path, baseline_reproduction_pass=False)
    _forbid_qlib_start(monkeypatch)
    with pytest.raises(ValueError, match="baseline"):
        release.seal_paper_release(root=tmp_path)


def test_changed_research_implementation_blocks_before_loading_models(monkeypatch, tmp_path):
    path = tmp_path / "src/workflow/research.py"
    path.parent.mkdir(parents=True)
    path.write_text("original = True\n")
    digest = release.sha256(path)
    _source(tmp_path, source_sha256={"src/workflow/research.py": digest})
    path.write_text("original = False\n")
    _forbid_qlib_start(monkeypatch)
    with pytest.raises(ValueError, match="implementation changed"):
        release.seal_paper_release(root=tmp_path)


@pytest.mark.parametrize("unsealed", [False, True])
def test_all_source_hashes_are_verified_before_any_model_deserialization(monkeypatch, tmp_path, unsealed):
    import qlib
    import qlib.workflow

    recorder_dir = tmp_path / "recorder/run"
    artifacts = recorder_dir / "artifacts"
    objects = ("models/selected_action.pkl", "models/selected_spec.pkl", "inputs/prepared_inputs.pkl", "training/expanded_model_training.pkl")
    hashes = {}
    for name in objects:
        artifact = artifacts / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"opaque immutable artifact")
        hashes[name] = release.sha256(artifact)
    if unsealed:
        del hashes["inputs/prepared_inputs.pkl"]
    else:
        (artifacts / "training/expanded_model_training.pkl").write_bytes(b"changed after research")
    _source(tmp_path, recorder_uri=str(recorder_dir.parent), recorder_local_dir=str(recorder_dir),
            recorder_id="run", experiment="research", sealed_source_objects=hashes)

    class Recorder:
        def get_local_dir(self):
            return str(recorder_dir)

        def load_object(self, *args, **kwargs):
            pytest.fail("Deserialization occurred before verifying every sealed source object")

    class Workflow:
        @contextmanager
        def uri_context(self, uri):
            yield

        def get_recorder(self, **kwargs):
            return Recorder()

    monkeypatch.setattr(qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(qlib.workflow, "R", Workflow())
    with pytest.raises(ValueError, match="unsealed or changed research object"):
        release.seal_paper_release(root=tmp_path)
