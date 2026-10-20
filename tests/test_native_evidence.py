import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.core.native_evidence import finalize_evidence, native_dataset, prepare_groups
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.qlib_native import build_qlib_store_from_frame


def _frame():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["SOXL", "SOXX"]],
        names=["datetime", "instrument"],
    )
    columns = pd.MultiIndex.from_tuples(
        [("feature", "x"), ("label", "y"), ("feature", "z")]
    )
    return pd.DataFrame([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
                        index=index, columns=columns)


def test_native_dataset_raw_roundtrip_keeps_group_order_and_typed_empty():
    source = _frame()
    dataset = native_dataset(source, segments={
        "all": (None, None), "empty": ("2030-01-01", "2030-01-02")})
    restored = pickle.loads(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(prepare_groups(restored, "all"), source)
    empty = prepare_groups(restored, "empty")
    assert empty.empty
    assert empty.columns.equals(source.columns)
    assert empty.dtypes.tolist() == source.dtypes.tolist()


def test_finalizer_inventories_and_verified_loader_reads_native_recorder(tmp_path, monkeypatch):
    import qlib
    from qlib.workflow import R

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    card = tmp_path / "card"
    card.mkdir()
    provider = tmp_path / "provider"
    raw = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "instrument": ["SOXX", "SOXX"], "open": [100., 101.], "close": [100., 101.]})
    build_qlib_store_from_frame(raw, provider_uri=provider, as_of="2024-01-03")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    experiment = "native_evidence_finalizer_contract"
    recorder_uri = (card / "qlib_recorder").resolve()
    with R.uri_context(str(recorder_uri)):
        with R.start(experiment_name=experiment):
            recorder = R.get_recorder()
            recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
            recorder.save_objects(artifact_path="analysis", **{"sample.pkl": {"value": 42}})
            recorder.save_objects(artifact_path="implementation", **{"code.pkl": "source"})
            recorder.log_metrics(registered_artifact_count=2)
    binding = card / "nas_binding.json"
    binding.write_text("{}")
    calls = {}

    def fake_archive(_binding_path, **kwargs):
        calls.update(kwargs)
        return {"sha256": "archive-hash"}

    monkeypatch.setattr("src.data.nas_market_data.archive_nas_recorder", fake_archive)
    result = finalize_evidence(recorder, card=card, summary={"experiment": experiment,
        "recorder_uri": str(recorder_uri), "source_identity": {"sha256": "source-hash"},
        "transport": {"host": "nas"}}, transport_binding=binding, workspace_root=tmp_path,
        include_code_artifacts=False)
    manifest = json.loads((card / "verification_manifest.json").read_text())
    summary = json.loads((card / "summary.json").read_text())
    card_local = recorder.get_local_dir()
    artifact_root = Path(card_local) / "artifacts"
    code_path = (artifact_root / "implementation" / "code.pkl").relative_to(card).as_posix()
    sample_path = (artifact_root / "analysis" / "sample.pkl").relative_to(card).as_posix()
    all_paths = {path.relative_to(card).as_posix() for path in artifact_root.rglob("*") if path.is_file()}
    assert {entry["path"] for entry in manifest["entries"]} == all_paths
    assert {entry["path"] for entry in manifest["entries"] if entry["kind"] == "code"} == {
        path for path in all_paths if "/implementation/" in path or "/artifacts/code_" in path}
    assert code_path in manifest["code_artifact_paths"]
    assert sample_path in {entry["path"] for entry in manifest["entries"]}
    assert manifest["code_artifacts_required"] is False
    assert manifest["code_artifacts_included"] is True
    assert result["source_identity"]["sha256"] == summary["source_identity"]["sha256"] == "source-hash"
    assert result["transport"]["host"] == "nas"
    assert calls["recorder_id"] == recorder_id and calls["experiment_id"] == experiment_id
    assert result["metrics"]["registered_artifact_count"] == 2
    objects, _ = load_verified_recorder_objects(card, expected_recorder_id=recorder_id,
        artifacts=["analysis/sample.pkl"])
    assert objects["analysis/sample.pkl"] == {"value": 42}
    with pytest.raises(FileExistsError, match="sealed card"):
        finalize_evidence(recorder, card=card, summary={}, transport_binding=binding,
                          workspace_root=tmp_path)
