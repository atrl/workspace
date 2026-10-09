import gzip
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from src.core import recorder_artifacts as artifacts
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    import qlib
    from qlib.workflow import R

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    provider = (tmp_path / "provider").resolve()
    raw = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                        "instrument": ["SOXX", "SOXX"], "open": [100., 101.], "close": [100., 101.]})
    build_qlib_store_from_frame(raw, provider_uri=provider, as_of="2024-01-03")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    # Qlib's file-URI experiment lock strips the leading slash, creating a
    # repo-relative private/.../filelock on macOS. A resolved local tracking
    # path uses MLflow's local store without that faulty URI lock branch.
    with R.uri_context(str((tmp_path / "recorder").resolve())):
        with R.start(experiment_name="compressed_artifact_contract"):
            yield R.get_recorder()


def _value():
    return {"frame": pd.DataFrame({"return": [np.nan, .0123456789012345, -.03],
                                  "available_at": pd.date_range("2024-01-02", periods=3, tz="UTC")}),
            "matrix": np.arange(24, dtype=np.float64).reshape(3, 2, 4),
            "metadata": {"experiment": "synthetic", "not_a_local_sidecar": True}}


def test_real_qlib_recorder_roundtrip_download_signature_hash_and_lossless_types(recorder, tmp_path):
    value = _value()
    identity = artifacts.save_compressed_object(recorder, name="evidence.pkl.gz", artifact_path="inputs/frozen", value=value)
    assert identity["artifact"] == "inputs/frozen/evidence.pkl.gz"
    assert identity["artifact"] in recorder.list_artifacts("inputs/frozen")
    destination = (tmp_path / "download").resolve()
    destination.mkdir()
    # Official API: download_artifact(path: str, dst_path: str | None) -> str.
    path = recorder.download_artifact(identity["artifact"], dst_path=str(destination))
    packed = Path(path).read_bytes()
    raw = gzip.decompress(packed)
    assert identity["sha256"] == hashlib.sha256(packed).hexdigest()
    assert identity["uncompressed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert identity["bytes"] == len(packed)
    assert identity["uncompressed_bytes"] == len(raw)
    assert identity["encoding"] == "gzip(pickle)"
    loaded = artifacts.load_compressed_object(recorder, identity)
    pd.testing.assert_frame_equal(loaded["frame"], value["frame"], check_exact=True)
    np.testing.assert_array_equal(loaded["matrix"], value["matrix"])
    assert loaded["metadata"] == value["metadata"]


def test_compression_identity_is_deterministic_and_local_staging_is_removed(recorder, monkeypatch):
    logged = []
    log_artifact = recorder.log_artifact

    def capture(local_path, artifact_path=None):
        assert Path(local_path).is_file()
        logged.append(Path(local_path))
        return log_artifact(local_path, artifact_path=artifact_path)

    monkeypatch.setattr(recorder, "log_artifact", capture)
    value = _value()
    first = artifacts.save_compressed_object(recorder, name="first.pkl.gz", artifact_path="data", value=value)
    second = artifacts.save_compressed_object(recorder, name="second.pkl.gz", artifact_path="data", value=value)
    assert first["sha256"] == second["sha256"]
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    assert all(not path.exists() for path in logged)


@pytest.mark.parametrize("bad_field", ["sha256", "uncompressed_sha256"])
def test_either_digest_mismatch_blocks_before_unpickling(recorder, monkeypatch, bad_field):
    identity = artifacts.save_compressed_object(recorder, name="frozen.pkl.gz", artifact_path="data", value=_value())
    identity[bad_field] = "0" * 64
    calls = []

    def forbidden_unpickle(raw):
        calls.append(True)
        raise AssertionError("unpickle must not run before identity checks")

    monkeypatch.setattr(artifacts.pickle, "loads", forbidden_unpickle)
    with pytest.raises(ValueError, match="Recorder artifact identity differs"):
        artifacts.load_compressed_object(recorder, identity)
    assert calls == []


def test_changed_recorder_bytes_are_rejected_using_original_identity(recorder, tmp_path):
    identity = artifacts.save_compressed_object(recorder, name="frozen.pkl.gz", artifact_path="data", value=_value())
    replaced = (tmp_path / "frozen.pkl.gz").resolve()
    replaced.write_bytes(gzip.compress(pickle.dumps({"changed": True}), mtime=0))
    recorder.log_artifact(str(replaced), artifact_path="data")
    with pytest.raises(ValueError, match="compressed Recorder artifact identity differs"):
        artifacts.load_compressed_object(recorder, identity)


@pytest.mark.parametrize("name", ["../escape.pkl.gz", "nested/file.pkl.gz", "/tmp/file.pkl.gz", "bare.pkl", ""])
def test_invalid_artifact_names_are_rejected_before_recorder_access(name):
    with pytest.raises(ValueError, match="blocked:"):
        artifacts.save_compressed_object(None, name=name, artifact_path="data", value={})
