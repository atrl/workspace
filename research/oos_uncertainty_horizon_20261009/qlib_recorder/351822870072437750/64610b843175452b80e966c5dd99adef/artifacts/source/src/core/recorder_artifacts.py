"""Lossless compressed artifacts owned by a Qlib Recorder."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import pickle
import tempfile


def save_compressed_object(recorder, *, name, artifact_path, value):
    if Path(name).name != name or not name.endswith(".pkl.gz"):
        raise ValueError("blocked: compressed artifact requires a simple .pkl.gz name")
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    packed = gzip.compress(raw, mtime=0)
    with tempfile.TemporaryDirectory(prefix="qlib-evidence-") as directory:
        path = Path(directory) / name
        path.write_bytes(packed)
        recorder.log_artifact(str(path), artifact_path=artifact_path)
    return {"artifact": str(Path(artifact_path) / name), "encoding": "gzip(pickle)",
        "sha256": hashlib.sha256(packed).hexdigest(), "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(packed), "uncompressed_bytes": len(raw)}


def load_compressed_object(recorder, identity):
    """Load trusted project evidence only after checking its exact byte identity."""
    with tempfile.TemporaryDirectory(prefix="qlib-evidence-read-") as directory:
        downloaded = recorder.download_artifact(identity["artifact"], dst_path=directory)
        packed = Path(downloaded).read_bytes()
    if hashlib.sha256(packed).hexdigest() != identity["sha256"]:
        raise ValueError("blocked: compressed Recorder artifact identity differs")
    raw = gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != identity["uncompressed_sha256"]:
        raise ValueError("blocked: decoded Recorder artifact identity differs")
    return pickle.loads(raw)
