"""Hash-bound NAS market snapshots; local copies are disposable consumers."""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("blocked: unsafe NAS snapshot member")
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssh(ssh: dict, command: str, *, input: str | None = None) -> str:
    result = subprocess.run(["ssh", "-p", str(ssh["port"]), "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=5", ssh["host"], command],
                            input=input, text=True, capture_output=True, timeout=300)
    if result.returncode:
        raise ValueError("needs_data: NAS SSH operation failed")
    return result.stdout


def read_literal_env(path: Path) -> dict[str, str]:
    result = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if separator:
            pieces = shlex.split(value, comments=True)
            if len(pieces) > 1:
                raise ValueError("blocked: nonliteral operator configuration")
            result[key] = pieces[0] if pieces else ""
    return result


def nas_api_call(*, origin: str, path: str, token: str, method: str = "GET",
                 payload: dict | None = None, upload: Path | None = None,
                 executor_id: str = "soxl-minute-data") -> dict:
    """Use native curl for macOS LAN access; credentials stay on stdin."""
    if not token or any(c in token for c in '\r\n"\\'):
        raise ValueError("needs_auth: valid NAS registry token required")
    config = 'header = "Authorization: Bearer ' + token + '"\n'
    command = ["curl", "--noproxy", "*", "--config", "-", "--connect-timeout", "5",
               "--max-time", "300", "--fail-with-body", "-sS", "-X", method,
               "-H", "Content-Type: application/json" if upload is None else "Content-Type: application/gzip",
               "-H", "X-Executor-Id: " + executor_id]
    if upload is not None:
        command += ["--data-binary", "@" + str(Path(upload).resolve())]
    elif payload is not None:
        # Payload is nonsensitive metadata; JSON remains a structured curl arg.
        command += ["--data-binary", json.dumps(payload, ensure_ascii=False)]
    command += [origin.rstrip("/") + path]
    response = subprocess.run(command, input=config, text=True, capture_output=True)
    if response.returncode:
        raise RuntimeError(f"needs_data: NAS API {method} {path} failed (curl {response.returncode})")
    return json.loads(response.stdout)


def _settings(root: Path) -> dict:
    config = json.loads((root / "data/nas_sources.json").read_text())
    release = read_literal_env(Path(config["release_env"]))
    auth = read_literal_env(Path(config["registry_env"]))
    return {**config, "origin": release["YANXUAN_NAS_API_ORIGIN"],
            "ssh": {"host": release["YANXUAN_NAS_HOST"],
                    "port": int(release["YANXUAN_NAS_SSH_PORT"])},
            "token": auth["RESEARCH_REGISTRY_TOKEN"]}


def _latest_verified_snapshot(settings: dict) -> dict:
    # sync.current is also refreshed by the daily updater. Its source index is
    # reconstructible from immutable NAS snapshots, never from local caches.
    source_manifest_path = settings.get(
        "source_manifest_path", "data/raw/external/semis_minute_refresh/source_manifest.json"
    )
    script = """import json,hashlib
from pathlib import Path
parent=Path(PARENT)
expected_source_manifest=SOURCE_MANIFEST_PATH
candidates=[]
for path in parent.glob('*/nas_binding.json'):
 data=json.loads(path.read_text())
 if data.get('verified') is not True or data.get('authority')!='yanxuan_nas':continue
 if data['snapshot_root']!=str(path.parent):continue
 if data.get('source_manifest_path')!=expected_source_manifest:continue
 source=path.parent/expected_source_manifest
 if hashlib.sha256(source.read_bytes()).hexdigest()!=data['source_id']:continue
 candidates.append(data)
print(json.dumps(max(candidates,key=lambda item:(item['as_of'],item['verified_at'])) if candidates else {}))
""".replace("PARENT", repr(settings["snapshot_parent"])).replace(
        "SOURCE_MANIFEST_PATH", repr(source_manifest_path)
    )
    binding = json.loads(_ssh(settings["ssh"], "python3 -", input=script))
    if not binding:
        raise ValueError("needs_data: no verified NAS minute snapshot")
    return binding


def load_current_nas_binding(*, workspace_root: Path) -> dict:
    """Resolve the current input from NAS DuckDB, not a local mutable index."""
    settings = _settings(Path(workspace_root))
    state = nas_api_call(origin=settings["origin"], token=settings["token"],
                         path="/api/research/control/state/" + settings["state_key"])
    source = next((item for item in state.get("source_updates", [])
                   if item["name"] == settings["source_name"]), None)
    if source is None:
        binding = _latest_verified_snapshot(settings)
        metadata = binding.get("metadata", {})
        register_nas_source(binding, workspace_root=workspace_root,
            observation={**metadata, "quality_status": metadata.get("quality_status", "needs_data"),
                         "index_reconstructed_from_verified_nas_snapshots": True},
            error_count=int(metadata.get("malformed_rows", 1)))
        binding["ssh"] = settings["ssh"]
        return binding
    observation = source["observation"]
    encoded = _ssh(settings["ssh"], "cat " + shlex.quote(observation["binding_path"]))
    if hashlib.sha256(encoded.encode()).hexdigest() != observation["binding_sha256"]:
        raise ValueError("blocked: NAS binding hash differs from registry")
    binding = json.loads(encoded)
    if binding["source_id"] != observation["source_id"]:
        raise ValueError("blocked: NAS source identity differs from registry")
    # DHCP settings are operational configuration, not immutable data identity.
    binding["ssh"] = settings["ssh"]
    return binding


def register_nas_source(binding: dict, *, workspace_root: Path,
                        observation: dict, error_count: int = 0) -> dict:
    settings = _settings(Path(workspace_root))
    encoded = _ssh(settings["ssh"], "cat " + shlex.quote(binding["snapshot_root"] + "/nas_binding.json"))
    persisted = json.loads(encoded)
    if persisted.get("verified") is not True or persisted["source_id"] != binding["source_id"]:
        raise ValueError("blocked: NAS snapshot has not been verified")
    source = {"name": settings["source_name"], "required": False,
              "status": "warning" if error_count else "ok", "error_count": error_count,
              "observation": {**observation, "data_owner": "yanxuan_nas",
                  "source_id": binding["source_id"], "as_of": binding["as_of"],
                  "binding_path": binding["snapshot_root"] + "/nas_binding.json",
                  "binding_sha256": hashlib.sha256(encoded.encode()).hexdigest()}}
    path = "/api/research/control/state/" + settings["state_key"]
    # The existing API accepts a full state. Read immediately before merging;
    # preserve every unrelated source entry and every other control field.
    state = nas_api_call(origin=settings["origin"], path=path, token=settings["token"])
    state.pop("_state", None)
    entries = state.setdefault("source_updates", [])
    state["source_updates"] = [item for item in entries if item["name"] != source["name"]] + [source]
    nas_api_call(origin=settings["origin"], path=path, token=settings["token"], method="PUT",
                 payload={"payload": state, "updated_by": "soxl-minute-data"})
    readback = nas_api_call(origin=settings["origin"], path=path, token=settings["token"])
    if source not in readback.get("source_updates", []):
        raise ValueError("needs_data: NAS source registration readback differs")
    return source


def materialize_nas_snapshot(binding: Path | dict, *, workspace_root: Path,
                             include_derived: bool = False) -> dict:
    """Restore pinned source bytes from NAS; no upstream/local-source fallback."""
    manifest = binding if isinstance(binding, dict) else json.loads(Path(binding).read_text())
    root = Path(workspace_root).resolve()
    if manifest.get("authority") != "yanxuan_nas" or manifest.get("verified") is not True:
        raise ValueError("blocked: verified NAS source binding required")
    ssh = manifest["ssh"]
    imported = 0
    for entry in manifest["objects"]:
        if entry.get("kind") == "derived" and not include_derived:
            continue
        relative = _safe_relative(entry["local_path"])
        remote = entry.get("snapshot_root", manifest["snapshot_root"]).rstrip("/") + "/" + _safe_relative(entry["snapshot_path"])
        destination = root / relative
        if not destination.resolve().is_relative_to(root):
            raise ValueError("blocked: NAS cache destination escapes workspace")
        if destination.exists() and _sha256(destination) == entry["sha256"]:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".nas-partial")
        command = ["ssh", "-p", str(ssh["port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   ssh["host"], "cat " + shlex.quote(remote)]
        with temporary.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE)
        if result.returncode or temporary.stat().st_size != entry["bytes"] or _sha256(temporary) != entry["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError("blocked: NAS source unavailable or content hash differs")
        temporary.replace(destination)
        imported += 1
    return {"authority": "yanxuan_nas", "source_id": manifest["source_id"], "restored_objects": imported}


def publish_nas_snapshot(*, workspace_root: Path, source_manifest: Path,
                         files: dict[str, str], as_of: str,
                         inherited_objects: list[dict] | None = None,
                         source_metadata: dict | None = None) -> dict:
    """Publish an immutable verified snapshot; Git is never a data transport."""
    root = Path(workspace_root).resolve()
    settings = _settings(root)
    source_id = _sha256(source_manifest)
    binding = {"schema_version": "nas_market_snapshot_v1", "authority": "yanxuan_nas",
               "source_id": source_id, "as_of": as_of, "ssh": settings["ssh"],
               "snapshot_root": settings["snapshot_parent"].rstrip("/") + "/" + source_id,
               "source_manifest_path": str(source_manifest.relative_to(root)), "verified": False,
               "metadata": source_metadata or {},
               "objects": []}
    for relative, kind in files.items():
        relative = _safe_relative(relative)
        path = root / relative
        binding["objects"].append({"local_path": relative, "snapshot_path": relative,
                                   "kind": kind, "sha256": _sha256(path), "bytes": path.stat().st_size})
    binding["objects"].extend(inherited_objects or [])
    stage = settings["snapshot_parent"].rstrip("/") + "/.incoming-" + uuid.uuid4().hex
    _ssh(settings["ssh"], "mkdir -p " + shlex.quote(stage))
    with tempfile.TemporaryDirectory(prefix="soxl-nas-") as temporary:
        archive = Path(temporary) / "snapshot.tar"
        with tarfile.open(archive, "w") as output:
            for relative in files:
                output.add(root / _safe_relative(relative), arcname=relative, recursive=False)
        command = ["ssh", "-p", str(settings["ssh"]["port"]), "-o", "BatchMode=yes",
                   "-o", "ConnectTimeout=5", settings["ssh"]["host"], "tar -xf - -C " + shlex.quote(stage)]
        with archive.open("rb") as stream:
            result = subprocess.run(command, stdin=stream, capture_output=True, timeout=300)
        if result.returncode:
            raise ValueError("needs_data: NAS snapshot transfer failed")
    script = """import json,hashlib,os,shutil,datetime
from pathlib import Path
binding=BOUND
stage=Path(STAGE)
final=Path(binding['snapshot_root'])
base=final if final.exists() else stage
for obj in binding['objects']:
 p=Path(obj.get('snapshot_root',str(base)))/obj['snapshot_path']
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 if p.stat().st_size!=obj['bytes'] or h.hexdigest()!=obj['sha256']:raise ValueError('NAS object hash differs')
binding['verified']=True
binding['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
if final.exists():
 saved=json.loads((final/'nas_binding.json').read_text())
 if saved['source_id']!=binding['source_id'] or saved['objects']!=binding['objects']:raise ValueError('NAS snapshot identity collision')
 binding=saved
 shutil.rmtree(stage)
else:
 (stage/'nas_binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\\n')
 os.rename(stage,final)
print(json.dumps(binding))
""".replace("BOUND", repr(binding)).replace("STAGE", repr(stage))
    return json.loads(_ssh(settings["ssh"], "python3 -", input=script))


def materialize_nas_evidence(binding: Path, *, card: Path) -> dict:
    """Restore the original MLflow file Recorder from its NAS artifact."""
    manifest = json.loads(Path(binding).read_text())
    evidence = manifest["evidence"]
    with tempfile.TemporaryDirectory(prefix="soxl-recorder-") as temporary:
        archive = Path(temporary) / "evidence.tar.gz"
        ssh = manifest["ssh"]
        with archive.open("wb") as output:
            result = subprocess.run(["ssh", "-p", str(ssh["port"]), "-o", "BatchMode=yes",
                ssh["host"], "cat " + shlex.quote(evidence["nas_path"])], stdout=output, stderr=subprocess.PIPE)
        if result.returncode or _sha256(archive) != evidence["sha256"]:
            raise ValueError("blocked: NAS Recorder artifact hash differs")
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                if _safe_relative(member.name).split("/", 1)[0] != "qlib_recorder" or not (member.isdir() or member.isfile()):
                    raise ValueError("blocked: unsafe NAS Recorder archive member")
            source.extractall(card, filter="data")
    # File-store metadata may retain the original absolute artifact URI.
    # Rebase only disposable metadata; recorded object bytes and IDs stay exact.
    import yaml

    tracking = (Path(card) / "qlib_recorder").resolve()
    experiment = tracking / evidence["experiment_id"]
    for path, key, destination in [
        (experiment / "meta.yaml", "artifact_location", experiment),
        (experiment / evidence["recorder_id"] / "meta.yaml", "artifact_uri",
         experiment / evidence["recorder_id"] / "artifacts"),
    ]:
        metadata = yaml.safe_load(path.read_text())
        metadata[key] = destination.as_uri()
        path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return {"recorder_id": evidence["recorder_id"], "archive_sha256": evidence["sha256"]}


def archive_nas_recorder(binding: Path, *, workspace_root: Path, card: Path,
                         recorder_id: str, experiment_id: str) -> dict:
    """Archive a completed native Recorder through the existing NAS API."""
    settings = _settings(Path(workspace_root))
    experiment = Path(card) / "qlib_recorder" / experiment_id
    with tempfile.TemporaryDirectory(prefix="soxl-recorder-") as temporary:
        archive = Path(temporary) / "evidence.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(experiment / "meta.yaml", arcname="qlib_recorder/" + experiment_id + "/meta.yaml")
            output.add(experiment / recorder_id, arcname="qlib_recorder/" + experiment_id + "/" + recorder_id)
        digest = _sha256(archive)
        receipt = nas_api_call(origin=settings["origin"], token=settings["token"], method="PUT",
            path="/api/research/artifacts/" + digest, upload=archive,
            executor_id="soxl-minute-research")
        evidence = receipt["artifact"]
        if evidence["sha256"] != digest or evidence["size_bytes"] != archive.stat().st_size:
            raise ValueError("blocked: NAS Recorder upload receipt differs")
    evidence.update(nas_path=settings["artifact_root"].rstrip("/") + "/" + digest[:2] + "/" + digest[2:4] + "/" + digest,
                    recorder_id=recorder_id, experiment_id=experiment_id, archive_root="qlib_recorder")
    manifest = json.loads(Path(binding).read_text())
    manifest["evidence"] = evidence
    Path(binding).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return evidence
