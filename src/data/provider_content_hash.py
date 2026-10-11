"""Deterministic provider content hash (read-only, reproducible).

The plan freezes a content hash of each Qlib provider store so the H20 work cannot
silently drift onto a different price/execution snapshot. This module is the single
pure implementation: any reviewer can call ``provider_content_hash(store)`` and get the
same hash and file inventory.

Serialization (exact, versioned)
--------------------------------
- File selection: every regular file under ``calendars/``, ``instruments/`` and
  ``features/`` (recursive), plus ``qlib_store_manifest.json`` at the store root.
- Exclusions: file name ``qlib.yaml``; any path component in ``{"cache", "__pycache__",
  ".cache"}``.
- Inventory: one record ``{"path": <posix relative path>, "sha256": <64 hex>}`` per file,
  sorted ascending by the UTF-8 byte order of ``path``.
- Digest input: for each record in sorted order, append
  ``path.encode("utf-8")`` + ``b"\\x00"`` + ``sha256.encode("ascii")`` + ``b"\\n"``
  (no JSON, no separators beyond those bytes, always a trailing newline).
- Result: ``hashlib.sha256(concat).hexdigest()``.

Files are never written; the hash is computed from existing provider bytes only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "provider_content_hash_v1"
SCAN_KINDS = ("calendars", "instruments", "features")
EXCLUDE_NAMES = {"qlib.yaml"}
EXCLUDE_DIRS = {"cache", "__pycache__", ".cache"}
ROOT_MANIFEST = "qlib_store_manifest.json"
SERIALIZATION = "sorted(path)[path.utf8 + 0x00 + sha256.ascii + 0x0a]"
SEPARATOR_BYTES = b"\x00"
TERMINATOR_BYTES = b"\n"
ENCODING = "utf-8/ascii"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provider_inventory(store: Path) -> list[dict]:
    """Sorted inventory of the included files (read-only)."""
    store = Path(store)
    if not store.is_dir():
        raise ValueError(f"blocked: provider store not found: {store}")
    records = []
    for kind in SCAN_KINDS:
        base = store / kind
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.name in EXCLUDE_NAMES or any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            records.append({"path": path.relative_to(store).as_posix(), "sha256": _sha256(path)})
    manifest = store / ROOT_MANIFEST
    if manifest.exists():
        records.append({"path": ROOT_MANIFEST, "sha256": _sha256(manifest)})
    records.sort(key=lambda record: record["path"].encode("utf-8"))
    return records


def serialize_inventory(records: list[dict]) -> bytes:
    payload = bytearray()
    for record in sorted(records, key=lambda item: item["path"].encode("utf-8")):
        payload += record["path"].encode("utf-8")
        payload += SEPARATOR_BYTES
        payload += record["sha256"].encode("ascii")
        payload += TERMINATOR_BYTES
    return bytes(payload)


def provider_content_hash(store: Path) -> tuple[str, list[dict]]:
    """Return ``(content_hash, inventory)`` for a provider store."""
    records = provider_inventory(store)
    digest = hashlib.sha256(serialize_inventory(records)).hexdigest()
    return digest, records


def scope_document(store: Path) -> dict:
    digest, records = provider_content_hash(store)
    return {
        "schema_version": SCHEMA_VERSION,
        "serialization": SERIALIZATION,
        "separator_bytes": "0x00",
        "terminator_bytes": "0x0a",
        "encoding": ENCODING,
        "scan_kinds": list(SCAN_KINDS),
        "exclude_names": sorted(EXCLUDE_NAMES),
        "exclude_dirs": sorted(EXCLUDE_DIRS),
        "root_manifest": ROOT_MANIFEST,
        "file_count": len(records),
        "content_sha256": digest,
        "inventory": records,
    }


def write_scope_sidecar(store: Path, destination: Path) -> dict:
    document = scope_document(store)
    Path(destination).write_text(json.dumps(document, indent=2, ensure_ascii=True, sort_keys=False))
    return document
