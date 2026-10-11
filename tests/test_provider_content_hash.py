"""Reproducibility contract for the frozen provider content hash (read-only).

Any reviewer can recompute the same hash with the same pure function; failures here
block fit authorization. The test never writes to a provider.
"""
import json
from pathlib import Path

import pytest

from src.data import provider_content_hash as PH

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "h20_matched_forecast_execution_20261010"
PLAN = CARD / "preregister_plan.json"
SIDECAR = CARD / "provider_content_hash_scope.json"
STORES = {
    "published_futu_affine_20260925_v1": "data/qlib/published_futu_affine_20260925_v1",
    "legacy_soxl_execution_20260925_v1": "data/qlib/legacy_soxl_execution_20260925_v1",
}


def test_serialization_is_deterministic_and_documented():
    records = [{"path": "b/x.bin", "sha256": "aa"}, {"path": "a/y.bin", "sha256": "bb"}]
    once = PH.serialize_inventory(records)
    twice = PH.serialize_inventory(list(reversed(records)))
    assert once == twice  # sorted by path
    assert once == b"a/y.bin\x00bb\nb/x.bin\x00aa\n"  # exact bytes: path + 0x00 + sha + 0x0a
    assert PH.SEPARATOR_BYTES == b"\x00" and PH.TERMINATOR_BYTES == b"\n"


@pytest.mark.skipif(not Path(STORES["published_futu_affine_20260925_v1"]).exists(),
                    reason="provider store not present")
def test_provider_hashes_match_plan_and_sidecar():
    plan = json.loads(PLAN.read_text())
    sidecar = json.loads(SIDECAR.read_text())
    frozen = plan["source_binding"]["provider_content_sha256"]
    assert plan["source_binding"]["provider_content_hash_sidecar"] == "provider_content_hash_scope.json"
    for name, rel in STORES.items():
        digest, records = PH.provider_content_hash(ROOT / rel)
        assert digest == frozen[name]
        assert sidecar["stores"][name]["content_sha256"] == digest
        assert sidecar["stores"][name]["file_count"] == len(records)
        assert sidecar["stores"][name]["inventory"] == records
    assert sidecar["serialization"] == PH.SERIALIZATION
    assert set(sidecar["stores"]) == set(STORES)


@pytest.mark.skipif(not Path(STORES["published_futu_affine_20260925_v1"]).exists(),
                    reason="provider store not present")
def test_inventory_scope_and_counts():
    for name, rel in STORES.items():
        records = PH.provider_inventory(ROOT / rel)
        paths = [r["path"] for r in records]
        assert paths == sorted(paths, key=lambda p: p.encode("utf-8"))
        assert all(not p.endswith("qlib.yaml") for p in paths)
        assert all("cache" not in p.split("/") for p in paths)
    assert len(PH.provider_inventory(ROOT / STORES["published_futu_affine_20260925_v1"])) == 13
    assert len(PH.provider_inventory(ROOT / STORES["legacy_soxl_execution_20260925_v1"])) == 14
