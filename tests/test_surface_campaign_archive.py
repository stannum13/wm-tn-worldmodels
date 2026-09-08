import json

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from scripts.run_surface_frontier_challenge import seed_value
from scripts.verify_surface_campaigns import check_counts, check_seeds, check_source


def test_archive_rejects_count_ler_disagreement():
    check_counts({"model": {"errors_per_stream": [1, 2], "ler": 3 / 8}}, 2, 4)
    with pytest.raises(ValueError, match="LER/count mismatch"):
        check_counts({"model": {"errors_per_stream": [1, 2], "ler": 2 / 8}}, 2, 4)
    with pytest.raises(ValueError, match="invalid stream"):
        check_counts({"model": {"errors_per_stream": [1, 5], "ler": 6 / 8}}, 2, 4)


def test_archive_requires_exact_seed_role_manifest():
    keys = [("heldout", "nominal", "stim", 0)]
    name = json.dumps(keys[0], separators=(",", ":"))
    manifest = {name: seed_value(42, 0, 5, *keys[0])}
    check_seeds(manifest, 42, 0, 5, keys)
    with pytest.raises(ValueError, match="namespace"):
        check_seeds(manifest, 42, 1, 5, keys)
    with pytest.raises(ValueError, match="namespace"):
        check_seeds({}, 42, 0, 5, keys)


def test_archive_detects_source_artifact_mutation(tmp_path):
    import hashlib

    path = tmp_path / "source.json"
    path.write_text("{}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    check_source(path, digest)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        check_source(path, digest)
