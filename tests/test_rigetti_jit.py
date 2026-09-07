import numpy as np
import pytest

numba = pytest.importorskip("numba")
pymatching = pytest.importorskip("pymatching")

from ptwm.rigetti import compile_frontier_decoder
from ptwm.rigetti_jit import decode_frontier_batch, decode_frontier_one, pack_frontier_schedule


def test_compiled_frontier_matches_python_and_pymatching():
    matching = pymatching.Matching()
    matching.add_edge(0, 1, fault_ids={0}, weight=1.13)
    matching.add_edge(1, 2, weight=2.07)
    matching.add_boundary_edge(0, weight=3.19)
    matching.add_boundary_edge(2, fault_ids={0}, weight=0.71)
    plan = compile_frontier_decoder(
        matching, {node: [0.0, 0.0, float(node)] for node in range(3)}
    )
    schedule = pack_frontier_schedule(plan)
    weights = np.asarray([attributes["weight"] for _, _, attributes in matching.edges()])
    syndromes = np.asarray([
        [(value >> bit) & 1 for bit in range(3)] for value in range(8)
    ], dtype=np.uint8)
    compiled, margins = decode_frontier_batch(
        syndromes, np.broadcast_to(weights, (len(syndromes), len(weights))), *schedule
    )
    for row, syndrome in enumerate(syndromes):
        expected = int(matching.decode(syndrome)[0])
        python, margin = plan.decode(syndrome, weights)
        one, compiled_margin = decode_frontier_one(syndrome, weights, *schedule)
        assert compiled[row] == one == python == expected
        assert margins[row] == pytest.approx(compiled_margin)
        assert compiled_margin == pytest.approx(margin)
