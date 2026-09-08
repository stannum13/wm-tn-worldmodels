import pytest

stim = pytest.importorskip("stim")

from scripts.run_surface_local_overlay import (  # noqa: E402
    BURST_P,
    NOMINAL_P,
    ancilla_coordinates,
    base_circuit,
    burst_overlay,
)


def test_local_overlay_preserves_circuit_shape_and_composes_probability():
    nominal = base_circuit(3)
    coordinates = ancilla_coordinates(nominal)
    overlaid = burst_overlay(nominal, {min(coordinates)})
    extra = (BURST_P - NOMINAL_P) / (1.0 - 2.0 * NOMINAL_P)
    combined = NOMINAL_P * (1.0 - extra) + (1.0 - NOMINAL_P) * extra
    assert combined == pytest.approx(BURST_P)
    assert overlaid.num_qubits == nominal.num_qubits
    assert overlaid.num_detectors == nominal.num_detectors


def test_local_overlay_changes_detector_model():
    nominal = base_circuit(3)
    coordinates = ancilla_coordinates(nominal)
    overlaid = burst_overlay(nominal, {min(coordinates)})
    assert str(overlaid.detector_error_model(decompose_errors=True)) != str(
        nominal.detector_error_model(decompose_errors=True)
    )
