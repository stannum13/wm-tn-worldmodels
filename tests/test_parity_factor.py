import numpy as np
import pytest

from ptwm.parity_factor import (
    compile_parity_factor_decoder,
    decode_parity_factor,
    decoder_tables,
    generate_factor_records,
)


def test_correct_factor_changes_ambiguous_four_detector_decision():
    syndrome = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    tables = decoder_tables()
    assert decode_parity_factor(tables["base"], syndrome)[0] == 0
    assert decode_parity_factor(tables["insert_correct_factor"], syndrome)[0] == 1
    assert decode_parity_factor(tables["insert_wrong_factor"], syndrome)[0] == 0


def test_factor_generator_reproducible_and_null_has_zero_logical_labels():
    first = generate_factor_records(seed=4, episodes=3, horizon=20, factor_probability=0.0)
    second = generate_factor_records(seed=4, episodes=3, horizon=20, factor_probability=0.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(first[1] == 0)


def test_invalid_factor_probability_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        compile_parity_factor_decoder(np.asarray([[1]], dtype=np.uint8), np.asarray([0]), np.asarray([0.5]))

