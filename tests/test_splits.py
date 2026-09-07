from ptwm.splits import leakage_safe_split


def _records():
    return [
        {"sequence_id": "a", "bias": 0.1, "control_family": "x", "t": t}
        for t in range(3)
    ] + [
        {"sequence_id": "b", "bias": 0.2, "control_family": "x", "t": t}
        for t in range(3)
    ] + [
        {"sequence_id": "c", "bias": 0.3, "control_family": "y", "t": t}
        for t in range(3)
    ]


def test_split_never_separates_points_from_one_sequence():
    train, test = leakage_safe_split(_records(), test_fraction=1 / 3, seed=7)
    assert {r["sequence_id"] for r in train}.isdisjoint({r["sequence_id"] for r in test})
    assert {r["sequence_id"] for r in test} == {"c"}


def test_split_can_force_a_control_family_into_test_set():
    train, test = leakage_safe_split(
        _records(), test_fraction=1 / 3, seed=7,
        test_group_predicate=lambda rows: rows[0]["control_family"] == "y",
    )
    assert all(r["control_family"] == "y" for r in test)
    assert all(r["control_family"] != "y" for r in train)
