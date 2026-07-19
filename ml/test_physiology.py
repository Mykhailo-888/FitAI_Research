import numpy as np
import pytest

from ml.bioenergetics.physiology import PhysiologyState


def test_valid_bai_and_input_is_not_mutated():
    bai = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    original = bai.copy()
    state = PhysiologyState(bai)

    assert np.array_equal(bai, original)
    assert set(state.to_dict()) == {
        "fatigue", "recovery", "stress", "performance", "readiness", "adaptation"
    }


@pytest.mark.parametrize("bai", ([0.0] * 3, [[0.0] * 4], [0.0] * 5))
def test_invalid_bai_shape(bai):
    with pytest.raises(ValueError, match="4 values"):
        PhysiologyState(bai)


@pytest.mark.parametrize(
    "bai",
    ([np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [-np.inf, 0, 0, 0]),
)
def test_non_finite_bai_is_rejected(bai):
    with pytest.raises(ValueError, match="finite"):
        PhysiologyState(bai)


def test_output_is_deterministic_and_bounded():
    bai = [0.5, -0.25, 1.25, -2.0]
    first = PhysiologyState(bai).to_dict()
    second = PhysiologyState(bai).to_dict()

    assert first == second
    assert all(0.0 <= value <= 100.0 for value in first.values())
    assert first["fatigue"] == pytest.approx(100.0 - first["adaptation"])
    assert first["readiness"] == first["performance"]


@pytest.mark.parametrize("bai", ([21.0, 0, 0, 0], ["1", "2", "3", "4"]))
def test_unsupported_range_or_dtype_is_rejected(bai):
    with pytest.raises((TypeError, ValueError)):
        PhysiologyState(bai)
