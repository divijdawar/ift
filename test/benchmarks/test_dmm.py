import numpy as np
import pytest

from benchmarks.dmm import elastic_dmm


@pytest.fixture
def one_bin_inputs():
    frequencies = np.array([[0.0, 0.0, 0.0, 0.75], [0.75, 0.75, 0.75, 0.75]])
    common_mesh = {
        "qpoints": np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        "q_weights": np.array([0.25, 0.75]),
        "frequencies_thz": frequencies,
    }

    velocities_a = np.zeros((2, 4, 3))
    velocities_a[0, :2] = [[5000.0, 1000.0, 0.0], [3000.0, 0.0, 0.0]]
    velocities_a[1, 0] = [-4000.0, 0.0, 0.0]

    velocities_b = np.zeros((2, 4, 3))
    velocities_b[0, 0] = [0.0, 4000.0, 0.0]
    velocities_b[1, [0, 3]] = [[0.0, -2000.0, 0.0], [0.0, -6000.0, 0.0]]

    return {
        "mesh_a": {
            **common_mesh,
            "material_id": "A",
            "group_velocities_m_s": velocities_a,
            "primitive_volume_m3": 1e-29,
        },
        "mesh_b": {
            **common_mesh,
            "material_id": "B",
            "group_velocities_m_s": velocities_b,
            "primitive_volume_m3": 2e-29,
        },
        "normal_a": [2.0, 0.0, 0.0],
        "normal_b": [0.0, -3.0, 0.0],
        "temperatures_k": np.array([10.0, 300.0]),
        "bin_width_thz": 1.0,
    }

def test_one_bin_matches_hand_calculation(one_bin_inputs):
    result = elastic_dmm(**one_bin_inputs)

    # A contributes 0.25 * 8000 / 1e-29. B contributes 0.75 * 8000 / 2e-29.
    flux_scale = 1e32
    bin_width_rad_s = 2 * np.pi * 1e12
    np.testing.assert_allclose(result["phi_a"] * bin_width_rad_s, [2 * flux_scale])
    np.testing.assert_allclose(result["phi_b"] * bin_width_rad_s, [3 * flux_scale])
    np.testing.assert_allclose(result["alpha_ab"], [0.6])
    np.testing.assert_allclose(result["alpha_ba"], [0.4])

    # The bin center is 0.5 THz. Evaluate the heat capacity using ordinary frequency.
    boltzmann = 1.380649e-23
    planck = 6.62607015e-34
    x = planck * 0.5e12 / (boltzmann * one_bin_inputs["temperatures_k"])
    heat_capacity = boltzmann * x**2 * np.exp(x) / (np.exp(x) - 1)**2
    expected_conductance = 1.2 * flux_scale * heat_capacity
    np.testing.assert_allclose(result["g_w_m2_k"], expected_conductance)
    np.testing.assert_allclose(result["r_k_m2_k_w"], 1 / expected_conductance, atol=0)

def test_no_spectral_overlap(one_bin_inputs):
    mesh_b = one_bin_inputs["mesh_b"]
    mesh_b["frequencies_thz"] = np.full_like(mesh_b["frequencies_thz"], 2.75)

    result = elastic_dmm(**one_bin_inputs)

    np.testing.assert_array_equal(result["g_w_m2_k"], [0.0, 0.0])
    np.testing.assert_array_equal(result["r_k_m2_k_w"], [np.inf, np.inf])
    np.testing.assert_array_equal(result["alpha_ab"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result["alpha_ba"], [1.0, 0.0, 0.0])

def test_swapping_materials_and_normals_preserves_conductance(one_bin_inputs):
    forward = elastic_dmm(**one_bin_inputs)
    swapped = {
        **one_bin_inputs,
        "mesh_a": one_bin_inputs["mesh_b"],
        "mesh_b": one_bin_inputs["mesh_a"],
        "normal_a": one_bin_inputs["normal_b"],
        "normal_b": one_bin_inputs["normal_a"],
    }
    reverse = elastic_dmm(**swapped)

    np.testing.assert_allclose(forward["g_w_m2_k"], reverse["g_w_m2_k"])
    np.testing.assert_allclose(forward["alpha_ab"], reverse["alpha_ba"])
    np.testing.assert_allclose(forward["alpha_ba"], reverse["alpha_ab"])

def test_tiny_negative_gamma_acoustic_frequency_matches_zero(one_bin_inputs):
    expected = elastic_dmm(**one_bin_inputs)
    mesh_a = one_bin_inputs["mesh_a"]
    frequencies = mesh_a["frequencies_thz"].copy()
    frequencies[0, 0] = -1e-6
    mesh_a["frequencies_thz"] = frequencies

    result = elastic_dmm(**one_bin_inputs)

    np.testing.assert_array_equal(result["phi_a"], expected["phi_a"])
    np.testing.assert_array_equal(result["g_w_m2_k"], expected["g_w_m2_k"])

@pytest.mark.parametrize(
    ("q_index", "branch", "frequency"),
    [
        pytest.param(1, 0, -1e-6, id="away-from-gamma"),
        pytest.param(0, 3, -1e-6, id="gamma-optical"),
        pytest.param(0, 0, -1e-3, id="gamma-outside-tolerance"),
    ],
)
def test_rejects_imaginary_frequencies(one_bin_inputs, q_index, branch, frequency):
    mesh_a = one_bin_inputs["mesh_a"]
    frequencies = mesh_a["frequencies_thz"].copy()
    frequencies[q_index, branch] = frequency
    mesh_a["frequencies_thz"] = frequencies

    with pytest.raises(ValueError, match="imaginary phonon frequencies"):
        elastic_dmm(**one_bin_inputs)

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("normal_a", [0.0, 0.0, 0.0], "interface normal", id="zero-normal"),
        pytest.param("q_weights", [0.25, 0.5], "q-weights", id="unnormalized-weights"),
        pytest.param("group_velocities_m_s", np.zeros((2, 4, 2)), "group velocities", id="velocity-shape"),
        pytest.param("primitive_volume_m3", 0.0, "primitive-cell volume", id="zero-volume"),
        pytest.param("temperatures_k", [0.0, 300.0], "Temperatures", id="zero-temperature"),
        pytest.param("bin_width_thz", 0.0, "Bin width", id="zero-bin-width"),
    ],
)
def test_rejects_invalid_inputs(one_bin_inputs, field, value, message):
    if field in one_bin_inputs:
        one_bin_inputs[field] = value
    else:
        one_bin_inputs["mesh_a"][field] = value

    with pytest.raises(ValueError, match=message):
        elastic_dmm(**one_bin_inputs)
