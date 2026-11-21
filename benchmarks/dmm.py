import argparse
import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_THZ_TO_RAD_S = 2 * np.pi * 1e12

PHONON_METHOD = "pheasy"
PHONON_FIELDS = (
    "structure",
    "supercell_matrix",
    "primitive_matrix",
    "born",
    "epsilon_electronic",
    "epsilon_static",
)

def fetch_material_phonons(mpr: Any, material_id: str, phonon_task_id: str | None = None) -> dict[str, Any]:
    """Fetch the structure and force constants from the same phonon task."""
    summaries = mpr.materials.summary.search(material_ids=[material_id], fields=["material_id", "phonon_IDs"])
    if not summaries:
        raise ValueError(f"{material_id}: material not found in Materials Project.")

    phonon_ids = summaries[0].get("phonon_IDs") or {}
    task_ids = phonon_ids.get(PHONON_METHOD) or []
    if not task_ids:
        raise ValueError(f"{material_id}: no Pheasy phonon task found.")

    if phonon_task_id is None and len(task_ids) == 1:
        phonon_task_id = task_ids[0]
    elif phonon_task_id not in task_ids:
        choices = ", ".join(sorted(task_ids))
        raise ValueError(f"{material_id}: choose a Pheasy task using --task-a or --task-b. Available tasks: {choices}.")

    phonons = mpr.materials.phonon
    task = phonons.search(
        identifiers=[phonon_task_id],
        phonon_method=PHONON_METHOD,
        all_fields=False,
        fields=["identifier", *PHONON_FIELDS],
    )[0]

    force_constants = np.asarray(phonons.get_forceconstants_from_phonon_id(phonon_task_id, PHONON_METHOD), dtype=float)
    if (
        force_constants.ndim != 4
        or force_constants.shape[-2:] != (3, 3)
        or force_constants.size == 0
        or not np.isfinite(force_constants).all()
    ):
        raise ValueError(f"{material_id} ({phonon_task_id}): invalid force constants.")

    return {
        "material_id": material_id,
        "phonon_task_id": phonon_task_id,
        **{field: task.get(field) for field in PHONON_FIELDS},
        "force_constants": force_constants,
    }

def reconstruct_mesh(material: dict[str, Any], mesh_numbers: list[int]) -> dict[str, Any]:
    """Return the mode data and crystal geometry needed for computing flux."""
    from phonopy import Phonopy
    from phonopy.physical_units import get_physical_units
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.symmetry import symmetrize_borns_and_epsilon

    material_id = material["material_id"]
    structure = material["structure"]
    sites = structure["sites"]
    if any(len(site["species"]) != 1 or site["species"][0]["occu"] != 1 for site in sites):
        raise ValueError(f"{material_id}: phonon reconstruction requires fully occupied sites.")

    cell = PhonopyAtoms(
        symbols=[site["species"][0]["element"] for site in sites],
        cell=structure["lattice"]["matrix"],
        scaled_positions=[site["abc"] for site in sites],
    )
    if material.get("primitive_matrix") is None or material.get("supercell_matrix") is None:
        raise ValueError(f"{material_id}: both the primitive and supercell matrices are required.")
    phonon = Phonopy(
        cell,
        supercell_matrix=material["supercell_matrix"],
        primitive_matrix=material["primitive_matrix"],
        calculator="vasp",
    )
    force_constants = np.asarray(material["force_constants"], dtype=float, order="C")
    atoms = len(phonon.supercell)
    if force_constants.shape != (atoms, atoms, 3, 3) or not np.isfinite(force_constants).all():
        raise ValueError(f"{material_id}: force constants do not match the reconstructed supercell.")
    phonon.force_constants = force_constants

    born = material.get("born")
    if born is not None and not np.allclose(born, 0):
        dielectric = material.get("epsilon_static")
        if dielectric is None:
            raise ValueError(f"{material_id}: the Pheasy high-frequency dielectric tensor is missing.")
        primitive_born, dielectric = symmetrize_borns_and_epsilon(
            borns=np.asarray(born, dtype=float),
            epsilon=np.asarray(dielectric, dtype=float),
            ucell=phonon.unitcell,
            primitive_matrix=phonon.primitive_matrix,
            supercell_matrix=phonon.supercell_matrix,
        )
        units = get_physical_units()
        phonon.nac_params = {"born": primitive_born, "dielectric": dielectric, "factor": units.Hartree * units.Bohr}

    mesh = phonon.run_mesh(mesh_numbers, is_mesh_symmetry=False, is_gamma_center=True, with_group_velocities=True)
    # Nq is the number of q-points. Nb is the number of phonon branches.
    return {
        "material_id": material_id,
        "phonon_task_id": material["phonon_task_id"],
        "mesh_numbers": mesh_numbers,
        "qpoints": mesh.qpoints,  # (Nq, 3)
        "q_weights": mesh.weights / mesh.weights.sum(),  # (Nq,)
        "frequencies_thz": mesh.frequencies,  # (Nq, Nb)
        "group_velocities_m_s": mesh.group_velocities * 100.0,  # (Nq, Nb, 3)
        "primitive_volume_m3": phonon.primitive.volume * 1e-30,  # Scalar
        "input_lattice_angstrom": cell.cell,  # (3, 3)
        "primitive_lattice_angstrom": phonon.primitive.cell,  # (3, 3)
        "nac_applied": phonon.nac_params is not None,
    }

def _angular_frequencies(mesh: dict[str, Any]) -> np.ndarray:
    material = mesh.get("material_id", "Mesh")
    frequencies = np.asarray(mesh["frequencies_thz"], dtype=float)
    if frequencies.ndim != 2 or frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError(f"{material}: frequencies must be a finite, nonempty (Nq, Nb) array.")

    negative_q, negative_branch = np.nonzero(frequencies < 0.0)
    if negative_q.size:
        qpoints = np.asarray(mesh.get("qpoints"), dtype=float)
        if qpoints.shape != (frequencies.shape[0], 3) or not np.isfinite(qpoints).all():
            raise ValueError(f"{material}: checking negative frequencies requires finite (Nq, 3) q-points.")
        q = qpoints[negative_q]
        at_gamma = np.all(np.abs(q - np.rint(q)) <= 1e-8, axis=1)
        # Phonopy orders the acoustic modes first at Gamma.
        if not np.all(at_gamma & (negative_branch < 3) & (frequencies[negative_q, negative_branch] >= -1e-5)):
            raise ValueError(f"{material}: imaginary phonon frequencies outside the Gamma acoustic tolerance.")
        frequencies = np.maximum(frequencies, 0.0)

    return frequencies * _THZ_TO_RAD_S

def _incident_flux_per_bin(
    mesh: dict[str, Any], 
    omega: np.ndarray, 
    normal: ArrayLike, 
    edges: np.ndarray
) -> np.ndarray:
    material = mesh.get("material_id", "Mesh")
    velocities = np.asarray(mesh["group_velocities_m_s"], dtype=float)
    weights = np.asarray(mesh["q_weights"], dtype=float)
    volume = float(mesh["primitive_volume_m3"])
    if (
        velocities.shape != (*omega.shape, 3)
        or not np.isfinite(velocities).all()
        or weights.shape != (omega.shape[0],)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(f"{material}: expected finite (Nq, Nb, 3) group velocities and finite, nonnegative (Nq,) q-weights summing to one.")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"{material}: primitive-cell volume must be finite and positive.")

    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not direction.any():
        raise ValueError(f"{material}: interface normal must be a finite, nonzero Cartesian vector.")
    direction = direction / np.max(np.abs(direction))
    direction /= np.linalg.norm(direction)
    incident = np.maximum(velocities @ direction, 0.0)
    mode_flux = weights[:, None] * incident / volume
    flux, _ = np.histogram(omega.ravel(), bins=edges, weights=mode_flux.ravel())
    return flux

def elastic_dmm(
    mesh_a: dict[str, Any],
    mesh_b: dict[str, Any],
    normal_a: ArrayLike,
    normal_b: ArrayLike,
    temperatures_k: ArrayLike,
    bin_width_thz: float,
) -> dict[str, np.ndarray]:
    """Compute elastic DMM conductance from two full Brillouin-zone meshes.

    Each Cartesian normal points toward the interface in its mesh's frame.
    The incident spectra are per unit angular frequency. Conductance and
    resistance are area-specific and follow the supplied temperature order.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures_k, dtype=float))
    if (
        temperatures.ndim != 1
        or temperatures.size == 0
        or not np.isfinite(temperatures).all()
        or np.any(temperatures <= 0.0)
    ):
        raise ValueError("Temperatures must be a finite, positive scalar or nonempty one-dimensional array in kelvin.")
    if not np.isfinite(bin_width_thz) or bin_width_thz <= 0.0:
        raise ValueError("Bin width must be finite and positive in THz.")

    omega_a = _angular_frequencies(mesh_a)
    omega_b = _angular_frequencies(mesh_b)
    width = bin_width_thz * _THZ_TO_RAD_S
    max_omega = max(omega_a.max(), omega_b.max())
    n_bins = int(np.floor(max_omega / width)) + 1
    edges = np.arange(n_bins + 1) * width
    centers = (edges[:-1] + edges[1:]) / 2
    flux_a = _incident_flux_per_bin(mesh_a, omega_a, normal_a, edges)
    flux_b = _incident_flux_per_bin(mesh_b, omega_b, normal_b, edges)

    total = flux_a + flux_b
    populated = total > 0.0
    alpha_ab = np.divide(flux_b, total, out=np.zeros_like(total), where=populated)
    alpha_ba = np.divide(flux_a, total, out=np.zeros_like(total), where=populated)
    transmitted = flux_a * alpha_ab  # Bin-integrated flux, including its frequency width.

    k_b = 1.380649e-23
    hbar = 6.62607015e-34 / (2 * np.pi)
    with np.errstate(over="ignore"):
        x = (hbar / k_b) * centers[None, :] / temperatures[:, None]
    ratio = (x == 0.0).astype(float)
    finite_positive = (x > 0.0) & np.isfinite(x)
    finite_x = x[finite_positive]
    ratio[finite_positive] = finite_x * np.exp(-finite_x / 2) / -np.expm1(-finite_x)
    conductance = (k_b * ratio**2) @ transmitted
    resistance = np.divide(
        1.0, conductance, out=np.full_like(conductance, np.inf), where=conductance > 0.0
    )

    return {
        "temperatures_k": temperatures,
        "omega_edges_rad_s": edges,
        "omega_centers_rad_s": centers,
        "phi_a": flux_a / np.diff(edges),
        "phi_b": flux_b / np.diff(edges),
        "alpha_ab": alpha_ab,
        "alpha_ba": alpha_ba,
        "g_w_m2_k": conductance,
        "r_k_m2_k_w": resistance,
    }

def main(argv: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Reconstruct phonon meshes and optionally compute elastic DMM conductance.")
    parser.add_argument("material_a", help="Materials Project ID of the first bulk solid")
    parser.add_argument("material_b", help="Materials Project ID of the second bulk solid")
    parser.add_argument("--task-a", help="Pheasy phonon task ID for the first solid")
    parser.add_argument("--task-b", help="Pheasy phonon task ID for the second solid")
    parser.add_argument("--mesh", type=int, nargs=3, default=[12, 12, 12], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--normal-a", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Cartesian normal pointing toward the interface from the first solid")
    parser.add_argument("--normal-b", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Cartesian normal pointing toward the interface from the second solid")
    parser.add_argument("--temperatures", type=float, nargs="+", metavar="K", help="DMM temperatures in kelvin (default: 30 300 700)")
    parser.add_argument("--bin-width", type=float, metavar="THZ", help="DMM frequency bin width in THz (default: 0.1)")
    args = parser.parse_args(argv)
    if min(args.mesh) < 1:
        parser.error("Mesh dimensions must be positive.")
    run_dmm = any(value is not None for value in (args.normal_a, args.normal_b, args.temperatures, args.bin_width))
    if run_dmm and (args.normal_a is None or args.normal_b is None):
        parser.error("Elastic DMM requires both --normal-a and --normal-b.")

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        parser.error("Set MP_API_KEY to your Materials Project API key.")

    try:
        from mp_api.client import MPRester
        from mp_api.client.core.exceptions import MPRestError
    except ModuleNotFoundError as error:
        if error.name != "mp_api":
            raise
        parser.error("Install the Materials Project client using uv sync --extra mp.")

    try:
        with MPRester(api_key=api_key, use_document_model=False) as mpr:
            material_a = fetch_material_phonons(mpr, args.material_a, args.task_a)
            material_b = fetch_material_phonons(mpr, args.material_b, args.task_b)
        mesh_a = reconstruct_mesh(material_a, args.mesh)
        mesh_b = reconstruct_mesh(material_b, args.mesh)
        result = None
        if run_dmm:
            result = elastic_dmm(
                mesh_a,
                mesh_b,
                normal_a=args.normal_a,
                normal_b=args.normal_b,
                temperatures_k=args.temperatures if args.temperatures is not None else [30, 300, 700],
                bin_width_thz=args.bin_width if args.bin_width is not None else 0.1,
            )
    except ModuleNotFoundError as error:
        if error.name != "phonopy":
            raise
        parser.error("Install Phonopy using uv sync --extra phonons.")
    except (MPRestError, OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    for mesh in (mesh_a, mesh_b):
        qpoints, branches = mesh["frequencies_thz"].shape
        print(f"{mesh['material_id']}: reconstructed {qpoints} q-points and {branches} branches")
    if result is not None:
        print("\n  T (K)    G (MW/m²/K)    R_K (m² K/W)")
        for temperature, conductance, resistance in zip(
            result["temperatures_k"], result["g_w_m2_k"], result["r_k_m2_k_w"], strict=True
        ):
            print(f"{temperature:7g}    {conductance / 1e6:11.3f}    {resistance:.4e}")
    return mesh_a, mesh_b

if __name__ == "__main__":
    main()
