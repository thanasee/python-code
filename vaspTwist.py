#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import multiprocessing as mp
from numba import njit


def usage():
    """Print usage information and exit."""
    text = """
Usage:
  vaspTwist.py match    POSCAR_1             Homobilayer  — search commensurate cells
  vaspTwist.py match    POSCAR_1  POSCAR_2   Heterobilayer — search commensurate cells
  vaspTwist.py generate POSCAR_1             Generate POSCARs from existing TWIST_LIST.dat
  vaspTwist.py generate POSCAR_1  POSCAR_2   Generate POSCARs from existing TWIST_LIST.dat

Workflow:
  1. Run 'match'    — searches twist angles 0–180° with step 0.1°, writes TWIST_LIST.dat
                      sorted by strain.  No POSCAR is written at this stage.
  2. Run 'generate' — reads TWIST_LIST.dat, shows the candidate table, prompts for
                      index selection, then writes one POSCAR per stacking configuration.

Only monolayer POSCAR files with vacuum space in the z-direction are supported.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

MAX_ATOMS  = 200    # Maximum number of atoms in the bilayer supercell
THETA_MIN  = 0.0    # Minimum twist angle (degrees)
THETA_MAX  = 180.0  # Maxximum twist angle (degrees)
THETA_STEP = 0.1    # Twist angle search step (degrees)
MAX_STRAIN = 0.05   # Maximum symmetric relative distance for vector coincidence
TWIST_LIST_FILE = "TWIST_LIST.dat"

def read_POSCAR(filepath):
    """Read a VASP POSCAR file and return its contents as a dictionary.

    Supports both VASP4 (no element line) and VASP5 (with element line) formats,
    scalar and negative (volume-based) scaling factors, a 3-component scaling
    vector, Selective Dynamics, and both Direct and Cartesian coordinate modes.

    Parameters
    ----------
    filepath : str
        Path to the POSCAR file to read.

    Returns
    -------
    dict with keys:
        lattice_matrix      : np.ndarray, shape (3, 3)  — lattice vectors in Å
        elements            : list[str]                 — element symbols
        atom_counts         : list[int]                 — number of atoms per element
        total_atoms         : int                       — total number of atoms
        positions_cartesian : np.ndarray, shape (N, 3)  — Cartesian coordinates in Å
        positions_direct    : np.ndarray, shape (N, 3)  — fractional coordinates
        species             : list[str]                 — element symbol per atom
        selective_dynamics  : bool                      — whether Selective Dynamics is present
        flags               : np.ndarray or None        — T/F flags per atom, or None
    """

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    with open(filepath, 'r') as poscar:
        lines = poscar.readlines()

    if len(lines[1].split()) == 1:
        raw_scale = float(lines[1])
        raw_lattice_matrix = np.array([[float(x) for x in line.split()]
                                       for line in lines[2:5]])
        if raw_scale < 0:
            volume = np.abs(np.linalg.det(raw_lattice_matrix))
            scale = np.cbrt(np.abs(raw_scale) / volume)
        elif raw_scale == 0:
            print("ERROR! The scaling factor must be not zero.")
            exit(1)
        else:
            scale = raw_scale
        lattice_matrix = raw_lattice_matrix * scale
    elif len(lines[1].split()) == 3:
        scale = np.array(list(map(float, lines[1].split())))
        lattice_matrix = np.array([[float(x) * scale[i] for i, x in enumerate(line.split())]
                                   for line in lines[2:5]])
    else:
        print("ERROR! The scaling factor must be 1 or 3 components.")
        exit(1)

    elements = []
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
        for i in range(len(lines[5].split())):
            while True:
                name = input(f"Enter the name of species No. {i + 1:>3}: ").strip()
                if name.isalpha():
                    break
                else:
                    print("The name of species must be alphabetic characters only.")
            elements.append(name)
        atom_counts = [int(x) for x in lines[5].split()]
        selective_dynamics = lines[6].lower().startswith('s')
        position_start = 8 if selective_dynamics else 7
    else:
        raw_elements = lines[5].split()
        for name in raw_elements:
            elements.append(name.split('/')[0].split('_')[0])
        atom_counts = [int(x) for x in lines[6].split()]
        selective_dynamics = lines[7].lower().startswith('s')
        position_start = 9 if selective_dynamics else 8

    total_atoms = sum(atom_counts)
    position_stop = position_start + total_atoms

    positions = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(position_start, position_stop)])

    species = [x for i, x in enumerate(elements)
               for _ in range(atom_counts[i])]

    flags = None
    if selective_dynamics:
        flags = np.array([[x for x in lines[i].split()[3:6]]
                          for i in range(position_start, position_stop)])

    is_direct = lines[position_start - 1].strip().lower().startswith('d')
    if is_direct:
        positions_direct = positions % 1.0
        positions_cartesian = direct_to_cartesian(lattice_matrix, positions_direct)
    else:
        positions_cartesian = positions * scale
        positions_direct = cartesian_to_direct(lattice_matrix, positions_cartesian)

    return {"lattice_matrix":      lattice_matrix,
            "elements":            elements,
            "atom_counts":         atom_counts,
            "total_atoms":         total_atoms,
            "positions_cartesian": positions_cartesian,
            "positions_direct":    positions_direct,
            "species":             species,
            "selective_dynamics":  selective_dynamics,
            "flags":               flags if selective_dynamics else None}


def direct_to_cartesian(lattice_matrix, positions_direct):
    """Convert fractional (Direct) coordinates to Cartesian coordinates.

    Uses the relation:  r_cart = r_direct @ lattice_matrix

    Parameters
    ----------
    lattice_matrix    : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_direct  : np.ndarray, shape (N, 3) — fractional coordinates

    Returns
    -------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    """

    positions = positions_direct % 1.0
    positions_cartesian = positions @ lattice_matrix

    return positions_cartesian


def cartesian_to_direct(lattice_matrix, positions_cartesian):
    """Convert Cartesian coordinates to fractional (Direct) coordinates.

    Uses the relation:  r_direct = r_cart @ lattice_matrix⁻¹

    Parameters
    ----------
    lattice_matrix      : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å

    Returns
    -------
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates in [0, 1)
    """

    positions_direct = (positions_cartesian @ np.linalg.inv(lattice_matrix)) % 1.0

    return positions_direct


def check_elements(elements):
    """Check for duplicate element symbols and prompt the user for a canonical order.

    If duplicate symbols are found (e.g. ['Mo', 'S', 'Mo']), the user is asked
    to specify the desired ordering of the unique species. An empty input accepts
    the default order (first-occurrence order).

    Parameters
    ----------
    elements : list[str] — element symbols as parsed from the POSCAR

    Returns
    -------
    list[str] or None
        The user-specified element order if duplicates were found, else None.
    """

    unique_elements = list(dict.fromkeys(elements))

    if len(elements) != len(unique_elements):
        print("\nFound duplicated elements in POSCAR!")
        print("Unique elements: [" + " ".join(unique_elements) + "]")
        while True:
            sort_elements = input("Enter the desired element order (separate by space): ").split()
            if len(sort_elements) == 0:
                print("Warning! Empty input — using default unique element order.")
                return unique_elements.copy()
            if (len(sort_elements) == len(unique_elements) and
                    set(sort_elements) == set(unique_elements)):
                return sort_elements
            print("ERROR! The species do not match the unique elements. Try again.")
    else:
        return None


def mapping_elements(elements, atom_counts, positions_cartesian, positions_direct,
                     species, selective_dynamics, flags, sort_elements=None):
    """Re-order atoms so that each element block is contiguous and sorted canonically.

    Groups atomic positions by element symbol, resolves any duplicate element
    entries via check_elements(), and returns arrays sorted according to the
    specified (or user-supplied) element order. This is required because some
    POSCARs interleave atoms of the same species across multiple blocks.

    Parameters
    ----------
    elements            : list[str]            — element symbols from POSCAR
    atom_counts         : list[int]            — atoms per element block
    positions_cartesian : np.ndarray (N, 3)    — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)    — fractional coordinates
    species             : list[str]            — per-atom element labels
    selective_dynamics  : bool                 — whether Selective Dynamics is used
    flags               : np.ndarray or None   — per-atom T/F flags
    sort_elements       : list[str] or None    — explicit element order (optional)

    Returns
    -------
    dict with keys:
        elements            : list[str]
        atom_counts         : list[int]
        positions_cartesian : np.ndarray (N, 3)
        positions_direct    : np.ndarray (N, 3)
        species             : list[str]
        flags               : np.ndarray or None
    """

    new_elements = elements.copy()
    new_atom_counts = atom_counts.copy()
    new_positions_cartesian = positions_cartesian.copy()
    new_positions_direct = positions_direct.copy()
    new_species = list(species).copy()
    new_flags = flags.copy() if (selective_dynamics and flags is not None) else None

    # Group positions and flags by element symbol using the per-atom species list.
    # This is correct even when positions are in layer order (not element order),
    # because we look up each atom's element from species[i] rather than assuming
    # the positions are already contiguous by element.
    elements_positions_cartesian = {}
    elements_positions_direct = {}
    elements_species = {}
    elements_flags = {} if selective_dynamics else None
    for idx in range(len(new_species)):
        element = new_species[idx]
        elements_positions_cartesian.setdefault(element, []).append(
            new_positions_cartesian[idx])
        elements_positions_direct.setdefault(element, []).append(
            new_positions_direct[idx])
        elements_species.setdefault(element, []).append(new_species[idx])
        if selective_dynamics and new_flags is not None:
            elements_flags.setdefault(element, []).append(new_flags[idx])

    if sort_elements is None:
        sort_elements = check_elements(elements)

    if sort_elements is not None:
        sort_positions_cartesian = []
        sort_positions_direct = []
        sort_species = []
        sort_flags = [] if selective_dynamics else None
        sort_atom_counts = []
        for element in sort_elements:
            sort_positions_cartesian.extend(elements_positions_cartesian[element])
            sort_positions_direct.extend(elements_positions_direct[element])
            sort_species.extend(elements_species[element])
            if selective_dynamics and elements_flags is not None:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))

        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        new_species = list(sort_species)
        if selective_dynamics and sort_flags is not None:
            new_flags = np.array(sort_flags)
        new_atom_counts = sort_atom_counts
        new_elements = sort_elements

    return {"elements":            new_elements,
            "atom_counts":         new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct":    new_positions_direct,
            "species":             new_species,
            "flags":               new_flags if selective_dynamics else None}


def define_labels(elements, atom_counts):
    """Generate per-atom labels used as comments in the POSCAR position block.

    Labels take the form '<Symbol><index>' with the index zero-padded to the
    width of the largest atom count plus one (e.g. 'Mo01', 'S003').

    Parameters
    ----------
    elements    : list[str]  — element symbols in canonical order
    atom_counts : list[int]  — number of atoms per element

    Returns
    -------
    labels : list[str] — one label per atom in the same order as the position arrays
    """

    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}"
              for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]

    return labels


def write_POSCAR(filepath, lattice_matrix, elements, atom_counts,
                 positions_direct, selective_dynamics, flags, labels):
    """Write a VASP5-format POSCAR file with Direct coordinates.

    The scale factor is always written as 1.0 because the lattice vectors
    are already stored in absolute Å units. Atom labels are appended as
    inline comments after each position line for readability.

    Parameters
    ----------
    filepath           : str
    lattice_matrix     : np.ndarray (3, 3)  — lattice vectors in Å
    elements           : list[str]          — element symbols in canonical order
    atom_counts        : list[int]          — atoms per element
    positions_direct   : np.ndarray (N, 3)  — fractional coordinates
    selective_dynamics : bool
    flags              : np.ndarray or None  — per-atom T/F flags
    labels             : list[str]          — per-atom comment labels
    """

    with open(filepath, 'w') as o:
        o.write("Generated by vaspTwist.py code\n")
        o.write(f"   {1.0:.14f}\n")
        for lattice in lattice_matrix:
            o.write(f"   {lattice[0]:20.16f}  {lattice[1]:20.16f}  {lattice[2]:20.16f}\n")
        o.write("   " + "    ".join(elements) + " \n")
        o.write("     " + "    ".join(map(str, atom_counts)) + "\n")
        if selective_dynamics:
            o.write("Selective dynamics\n")
        o.write("Direct\n")
        if selective_dynamics:
            for position, flag, label in zip(positions_direct, flags, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {flag[0]:s}   {flag[1]:s}   {flag[2]:s}"
                        f"   {label:>6s}\n")
        else:
            for position, label in zip(positions_direct, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {label:>6s}\n")




def unwrap(positions_direct):
    """Reconstruct a contiguous cluster by unwrapping periodic boundary conditions.

    Shifts all atoms into the minimum-image frame relative to atom[0], so that
    atoms split across a cell boundary are treated as geometrically contiguous.
    Interatomic distances are preserved exactly.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    reference : np.ndarray (3,)   — fractional coordinate of atom[0]
    unwrapped : np.ndarray (N, 3) — unwrapped fractional coordinates
    """

    reference = np.copy(positions_direct[0])
    delta = positions_direct - reference
    delta -= np.round(delta)

    return reference, reference + delta


def center_sheet(positions_direct):
    """Shift a 2D sheet so the vacuum direction is centered at z = 0.5.

    The vacuum axis is assumed to be z (index 2).

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """

    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = 2
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5

    return new % 1.0


@njit
def rotation_matrix(degree):
    """Construct a rotation matrix for rotation about the z-axis.

    Uses the Rodrigues rotation formula:
        R = cos θ · I + sin θ · (u×) + (1 − cos θ) · u⊗u
    where u = [0, 0, 1].

    Parameters
    ----------
    degree : float

    Returns
    -------
    rotate : np.ndarray (3, 3) — rotation matrix
    """

    radian = np.radians(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)
    u = np.array([0., 0., 1.])

    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)

    return rotate


@njit
def find_moire_vectors_chunk(bottom_lattice_matrix, top_lattice_matrix,
                             theta_chunk, combined_list, max_strain):
    """Search for near-coincident lattice vector pairs for a chunk of twist angles.

    Symmetric relative distance (CellMatch convention):
        ε = |v1 − v2| / (|v1| + |v2|)

    Parameters
    ----------
    bottom_lattice_matrix : np.ndarray (3, 3)
    top_lattice_matrix    : np.ndarray (3, 3)
    theta_chunk           : np.ndarray (K,)
    combined_list         : np.ndarray (M,)
    max_strain            : float

    Returns
    -------
    results : list of tuples (theta, n1, n2, m1, m2, rel_distance, v_layer1, v_layer2)
    """

    results = []
    for theta in theta_chunk:
        new_lattice_matrix = rotation_matrix(theta) @ top_lattice_matrix

        for n1 in combined_list:
            for n2 in combined_list:
                v_layer1 = n1 * bottom_lattice_matrix[0] + n2 * bottom_lattice_matrix[1]
                norm_v1 = np.linalg.norm(v_layer1)
                if norm_v1 == 0.0:
                    continue
                for m1 in combined_list:
                    for m2 in combined_list:
                        v_layer2 = m1 * new_lattice_matrix[0] + m2 * new_lattice_matrix[1]
                        norm_v2 = np.linalg.norm(v_layer2)
                        if norm_v2 == 0.0:
                            continue
                        rel_distance = np.linalg.norm(v_layer1 - v_layer2) / (norm_v1 + norm_v2)
                        if rel_distance <= max_strain:
                            results.append((theta, n1, n2, m1, m2,
                                            rel_distance, v_layer1, v_layer2))

    return results


def find_moire_vectors(bottom_lattice_matrix, top_lattice_matrix,
                       theta_min, theta_max, theta_step, n_max):
    """Search for commensurate moiré vectors over a range of twist angles.

    Integer indices search from -n_max to n_max inclusive (including 0),
    covering all lattice vectors: primitive (1,0), (0,1), and all combinations.

    Parameters
    ----------
    bottom_lattice_matrix : np.ndarray (3, 3)
    top_lattice_matrix    : np.ndarray (3, 3)
    theta_min             : float
    theta_max             : float
    theta_step            : float
    n_max                 : int

    Returns
    -------
    moire_vectors : list of tuples (theta, n1, n2, m1, m2, rel_distance, v_layer1, v_layer2)
    """

    theta_array = np.arange(theta_min, theta_max, theta_step)

    # Full range -n_max to n_max including 0, so (1,0) and (0,1) are always included.
    # Zero-vector (0,0) is rejected inside find_moire_vectors_chunk by the norm_v == 0 check.
    combined_list = np.arange(-n_max, n_max + 1)

    num_cores = mp.cpu_count()
    chunk_size = max(1, len(theta_array) // num_cores)
    theta_chunks = [theta_array[i:i + chunk_size]
                    for i in range(0, len(theta_array), chunk_size)]

    with mp.Pool(processes=num_cores) as pool:
        chunk_results = pool.starmap(
            find_moire_vectors_chunk,
            [(bottom_lattice_matrix, top_lattice_matrix, chunk, combined_list, MAX_STRAIN)
             for chunk in theta_chunks]
        )

    moire_vectors = [item for sublist in chunk_results for item in sublist]

    return moire_vectors


def build_supercell(lattice_matrix, positions_cartesian, species,
                    selective_dynamics, flags, A1, A2):
    """Tile a primitive cell into a moiré supercell defined by vectors A1 and A2.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3)
    positions_cartesian : np.ndarray (N, 3)
    species             : list[str]
    selective_dynamics  : bool
    flags               : np.ndarray or None
    A1, A2              : np.ndarray (3,)

    Returns
    -------
    dict with keys: lattice_matrix, positions_direct, species, flags
    """

    n_rep = int(np.ceil(max(np.linalg.norm(A1), np.linalg.norm(A2)) /
                        min(np.linalg.norm(lattice_matrix[0]),
                            np.linalg.norm(lattice_matrix[1])))) + 2

    supercell_positions_cartesian = []
    supercell_species = []
    supercell_flags = [] if selective_dynamics else None

    for n1 in range(-n_rep, n_rep + 1):
        for n2 in range(-n_rep, n_rep + 1):
            shift = n1 * lattice_matrix[0] + n2 * lattice_matrix[1]
            for atom_index in range(len(positions_cartesian)):
                supercell_positions_cartesian.append(positions_cartesian[atom_index] + shift)
                supercell_species.append(species[atom_index])
                if selective_dynamics:
                    supercell_flags.append(flags[atom_index])

    supercell_positions_cartesian = np.array(supercell_positions_cartesian)

    new_lattice_matrix = np.array([A1, A2, lattice_matrix[2]])

    new_positions_direct = cartesian_to_direct(new_lattice_matrix, supercell_positions_cartesian)
    inside_mask = ((new_positions_direct[:, 0] >= -1e-8) &
                   (new_positions_direct[:, 0] <   1.0 - 1e-8) &
                   (new_positions_direct[:, 1] >= -1e-8) &
                   (new_positions_direct[:, 1] <   1.0 - 1e-8))

    filtered_positions_direct = new_positions_direct[inside_mask] % 1.0
    filtered_species = [supercell_species[i]
                        for i in range(len(supercell_species)) if inside_mask[i]]
    filtered_flags = None
    if selective_dynamics:
        filtered_flags = np.array([supercell_flags[i]
                                   for i in range(len(supercell_flags)) if inside_mask[i]])

    return {"lattice_matrix":   new_lattice_matrix,
            "positions_direct": filtered_positions_direct,
            "species":          filtered_species,
            "flags":            filtered_flags if selective_dynamics else None}


def build_twisted_bilayer(layer1, layer2_rotated, selective_dynamics,
                          A1_bottom, A2_bottom, A1_top, A2_top):
    """Stack two supercell layers into a twisted bilayer with a 3.5 Å interlayer gap.

    Uses a common averaged lattice  A_common = (A_bottom + A_top) / 2  so that
    neither layer is privileged and both are strained equally to the common cell.
    Each layer's atoms are expressed in Cartesian coordinates from their own
    matched supercell vectors, then re-expressed in the common lattice.

    Parameters
    ----------
    layer1             : dict — bottom layer from build_supercell (A1_bottom, A2_bottom)
    layer2_rotated     : dict — top layer from build_supercell (A1_bottom, A2_bottom)
    selective_dynamics : bool
    A1_bottom, A2_bottom : np.ndarray (3,) — bottom matched supercell vectors
    A1_top,    A2_top    : np.ndarray (3,) — top matched supercell vectors

    Returns
    -------
    dict with keys: lattice_matrix, positions_direct, species, flags
    """

    # Common averaged lattice — neither layer is privileged
    A1_common = (A1_bottom + A1_top) / 2.0
    A2_common = (A2_bottom + A2_top) / 2.0
    a3_common = layer1["lattice_matrix"][2].copy()
    common_lattice = np.array([A1_common, A2_common, a3_common])

    # Convert each layer's atoms from their own supercell to Cartesian
    layer1_cartesian = direct_to_cartesian(layer1["lattice_matrix"], layer1["positions_direct"])
    layer2_cartesian = direct_to_cartesian(layer2_rotated["lattice_matrix"], layer2_rotated["positions_direct"])

    # Stack with interlayer gap
    z_max_layer1 = np.max(layer1_cartesian[:, 2])
    z_min_layer2 = np.min(layer2_cartesian[:, 2])
    interlayer_gap = 3.5
    layer2_cartesian[:, 2] += z_max_layer1 - z_min_layer2 + interlayer_gap

    combined_cartesian = np.vstack((layer1_cartesian, layer2_cartesian))
    combined_species = list(layer1["species"]) + list(layer2_rotated["species"])

    combined_flags = None
    if selective_dynamics and layer1["flags"] is not None and layer2_rotated["flags"] is not None:
        combined_flags = np.vstack((layer1["flags"], layer2_rotated["flags"]))

    # Express all atoms in the common lattice and center along z-axis
    combined_positions_direct = cartesian_to_direct(common_lattice, combined_cartesian)
    combined_positions_direct = center_sheet(combined_positions_direct)

    return {"lattice_matrix":   common_lattice,
            "positions_direct": combined_positions_direct,
            "species":          combined_species,
            "flags":            combined_flags if selective_dynamics else None}


def collect_elements_and_counts(species, known_elements):
    """Collect unique elements present in species and their counts.

    Returns elements in the order they first appear in known_elements,
    restricted to those actually present in species.

    Parameters
    ----------
    species        : list[str]
    known_elements : list[str]

    Returns
    -------
    elements_order : list[str]
    atom_counts    : list[int]
    """

    elements_order = list(dict.fromkeys(e for e in known_elements if e in species))
    atom_counts    = [species.count(e) for e in elements_order]

    return elements_order, atom_counts


def metric_tensor(e1, e2, e3):
    """Compute the 3×3 metric tensor G where G[i,j] = ei · ej.

    Parameters
    ----------
    e1, e2, e3 : np.ndarray (3,)

    Returns
    -------
    G : np.ndarray (3, 3)
    """

    vecs = [e1, e2, e3]
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G[i, j] = vecs[i] @ vecs[j]

    return G


def calculate_strain(A1_bottom, A2_bottom, A1_top, A2_top):
    """Compute the Lagrangian finite strain between the bottom and top supercell vectors.

    Measures the physical strain needed to make the two supercells commensurate,
    i.e. the deformation from the top supercell to the bottom supercell.

    For a homobilayer at θ = 0°, both supercells are identical so strain = 0.
    As θ increases, the matched vectors diverge and strain increases.
    For a heterobilayer, strain reflects the lattice mismatch at the given angle.

    Builds 3×3 metric tensors G = [ei · ej] for both supercell vector sets,
    computes the Cholesky decomposition G = Rᵀ R, then evaluates:
        F = R_top · R_bottom⁻¹ − I     (deformation gradient)
        ε = ½(F + Fᵀ + FᵀF)            (Lagrangian strain tensor)
    Scalar measure: √(Σλᵢ² / 3) where λᵢ are eigenvalues of ε.

    The z-component of all vectors is set to 1.0 to avoid singularity
    (CellMatch convention for 2D systems).

    Parameters
    ----------
    A1_bottom, A2_bottom : np.ndarray (3,) — bottom supercell vectors in Å
    A1_top,    A2_top    : np.ndarray (3,) — top supercell vectors in Å

    Returns
    -------
    deformation : float — scalar Lagrangian strain measure
    """

    V1 = np.array([A1_bottom[0], A1_bottom[1], 0.0])
    V2 = np.array([A2_bottom[0], A2_bottom[1], 0.0])
    V3 = np.array([0.0,          0.0,          1.0])
    U1 = np.array([A1_top[0],    A1_top[1],    0.0])
    U2 = np.array([A2_top[0],    A2_top[1],    0.0])
    U3 = np.array([0.0,          0.0,          1.0])

    G_bottom = metric_tensor(V1, V2, V3)
    G_top    = metric_tensor(U1, U2, U3)

    R_bottom = np.linalg.cholesky(G_bottom).T
    R_top    = np.linalg.cholesky(G_top).T

    F = (R_top @ np.linalg.inv(R_bottom)) - np.eye(3)
    lagrangian = 0.5 * (F + F.T + F.T @ F)

    eigenvalues = np.linalg.eigvalsh(lagrangian)
    deformation = np.sqrt(np.sum(eigenvalues ** 2) / 3.0)

    return float(deformation)



def canonicalize_cell(A1, A2):
    """Build a canonical key for a 2D supercell that is invariant to row order
    and vector sign, so geometrically equivalent cells are treated as duplicates.

    Tries all combinations of sign and row order, rounds to 6 decimal places,
    and returns the lexicographically smallest flattened tuple as the key.

    Parameters
    ----------
    A1, A2 : np.ndarray (3,) — supercell lattice vectors (only xy used)

    Returns
    -------
    tuple — canonical hashable key
    """

    v1 = A1[:2]
    v2 = A2[:2]

    candidates = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            candidates.append(tuple(np.round(np.concatenate([s1 * v1, s2 * v2]), 6)))
            candidates.append(tuple(np.round(np.concatenate([s2 * v2, s1 * v1]), 6)))

    return min(candidates)


def build_candidates_for_theta(theta_key, vec_list, bottom_lattice_matrix, bottom_positions_cartesian,
                               bottom_species, bottom_selective_dynamics, bottom_flags,
                               top_lattice_matrix, top_positions_cartesian, top_species,
                               top_selective_dynamics, top_flags, sort_elements, known_elements):
    """Build and filter bilayer supercell candidates for a single twist angle.

    Implements the CellMatch (match_cells.py) filter pipeline:
        1. omjer1 = round(|A1×A2| / |a1×a2|) >= 1   (bottom area ratio is positive integer)
        2. omjer2 = round(|G1×G2| / |b1×b2|) >= 1   (top area ratio is positive integer)
        3. total_atoms = n_bottom*omjer1 + n_top*omjer2 <= MAX_ATOMS
        4. canonicalize_cell deduplication (per-theta)
    Atom count is computed arithmetically from omjer1/omjer2 — build_supercell is
    only called after all filters pass, keeping the loop fast.

    Supports both homobilayer (bottom == top) and heterobilayer (different layers).

    Parameters
    ----------
    theta_key                   : float
    vec_list                    : list of (rel_distance, v_bottom, v_top, n1, n2, m1, m2)
    bottom_lattice_matrix       : np.ndarray (3, 3)
    bottom_positions_cartesian  : np.ndarray (N, 3)
    bottom_species              : list[str]
    bottom_selective_dynamics   : bool
    bottom_flags                : np.ndarray or None
    top_lattice_matrix          : np.ndarray (3, 3)
    top_positions_cartesian     : np.ndarray (M, 3)
    top_species                 : list[str]
    top_selective_dynamics      : bool
    top_flags                   : np.ndarray or None
    sort_elements               : list[str] or None
    known_elements              : list[str]

    Returns
    -------
    theta_candidates : list[dict]
    """

    a1 = bottom_lattice_matrix[0]
    a2 = bottom_lattice_matrix[1]
    b1 = top_lattice_matrix[0]
    b2 = top_lattice_matrix[1]

    # Primitive cell cross products (z-component of 2D cross product)
    prim_cross_bottom = a1[0] * a2[1] - a1[1] * a2[0]
    prim_cross_top    = b1[0] * b2[1] - b1[1] * b2[0]

    rotated_top_lattice = rotation_matrix(theta_key) @ top_lattice_matrix

    selective_dynamics = bottom_selective_dynamics or top_selective_dynamics

    # Sort by rel_dist only — CellMatch convention (match_cells.py).
    # The cross-product filter below rejects degenerate (parallel) pairs.
    vec_list = sorted(vec_list, key=lambda x: x[0])

    theta_candidates = []
    seen_configurations = set()

    for i in range(len(vec_list)):
        rel_dist_i, A1_bottom, A1_top, n1_i, n2_i, m1_i, m2_i = vec_list[i]
        for j in range(i, len(vec_list)):   # j >= i, matching CellMatch; j==i rejected by cross_z check
            rel_dist_j, A2_bottom, A2_top, n1_j, n2_j, m1_j, m2_j = vec_list[j]

            A1_vec = A1_bottom
            A2_vec = A2_bottom

            # ------------------------------------------------------------------
            # CellMatch filter 1: non-degenerate bottom supercell
            # omjer1 = |A1×A2| / |a1×a2|  must be a positive integer
            # (equivalent to match_cells.py: omjer1 = round(abs(cross/prim_cross)))
            # ------------------------------------------------------------------
            sup_cross_bottom = A1_vec[0] * A2_vec[1] - A1_vec[1] * A2_vec[0]
            if abs(sup_cross_bottom) < 1e-10 or abs(prim_cross_bottom) < 1e-10:
                continue
            omjer1 = round(abs(sup_cross_bottom / prim_cross_bottom))
            if omjer1 < 1:
                continue

            # ------------------------------------------------------------------
            # CellMatch filter 2: non-degenerate top supercell
            # omjer2 = |G1×G2| / |b1×b2| using the rotated top vectors
            # ------------------------------------------------------------------
            G1_vec = A1_top
            G2_vec = A2_top
            sup_cross_top = G1_vec[0] * G2_vec[1] - G1_vec[1] * G2_vec[0]
            if abs(sup_cross_top) < 1e-10 or abs(prim_cross_top) < 1e-10:
                continue
            omjer2 = round(abs(sup_cross_top / prim_cross_top))
            if omjer2 < 1:
                continue

            # ------------------------------------------------------------------
            # CellMatch filter 3: total atom count within MAX_ATOMS
            # Computed arithmetically — no build_supercell needed at this stage
            # ------------------------------------------------------------------
            n_bottom_prim = len(bottom_species)
            n_top_prim    = len(top_species)
            total_atoms_bilayer = n_bottom_prim * omjer1 + n_top_prim * omjer2
            if total_atoms_bilayer > MAX_ATOMS:
                continue

            # ------------------------------------------------------------------
            # Deduplicate using canonical cell key (invariant to row order/sign)
            # ------------------------------------------------------------------
            config_key = canonicalize_cell(A1_vec, A2_vec)
            if config_key in seen_configurations:
                continue
            seen_configurations.add(config_key)

            # ------------------------------------------------------------------
            # Compute Lagrangian strain (after cheap filters)
            # ------------------------------------------------------------------
            lagrangian_strain = calculate_strain(A1_bottom, A2_bottom, A1_top, A2_top)

            # ------------------------------------------------------------------
            # Build supercells — only reached after all filters pass
            # ------------------------------------------------------------------
            layer1 = build_supercell(bottom_lattice_matrix, bottom_positions_cartesian,
                                      bottom_species, bottom_selective_dynamics, bottom_flags,
                                      A1_vec, A2_vec)

            # Top layer: rotation applied to both lattice and positions so they
            # remain in the same rotated Cartesian frame.
            top_positions_rotated = top_positions_cartesian @ rotation_matrix(theta_key).T
            layer2_rotated = build_supercell(rotated_top_lattice, top_positions_rotated,
                                              top_species, top_selective_dynamics, top_flags,
                                              A1_vec, A2_vec)

            bilayer = build_twisted_bilayer(layer1, layer2_rotated, selective_dynamics,
                                             A1_bottom, A2_bottom, A1_top, A2_top)

            order_ref = sort_elements if sort_elements is not None else list(dict.fromkeys(known_elements))
            elements_order, atom_counts = collect_elements_and_counts(
                bilayer["species"], order_ref)

            theta_candidates.append({"theta":                  theta_key,
                                      "A1_vec":                 A1_vec,
                                      "A2_vec":                 A2_vec,
                                      "strain":                 lagrangian_strain,
                                      "area_ratio_bottom":      omjer1,
                                      "area_ratio_top":         omjer2,
                                      "n_bottom":               len(layer1["species"]),
                                      "indices1":               (n1_i, n2_i, n1_j, n2_j),
                                      "indices2":               (m1_i, m2_i, m1_j, m2_j),
                                      "bilayer_lattice_matrix": bilayer["lattice_matrix"],
                                      "bilayer_positions":      bilayer["positions_direct"],
                                      "bilayer_species":        bilayer["species"],
                                      "bilayer_flags":          bilayer["flags"],
                                      "elements_order":         elements_order,
                                      "total_atoms":            total_atoms_bilayer,
                                      "selective_dynamics":     selective_dynamics})

    return theta_candidates


def get_2d_lattice_type(lattice_matrix):
    """Classify the 2D Bravais lattice type from the in-plane lattice vectors.

    Compares the in-plane lattice lengths (a, b) and the angle γ between them
    to identify the crystal system according to the standard 2D classification:

        hexagonal   : γ = 60° or 120°  and  a = b
        square      : γ = 90°           and  a = b
        rectangular : γ = 90°           and  a ≠ b
        oblique     : all other cases

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3) — row vectors of the lattice in Å

    Returns
    -------
    str : one of 'hexagonal', 'square', 'rectangular', 'oblique'
    """

    length_a = np.linalg.norm(lattice_matrix[0])
    length_b = np.linalg.norm(lattice_matrix[1])
    gamma = np.degrees(np.arccos(np.clip((lattice_matrix[0] @ lattice_matrix[1]) /
                                         (length_a * length_b), -1.0, 1.0)))

    if np.abs(gamma - 90.0) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif ((np.abs(gamma - 60.0) < 1e-5 or np.abs(gamma - 120.0) < 1e-5)
          and np.abs(length_a - length_b) < 1e-8):
        return 'hexagonal'
    else:
        return 'oblique'


def get_shift_grid(lattice_type):
    """Return the high-symmetry stacking shift points for the given 2D lattice type.

    The shifts correspond to the irreducible high-symmetry points of the
    stacking space for each 2D Bravais lattice:

        hexagonal   : AA (0,0),  AB (1/3,2/3),  AB' (2/3,1/3)
        square      : AA (0,0),  AB (1/2,0),    AC  (1/2,1/2)
        rectangular : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC (1/2,1/2)
        oblique     : AA (0,0),  AB (1/2,0),    AB' (0,1/2),  AC (1/2,1/2)

    Parameters
    ----------
    lattice_type : str — one of 'hexagonal', 'square', 'rectangular', 'oblique'

    Returns
    -------
    list of (float, float, str) — (shift_a, shift_b, label)
    """

    if lattice_type == "hexagonal":
        return [(0.0,       0.0,       "AA"),
                (1.0 / 3.0, 2.0 / 3.0, "AB"),
                (2.0 / 3.0, 1.0 / 3.0, "AB_prime")]

    elif lattice_type == "square":
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB"),
                (0.5, 0.5, "AC")]

    else:
        # rectangular or oblique
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB_x"),
                (0.0, 0.5, "AB_y"),
                (0.5, 0.5, "AC")]


def write_twist_list(filepath, candidates, bottom_file, top_file):
    """Write all search candidates to TWIST_LIST.dat, sorted by strain.

    Format mirrors CellMatch's results.dat with an added Theta column.
    The header records the input filenames so 'generate' mode can verify
    the same files are used.

    Parameters
    ----------
    filepath     : str
    candidates   : list[dict]  — sorted by strain (ascending)
    bottom_file  : str
    top_file     : str
    """

    SEP = "-" * 109

    with open(filepath, 'w') as o:
        o.write("# TWIST_LIST generated by vaspTwist.py\n")
        o.write(f"# bottom = {bottom_file}\n")
        o.write(f"# top    = {top_file}\n")
        o.write(f"# Total candidates: {len(candidates)}\n")
        o.write(f"# {'---':->52} RESULTS {'---':->49}\n")
        o.write(f"# {SEP}\n")
        o.write(f"# {'|':1}{'index':^7}{'|':1}{'theta (deg)':^18}{'|':1}{'strain':^18}{'|':1}"
                f"{'atoms':^9}{'|':1}{'surf_ratio':^12}{'|':1}"
                f"{'indices1':^23}{'|':1}{'indices2':^23}{'|':1}\n")
        o.write(f"# {SEP}\n")
        for no, c in enumerate(candidates, start=1):
            i11, i12, i21, i22 = c["indices1"]
            j11, j12, j21, j22 = c["indices2"]
            r1, r2 = c["area_ratio_bottom"], c["area_ratio_top"]
            o.write(f"| {no:>6}  |  {c['theta']:>14.8f}  |  {c['strain']:>14.8f}  |"
                    f"  {c['total_atoms']:>6}   |"
                    f"  {r1:>4}  {r2:>4}  |"
                    f"  {i11:>4} {i12:>4} {i21:>4} {i22:>4}  |"
                    f"  {j11:>4} {j12:>4} {j21:>4} {j22:>4}  |\n")
        o.write(f"# {SEP}\n")


def read_twist_list_header(filepath):
    """Read only the header of TWIST_LIST.dat to extract input filenames.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    bottom_file : str
    top_file    : str
    """

    bottom_file = None
    top_file    = None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("# bottom ="):
                bottom_file = line.split("=", 1)[1].strip()
            elif line.startswith("# top    ="):
                top_file = line.split("=", 1)[1].strip()
            if bottom_file is not None and top_file is not None:
                break
    return bottom_file, top_file


def write_output_list(filepath, output_records):
    """Write a summary of all generated POSCAR files to a plain-text list file.

    Each line contains the relative path to a POSCAR and its key parameters.

    Parameters
    ----------
    filepath       : str
    output_records : list[dict]
    """

    with open(filepath, 'w') as o:
        o.write("# POSCAR output list generated by vaspTwist.py\n")
        o.write(f"# Total POSCAR files: {len(output_records)}\n")
        o.write("#\n")
        o.write(f"# {'Theta(deg)':>10}  {'Atoms':>6}"
                f"  {'Stack':<12}  {'Strain':>12}"
                f"  {'indices1':>20}  {'indices2':>20}  {'Path'}\n")
        o.write("# " + "-" * 150 + "\n")
        for rec in output_records:
            i11, i12, i21, i22 = rec["indices1"]
            j11, j12, j21, j22 = rec["indices2"]
            o.write(f"  {rec['theta']:>10.4f}  {rec['total_atoms']:>6}"
                    f"  {rec['stack']:<12}  {rec['strain']:>12.8f}"
                    f"  {i11:>4} {i12:>4} {i21:>4} {i22:>4}"
                    f"  {j11:>4} {j12:>4} {j21:>4} {j22:>4}"
                    f"  {rec['path']}\n")


def display_candidates(candidates):
    """Print a numbered table of candidates."""

    print(f"\nFound {len(candidates)} candidate(s) with <= {MAX_ATOMS} atoms:\n")
    print(f"  {'No.':>4}  {'Theta (deg)':>12}  {'Atoms':>5}  {'Ratio1':>6}  {'Ratio2':>6}"
          f"  {'Strain':>14}"
          f"  {'|A1| (Ang)':>10}  {'|A2| (Ang)':>10}  {'Lattice':>12}  {'Stackings':>9}"
          f"  {'indices1':>20}  {'indices2':>20}")
    print("  " + "-" * 150)
    for index, c in enumerate(candidates):
        norm_A1      = np.linalg.norm(c["A1_vec"])
        norm_A2      = np.linalg.norm(c["A2_vec"])
        lattice_type = get_2d_lattice_type(c["bilayer_lattice_matrix"])
        n_stackings  = len(get_shift_grid(lattice_type))
        i11, i12, i21, i22 = c["indices1"]
        j11, j12, j21, j22 = c["indices2"]
        print(f"  {index + 1:>4}  {c['theta']:>12.4f}  {c['total_atoms']:>5}"
              f"  {c['area_ratio_bottom']:>6}  {c['area_ratio_top']:>6}"
              f"  {c['strain']:>14.8f}"
              f"  {norm_A1:>10.4f}  {norm_A2:>10.4f}  {lattice_type:>12}  {n_stackings:>9}"
              f"  {i11:>4} {i12:>4} {i21:>4} {i22:>4}"
              f"  {j11:>4} {j12:>4} {j21:>4} {j22:>4}")


def prompt_selection(candidates):
    """Prompt the user to select candidates to generate.

    Parameters
    ----------
    candidates : list[dict]

    Returns
    -------
    chosen : list[int] — zero-based indices of selected candidates
    """

    total_poscars = sum(
        len(get_shift_grid(get_2d_lattice_type(c["bilayer_lattice_matrix"])))
        for c in candidates
    )
    print(f"\n  Selecting 'all' will write {total_poscars} POSCAR(s) total.")
    print( "  Enter 'none' to finish without generating any POSCAR.\n")

    while True:
        raw = input("Enter candidate numbers to generate (e.g. 1 3 5), 'all', or 'none': ").strip()
        if raw.lower() == 'none':
            return []
        if raw.lower() == 'all':
            return list(range(len(candidates)))
        try:
            chosen = [int(x) - 1 for x in raw.split()]
            if all(0 <= i < len(candidates) for i in chosen):
                return chosen
            print(f"ERROR! Numbers must be between 1 and {len(candidates)}.")
        except ValueError:
            print("ERROR! Enter integers separated by spaces, 'all', or 'none'.")


def run_search(bottom, top, sort_elements, known_elements):
    """Run the full moiré vector search and candidate-building pipeline.

    Returns candidates sorted by strain (ascending), one per twist angle
    (keeping the smallest-atom candidate for each theta).

    Parameters
    ----------
    bottom, top   : dict        — from read_POSCAR
    sort_elements : list or None
    known_elements : list[str]

    Returns
    -------
    candidates : list[dict]  — sorted by strain
    """

    primitive_atoms = bottom["total_atoms"] + top["total_atoms"]
    n_max = max(1, int(np.ceil(np.sqrt(MAX_ATOMS / primitive_atoms))))
    print(f"Auto-detected N_MAX = {n_max}"
          f"  (MAX_ATOMS={MAX_ATOMS} / {primitive_atoms} primitive atoms, ceil(sqrt) → {n_max})")

    print(f"Searching moiré vectors ({THETA_MIN:.1f} to {THETA_MAX:.1f} deg, step = {THETA_STEP:.1f} deg)...")
    moire_vectors = find_moire_vectors(bottom["lattice_matrix"], top["lattice_matrix"],
                                       THETA_MIN, THETA_MAX, THETA_STEP, n_max)

    if len(moire_vectors) == 0:
        print("No commensurate moiré vectors found with the given parameters.")
        exit(0)

    print(f"Found {len(moire_vectors)} raw vector pair(s). Building candidates (<= {MAX_ATOMS} atoms)...\n")

    vectors_by_theta = {}
    for result in moire_vectors:
        theta    = result[0]
        n1, n2   = result[1], result[2]
        m1, m2   = result[3], result[4]
        rel_dist = result[5]
        v_bottom = np.array(result[6])
        v_top    = np.array(result[7])
        vectors_by_theta.setdefault(round(theta, 4), []).append(
            (rel_dist, v_bottom, v_top, n1, n2, m1, m2)
        )

    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        per_theta_results = pool.starmap(
            build_candidates_for_theta,
            [(theta_key, vec_list,
              bottom["lattice_matrix"], bottom["positions_cartesian"],
              bottom["species"], bottom["selective_dynamics"], bottom["flags"],
              top["lattice_matrix"], top["positions_cartesian"],
              top["species"], top["selective_dynamics"], top["flags"],
              sort_elements, known_elements)
             for theta_key, vec_list in vectors_by_theta.items()]
        )

    all_candidates = [c for sublist in per_theta_results for c in sublist]

    # Keep only the smallest-atom candidate per twist angle
    best_per_theta = {}
    for c in all_candidates:
        tk = c["theta"]
        if tk not in best_per_theta or c["total_atoms"] < best_per_theta[tk]["total_atoms"]:
            best_per_theta[tk] = c
    candidates = list(best_per_theta.values())

    if len(candidates) == 0:
        print(f"No candidates found with <= {MAX_ATOMS} atoms. Try relaxing MAX_STRAIN or MAX_ATOMS.")
        exit(0)

    # Sort by strain (ascending) — matches results.dat convention from CellMatch
    candidates.sort(key=lambda c: c["strain"])
    return candidates


def generate_poscars(chosen_indices, candidates, sort_elements, known_elements, working_dir):
    """Build and write POSCARs for the chosen candidates.

    Parameters
    ----------
    chosen_indices : list[int]  — zero-based indices into candidates
    candidates     : list[dict]
    sort_elements  : list or None
    known_elements : list[str]
    working_dir    : str

    Returns
    -------
    output_records : list[dict]
    """

    output_records = []
    written_count  = 0

    for list_no, index in enumerate(chosen_indices, start=1):
        candidate          = candidates[index]
        theta              = candidate["theta"]
        lattice_matrix     = candidate["bilayer_lattice_matrix"]
        total_atoms        = candidate["total_atoms"]
        ratio1             = candidate["area_ratio_bottom"]
        ratio2             = candidate["area_ratio_top"]
        strain             = candidate["strain"] * 100.0
        n_bottom           = candidate["n_bottom"]
        selective_dynamics = candidate["selective_dynamics"]

        elements_order, atom_counts = collect_elements_and_counts(
            candidate["bilayer_species"],
            sort_elements if sort_elements is not None else list(dict.fromkeys(known_elements))
        )

        lattice_type = get_2d_lattice_type(lattice_matrix)
        shifts       = get_shift_grid(lattice_type)

        base_dir = os.path.join(working_dir,
                                f"{list_no}_twist_{theta:.4f}deg_{total_atoms}atoms")
        os.makedirs(base_dir, exist_ok=True)

        print(f"\n  [No.{index + 1}] theta = {theta:.4f} deg | {total_atoms} atoms"
              f" | ratio1 = {ratio1}  ratio2 = {ratio2}"
              f" | strain = {strain:.4f}% | lattice = {lattice_type}")

        for i, (shift_a, shift_b, stack_label) in enumerate(shifts, start=1):

            # Apply stacking shift to top-layer atoms BEFORE mapping_elements.
            # n_bottom correctly identifies the boundary between bottom and top
            # atoms in the raw (layer-ordered) bilayer_positions array.
            # After mapping_elements, atoms are grouped by element, so n_bottom
            # would no longer identify the top layer correctly.
            raw_positions = candidate["bilayer_positions"].copy()
            raw_positions[n_bottom:, 0] = (raw_positions[n_bottom:, 0] + shift_a) % 1.0
            raw_positions[n_bottom:, 1] = (raw_positions[n_bottom:, 1] + shift_b) % 1.0

            raw_cartesian = direct_to_cartesian(lattice_matrix, raw_positions)

            mapping = mapping_elements(elements_order, atom_counts, raw_cartesian, raw_positions,
                                       candidate["bilayer_species"], selective_dynamics, candidate["bilayer_flags"],
                                       sort_elements)

            labels = define_labels(mapping["elements"], mapping["atom_counts"])

            output_dir  = os.path.join(base_dir, f"{i}_{stack_label}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "POSCAR")

            write_POSCAR(output_path, lattice_matrix, mapping["elements"], mapping["atom_counts"],
                         mapping["positions_direct"], selective_dynamics, mapping["flags"], labels)

            rel_path = os.path.relpath(output_path, working_dir)
            output_records.append({"path":        rel_path,
                                   "theta":       theta,
                                   "total_atoms": total_atoms,
                                   "stack":       stack_label,
                                   "strain":      strain,
                                   "indices1":    candidate["indices1"],
                                   "indices2":    candidate["indices2"]})

            print(f"    Written: {rel_path}  (stacking = {stack_label},"
                  f" shift = ({shift_a:.4f}, {shift_b:.4f}))")
            written_count += 1

    return output_records


def match_mode(bottom_file, top_file):
    """'match' mode: search all twist angles and write TWIST_LIST.dat."""

    working_dir = os.getcwd()
    bottom = read_POSCAR(bottom_file)
    top    = read_POSCAR(top_file) if top_file != bottom_file else bottom
    is_hetero = (top_file != bottom_file)

    if is_hetero:
        print(f"\nHeterobilayer match: {bottom_file}  +  {top_file}")
    else:
        print(f"\nHomobilayer match: {bottom_file}")

    bilayer_elements = bottom["elements"] + top["elements"]
    sort_elements    = check_elements(bilayer_elements)
    known_elements   = sort_elements if sort_elements is not None else bilayer_elements

    candidates = run_search(bottom, top, sort_elements, known_elements)

    display_candidates(candidates)

    match_list_path = os.path.join(working_dir, TWIST_LIST_FILE)
    write_twist_list(match_list_path, candidates, bottom_file, top_file)

    print(f"\nFound {len(candidates)} candidate(s). Results written to {TWIST_LIST_FILE}")
    print("Run 'vaspTwist.py generate' with the same input files to create POSCARs.\n")


def generate_mode(bottom_file, top_file):
    """'generate' mode: read TWIST_LIST.dat, prompt for selection, write POSCARs."""

    working_dir     = os.getcwd()
    match_list_path = os.path.join(working_dir, TWIST_LIST_FILE)

    # Verify TWIST_LIST.dat exists
    if not os.path.exists(match_list_path):
        print(f"\nERROR! {TWIST_LIST_FILE} not found in {working_dir}.")
        print("Run 'vaspTwist.py match' first to generate it.\n")
        exit(1)

    # Verify input files match what was used during match mode
    saved_bottom, saved_top = read_twist_list_header(match_list_path)
    if saved_bottom is None or saved_top is None:
        print(f"\nERROR! {TWIST_LIST_FILE} has no valid header. Please re-run match mode.\n")
        exit(1)

    if os.path.abspath(bottom_file) != os.path.abspath(saved_bottom) or \
       os.path.abspath(top_file)    != os.path.abspath(saved_top):
        print(f"\nWARNING! Input files do not match those recorded in {TWIST_LIST_FILE}.")
        print(f"  Recorded bottom : {saved_bottom}")
        print(f"  Provided bottom : {bottom_file}")
        print(f"  Recorded top    : {saved_top}")
        print(f"  Provided top    : {top_file}")
        print("Please re-run 'vaspTwist.py match' with the correct input files, or")
        print("use the same files that were used during match mode.\n")
        exit(1)

    # Re-run the search to reconstruct candidates in memory
    # (TWIST_LIST.dat stores display info; full candidate dicts are needed for POSCAR building)
    bottom = read_POSCAR(bottom_file)
    top    = read_POSCAR(top_file) if top_file != bottom_file else bottom
    is_hetero = (top_file != bottom_file)

    if is_hetero:
        print(f"\nHeterobilayer generate: {bottom_file}  +  {top_file}")
    else:
        print(f"\nHomobilayer generate: {bottom_file}")

    bilayer_elements = bottom["elements"] + top["elements"]
    sort_elements    = check_elements(bilayer_elements)
    known_elements   = sort_elements if sort_elements is not None else bilayer_elements

    candidates = run_search(bottom, top, sort_elements, known_elements)

    display_candidates(candidates)
    chosen_indices = prompt_selection(candidates)

    if len(chosen_indices) == 0:
        print("\nNo candidates selected. Finished without writing any POSCAR.\n")
        exit(0)

    output_records = generate_poscars(chosen_indices, candidates, sort_elements, known_elements, working_dir)

    twist_list_path = os.path.join(working_dir, TWIST_LIST_FILE)
    write_output_list(twist_list_path, output_records)

    print(f"\nFinished! Written {len(output_records)} POSCAR(s).")
    print(f"Output list written to {TWIST_LIST_FILE}\n")


def main():
    """Dispatch to match or generate mode based on first argument."""

    if '-h' in argv or '--help' in argv:
        usage()

    # Expected: vaspTwist.py <mode> <bottom> [top]
    if len(argv) not in (3, 4):
        usage()

    mode        = argv[1].lower()
    bottom_file = argv[2]
    top_file    = argv[3] if len(argv) == 4 else argv[2]

    if mode == "match":
        match_mode(bottom_file, top_file)
    elif mode == "generate":
        generate_mode(bottom_file, top_file)
    else:
        print(f"\nERROR! Unknown mode '{argv[1]}'. Use 'match' or 'generate'.\n")
        usage()


if __name__ == "__main__":
    main()
