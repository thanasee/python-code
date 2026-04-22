#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
from ase.io import read


def usage():
    """Print usage information and exit."""

    text = """
Usage: vaspVibration.py <POSCAR input> <input file> [scaling factor]

This script extracts vibrational modes and writes them in XSF format.
Supports both VASP (OUTCAR) and Phonopy (band.yaml/mesh.yaml) outputs with write eigenvectors.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


def read_structure(poscar_file):
    """Read atomic structure from a POSCAR file using ASE.

    Parameters
    ----------
    poscar_file : str
        Path to the POSCAR file.

    Returns
    -------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    """

    if not os.path.exists(poscar_file):
        print(f"ERROR!\nFile: {poscar_file} does not exist.")
        exit(0)

    return read(poscar_file)


def read_modes_outcar(outcar_file, structure):
    """Read vibrational frequencies and eigenvectors from a VASP OUTCAR file.

    Locates the 'Eigenvectors and eigenvalues of the dynamical matrix'
    section, extracts all normal mode frequencies and mass-unweighted
    Cartesian displacement vectors.

    Parameters
    ----------
    outcar_file : str
        Path to the OUTCAR file.
    structure : ase.Atoms
        ASE Atoms object, used to obtain atomic masses for mass-unweighting.

    Returns
    -------
    frequency : np.ndarray, shape (n_modes,)
        Vibrational frequencies in THz (2PiTHz convention from OUTCAR).
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Mass-unweighted Cartesian displacement vectors for each normal mode.
    """

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(0)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    total_ions = None
    for line in outcar_lines:
        if 'NIONS =' in line:
            total_ions = int(line.split()[-1])
            break

    if total_ions is None:
        print("ERROR!\n'NIONS' not found in OUTCAR.")
        exit(0)

    index_start = None
    frequency_index = []
    for i, line in enumerate(outcar_lines):
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in line:
            index_start = i + 2
        if '2PiTHz' in line:
            frequency_index.append(i)

    if index_start is None or len(frequency_index) == 0:
        print("ERROR!\nEigenvector block not found in OUTCAR.")
        exit(0)

    index_stop = frequency_index[-1] + total_ions + 2

    frequency = np.array(
        [line.split()[-8] for line in outcar_lines[index_start:index_stop] if '2PiTHz' in line],
        dtype=float
    )
    modes = [line.split()[3:6] for line in outcar_lines[index_start:index_stop]
             if ('dx' not in line) and ('2PiTHz' not in line)]
    modes = np.array([m for m in modes if len(m) > 0], dtype=float)
    modes = modes.reshape((-1, total_ions, 3))

    # Mass-unweight: OUTCAR stores mass-weighted eigenvectors; divide by sqrt(m)
    modes /= np.sqrt(structure.get_masses()[None, :, None])

    return frequency, modes


def read_modes_phonopy(yaml_file):
    """Read vibrational frequencies and eigenvectors from a Phonopy YAML file.

    Parses band.yaml or mesh.yaml by string splitting. Only the first
    q-point (index 0) is used. Only the real part of each eigenvector
    component is extracted; the imaginary part is discarded.

    Parameters
    ----------
    yaml_file : str
        Path to the Phonopy YAML file (band.yaml or mesh.yaml).

    Returns
    -------
    frequency : np.ndarray, shape (n_bands,)
        Phonon frequencies at q-point 0 (THz).
    modes : np.ndarray, shape (n_bands, n_atoms, 3)
        Real part of eigenvectors at q-point 0.
    """

    if not os.path.exists(yaml_file):
        print(f"ERROR!\nFile: {yaml_file} does not exist.")
        exit(0)

    with open(yaml_file, 'r') as f:
        content = f.read()

    q_points = content.split('phonon:')[1].split('q-position:')[1:]

    if len(q_points) == 0:
        print("ERROR!\nNo q-point data found in YAML file.")
        exit(0)

    bands_per_qpoint = []
    for qp in q_points:
        bands_per_qpoint.append(qp.split('frequency:')[1:])

    nbands  = len(bands_per_qpoint[0])
    natoms  = len(bands_per_qpoint[0][0].split('atom')[1:])

    # Extract frequencies and real eigenvector components at q-point 0
    frequency = np.array(
        [float(bands_per_qpoint[0][b].split('eigenvector')[0]) for b in range(nbands)],
        dtype=float
    )

    modes = np.zeros((nbands, natoms, 3), dtype=float)
    for b in range(nbands):
        atom_blocks = bands_per_qpoint[0][b].split('atom')[1:]
        for a in range(natoms):
            for d in range(3):
                modes[b, a, d] = float(atom_blocks[a].split('[')[d + 1].split(',')[0])

    return frequency, modes


def write_xsf_modes(structure, modes, scale, is_phonopy):
    """Write each vibrational mode to a separate XSF file.

    Each output file mode_N.xsf contains the crystal lattice vectors
    and atomic positions augmented with displacement vectors, suitable
    for visualization in VESTA or XCrySDen. Mode numbering follows
    Phonopy order (ascending) for Phonopy input and VASP order
    (descending, highest frequency first) for OUTCAR input.

    Parameters
    ----------
    structure : ase.Atoms
        ASE Atoms object containing the crystal structure.
    modes : np.ndarray, shape (n_modes, n_atoms, 3)
        Displacement vectors for each mode (mass-unweighted).
    scale : float
        Scaling factor applied to displacement vectors for visualization.
    is_phonopy : bool
        If True, number modes in ascending order (Phonopy convention).
        If False, number modes in descending order (VASP convention).
    """

    total_modes = modes.shape[0]
    total_atoms = len(structure)
    symbols     = structure.get_chemical_symbols()

    if modes.shape[1] != total_atoms:
        print("ERROR!\nShape mismatch between eigenvectors and atomic positions.")
        exit(0)

    for j in range(total_modes):
        vector = modes[j] * scale
        positions_vector = np.hstack((structure.positions, vector))

        mode_index  = j + 1 if is_phonopy else total_modes - j
        output_name = f"mode_{mode_index:d}.xsf"

        with open(output_name, 'w') as o:
            o.write("CRYSTAL\n")
            o.write("PRIMVEC\n")
            o.write("\n".join(
                ' '.join(f'{a:20.16f}' for a in lattice_row)
                for lattice_row in structure.cell
            ))
            o.write("\nPRIMCOORD\n")
            o.write(f"{total_atoms:3d} 1\n")
            o.write("\n".join(
                f'{symbols[k]:3s}' + ' '.join(f'{a:20.16f}' for a in positions_vector[k])
                for k in range(total_atoms)
            ))


def main():
    """Parse arguments, read inputs, and write vibrational mode XSF files."""

    if '-h' in argv or not (3 <= len(argv) <= 4):
        usage()

    poscar_file = argv[1]
    input_file  = argv[2]
    scale       = float(argv[3]) if len(argv) == 4 else 1.0

    structure   = read_structure(poscar_file)
    is_phonopy  = input_file.endswith(".yaml")

    if is_phonopy:
        _, modes = read_modes_phonopy(input_file)
    else:
        _, modes = read_modes_outcar(input_file, structure)

    write_xsf_modes(structure, modes, scale, is_phonopy)


if __name__ == '__main__':
    main()
