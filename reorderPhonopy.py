#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


FREQ_WINDOW = 0.5       # THz — maximum frequency difference allowed when matching modes
OVERLAP_TOL = 1.0e-6    # minimum improvement in overlap to update the best candidate


def usage():
    """
    Print usage information and exit.
    """
    print("""
Usage: reorderBand.py <band.yaml>

Reorder phonon branches in a Phonopy band.yaml file to remove
crossing artifacts caused by frequency-based branch sorting.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def compute_overlap(local_mode, normal_mode):
    """
    Compute the magnitude of the complex inner product between two eigenvectors.

    Each eigenvector is stored as a nested list of shape (n_atoms, 3, 2),
    where the last axis holds [real, imag] components. The inner product is
    evaluated as the Hermitian dot product summed over all atoms and Cartesian
    components.

    Parameters
    ----------
    local_mode : list of shape (n_atoms, 3, 2)
        Eigenvector at the current q-point. Each element is [real, imag].
    normal_mode : list of shape (n_atoms, 3, 2)
        Reference eigenvector from the previous q-point. Each element is [real, imag].

    Returns
    -------
    overlap : float
        Magnitude |<local|normal>| of the complex inner product.
        Ranges from 0 (orthogonal) to 1 (identical, if normalized).
    """
    real_part = 0.0
    imag_part = 0.0

    for lmode_atom, nmode_atom in zip(local_mode, normal_mode):
        for ldim, ndim in zip(lmode_atom, nmode_atom):
            real_part += ldim[0] * ndim[0] + ldim[1] * ndim[1]
            imag_part += ldim[1] * ndim[0] - ldim[0] * ndim[1]

    overlap = np.sqrt(real_part ** 2 + imag_part ** 2)
    return overlap


def reorder_modes(local_freqs, local_modes, normal_freqs, normal_modes):
    """
    Reorder phonon modes at a q-point to match the branch ordering of a reference q-point.

    For each reference (normal) mode, the best matching local mode is found by
    maximizing eigenvector overlap among candidates within a frequency window.
    Each local mode can be assigned to at most one reference mode (one-to-one matching).

    If no candidate passes the frequency window filter, the mode retains its
    original index as a silent fallback.

    Parameters
    ----------
    local_freqs : list of float
        Phonon frequencies (THz) at the current q-point to be reordered.
    local_modes : list of shape (n_modes, n_atoms, 3, 2)
        Eigenvectors at the current q-point.
    normal_freqs : list of float
        Reference phonon frequencies (THz) from the previous q-point.
    normal_modes : list of shape (n_modes, n_atoms, 3, 2)
        Reference eigenvectors from the previous q-point.

    Returns
    -------
    reordered_freqs : list of float
        Phonon frequencies reordered to match the reference branch ordering.
    reordered_modes : list of shape (n_modes, n_atoms, 3, 2)
        Eigenvectors reordered to match the reference branch ordering.
    """
    reordered_freqs = []
    reordered_modes = []
    claimed = []

    for j, nmode in enumerate(normal_modes):
        best_index   = j
        best_overlap = 0.0

        for k, lmode in enumerate(local_modes):
            if k in claimed:
                continue
            if np.fabs(normal_freqs[j] - local_freqs[k]) > FREQ_WINDOW:
                continue

            overlap = compute_overlap(lmode, nmode)
            if (overlap - best_overlap) > OVERLAP_TOL:
                best_index   = k
                best_overlap = overlap

        claimed.append(best_index)
        reordered_freqs.append(local_freqs[best_index])
        reordered_modes.append(local_modes[best_index])

    return reordered_freqs, reordered_modes


def read_band_yaml(filepath):
    """
    Parse a Phonopy band.yaml file and return reordered phonon frequencies.

    Eigenvector-based branch reordering is applied across all q-points using
    a sliding reference window: each q-point is matched against the already-
    reordered previous q-point. The first q-point on each continuous path
    segment is used as the initial reference.

    Parameters
    ----------
    filepath : str
        Path to the Phonopy band.yaml output file.

    Returns
    -------
    distances : np.ndarray of shape (n_qpoints,)
        Cumulative path distances along the reciprocal-space path (Angstrom^-1).
    frequencies : np.ndarray of shape (n_qpoints, n_bands)
        Reordered phonon frequencies (THz) at each q-point.
    segment_nqpoint : list of int
        Number of q-points in each path segment.
    """
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=Loader)
    
    if 'eigenvector' not in data['phonon'][0]['band'][0]:
        print("ERROR! The band.yaml file does not contain eigenvectors.")
        print("Please run Phonopy with the --eigenvectors flag and try again.")
        exit(1)

    raw_freqs   = []
    raw_modes   = []
    distances   = []

    for point in data['phonon']:
        distances.append(point['distance'])
        raw_freqs.append([b['frequency']   for b in point['band']])

        raw_modes.append([b['eigenvector'] for b in point['band']])

    ordered_freqs = []
    normal_freqs  = raw_freqs[0]
    normal_modes  = raw_modes[0]

    for i in range(len(raw_freqs)):
        if i == 0:
            ordered_freqs.append(raw_freqs[0])
            continue

        local_freqs = raw_freqs[i]
        local_modes = raw_modes[i]

        reordered_freqs, reordered_modes = reorder_modes(local_freqs, local_modes,
                                                         normal_freqs, normal_modes)

        ordered_freqs.append(reordered_freqs)
        normal_freqs = reordered_freqs
        normal_modes = reordered_modes

    return (
        np.array(distances),
        np.array(ordered_freqs),
        data['segment_nqpoint']
    )


def write_band_dat(distances, frequencies, segment_nqpoint):
    """
    Print reordered phonon band data to stdout in xmgrace format.

    Each phonon band is printed as a block of (distance, frequency) pairs.
    Blank lines separate path segments within a band; double blank lines
    separate consecutive bands.

    Parameters
    ----------
    distances : np.ndarray of shape (n_qpoints,)
        Cumulative path distances along the reciprocal-space path (Angstrom^-1).
    frequencies : np.ndarray of shape (n_qpoints, n_bands)
        Reordered phonon frequencies (THz) at each q-point.
    segment_nqpoint : list of int
        Number of q-points in each path segment.
    """
    with open('band.dat', 'w') as f:
        for band_freqs in frequencies.T:
            q = 0
            for nq in segment_nqpoint:
                for dis, freq in zip(distances[q:q + nq], band_freqs[q:q + nq]):
                    f.write("%f %f\n" % (dis, freq))
                q += nq
                f.write('\n')
            f.write('\n')


def main():
    """
    Parse arguments, read the band.yaml file, reorder phonon modes, and write the output.
    """
    if '-h' in argv or len(argv) != 2:
        usage()

    filepath = argv[1]

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    distances, frequencies, segment_nqpoint = read_band_yaml(filepath)
    write_band_dat(distances, frequencies, segment_nqpoint)


if __name__ == "__main__":
    main()
