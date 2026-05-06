#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    """Print usage information and exit."""
    print("""
Usage: getQPATH.py <band.dat input>

This script read second line in band.dat file
which generated from phonopy-bandplot --gnuplot command
and write QLINES.dat in same format with KLINES.dat from VASPKIT

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_band_dat(filepath):
    """
    Read q-point path positions and frequency data from a phonopy band.dat file.
 
    Parameters
    ----------
    filepath : str
        Path to the band.dat file produced by ``phonopy-bandplot --gnuplot``.
 
    Returns
    -------
    q_points : numpy.ndarray, shape (N,)
        Q-path distances (1/Angstrom) extracted from the second line of the file.
    fmin : float
        Floor of the minimum frequency (THz) found in the data columns.
    fmax : float
        Ceiling of the maximum frequency (THz) found in the data columns.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
 
    q_points = np.array([float(x) for x in lines[1].split()[1:]])
 
    freqs = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            freqs.append(float(parts[1]))
        except ValueError:
            pass
 
    if not freqs:
        raise ValueError("No frequency data found in input file (2nd column).")
 
    fmin = np.floor(min(freqs))
    fmax = np.ceil(max(freqs))
 
    return q_points, fmin, fmax


def write_QLINES(q_points, fmin, fmax, output_path="QLINES.dat"):
    """
    Write the q-path boundary file QLINES.dat.
 
    For each interior high-symmetry q-point, three lines are written that
    trace a vertical tick from ``fmin`` up to ``fmax`` and back down.
    The outer box and the zero-frequency axis are appended at the end.
 
    Parameters
    ----------
    q_points : numpy.ndarray, shape (N,)
        Q-path distances (1/Angstrom) of the high-symmetry points.
    fmin : float
        Lower frequency boundary (THz).
    fmax : float
        Upper frequency boundary (THz).
    output_path : str, optional
        Destination file path (default: ``"QLINES.dat"``).
    """
    with open(output_path, 'w') as o:
        o.write("#Q-Path(1/A) Frequency-Window(THz)\n")
 
        # Left edge baseline
        o.write(f"{q_points[0]:12.8f}{fmin:13.6f}\n")
 
        # Vertical ticks at interior high-symmetry points
        for x in q_points[1:-1]:
            o.write(f"{x:12.8f}{fmin:13.6f}\n")
            o.write(f"{x:12.8f}{fmax:13.6f}\n")
            o.write(f"{x:12.8f}{fmin:13.6f}\n")
 
        # Outer box: right edge then top and back to origin
        o.write(f"{q_points[-1]:12.8f}{fmin:13.6f}\n")
        o.write(f"{q_points[-1]:12.8f}{fmax:13.6f}\n")
        o.write(f"{q_points[0]:12.8f}{fmax:13.6f}\n")
        o.write(f"{q_points[0]:12.8f}{fmin:13.6f}\n")
 
        # Zero-frequency axis
        o.write(f"{q_points[0]:12.8f}{0.0:13.6f}\n")
        o.write(f"{q_points[-1]:12.8f}{0.0:13.6f}\n")
 
    print(f"Written: {output_path}")


def main():
    """Parse arguments, read band.dat, write QLINES.dat."""
    if '-h' in argv or len(argv) != 2:
        usage()
 
    input_file = argv[1]
    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(1)
 
    q_points, fmin, fmax = read_band_dat(input_file)
    write_QLINES(q_points, fmin, fmax)
 
 
if __name__ == '__main__':
    main()
