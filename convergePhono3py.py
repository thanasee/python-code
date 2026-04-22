#!/usr/bin/env python

from sys import argv, exit
import os, re
import numpy as np
import h5py as h5


def usage():
    """Print usage information and exit."""
    text = """
Usage: convergePhono3py.py

This script obtain lattice thermal conductivity depends on q-mesh from HDF5 files

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


def find_kappa_files():
    """Find and return a sorted list of phono3py kappa HDF5 files in the current directory.

    Scans the current directory for files matching the pattern 'kappa-m*.hdf5',
    which are the default output files from phono3py thermal conductivity calculations.
    Files are sorted in ascending order by the mesh number extracted from the filename
    (e.g., kappa-m20.hdf5 < kappa-m30.hdf5 < kappa-m40.hdf5).

    Returns
    -------
    list of str
        Sorted list of matching filenames.

    Exits
    -----
    Exits with an error message if no matching files are found.
    """

    files = [f for f in os.listdir() if f.startswith("kappa-m") and f.endswith(".hdf5")]
    if not files:
        print("ERROR! kappa hdf5 files are not found.")
        exit(0)
    return sorted(files, key=mesh_number)


def mesh_number(filename):
    """Extract the integer mesh number from a kappa-mXXX.hdf5 filename."""
    match = re.search(r"kappa-m(\d+)\.hdf5", filename)
    return int(match.group(1)) if match else float('inf')


def load(f, key):
    """Load a dataset from an open HDF5 file and suppress numerical noise."""
    if key not in f:
        return None
    arr = np.array(f[key])
    return np.where(arr < 1e-12, 0.0, arr)


def read_HDF5(filepath):
    """Read thermal conductivity data from a phono3py HDF5 output file.

    Loads the q-mesh, temperature grid, and all available kappa arrays from the file.
    Values below 1e-12 are set to zero to suppress numerical noise.
    Keys not present in the file are returned as None.

    Supported phono3py calculation modes and their corresponding keys:
        --br / --lbte          : 'kappa', 'kappa_RTA'
        --wigner --br or --lbte: 'kappa_C'
        --wigner --br          : 'kappa_P_RTA', 'kappa_TOT_RTA'
        --wigner --lbte        : 'kappa_P_exact', 'kappa_TOT_exact'

    Parameters
    ----------
    filepath : str
        Path to the phono3py HDF5 file (e.g., 'kappa-m20.hdf5').

    Returns
    -------
    dict
        Dictionary with keys: 'mesh', 'temperature', 'kappa', 'kappa_RTA',
        'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA', 'kappa_P_exact', 'kappa_TOT_exact'.
        Each value is a numpy array or None if not present in the file.
    """

    with h5.File(filepath, 'r') as f:
        data = {'mesh'           : np.array(f["mesh"])        if 'mesh'        in f else None,
                'temperature'    : np.array(f["temperature"]) if 'temperature' in f else None,
                # --br / --lbte
                'kappa'          : load(f, 'kappa'),
                'kappa_RTA'      : load(f, 'kappa_RTA'),
                # --wigner --br or --wigner --lbte
                'kappa_C'        : load(f, 'kappa_C'),
                # --wigner --br
                'kappa_P_RTA'    : load(f, 'kappa_P_RTA'),
                'kappa_TOT_RTA'  : load(f, 'kappa_TOT_RTA'),
                # --wigner --lbte
                'kappa_P_exact'  : load(f, 'kappa_P_exact'),
                'kappa_TOT_exact': load(f, 'kappa_TOT_exact')}

    return data


def validate(data, filepath):
    """Check that the HDF5 file contains at least one usable set of kappa variables.

    The script requires either:
        - 'kappa' (from --lbte), or
        - the complete Wigner-RTA set: 'kappa_C', 'kappa_P_RTA', 'kappa_TOT_RTA'
          (from --wigner --br)

    Parameters
    ----------
    data : dict
        Data dictionary returned by read_HDF5().
    filepath : str
        Filename used in the error message if validation fails.

    Exits
    -----
    Exits with an error message if neither condition is satisfied.
    """
    k   = data['kappa']
    kC  = data['kappa_C']
    kPR = data['kappa_P_RTA']
    kTR = data['kappa_TOT_RTA']
    if k is None and (kC is None or kPR is None or kTR is None):
        print(f"ERROR! Essential variables are not found in {filepath}.")
        exit(0)


def choose_temperature(temperature, filepath, last_temperature, current_temp):
    """Select the target temperature for a given file.

    If the temperature grid of the current file differs from the previous file
    (or this is the first file), the user is prompted to choose a temperature.
    If the grid is unchanged, the previously chosen temperature is reused silently.
    If the file contains only one temperature, it is selected automatically.

    Parameters
    ----------
    temperature : numpy.ndarray
        Temperature grid from the current HDF5 file.
    filepath : str
        Filename shown in the prompt when asking the user to choose.
    last_temperature : numpy.ndarray or None
        Temperature grid from the previous file, or None for the first file.
    current_temp : float or None
        The temperature chosen for the previous file, reused if grid is unchanged.

    Returns
    -------
    float
        The selected target temperature in Kelvin.
    """
    if last_temperature is None or not np.array_equal(temperature, last_temperature):
        if len(temperature) == 1:
            return temperature[0]
        else:
            print(f"\n[{filepath}] List of temperatures:")
            print("   ".join(map(str, temperature)))
            return float(input("Choose temperature: "))

    return current_temp


def get_temp_index(temperature, target_temp, filepath):
    """Return the array index corresponding to the target temperature.

    Parameters
    ----------
    temperature : numpy.ndarray
        Temperature grid from the current HDF5 file.
    target_temp : float
        The target temperature in Kelvin.
    filepath : str
        Filename used in the error message if the temperature is not found.

    Returns
    -------
    int
        Index of target_temp in the temperature array.

    Exits
    -----
    Exits with an error message if target_temp is not found in the array.
    """
    index = np.where(temperature == target_temp)[0]
    if len(index) == 0:
        print(f"ERROR! Temperature {target_temp} not found in {filepath}.")
        exit(0)
    return index[0]


def extract_row(mesh, kappa_arr, temp_index):
    """Extract one data row for a given temperature from a kappa array.

    Builds a tuple containing the three q-mesh dimensions followed by the
    six independent components of the thermal conductivity tensor at the
    specified temperature index.

    Parameters
    ----------
    mesh : numpy.ndarray
        Array of three integers [Q_x, Q_y, Q_z] representing the q-mesh dimensions.
    kappa_arr : numpy.ndarray
        Kappa array of shape (n_temperatures, 6), where the 6 components are
        xx, yy, zz, yz, xz, xy.
    temp_index : int
        Index of the target temperature in the kappa array.

    Returns
    -------
    tuple of float
        (Q_x, Q_y, Q_z, xx, yy, zz, yz, xz, xy)
    """
    return (mesh[0], mesh[1], mesh[2],
            kappa_arr[temp_index, 0], kappa_arr[temp_index, 1], kappa_arr[temp_index, 2],
            kappa_arr[temp_index, 3], kappa_arr[temp_index, 4], kappa_arr[temp_index, 5])


def write_dat(filename, rows, display=False):
    """Write kappa-vs-mesh data to a .dat file and optionally print to stdout.

    Each row contains the q-mesh dimensions (Q_x, Q_y, Q_z) and the six
    thermal conductivity tensor components (xx, yy, zz, yz, xz, xy) in W/m-K.

    Parameters
    ----------
    filename : str
        Output filename (e.g., 'KappaVsMesh.dat').
    rows : numpy.ndarray
        2D array of shape (n_meshes, 9) where each row is
        (Q_x, Q_y, Q_z, xx, yy, zz, yz, xz, xy).
    display : bool, optional
        If True, also print the filename, header, and data to stdout.
        Default is False.
    """
    header1 = "# Thermal conductivity(W/m-K) vs Q-Mesh"
    header2 = "#  Q_x   Q_y   Q_z        xx          yy          zz          yz          xz          xy"
    row_fmt = (" {0:>5.0f} {1:>5.0f} {2:>5.0f}"
               "  {3:>10.3f}  {4:>10.3f}  {5:>10.3f}  {6:>10.3f}  {7:>10.3f}  {8:>10.3f}")

    with open(filename, 'w') as o:
        o.write(header1 + "\n")
        o.write(header2 + "\n")
        for item in rows:
            o.write(row_fmt.format(*item) + "\n")
        o.write("\n")

    if display:
        print(filename)
        print(header1)
        print(header2)
        for item in rows:
            print(row_fmt.format(*item))
        print("")


def main():
    """Parses arguments, sort kappa HDF5 files, and write outputs."""
    if '-h' in argv or len(argv) != 1:
        usage()

    files = find_kappa_files()

    # Accumulators: one list of row-tuples per kappa type
    collected = {'kappa'          : [],
                 'kappa_RTA'      : [],
                 'kappa_C'        : [],
                 'kappa_P_RTA'    : [],
                 'kappa_TOT_RTA'  : [],
                 'kappa_P_exact'  : [],
                 'kappa_TOT_exact': []}

    last_temperature = None
    target_temp      = None

    for filepath in files:
        data        = read_HDF5(filepath)
        validate(data, filepath)
        temperature = data['temperature']
        target_temp = choose_temperature(temperature, filepath, last_temperature, target_temp)
        last_temperature = temperature

        temp_index = get_temp_index(temperature, target_temp, filepath)
        mesh       = data['mesh']

        for key in collected:
            if data[key] is not None:
                collected[key].append(extract_row(mesh, data[key], temp_index))

    # Output configuration: filename and whether to display to stdout
    output_map = {'kappa'          : ('KappaVsMesh.dat',           True),
                  'kappa_RTA'      : ('Kappa_RTAVsMesh.dat',       False),
                  'kappa_C'        : ('Kappa_CVsMesh.dat',         False),
                  'kappa_P_RTA'    : ('Kappa_P_RTAVsMesh.dat',     True),
                  'kappa_TOT_RTA'  : ('Kappa_TOT_RTAVsMesh.dat',   False),
                  'kappa_P_exact'  : ('Kappa_P_exactVsMesh.dat',   False),
                  'kappa_TOT_exact': ('Kappa_TOT_exactVsMesh.dat', False)}

    for key, (filename, display) in output_map.items():
        rows = collected[key]
        if rows:
            write_dat(filename, np.array(rows), display=display)


if __name__ == "__main__":
    main()
