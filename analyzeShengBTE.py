#!/usr/bin/env python

from sys import argv, exit
import os
import re
import glob
import numpy as np


def usage():
    """Print usage information and exit."""
    text = """
Usage: analyzeShengBTE.py

This script extracts thermal transport data from ShengBTE (and FourPhonon) output files.
Run from the ShengBTE output directory.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


TENSOR_HEADER  = ("#  T(K)          xx          xy          xz"
                  "          yx          yy          yz"
                  "          zx          zy          zz")
FREQ_HEADER    = ("#  Frequency(THz)    xx            xy            xz"
                  "            yx            yy            yz"
                  "            zx            zy            zz")
MFP_HEADER     = ("#  MFP(nm)           xx            xy            xz"
                  "            yx            yy            yz"
                  "            zx            zy            zz")
GAMMA_HEADER   = "#  Frequency(THz)    Gamma(ps-1)"
TAU_HEADER     = "#  Frequency(THz)    Tau(ps)"
GRUN_HEADER    = "#  Frequency(THz)    Gruneisen"
P3_HEADER      = "#  Frequency(THz)    P3            P3_plus       P3_minus"
P4_HEADER      = "#  Frequency(THz)    P4            P4_plusplus   P4_plusminus  P4_minusminus"
 
 
def _fmt9(row):
    """Format 9 kappa tensor components (full 3x3 row-major)."""
    return "".join(f"  {v:>12.4f}" for v in row)

 
 
def _write_blocks(f, rows, band_indices=None):
    """
    Write rows to an open file handle, inserting a blank line between
    each phonon band.

    Band separation is detected from band_indices: a blank line is inserted
    whenever band_indices[i] differs from band_indices[i-1].
    All per-mode files share the same mode ordering as BTE.kappa, so the
    band index array from BTE.kappa col0 applies universally.

    Produces xmgrace-compatible multiset format (one dataset per phonon band).

    Parameters
    ----------
    f : file object
        Open file handle to write to.
    rows : list of str
        Pre-formatted data lines (without trailing newline).
    band_indices : array-like or None
        Integer band index per row (from BTE.kappa col0).
        If None, rows are written without separation.
    """
    for i, row in enumerate(rows):
        if band_indices is not None and i > 0 and band_indices[i] != band_indices[i - 1]:
            f.write("\n")
        f.write(row + "\n")


def _to_THz(omega_rad_ps):
    """Convert angular frequency rad/ps to THz."""
    return omega_rad_ps / (2.0 * np.pi)


def detect_temperature_dirs(base_dir):
    """
    Find all T<int>K subdirectories in base_dir.
 
    Parameters
    ----------
    base_dir : str
        Root ShengBTE output directory.
 
    Returns
    -------
    dict
        {temperature_K (int): absolute_path (str)}, sorted ascending.
    """
    dirs = {}
    for d in glob.glob(os.path.join(base_dir, "T*K")):
        m = re.search(r"T(\d+)K$", d)
        if m and os.path.isdir(d):
            dirs[int(m.group(1))] = d
    if not dirs:
        print(f"ERROR: No T*K subdirectories found in '{base_dir}'.")
        print("       Make sure you are pointing to the ShengBTE output directory.")
        exit(1)
    return dict(sorted(dirs.items()))
 
 
def select_temperatures(temp_dirs):
    """
    Ask the user which temperatures to process when more than one is found.
 
    Displays a numbered list and accepts: individual numbers, comma-separated
    numbers, ranges (e.g. 1-3), 'all', or 'enter' (defaults to all).
 
    Parameters
    ----------
    temp_dirs : dict
        {temperature_K (int): path (str)}.
 
    Returns
    -------
    dict
        Subset of temp_dirs selected by the user.
    """
    temps = sorted(temp_dirs.keys())
    if len(temps) == 1:
        return temp_dirs
 
    print(f"\nFound {len(temps)} temperatures:")
    for i, T in enumerate(temps, 1):
        print(f"  [{i:>2d}]  {T} K   ({temp_dirs[T]})")
 
    print("\nEnter temperature index/indices to process")
    print("  Examples:  2        (one temperature)")
    print("             1,3,5    (multiple)")
    print("             2-5      (range)")
    print("             all      (all temperatures, default)")
    raw = input("Selection [all]: ").strip()
 
    if not raw or raw.lower() == 'all':
        return temp_dirs
 
    selected_idx = set()
    for token in raw.replace(' ', '').split(','):
        if '-' in token:
            parts = token.split('-')
            try:
                lo, hi = int(parts[0]), int(parts[1])
                if lo > hi:
                    print(f"  WARNING: Invalid range '{token}' (start > end) — skipping.")
                else:
                    selected_idx.update(range(lo, hi + 1))
            except (ValueError, IndexError):
                print(f"  WARNING: Cannot parse range '{token}' — skipping.")
        else:
            try:
                selected_idx.add(int(token))
            except ValueError:
                print(f"  WARNING: Cannot parse '{token}' — skipping.")
 
    result = {}
    for idx in sorted(selected_idx):
        if 1 <= idx <= len(temps):
            T = temps[idx - 1]
            result[T] = temp_dirs[T]
        else:
            print(f"  WARNING: Index {idx} out of range — skipping.")
 
    if not result:
        print("  No valid selection — processing all temperatures.")
        return temp_dirs
    return result
 
 
def detect_fourphonon(base_dir, temp_dirs):
    """
    Return True if FourPhonon output files are present.
 
    Checks BTE.w4_* inside T*K/ subdirectories (temperature-dependent)
    and BTE.P4 in base_dir (temperature-independent).
 
    Parameters
    ----------
    base_dir : str
        ShengBTE root directory (checked for BTE.P4).
    temp_dirs : dict
        {temperature_K (int): path (str)} (checked for BTE.w4_*).
 
    Returns
    -------
    bool
    """
    # FourPhonon replaces BTE.w_anharmonic with BTE.w_3ph inside T*K/
    # and writes BTE.w_4ph for four-phonon rates.
    # BTE.P4 (temperature-independent) is also a reliable marker.
    markers_tdir = ["BTE.w_3ph", "BTE.w_4ph"]
    for tdir in temp_dirs.values():
        for m in markers_tdir:
            if os.path.isfile(os.path.join(tdir, m)):
                return True
    if os.path.isfile(os.path.join(base_dir, "BTE.P4")):
        return True
    return False
 
 

# ===========================================================================
# Low-level file I/O
# ===========================================================================

def _readcols(path, label=""):
    """
    Read a ShengBTE plain-text file into a 2-D NumPy array.

    Skips blank lines and lines beginning with '#'.
    Always returns a 2-D array regardless of the number of data rows,
    so single-row files are handled without special-casing.

    Parameters
    ----------
    path : str
        Absolute path to the file.
    label : str
        Name shown in warning messages when the file is absent or empty.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_rows, n_cols).  None if the file is missing or empty.
    """
    if not os.path.isfile(path):
        print(f"  WARNING: '{label or os.path.basename(path)}' not found — skipping.")
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        print(f"  WARNING: '{label or os.path.basename(path)}' is empty — skipping.")
        return None
    return np.array(rows)


def _read_scalar(directory, fname):
    """
    Read a single-value ShengBTE summary file (e.g. BTE.P3_total).

    Parameters
    ----------
    directory : str
    fname : str

    Returns
    -------
    float or None
    """
    data = _readcols(os.path.join(directory, fname))
    return float(data.flat[0]) if data is not None else None


# ===========================================================================
# Temperature-independent readers  (base_dir / root directory)
# ===========================================================================

def read_omega(base_dir):
    """
    Read BTE.omega: phonon angular frequencies (rad/ps).

    File layout: (n_qpoints, n_bands)
      Each row  = one irreducible q-point.
      Each col  = one phonon band.
    Flattened column-major ('F' order) so the resulting 1-D array has the
    same mode ordering as BTE.kappa (q-index changes first per band block).

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,), units rad/ps.
    """
    data = _readcols(os.path.join(base_dir, "BTE.omega"), "BTE.omega")
    if data is None:
        return None
    return data.flatten('F')


def read_band_idx(base_dir):
    """
    Derive the phonon band index for every mode from BTE.omega.

    BTE.omega has shape (n_qpoints, n_bands).  After column-major flattening
    the band index repeats n_qpoints times for each band:
      [0, 0, ..., 0,  1, 1, ..., 1,  ...,  n_bands-1, ...]

    Used by all per-mode writers to insert blank lines between bands
    (xmgrace multiset format).

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    numpy.ndarray or None
        Integer band indices, shape (n_modes,).
    """
    data = _readcols(os.path.join(base_dir, "BTE.omega"), "BTE.omega")
    if data is None:
        return None
    n_qpoints, n_bands = data.shape
    return np.repeat(np.arange(n_bands), n_qpoints)


def read_gruneisen(base_dir):
    """
    Read BTE.gruneisen: mode Gruneisen parameters.

    File layout: (n_qpoints, n_bands) — identical structure to BTE.omega.
    Flattened column-major to align with BTE.kappa mode ordering.

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    numpy.ndarray or None
        Gruneisen values, shape (n_modes,).
    """
    data = _readcols(os.path.join(base_dir, "BTE.gruneisen"), "BTE.gruneisen")
    if data is None:
        return None
    return data.flatten('F')


def read_phase_space_3ph(base_dir):
    """
    Read 3-phonon phase space data.

    All files are temperature-independent and live in base_dir.
    Per-mode files contain a single column (phase space value).
    Mode ordering is the same as BTE.omega; frequency axis is taken
    externally from BTE.omega.

    Files:
      BTE.P3, BTE.P3_plus, BTE.P3_minus          (per-mode, single col)
      BTE.P3_total, BTE.P3_plus_total,
      BTE.P3_minus_total                          (integrated scalar)

    3ph channels:
      plus  (+): absorption   lambda + lambda' -> lambda''
      minus (-): emission     lambda -> lambda' + lambda''

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    dict
        'p3', 'p3_plus', 'p3_minus'         -> ndarray (n_modes,) or None
        'p3_total', 'p3_plus_total',
        'p3_minus_total'                     -> float or None
    """
    def _col0(fname):
        data = _readcols(os.path.join(base_dir, fname), fname)
        return data[:, 0] if data is not None else None

    return {
        'p3'             : _col0("BTE.P3"),
        'p3_plus'        : _col0("BTE.P3_plus"),
        'p3_minus'       : _col0("BTE.P3_minus"),
        'p3_total'       : _read_scalar(base_dir, "BTE.P3_total"),
        'p3_plus_total'  : _read_scalar(base_dir, "BTE.P3_plus_total"),
        'p3_minus_total' : _read_scalar(base_dir, "BTE.P3_minus_total"),
    }


def read_phase_space_4ph(base_dir):
    """
    Read 4-phonon phase space data (FourPhonon).

    All files are temperature-independent and live in base_dir.
    Per-mode files contain a single column (phase space value).
    A glob-based lookup is used to tolerate filename whitespace quirks
    on Lustre filesystems.

    Files:
      BTE.P4, BTE.P4_plusplus, BTE.P4_plusminus,
      BTE.P4_minusminus                           (per-mode, single col)
      BTE.P4_total, BTE.P4_plusplus_total,
      BTE.P4_plusminus_total, BTE.P4_minusminus_total   (integrated scalar)

    4ph channels:
      plusplus   (++): recombination   lambda + lambda' + lambda'' -> lambda'''
      plusminus  (+-): redistribution  lambda + lambda' -> lambda'' + lambda'''
      minusminus (--): splitting       lambda -> lambda' + lambda'' + lambda'''

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    dict
        'p4', 'p4_plusplus', 'p4_plusminus', 'p4_minusminus'
                                             -> ndarray (n_modes,) or None
        'p4_total', 'p4_plusplus_total',
        'p4_plusminus_total', 'p4_minusminus_total'
                                             -> float or None
    """
    # Glob-based map strips whitespace to handle Lustre filename encoding issues
    p4_map = {os.path.basename(f).strip(): f
               for f in glob.glob(os.path.join(base_dir, "BTE.P4*"))}

    def _col0(fname):
        path = p4_map.get(fname, os.path.join(base_dir, fname))
        data = _readcols(path, fname)
        return data[:, 0] if data is not None else None

    return {
        'p4'                  : _col0("BTE.P4"),
        'p4_plusplus'         : _col0("BTE.P4_plusplus"),
        'p4_plusminus'        : _col0("BTE.P4_plusminus"),
        'p4_minusminus'       : _col0("BTE.P4_minusminus"),
        'p4_total'            : _read_scalar(base_dir, "BTE.P4_total"),
        'p4_plusplus_total'   : _read_scalar(base_dir, "BTE.P4_plusplus_total"),
        'p4_plusminus_total'  : _read_scalar(base_dir, "BTE.P4_plusminus_total"),
        'p4_minusminus_total' : _read_scalar(base_dir, "BTE.P4_minusminus_total"),
    }


def read_kappa_tensor_vs_T(base_dir, method):
    """
    Read BTE.KappaTensorVsT_{method}: total kappa tensor vs. temperature.

    ShengBTE writes the full 3x3 tensor in row-major order for each
    temperature.

    File columns: T(K)  k11  k12  k13  k21  k22  k23  k31  k32  k33
                  (10 columns total: T + 9 tensor components)

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.
    method : str
        'RTA', 'CONV', or 'sg'.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_temps, 10).
    """
    fname = f"BTE.KappaTensorVsT_{method}"
    return _readcols(os.path.join(base_dir, fname), fname)


# ===========================================================================
# Temperature-dependent readers  (T*K/ subdirectories)
# ===========================================================================

def read_kappa_mode(temp_dir, n_modes):
    """
    Read per-mode kappa tensor from BTE.kappa.

    BTE.kappa stores one row per mode per convergence iteration.
    Only the last n_modes rows (final iteration) are returned.

    File columns: band_index  k11  k12  k13  k21  k22  k23  k31  k32  k33
      col0  = integer band index (used for blank-line separation in output)
      cols 1-9 = full 3x3 kappa tensor in row-major order [W/(m K)]

    Frequency axis is taken externally from BTE.omega.

    Parameters
    ----------
    temp_dir : str
        T*K subdirectory.
    n_modes : int
        Total number of phonon modes (n_qpoints * n_bands).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or (None, None)
        (band_idx, tensor)
        band_idx shape (n_modes,) int
        tensor   shape (n_modes, 9)
    """
    data = _readcols(os.path.join(temp_dir, "BTE.kappa"), "BTE.kappa")
    if data is None:
        return None, None
    data = data[-n_modes:]                     # last iteration only
    return data[:, 0].astype(int), data[:, 1:]


def read_scattering_rate(temp_dir, base_dir, fourphonon=False):
    """
    Read phonon scattering rates.

    File columns (for all BTE.w* files): omega(rad/ps)  rate(ps-1)
    Only the rate column (col1) is returned; the frequency axis is taken
    externally from BTE.omega.

    Files and their locations:
      BTE.w              temp_dir   total RTA scattering rate
      BTE.w_anharmonic   temp_dir   3ph contribution (3ph-only runs)
      BTE.w_3ph          temp_dir   3ph contribution (FourPhonon runs)
      BTE.w_isotopic     base_dir   isotopic contribution (temperature-independent)
      BTE.w_final        temp_dir   total converged scattering rate

    Parameters
    ----------
    temp_dir : str
        T*K subdirectory.
    base_dir : str
        ShengBTE root directory (for BTE.w_isotopic).
    fourphonon : bool
        When True, read BTE.w_3ph instead of BTE.w_anharmonic.

    Returns
    -------
    dict
        'total'      -> ndarray (n_modes,) or None  [BTE.w]
        'anharmonic' -> ndarray (n_modes,) or None  [BTE.w_anharmonic or BTE.w_3ph]
        'isotopic'   -> ndarray (n_modes,) or None  [BTE.w_isotopic]
        'final'      -> ndarray (n_modes,) or None  [BTE.w_final]
        Units: ps-1.
    """
    def _rates(directory, fname):
        """Return rate column (col1) from a 2-column scattering rate file."""
        data = _readcols(os.path.join(directory, fname), fname)
        return data[:, 1] if data is not None else None

    anharmonic_file = "BTE.w_3ph" if fourphonon else "BTE.w_anharmonic"

    return {
        'total'      : _rates(temp_dir, "BTE.w"),
        'anharmonic' : _rates(temp_dir, anharmonic_file),
        'isotopic'   : _rates(base_dir, "BTE.w_isotopic"),
        'final'      : _rates(temp_dir, "BTE.w_final"),
    }


def read_w4(temp_dir):
    """
    Read 4-phonon scattering rates from BTE.w_4ph (FourPhonon).

    File columns: omega(rad/ps)  rate(ps-1)
    Only the rate column (col1) is returned; the frequency axis is taken
    externally from BTE.omega.

    Parameters
    ----------
    temp_dir : str
        T*K subdirectory.

    Returns
    -------
    numpy.ndarray or None
        Scattering rates, shape (n_modes,), units ps-1.
    """
    data = _readcols(os.path.join(temp_dir, "BTE.w_4ph"), "BTE.w_4ph")
    return data[:, 1] if data is not None else None


def read_cumulative_kappa_mfp(temp_dir):
    """
    Read BTE.cumulative_kappa_tensor: cumulative kappa vs. mean free path.

    File columns: mfp(nm)  k11  k12  k13  k21  k22  k23  k31  k32  k33
                  (10 columns total)

    Parameters
    ----------
    temp_dir : str

    Returns
    -------
    numpy.ndarray or None
        Shape (n_points, 10): col0=MFP(nm), cols 1-9 = 3x3 kappa tensor.
    """
    return _readcols(os.path.join(temp_dir, "BTE.cumulative_kappa_tensor"),
                     "BTE.cumulative_kappa_tensor")


def read_cumulative_kappa_freq(temp_dir):
    """
    Read BTE.cumulative_kappaVsOmega_tensor: cumulative kappa vs. frequency.

    File columns: omega(rad/ps)  k11  k12  k13  k21  k22  k23  k31  k32  k33
                  (10 columns total)

    Parameters
    ----------
    temp_dir : str

    Returns
    -------
    numpy.ndarray or None
        Shape (n_points, 10): col0=omega(rad/ps), cols 1-9 = 3x3 kappa tensor.
    """
    return _readcols(os.path.join(temp_dir, "BTE.cumulative_kappaVsOmega_tensor"),
                     "BTE.cumulative_kappaVsOmega_tensor")


def write_kappa_tensor_vs_T(outpath, label, kappa_data):
    """
    Write thermal conductivity tensor vs. temperature.
 
    Parameters
    ----------
    outpath : str
    label : str
    kappa_data : numpy.ndarray
        Shape (n_temps, 10): col0=T(K), cols 1-9 = 3x3 kappa tensor (row-major).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(TENSOR_HEADER + "\n")
        for row in kappa_data:
            o.write(f"{row[0]:>7.1f}" + _fmt9(row[1:10]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_kappa_mode(outpath, label, freq_THz, kappa_mode_data, band_indices=None):
    """
    Write mode-resolved thermal conductivity vs. frequency.

    Parameters
    ----------
    outpath : str
    label : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    kappa_mode_data : numpy.ndarray
        Shape (n_modes, 9): full 3x3 kappa tensor per mode (row-major).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(FREQ_HEADER + "\n")
        rows = [f"{freq:>16.6f}" + _fmt9(row)
                for freq, row in zip(freq_THz, kappa_mode_data)]
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_gruneisen(outpath, freq_THz, gruneisen, band_indices=None):
    """
    Write Gruneisen parameters vs. frequency.
 
    Parameters
    ----------
    outpath : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    gruneisen : numpy.ndarray
        Shape (n_modes,).
    """
    with open(outpath, 'w') as o:
        o.write("# Gruneisen parameter per mode\n")
        o.write(GRUN_HEADER + "\n")
        rows = [f"{freq:>16.6f}  {g:>14.6f}"
                for freq, g in zip(freq_THz, gruneisen)]
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_scattering_rate(outpath, label, freq_THz, rates, band_indices=None):
    """
    Write phonon scattering rates (Gamma) vs. frequency.
 
    Parameters
    ----------
    outpath : str
    label : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    rates : numpy.ndarray
        Scattering rates in ps-1, shape (n_modes,).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: ps-1\n")
        o.write(GAMMA_HEADER + "\n")
        rows = [f"{freq:>16.6f}  {g:>14.6e}"
                for freq, g in zip(freq_THz, rates)]
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_lifetime(outpath, label, freq_THz, rates, band_indices=None):
    """
    Write phonon lifetimes (tau = 1/Gamma) vs. frequency.
 
    Modes with zero or negative scattering rate (acoustic at Gamma) are
    written as tau = 0.0.
 
    Parameters
    ----------
    outpath : str
    label : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    rates : numpy.ndarray
        Scattering rates in ps-1, shape (n_modes,).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: ps\n")
        o.write(TAU_HEADER + "\n")
        rows = []
        for freq, g in zip(freq_THz, rates):
            tau = 1.0 / g if g > 1e-30 else 0.0
            rows.append(f"{freq:>16.6f}  {tau:>14.6e}")
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def _scalar_line(label, value):
    """Format a scalar total as a header comment line."""
    if value is not None:
        return f"# {label} = {value:.6e}\n"
    return f"# {label} = N/A\n"
 
 
def write_phase_space_3ph(outpath, freq_THz, ps, band_indices=None):
    """
    Write 3-phonon phase space vs. frequency.
 
    The integrated totals from BTE.P3_*_total are written as comment lines
    in the file header. The per-mode columns are P3, P3_plus, P3_minus.
 
    3ph channels:
      plus  (+): absorption  lambda + lambda' -> lambda''
      minus (-): emission    lambda -> lambda' + lambda''
 
    Parameters
    ----------
    outpath : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    ps : dict
        Output of read_phase_space_3ph(), containing per-mode arrays
        and scalar totals.
    """
    n = len(freq_THz)
    p3_arr  = ps['p3']       if ps['p3']       is not None else np.zeros(n)
    pp_arr  = ps['p3_plus']  if ps['p3_plus']  is not None else np.zeros(n)
    pm_arr  = ps['p3_minus'] if ps['p3_minus'] is not None else np.zeros(n)
 
    with open(outpath, 'w') as o:
        o.write("# 3-phonon phase space per mode\n")
        o.write("# plus  (+): absorption  lambda + lambda' -> lambda''\n")
        o.write("# minus (-): emission    lambda -> lambda' + lambda''\n")
        o.write(_scalar_line("P3_total",       ps['p3_total']))
        o.write(_scalar_line("P3_plus_total",  ps['p3_plus_total']))
        o.write(_scalar_line("P3_minus_total", ps['p3_minus_total']))
        if ps['p3']       is None: o.write("# WARNING: BTE.P3 not found — P3 column is zero-filled\n")
        if ps['p3_plus']  is None: o.write("# WARNING: BTE.P3_plus not found — P3_plus column is zero-filled\n")
        if ps['p3_minus'] is None: o.write("# WARNING: BTE.P3_minus not found — P3_minus column is zero-filled\n")
        o.write(P3_HEADER + "\n")
        rows = [f"{freq:>16.6f}  {tot:>14.6e}  {plus:>14.6e}  {minus:>14.6e}"
                for freq, tot, plus, minus in zip(freq_THz, p3_arr, pp_arr, pm_arr)]
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_phase_space_4ph(outpath, freq_THz, ps, band_indices=None):
    """
    Write 4-phonon phase space vs. frequency (FourPhonon).
 
    The integrated totals from BTE.P4_*_total are written as comment lines
    in the file header. The per-mode columns are P4, P4_plusplus, P4_plusminus,
    P4_minusminus.
 
    4ph channels:
      plusplus   (++): recombination   lambda + lambda' + lambda'' -> lambda'''
      plusminus  (+-): redistribution  lambda + lambda' -> lambda'' + lambda'''
      minusminus (--): splitting       lambda -> lambda' + lambda'' + lambda'''
 
    Parameters
    ----------
    outpath : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    ps : dict
        Output of read_phase_space_4ph(), containing per-mode arrays
        and scalar totals.
    """
    n = len(freq_THz)
    p4_arr  = ps['p4']            if ps['p4']            is not None else np.zeros(n)
    pp_arr  = ps['p4_plusplus']   if ps['p4_plusplus']   is not None else np.zeros(n)
    ppm_arr = ps['p4_plusminus']  if ps['p4_plusminus']  is not None else np.zeros(n)
    mm_arr  = ps['p4_minusminus'] if ps['p4_minusminus'] is not None else np.zeros(n)
 
    with open(outpath, 'w') as o:
        o.write("# 4-phonon phase space per mode (FourPhonon)\n")
        o.write("# plusplus   (++): recombination   lambda + lambda' + lambda'' -> lambda'''\n")
        o.write("# plusminus  (+-): redistribution  lambda + lambda' -> lambda'' + lambda'''\n")
        o.write("# minusminus (--): splitting       lambda -> lambda' + lambda'' + lambda'''\n")
        o.write(_scalar_line("P4_total",            ps['p4_total']))
        o.write(_scalar_line("P4_plusplus_total",   ps['p4_plusplus_total']))
        o.write(_scalar_line("P4_plusminus_total",  ps['p4_plusminus_total']))
        o.write(_scalar_line("P4_minusminus_total", ps['p4_minusminus_total']))
        if ps['p4']            is None: o.write("# WARNING: BTE.P4 not found — P4 column is zero-filled\n")
        if ps['p4_plusplus']   is None: o.write("# WARNING: BTE.P4_plusplus not found — P4_plusplus column is zero-filled\n")
        if ps['p4_plusminus']  is None: o.write("# WARNING: BTE.P4_plusminus not found — P4_plusminus column is zero-filled\n")
        if ps['p4_minusminus'] is None: o.write("# WARNING: BTE.P4_minusminus not found — P4_minusminus column is zero-filled\n")
        o.write(P4_HEADER + "\n")
        rows = [f"{freq:>16.6f}  {tot:>14.6e}  {pp:>14.6e}  {pm:>14.6e}  {mm:>14.6e}"
                for freq, tot, pp, pm, mm in zip(freq_THz, p4_arr, pp_arr, ppm_arr, mm_arr)]
        _write_blocks(o, rows, band_indices=band_indices)
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_cumulative_kappa_mfp(outpath, label, data):
    """
    Write cumulative kappa vs. mean free path.
 
    Parameters
    ----------
    outpath : str
    label : str
    data : numpy.ndarray
        Shape (n_points, 10): col0=mfp(nm), cols 1-9 = 3x3 kappa tensor (row-major).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(MFP_HEADER + "\n")
        for row in data:
            o.write(f"{row[0]:>16.4f}" + _fmt9(row[1:10]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_cumulative_kappa_freq(outpath, label, freq_THz, data):
    """
    Write cumulative kappa vs. frequency.
 
    Parameters
    ----------
    outpath : str
    label : str
    freq_THz : numpy.ndarray
        Shape (n_points,).
    data : numpy.ndarray
        Shape (n_points, 10): col0=omega(rad/ps), cols 1-9 = 3x3 kappa tensor (row-major).
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(FREQ_HEADER + "\n")
        for freq, row in zip(freq_THz, data):
            o.write(f"{freq:>16.6f}" + _fmt9(row[1:10]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")


def process_temperature(temp_K, temp_dir, base_dir, fourphonon,
                        freq=None):
    """
    Extract all available ShengBTE (and FourPhonon) quantities for one
    temperature and write the corresponding .dat files.

    Parameters
    ----------
    temp_K : int
        Temperature in Kelvin.
    temp_dir : str
        Absolute path to the T*K subdirectory.
    base_dir : str
        ShengBTE root directory (for temperature-independent files:
        BTE.omega, BTE.gruneisen).
    fourphonon : bool
        Whether FourPhonon output was detected.
    freq : numpy.ndarray or None, optional
        Pre-computed frequency array (THz). When provided, BTE.omega is not
        re-read for this temperature.


    Notes
    -----
    Phase space (temperature-independent) is written once in main(),
    not inside this function.
    """
    print(f"\n  T = {temp_K} K")

    # ----------------------------------------------------------------- omega
    if freq is None:
        omega = read_omega(base_dir)
        if omega is None:
            print(f"  BTE.omega missing — skipping T = {temp_K} K entirely.")
            return
        freq = _to_THz(omega)
 
    # --------------------------------------------------- mode kappa (per band)
    # BTE.kappa contains per-band contributions; single file for both 3ph and
    # 3ph+4ph runs — labelled accordingly based on fourphonon flag.
    band_idx_kappa = read_band_idx(base_dir)
    _, kappa_mode  = read_kappa_mode(
        temp_dir, n_modes=len(freq) if freq is not None else 0)
    if kappa_mode is not None:
        label = f"Mode kappa (3ph+4ph RTA) at {temp_K} K" if fourphonon \
                else f"Mode kappa (3ph RTA) at {temp_K} K"
        fname = "kappa_mode_4ph.dat" if fourphonon else "kappa_mode.dat"
        write_kappa_mode(
            os.path.join(temp_dir, fname),
            label, freq, kappa_mode, band_indices=band_idx_kappa)
    # ------------------------------------------------------------ Gruneisen
    gruneisen = read_gruneisen(base_dir)
    if gruneisen is not None:
        write_gruneisen(
            os.path.join(temp_dir, "gruneisen.dat"),
            freq, gruneisen, band_indices=band_idx_kappa)
 
    # ----------------------------------------- 3ph scattering rate & lifetime
    rates_dict_3ph = read_scattering_rate(temp_dir, base_dir, fourphonon=fourphonon)
    rates_3ph = rates_dict_3ph['total']
    if rates_3ph is not None:
        write_scattering_rate(
            os.path.join(temp_dir, "scattering_rate_3ph.dat"),
            f"3-phonon scattering rate Gamma at {temp_K} K",
            freq_3ph, rates_3ph)
        write_lifetime(
            os.path.join(temp_dir, "lifetime_3ph.dat"),
            f"3-phonon lifetime at {temp_K} K",
            freq_3ph, rates_3ph)
 
    # ----------------------------------------- 4ph scattering rate & lifetime
    if fourphonon:
        rates_4ph = read_w4(temp_dir)
        if rates_4ph is not None:
            write_scattering_rate(
                os.path.join(temp_dir, "scattering_rate_4ph.dat"),
                f"4-phonon scattering rate Gamma at {temp_K} K",
                freq_4ph, rates_4ph)
            write_lifetime(
                os.path.join(temp_dir, "lifetime_4ph.dat"),
                f"4-phonon lifetime at {temp_K} K",
                freq, rates_4ph, band_indices=band_idx_kappa)
 
    # ----------------------------------------- cumulative kappa vs. MFP
    cum_mfp = read_cumulative_kappa_mfp(temp_dir)
    if cum_mfp is not None:
        write_cumulative_kappa_mfp(
            os.path.join(temp_dir, "cumulative_kappa_mfp.dat"),
            f"Cumulative kappa vs. MFP at {temp_K} K",
            cum_mfp)
    # ----------------------------------------- cumulative kappa vs. frequency
    cum_freq_data = read_cumulative_kappa_freq(temp_dir)
    if cum_freq_data is not None:
        freq_cum = _to_THz(cum_freq_data[:, 0])
        write_cumulative_kappa_freq(
            os.path.join(temp_dir, "cumulative_kappa_freq.dat"),
            f"Cumulative kappa vs. frequency at {temp_K} K",
            freq_cum, cum_freq_data)


def main():
    if len(argv) > 1 and '-h' not in argv:
        print(f"ERROR: unexpected argument '{argv[1]}' — this script takes no arguments.")
        usage()
    if '-h' in argv:
        usage()
 
    base_dir   = os.path.abspath(os.getcwd())
    all_tdirs  = detect_temperature_dirs(base_dir)
    sel_tdirs  = select_temperatures(all_tdirs)
    fourphonon = detect_fourphonon(base_dir, sel_tdirs)
 
    print(f"\nShengBTE directory : {base_dir}")
    print(f"Temperatures       : {sorted(sel_tdirs.keys())} K")
    print(f"FourPhonon         : {'YES' if fourphonon else 'NO'}")
 
    # ------------------------------------------------ global kappa vs. T
    print("\n--- Kappa tensor vs. T ---")
    kappa_rta = read_kappa_tensor_vs_T(base_dir, "RTA")
    if kappa_rta is not None:
        write_kappa_tensor_vs_T(
            os.path.join(base_dir, "kappa_tensor_RTA.dat"),
            "Thermal conductivity tensor vs. T (3ph RTA)",
            kappa_rta)
 
    kappa_conv = read_kappa_tensor_vs_T(base_dir, "CONV")
    if kappa_conv is not None:
        write_kappa_tensor_vs_T(
            os.path.join(base_dir, "kappa_tensor_CONV.dat"),
            "Thermal conductivity tensor vs. T (3ph iterative)",
            kappa_conv)
 

 
    # ------------------------------------------------ phase space (once, temperature-independent)
    print("\n--- Phase space ---")
    omega_ref  = read_omega(base_dir)
    if omega_ref is not None:
        freq_ref = _to_THz(omega_ref)
        # Band index from BTE.omega shape — same ordering as all per-mode files
        band_idx_kappa_ref = read_band_idx(base_dir)
        ps3 = read_phase_space_3ph(base_dir)
        if any(ps3[k] is not None for k in ('p3', 'p3_plus', 'p3_minus')):
            write_phase_space_3ph(
                os.path.join(base_dir, "phase_space_3ph.dat"),
                freq_ref, ps3, band_indices=band_idx_kappa_ref)
        if fourphonon:
            ps4 = read_phase_space_4ph(base_dir)
            if any(ps4[k] is not None for k in ('p4', 'p4_plusplus',
                                                 'p4_plusminus', 'p4_minusminus')):
                write_phase_space_4ph(
                    os.path.join(base_dir, "phase_space_4ph.dat"),
                    freq_ref, ps4, band_indices=band_idx_kappa_ref)
    else:
        print("  WARNING: BTE.omega not found — phase space output skipped.")
 
    # ------------------------------------------------ per-temperature files
    print("\n--- Per-temperature quantities ---")
    for T in sorted(sel_tdirs.keys()):
        # Pass freq_ref for the first temperature to avoid re-reading BTE.omega
        pre_freq = freq_ref if (omega_ref is not None and T == sorted(sel_tdirs.keys())[0]) else None
        process_temperature(T, sel_tdirs[T], base_dir, fourphonon, freq=pre_freq)
 
    print("\nDone.")


if __name__ == "__main__":
    main()
