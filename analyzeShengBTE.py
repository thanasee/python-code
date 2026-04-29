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


TENSOR_HEADER  = ("#  T(K)          xx          yy          zz"
                  "          yz          xz          xy")
FREQ_HEADER    = ("#  Frequency(THz)    xx            yy            zz"
                  "            yz            xz            xy")
MFP_HEADER     = ("#  MFP(nm)           xx            yy            zz"
                  "            yz            xz            xy")
GAMMA_HEADER   = "#  Frequency(THz)    Gamma(ps-1)"
TAU_HEADER     = "#  Frequency(THz)    Tau(ps)"
GRUN_HEADER    = "#  Frequency(THz)    Gruneisen"
P3_HEADER      = "#  Frequency(THz)    P3            P3_plus       P3_minus"
P4_HEADER      = "#  Frequency(THz)    P4            P4_plusplus   P4_plusminus  P4_minusminus"
 
 
def _fmt6(row):
    """Format 6 kappa tensor components."""
    return "".join(f"  {v:>12.4f}" for v in row)
 
 
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
 
 
def _load(directory, fname, label=""):
    """
    Load a whitespace-delimited ShengBTE plain-text file.
 
    Parameters
    ----------
    directory : str
    fname : str
        Basename of the file.
    label : str
        Label used in the warning message.
 
    Returns
    -------
    numpy.ndarray or None
    """
    path = os.path.join(directory, fname)
    if not os.path.isfile(path):
        tag = label if label else fname
        print(f"  WARNING: '{tag}' not found — skipping.")
        return None
    return np.loadtxt(path)
 
 
def read_omega(base_dir):
    """
    Read BTE.omega: angular frequencies (rad/ps), one per mode.

    BTE.omega is temperature-independent and lives in the root ShengBTE
    directory, not inside T*K/ subdirectories.

    The file contains one angular frequency per line (single column).
    If the file has multiple columns the last column is taken as the
    frequency, consistent with ShengBTE output conventions.

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,).
    """
    data = _load(base_dir, "BTE.omega", "BTE.omega")
    if data is None:
        return None
    data = np.atleast_1d(data)
    if data.ndim == 2:
        return data[:, -1]   # take frequency column (last) if multi-column
    return data
 
 
def read_kappa_mode(temp_dir):
    """
    Read per-band kappa from BTE.kappa (inside a T*K/ subdirectory).

    ShengBTE writes one row per phonon band per convergence iteration.
    The final n_bands rows (last block) contain the RTA or converged values.
    Columns: omega(rad/ps)  kappa_xx  yy  zz  yz  xz  xy

    BTE.kappa is the only per-mode kappa file produced by both ShengBTE and
    FourPhonon; there is no separate BTE.kappa_RTA or BTE.kappa_4ph file.

    Parameters
    ----------
    temp_dir : str

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes, 7).
    """
    data = _load(temp_dir, "BTE.kappa", "BTE.kappa")
    if data is not None:
        data = np.atleast_2d(data)
    return data
 
 
def read_gruneisen(base_dir):
    """
    Read BTE.gruneisen: Gruneisen parameter per mode.

    BTE.gruneisen is temperature-independent and lives in the root ShengBTE
    directory, not inside T*K/ subdirectories.

    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,).
    """
    data = _load(base_dir, "BTE.gruneisen", "BTE.gruneisen")
    if data is None:
        return None
    data = np.atleast_1d(data)
    return data[:, -1] if data.ndim == 2 else data
 
 
def read_scattering_rate(temp_dir, fourphonon=False):
    """
    Read the total 3-phonon scattering rate for each mode.

    ShengBTE (3ph only) writes the total RTA scattering rate to BTE.w.
    When FourPhonon is present, BTE.w_anharmonic is replaced by BTE.w_3ph;
    in that case BTE.w_3ph is read instead.

    Fallback chain (3ph-only):  BTE.w  ->  BTE.w_anharmonic
    Fallback chain (FourPhonon): BTE.w_3ph  ->  BTE.w_anharmonic

    Parameters
    ----------
    temp_dir : str
    fourphonon : bool
        When True, prefer BTE.w_3ph over BTE.w_anharmonic.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,), units ps-1.
    """
    if fourphonon:
        # FourPhonon replaces BTE.w_anharmonic with BTE.w_3ph
        p = os.path.join(temp_dir, "BTE.w_3ph")
        if os.path.isfile(p):
            d = np.loadtxt(p); d = np.atleast_1d(d)
            return d[:, -1] if d.ndim == 2 else d
        print("  WARNING: 'BTE.w_3ph' not found — skipping 3ph scattering rate.")
        return None
    else:
        # Pure ShengBTE: total RTA rate is BTE.w; fallback to BTE.w_anharmonic
        p = os.path.join(temp_dir, "BTE.w")
        if os.path.isfile(p):
            d = np.loadtxt(p); d = np.atleast_1d(d)
            return d[:, -1] if d.ndim == 2 else d
        fallback = os.path.join(temp_dir, "BTE.w_anharmonic")
        if os.path.isfile(fallback):
            print("  INFO: BTE.w not found — using BTE.w_anharmonic.")
            d = np.loadtxt(fallback); d = np.atleast_1d(d)
            return d[:, -1] if d.ndim == 2 else d
        print("  WARNING: 'BTE.w' not found — skipping 3ph scattering rate.")
        return None
 
 
def read_w4(temp_dir):
    """
    Read the total 4-phonon scattering rate from BTE.w_4ph (FourPhonon).

    FourPhonon writes the combined 4-phonon scattering rate to BTE.w_4ph.
    Channel-resolved rates (BTE.w_4ph_plusplus, BTE.w_4ph_plusminus,
    BTE.w_4ph_minusminus) are not extracted separately here; the total
    is sufficient for scattering rate and lifetime outputs.

    Parameters
    ----------
    temp_dir : str

    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,), units ps-1.
    """
    path = os.path.join(temp_dir, "BTE.w_4ph")
    if os.path.isfile(path):
        d = np.loadtxt(path); d = np.atleast_1d(d)
        return d[:, -1] if d.ndim == 2 else d
    print("  WARNING: 'BTE.w_4ph' not found — skipping 4ph scattering rate.")
    return None
 
 
def _read_scalar(directory, fname):
    """
    Read a single-line ShengBTE file that contains exactly one scalar value.
 
    BTE.P3_total, BTE.P3_plus_total, BTE.P3_minus_total, BTE.P4_total, etc.
    each contain a single number — the integrated phase space over all modes.
 
    Parameters
    ----------
    directory : str
    fname : str
 
    Returns
    -------
    float or None
    """
    path = os.path.join(directory, fname)
    if not os.path.isfile(path):
        return None
    try:
        data = np.loadtxt(path)
        return float(data.flat[0])
    except Exception:
        return None
 
 
def read_phase_space_3ph(base_dir):
    """
    Read all 3-phonon phase space data.
 
    All files are temperature-independent and live in base_dir.
 
    Per-mode arrays (shape (n_modes,)):
      BTE.P3, BTE.P3_plus, BTE.P3_minus
 
    Integrated scalar totals (single value):
      BTE.P3_total, BTE.P3_plus_total, BTE.P3_minus_total
 
    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.
 
    Returns
    -------
    dict
        Keys: 'p3', 'p3_plus', 'p3_minus'        -> ndarray or None
              'p3_total', 'p3_plus_total',
              'p3_minus_total'                      -> float or None
    """
    def _load1d(primary, *alts):
        """
        Try primary filename first, then each alt in order, silently.
        Only warn if all candidates are absent.
        Returns 1-D ndarray (last column if multi-column) or None.
        """
        for fname in (primary,) + alts:
            path = os.path.join(base_dir, fname)
            if os.path.isfile(path):
                d = np.atleast_1d(np.loadtxt(path))
                return d[:, -1] if d.ndim == 2 else d
        tried = ', '.join((primary,) + alts)
        print(f"  WARNING: none of [{tried}] found — skipping.")
        return None

    return {
        'p3'             : _load1d("BTE.P3"),
        'p3_plus'        : _load1d("BTE.P3_plus"),
        'p3_minus'       : _load1d("BTE.P3_minus"),
        'p3_total'       : _read_scalar(base_dir, "BTE.P3_total"),
        'p3_plus_total'  : _read_scalar(base_dir, "BTE.P3_plus_total"),
        'p3_minus_total' : _read_scalar(base_dir, "BTE.P3_minus_total"),
    }
 
 
def read_phase_space_4ph(base_dir):
    """
    Read all 4-phonon phase space data (FourPhonon).
 
    All files are temperature-independent and live in base_dir.
 
    Per-mode arrays (shape (n_modes,)):
      BTE.P4, BTE.P4_plusplus, BTE.P4_plusminus, BTE.P4_minusminus
 
    Integrated scalar totals (single value):
      BTE.P4_total, BTE.P4_plusplus_total, BTE.P4_plusminus_total,
      BTE.P4_minusminus_total
 
    The three 4ph scattering channels are:
      plusplus   (++) -- recombination:    lambda + lambda' + lambda'' -> lambda'''
      plusminus  (+-) -- redistribution:  lambda + lambda' -> lambda'' + lambda'''
      minusminus (--) -- splitting:       lambda -> lambda' + lambda'' + lambda'''
 
    Parameters
    ----------
    base_dir : str
        ShengBTE root directory.
 
    Returns
    -------
    dict
        Keys: 'p4', 'p4_plusplus', 'p4_plusminus', 'p4_minusminus'
              -> ndarray or None
              'p4_total', 'p4_plusplus_total',
              'p4_plusminus_total', 'p4_minusminus_total'
              -> float or None
    """
    # Build a map of all BTE.P4* files actually present in base_dir,
    # stripping any whitespace from filenames to handle encoding quirks.
    p4_files = {}
    for f in glob.glob(os.path.join(base_dir, "BTE.P4*")):
        key = os.path.basename(f).strip()
        p4_files[key] = f

    def _load1d(primary, *alts):
        """
        Try primary filename first, then each alt in order, using the
        glob-based map to handle filename encoding or whitespace quirks.
        Only warn if all candidates are absent.
        Returns 1-D ndarray (last column if multi-column) or None.
        """
        for fname in (primary,) + alts:
            path = p4_files.get(fname) or os.path.join(base_dir, fname)
            if os.path.isfile(path):
                d = np.atleast_1d(np.loadtxt(path))
                return d[:, -1] if d.ndim == 2 else d
        tried = ', '.join((primary,) + alts)
        print(f"  WARNING: none of [{tried}] found — skipping.")
        print(f"    (BTE.P4* files found: {sorted(p4_files.keys()) or 'none'})")
        return None

    return {
        'p4'            : _load1d("BTE.P4"),
        'p4_plusplus'   : _load1d("BTE.P4_plusplus"),
        'p4_plusminus'  : _load1d("BTE.P4_plusminus"),
        'p4_minusminus' : _load1d("BTE.P4_minusminus"),
        'p4_total'            : _read_scalar(base_dir, "BTE.P4_total"),
        'p4_plusplus_total'   : _read_scalar(base_dir, "BTE.P4_plusplus_total"),
        'p4_plusminus_total'  : _read_scalar(base_dir, "BTE.P4_plusminus_total"),
        'p4_minusminus_total' : _read_scalar(base_dir, "BTE.P4_minusminus_total"),
    }
 
 
def read_cumulative_kappa_mfp(temp_dir):
    """
    Read BTE.cumulative_kappa_tensor: cumulative kappa vs. MFP.
 
    Columns: mfp(nm)  kappa_xx  yy  zz  yz  xz  xy
 
    Parameters
    ----------
    temp_dir : str
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_points, 7).
    """
    data = _load(temp_dir, "BTE.cumulative_kappa_tensor",
                 "BTE.cumulative_kappa_tensor")
    if data is not None:
        data = np.atleast_2d(data)
    return data
 
 
def read_cumulative_kappa_freq(temp_dir):
    """
    Read BTE.cumulative_kappaVsOmega_tensor: cumulative kappa vs. frequency.
 
    Columns: omega(rad/ps)  kappa_xx  yy  zz  yz  xz  xy
 
    Parameters
    ----------
    temp_dir : str
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_points, 7).
    """
    data = _load(temp_dir, "BTE.cumulative_kappaVsOmega_tensor",
                 "BTE.cumulative_kappaVsOmega_tensor")
    if data is not None:
        data = np.atleast_2d(data)
    return data
 
 
def read_kappa_tensor_vs_T(base_dir, method):
    """
    Read BTE.KappaTensorVsT_{method} from the root directory.

    Columns: T(K)  kappa_xx  yy  zz  yz  xz  xy

    np.loadtxt returns a 1-D array when the file contains only one
    temperature row (single-T run). ndmin=2 ensures the result is
    always 2-D so write_kappa_tensor_vs_T can iterate over rows safely.

    Parameters
    ----------
    base_dir : str
    method : str
        'RTA', 'CONV', or 'sg'.

    Returns
    -------
    numpy.ndarray or None
        Shape (n_temps, 7).
    """
    path = os.path.join(base_dir, f"BTE.KappaTensorVsT_{method}")
    if not os.path.isfile(path):
        print(f"  WARNING: 'BTE.KappaTensorVsT_{method}' not found — skipping.")
        return None
    return np.loadtxt(path, ndmin=2)


def write_kappa_tensor_vs_T(outpath, label, kappa_data):
    """
    Write thermal conductivity tensor vs. temperature.
 
    Parameters
    ----------
    outpath : str
    label : str
    kappa_data : numpy.ndarray
        Shape (n_temps, 7): T(K) + kappa_xx...xy  [W/(m K)].
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(TENSOR_HEADER + "\n")
        for row in kappa_data:
            o.write(f"{row[0]:>7.1f}" + _fmt6(row[1:7]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_kappa_mode(outpath, label, freq_THz, kappa_mode_data):
    """
    Write mode-resolved thermal conductivity vs. frequency.
 
    Parameters
    ----------
    outpath : str
    label : str
    freq_THz : numpy.ndarray
        Shape (n_modes,).
    kappa_mode_data : numpy.ndarray
        Shape (n_modes, 7): col0 = omega (replaced), cols 1-6 = kappa tensor.
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(FREQ_HEADER + "\n")
        for freq, row in zip(freq_THz, kappa_mode_data):
            o.write(f"{freq:>16.6f}" + _fmt6(row[1:7]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_gruneisen(outpath, freq_THz, gruneisen):
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
        for freq, g in zip(freq_THz, gruneisen):
            o.write(f"{freq:>16.6f}  {g:>14.6f}\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_scattering_rate(outpath, label, freq_THz, rates):
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
        for freq, g in zip(freq_THz, rates):
            o.write(f"{freq:>16.6f}  {g:>14.6e}\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_lifetime(outpath, label, freq_THz, rates):
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
        for freq, g in zip(freq_THz, rates):
            tau = 1.0 / g if g > 1e-30 else 0.0
            o.write(f"{freq:>16.6f}  {tau:>14.6e}\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def _scalar_line(label, value):
    """Format a scalar total as a header comment line."""
    if value is not None:
        return f"# {label} = {value:.6e}\n"
    return f"# {label} = N/A\n"
 
 
def write_phase_space_3ph(outpath, freq_THz, ps):
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
        for freq, tot, plus, minus in zip(freq_THz, p3_arr, pp_arr, pm_arr):
            o.write(f"{freq:>16.6f}  {tot:>14.6e}  {plus:>14.6e}  {minus:>14.6e}\n")
        o.write("\n")
    print(f"  Written: {outpath}")
 
 
def write_phase_space_4ph(outpath, freq_THz, ps):
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
        for freq, tot, pp, pm, mm in zip(freq_THz, p4_arr, pp_arr, ppm_arr, mm_arr):
            o.write(f"{freq:>16.6f}  {tot:>14.6e}  {pp:>14.6e}  {pm:>14.6e}  {mm:>14.6e}\n")
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
        Shape (n_points, 7): col0 = mfp(nm), cols 1-6 = kappa tensor.
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(MFP_HEADER + "\n")
        for row in data:
            o.write(f"{row[0]:>16.4f}" + _fmt6(row[1:7]) + "\n")
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
        Shape (n_points, 7): col0 = omega(rad/ps) replaced by freq_THz.
    """
    with open(outpath, 'w') as o:
        o.write(f"# {label}\n")
        o.write("# Units: W/(m K)\n")
        o.write(FREQ_HEADER + "\n")
        for freq, row in zip(freq_THz, data):
            o.write(f"{freq:>16.6f}" + _fmt6(row[1:7]) + "\n")
        o.write("\n")
    print(f"  Written: {outpath}")


def process_temperature(temp_K, temp_dir, base_dir, fourphonon, freq=None):
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
        re-read for this temperature. Used by main() to avoid reading the
        first temperature's BTE.omega twice.

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
    kappa_mode = read_kappa_mode(temp_dir)
    if kappa_mode is not None:
        label = f"Mode kappa (3ph+4ph RTA) at {temp_K} K" if fourphonon                 else f"Mode kappa (3ph RTA) at {temp_K} K"
        fname = "kappa_mode_4ph.dat" if fourphonon else "kappa_mode.dat"
        write_kappa_mode(
            os.path.join(temp_dir, fname),
            label, freq, kappa_mode)
 
    # ------------------------------------------------------------ Gruneisen
    gruneisen = read_gruneisen(base_dir)
    if gruneisen is not None:
        write_gruneisen(
            os.path.join(temp_dir, "gruneisen.dat"),
            freq, gruneisen)
 
    # ----------------------------------------- 3ph scattering rate & lifetime
    rates_3ph = read_scattering_rate(temp_dir, fourphonon=fourphonon)
    if rates_3ph is not None:
        write_scattering_rate(
            os.path.join(temp_dir, "scattering_rate_3ph.dat"),
            f"3-phonon scattering rate Gamma at {temp_K} K",
            freq, rates_3ph)
        write_lifetime(
            os.path.join(temp_dir, "lifetime_3ph.dat"),
            f"3-phonon lifetime at {temp_K} K",
            freq, rates_3ph)
 
    # ----------------------------------------- 4ph scattering rate & lifetime
    if fourphonon:
        rates_4ph = read_w4(temp_dir)
        if rates_4ph is not None:
            write_scattering_rate(
                os.path.join(temp_dir, "scattering_rate_4ph.dat"),
                f"4-phonon scattering rate Gamma at {temp_K} K",
                freq, rates_4ph)
            write_lifetime(
                os.path.join(temp_dir, "lifetime_4ph.dat"),
                f"4-phonon lifetime at {temp_K} K",
                freq, rates_4ph)
 
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
        ps3 = read_phase_space_3ph(base_dir)
        if any(ps3[k] is not None for k in ('p3', 'p3_plus', 'p3_minus')):
            write_phase_space_3ph(
                os.path.join(base_dir, "phase_space_3ph.dat"),
                freq_ref, ps3)
        if fourphonon:
            ps4 = read_phase_space_4ph(base_dir)
            if any(ps4[k] is not None for k in ('p4', 'p4_plusplus',
                                                 'p4_plusminus', 'p4_minusminus')):
                write_phase_space_4ph(
                    os.path.join(base_dir, "phase_space_4ph.dat"),
                    freq_ref, ps4)
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
