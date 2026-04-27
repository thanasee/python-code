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
    Find all T-<int>K subdirectories in base_dir.
 
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
    for d in glob.glob(os.path.join(base_dir, "T-*K")):
        m = re.search(r"T-(\d+)K$", d)
        if m and os.path.isdir(d):
            dirs[int(m.group(1))] = d
    if not dirs:
        print(f"ERROR: No T-*K subdirectories found in '{base_dir}'.")
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
 
    Checks BTE.w4_* inside T-*K/ subdirectories (temperature-dependent)
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
    markers_tdir = ["BTE.w4_total", "BTE.w4_anharmonic", "BTE.w4_isotopic"]
    for tdir in temp_dirs.values():
        for m in markers_tdir:
            if os.path.isfile(os.path.join(tdir, m)):
                return True
        if glob.glob(os.path.join(tdir, "BTE.w4_*")):
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
 
 
def read_omega(temp_dir):
    """
    Read BTE.omega: angular frequencies (rad/ps), one per mode.
 
    Parameters
    ----------
    temp_dir : str
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,).
    """
    return _load(temp_dir, "BTE.omega", "BTE.omega")
 
 
def read_kappa_mode(temp_dir, suffix):
    """
    Read per-mode kappa from BTE.kappa_{suffix}.
 
    ShengBTE columns: omega(rad/ps)  kappa_xx  yy  zz  yz  xz  xy
 
    Parameters
    ----------
    temp_dir : str
    suffix : str
        'RTA', 'CONV', or '4ph'.
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes, 7).
    """
    return _load(temp_dir, f"BTE.kappa_{suffix}", f"BTE.kappa_{suffix}")
 
 
def read_gruneisen(temp_dir):
    """
    Read BTE.gruneisen: Gruneisen parameter per mode.
 
    Parameters
    ----------
    temp_dir : str
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,).
    """
    return _load(temp_dir, "BTE.gruneisen", "BTE.gruneisen")
 
 
def read_scattering_rate(temp_dir, suffix):
    """
    Read 3ph phonon scattering rates (linewidths) from BTE.w_{suffix}.
 
    When suffix is 'total' and BTE.w_total is absent, falls back silently to
    BTE.w_anharmonic and prints a single informational message.
 
    Parameters
    ----------
    temp_dir : str
    suffix : str
        'total', 'anharmonic', or 'isotopic'.
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,), units ps-1.
    """
    path = os.path.join(temp_dir, f"BTE.w_{suffix}")
    if os.path.isfile(path):
        return np.loadtxt(path)
    if suffix == "total":
        fallback = os.path.join(temp_dir, "BTE.w_anharmonic")
        if os.path.isfile(fallback):
            print("  INFO: BTE.w_total not found — using BTE.w_anharmonic.")
            return np.loadtxt(fallback)
    print(f"  WARNING: 'BTE.w_{suffix}' not found — skipping.")
    return None
 
 
def read_w4(temp_dir, suffix="total"):
    """
    Read 4ph scattering rates from BTE.w4_{suffix} (FourPhonon).
 
    When suffix is 'total' and BTE.w4_total is absent, falls back silently to
    BTE.w4_anharmonic and prints a single informational message.
 
    Parameters
    ----------
    temp_dir : str
    suffix : str
 
    Returns
    -------
    numpy.ndarray or None
        Shape (n_modes,), units ps-1.
    """
    path = os.path.join(temp_dir, f"BTE.w4_{suffix}")
    if os.path.isfile(path):
        return np.loadtxt(path)
    if suffix == "total":
        fallback = os.path.join(temp_dir, "BTE.w4_anharmonic")
        if os.path.isfile(fallback):
            print("  INFO: BTE.w4_total not found — using BTE.w4_anharmonic.")
            return np.loadtxt(fallback)
    print(f"  WARNING: 'BTE.w4_{suffix}' not found — skipping.")
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
    return {
        'p3'             : _load(base_dir, "BTE.P3",             "BTE.P3"),
        'p3_plus'        : _load(base_dir, "BTE.P3_plus",        "BTE.P3_plus"),
        'p3_minus'       : _load(base_dir, "BTE.P3_minus",       "BTE.P3_minus"),
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
    return {
        'p4'                  : _load(base_dir, "BTE.P4",                  "BTE.P4"),
        'p4_plusplus'         : _load(base_dir, "BTE.P4_plusplus",         "BTE.P4_plusplus"),
        'p4_plusminus'        : _load(base_dir, "BTE.P4_plusminus",        "BTE.P4_plusminus"),
        'p4_minusminus'       : _load(base_dir, "BTE.P4_minusminus",       "BTE.P4_minusminus"),
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
    return _load(temp_dir, "BTE.cumulative_kappa_tensor",
                 "BTE.cumulative_kappa_tensor")
 
 
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
    return _load(temp_dir, "BTE.cumulative_kappaVsOmega_tensor",
                 "BTE.cumulative_kappaVsOmega_tensor")
 
 
def read_kappa_tensor_vs_T(base_dir, method):
    """
    Read BTE.KappaTensorVsT_{method} from the root directory.
 
    Columns: T(K)  kappa_xx  yy  zz  yz  xz  xy
 
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
    return _load(base_dir, f"BTE.KappaTensorVsT_{method}",
                 f"BTE.KappaTensorVsT_{method}")


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


def process_temperature(temp_K, temp_dir, fourphonon, freq=None):
    """
    Extract all available ShengBTE (and FourPhonon) quantities for one
    temperature and write the corresponding .dat files.
 
    Parameters
    ----------
    temp_K : int
        Temperature in Kelvin.
    temp_dir : str
        Absolute path to the T-*K subdirectory.
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
        omega = read_omega(temp_dir)
        if omega is None:
            print(f"  BTE.omega missing — skipping T = {temp_K} K entirely.")
            return
        freq = _to_THz(omega)
 
    # ------------------------------------------------------ mode kappa (3ph)
    kappa_rta = read_kappa_mode(temp_dir, "RTA")
    if kappa_rta is not None:
        write_kappa_mode(
            os.path.join(temp_dir, "kappa_mode.dat"),
            f"Mode kappa (3ph RTA) at {temp_K} K",
            freq, kappa_rta)
 
    # ------------------------------------------------------ mode kappa (4ph)
    if fourphonon:
        kappa_4ph = read_kappa_mode(temp_dir, "4ph")
        if kappa_4ph is not None:
            write_kappa_mode(
                os.path.join(temp_dir, "kappa_mode_4ph.dat"),
                f"Mode kappa (3ph+4ph RTA) at {temp_K} K",
                freq, kappa_4ph)
 
    # ------------------------------------------------------------ Gruneisen
    gruneisen = read_gruneisen(temp_dir)
    if gruneisen is not None:
        write_gruneisen(
            os.path.join(temp_dir, "gruneisen.dat"),
            freq, gruneisen)
 
    # ----------------------------------------- 3ph scattering rate & lifetime
    rates_3ph = read_scattering_rate(temp_dir, "total")
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
        rates_4ph = read_w4(temp_dir, "total")
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
 
    if fourphonon:
        kappa_4ph = read_kappa_tensor_vs_T(base_dir, "4ph")
        if kappa_4ph is not None:
            write_kappa_tensor_vs_T(
                os.path.join(base_dir, "kappa_tensor_4ph.dat"),
                "Thermal conductivity tensor vs. T (3ph+4ph RTA)",
                kappa_4ph)
 
    # ------------------------------------------------ phase space (once, temperature-independent)
    print("\n--- Phase space ---")
    first_tdir = sel_tdirs[sorted(sel_tdirs.keys())[0]]
    omega_ref  = read_omega(first_tdir)
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
        process_temperature(T, sel_tdirs[T], fourphonon, freq=pre_freq)
 
    print("\nDone.")


if __name__ == "__main__":
    main()
