#!/usr/bin/env python

from sys import argv, exit
import os
import subprocess
import numpy as np
import h5py as h5


def usage():
    """Print usage information and exit."""
    text = """
Usage: analyzePhono3py.py <kappa input> <gruneisen input (optional)>

This script obtain thermal properties form HDF5 files
some kind of thermal properties obtain at specified temperature

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


def read_kappa(file, key, threshold=None, abs_value=False):
    """Read a dataset from an open HDF5 file and optionally suppress numerical noise.

    Parameters
    ----------
    file      : h5py.File
        Open HDF5 file object.
    key       : str
        Dataset name to read.
    threshold : float or None
        If given, values satisfying the condition are set to zero.
        When abs_value=False : condition is values < threshold  (positive noise)
        When abs_value=True  : condition is |values| < threshold (±noise)
    abs_value : bool
        Use absolute-value comparison for the threshold condition.

    Returns
    -------
    numpy.ndarray or None
        Array of dataset values with noise suppressed, or None if key not found.
    """
    if key not in file:
        return None
    values = file[key][()]
    if threshold is not None:
        condition = np.abs(values) < threshold if abs_value else values < threshold
        return np.where(condition, 0.0, values)
    return values

# ── Column headers (preserved exactly from original script) ───────────────────
TENSOR_HEADER     = "#  T(K)        xx          yy          zz          yz          xz          xy"
FREQ_HEADER       = "#  Frequency(THz) xx            yy             zz           yz            xz            xy"
MFP_HEADER        = "#        MFP(A)       xx            yy            zz            yz            xz            xy"
CUM_FREQ_HEADER   = "#  Frequency(THz) xx            yy            zz            yz            xz            xy"
CUM_MFP_HEADER    = "#        MFP(A)       xx            yy            zz            yz            xz            xy"
GV_HEADER         = "#  Frequency(THz)    x               y               z"
GV_AMP_HEADER     = "#  Frequency(THz)    |v|"
GAMMA_HEADER      = "#  Frequency(THz) Gamma(ps-1)"
TAU_HEADER        = "#  Frequency(THz)    Tau(ps)"
CV_HEADER         = "#  T(K)    Cv(eV/K)"
GRUN_HEADER       = "#  Frequency(THz) Gruneisen"
TAU_TENSOR_HEADER = "#  T(K)           xx            yy            zz            yz            xz            xy"
TAU_TEMP_HEADER   = "#  T(K)          Tau(ps)"
PP_HEADER         = "#   Frequency(THz)  P3(eV^2)"


def write_tensor_vs_temperature(filepath, label, kappa_arr, temperature):
    """Write a 6-component kappa tensor (xx yy zz yz xz xy) vs temperature.

    Parameters
    ----------
    filepath    : str
        Output file path.
    label       : str
        Description written as the first comment line.
    kappa_arr   : ndarray, shape (nT, 6)
        Thermal conductivity tensor components for each temperature.
    temperature : ndarray, shape (nT,)
        Temperature values in K.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        o.write(TENSOR_HEADER + "\n")
        for temp, k in zip(temperature, kappa_arr):
            o.write(f"{temp:>7.1f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}\n")
        o.write("\n")


def write_band_tensor_vs_temperature(filepath, label, kappa_branch, temperature, note=None):
    """Write per-phonon-branch kappa tensor vs temperature.

    The first 3 branches are written individually (acoustic modes).
    All remaining optical branches are summed into a single entry before calling
    this function via compute_kappa_variant.

    Parameters
    ----------
    filepath     : str
        Output file path.
    label        : str
        Description written as the first comment line.
    kappa_branch : ndarray, shape (nT, nBand, 6)
        Per-branch thermal conductivity tensor.
    temperature  : ndarray, shape (nT,)
        Temperature values in K.
    note         : str or None
        Optional second comment line (e.g. "Sum all optical branch to one").
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        if note:
            o.write(f"# {note}\n")
        for band_index in range(kappa_branch.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TENSOR_HEADER + "\n")
            for temp, k in zip(temperature, kappa_branch[:, band_index, :]):
                o.write(f"{temp:>7.1f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}\n")
            o.write("\n")


def write_contribution_vs_temperature(filepath, label, contribute_all, temperature, frequency):
    """Write per-mode percentage contribution to kappa vs temperature.

    Parameters
    ----------
    filepath       : str
        Output file path.
    label          : str
        Description written as the first comment line.
    contribute_all : ndarray, shape (nT, nBand, 6)
        Percentage contribution of each phonon mode to the total kappa tensor.
    temperature    : ndarray, shape (nT,)
        Temperature values in K.
    frequency      : ndarray, shape (nQpt, nBand)
        Phonon frequencies; used only to determine the number of bands.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TENSOR_HEADER + "\n")
            for temp, k in zip(temperature, contribute_all[:, band_index, :]):
                o.write(f"{temp:>7.1f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}\n")
            o.write("\n")


def write_mode_vs_frequency(filepath, label, frequency, mode_weight, temp_index):
    """Write modal kappa (W/m-K per mode) vs phonon frequency at one temperature.

    Parameters
    ----------
    filepath    : str
        Output file path.
    label       : str
        Description written as the first comment line.
    frequency   : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    mode_weight : ndarray, shape (nT, nQpt, nBand, 6)
        Weight-normalized modal kappa tensor.
    temp_index  : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(FREQ_HEADER + "\n")
            for freq, k in zip(frequency[:, band_index],
                               mode_weight[temp_index, :, band_index, :]):
                o.write(f" {freq:>10.5f}  {k[0]:>12.5f}  {k[1]:>12.5f}  {k[2]:>12.4f}  {k[3]:>12.5f}  {k[4]:>12.5f}  {k[5]:>12.5f}\n")
        o.write("\n")


def write_mode_vs_mfp(filepath, label, mean_freepath, mode_weight, temp_index):
    """Write modal kappa (W/m-K per mode) vs mean free path at one temperature.

    Parameters
    ----------
    filepath      : str
        Output file path.
    label         : str
        Description written as the first comment line.
    mean_freepath : ndarray, shape (nT, nQpt, nBand)
        Phonon mean free path in Angstrom.
    mode_weight   : ndarray, shape (nT, nQpt, nBand, 6)
        Weight-normalized modal kappa tensor.
    temp_index    : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(mean_freepath.shape[2]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(MFP_HEADER + "\n")
            for mfp, k in zip(mean_freepath[temp_index, :, band_index],
                              mode_weight[temp_index, :, band_index, :]):
                o.write(f" {mfp:>14.4f}  {k[0]:>12.5f}  {k[1]:>12.5f}  {k[2]:>12.5f}  {k[3]:>12.5f}  {k[4]:>12.5f}  {k[5]:>12.5f}\n")
        o.write("\n")


def write_cumulative_vs_frequency(filepath, label, sorted_frequency, cum_data, temp_index):
    """Write cumulative kappa vs frequency (sorted ascending) at one temperature.

    Both the cumulative kappa and its spectral derivative (d kappa / d freq)
    are written using this same function — the caller passes the appropriate array.

    Parameters
    ----------
    filepath         : str
        Output file path.
    label            : str
        Description written as the first comment line.
    sorted_frequency : ndarray, shape (nQpt * nBand,)
        Flattened phonon frequencies sorted in ascending order (THz).
    cum_data         : ndarray, shape (nT, nQpt * nBand, 6)
        Cumulative (or derivative) kappa tensor, sorted to match sorted_frequency.
    temp_index       : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        o.write(CUM_FREQ_HEADER + "\n")
        for freq, ck in zip(sorted_frequency, cum_data[temp_index]):
            o.write(f" {freq:>10.5f}  {ck[0]:>12.5f}  {ck[1]:>12.5f}  {ck[2]:>12.5f}  {ck[3]:>12.5f}  {ck[4]:>12.5f}  {ck[5]:>12.5f}\n")
        o.write("\n")


def write_cumulative_vs_mfp(filepath, label, sorted_mfp, cum_data, temp_index):
    """Write cumulative kappa vs mean free path (sorted ascending) at one temperature.

    Both the cumulative kappa and its spectral derivative (d kappa / d MFP)
    are written using this same function — the caller passes the appropriate array.

    Parameters
    ----------
    filepath    : str
        Output file path.
    label       : str
        Description written as the first comment line.
    sorted_mfp  : ndarray, shape (nT, nQpt * nBand)
        Flattened mean free paths sorted in ascending order per temperature (Angstrom).
    cum_data    : ndarray, shape (nT, nQpt * nBand, 6)
        Cumulative (or derivative) kappa tensor, sorted to match sorted_mfp.
    temp_index  : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        o.write(CUM_MFP_HEADER + "\n")
        for mfp, ck in zip(sorted_mfp[temp_index], cum_data[temp_index]):
            o.write(f" {mfp:>14.4f}  {ck[0]:>12.5f}  {ck[1]:>12.5f}  {ck[2]:>12.5f}  {ck[3]:>12.5f}  {ck[4]:>12.5f}  {ck[5]:>12.5f}\n")
        o.write("\n")


def write_scattering_rate_vs_frequency(filepath, label, frequency, gamma, temp_index):
    """Write phonon scattering rate (linewidth) vs frequency at one temperature.

    Parameters
    ----------
    filepath   : str
        Output file path.
    label      : str
        Description written as the first comment line.
    frequency  : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    gamma      : ndarray, shape (nT, nQpt, nBand)
        Phonon linewidths (scattering rates) in ps⁻¹.
    temp_index : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GAMMA_HEADER + "\n")
            for freq, g in zip(frequency[:, band_index],
                               gamma[temp_index, :, band_index]):
                o.write(f" {freq:>10.5f}   {g:>14.4e}\n")
        o.write("\n")


def write_lifetime_vs_frequency(filepath, label, frequency, tau, temp_index):
    """Write phonon lifetime vs frequency at one temperature.

    Lifetime tau = 1 / (2 * 2π * gamma).

    Parameters
    ----------
    filepath   : str
        Output file path.
    label      : str
        Description written as the first comment line.
    frequency  : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    tau        : ndarray, shape (nT, nQpt, nBand)
        Phonon lifetimes in ps.
    temp_index : int
        Index into the temperature axis to extract.
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TAU_HEADER + "\n")
            for freq, t in zip(frequency[:, band_index],
                               tau[temp_index, :, band_index]):
                o.write(f" {freq:>10.5f}  {t:>14.4f}\n")
        o.write("\n")


def write_group_velocity(filepath, frequency, group_velocity):
    """Write phonon group velocity vector (vx, vy, vz) vs frequency.

    Temperature-independent quantity.

    Parameters
    ----------
    filepath       : str
        Output file path.
    frequency      : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    group_velocity : ndarray, shape (nQpt, nBand, 3)
        Group velocity Cartesian components in THz·Å.
    """
    with open(filepath, 'w') as o:
        o.write("# Group Velocity(THz * A) vs Frequency\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GV_HEADER + "\n")
            for freq, gv in zip(frequency[:, band_index],
                                group_velocity[:, band_index, :]):
                o.write(f" {freq:>10.5f}  {gv[0]:>14.4f}  {gv[1]:>14.4f}  {gv[2]:>14.4f}\n")
        o.write("\n")


def write_group_velocity_amplitude(filepath, frequency, group_velocity_amp):
    """Write phonon group velocity amplitude |v| vs frequency.

    Temperature-independent quantity.

    Parameters
    ----------
    filepath           : str
        Output file path.
    frequency          : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    group_velocity_amp : ndarray, shape (nQpt, nBand)
        Group velocity amplitude |v| = norm(vx, vy, vz) in THz·Å.
    """
    with open(filepath, 'w') as o:
        o.write("# Group Velocity amplitude(THz * A) vs Frequency\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GV_AMP_HEADER + "\n")
            for freq, amp in zip(frequency[:, band_index],
                                 group_velocity_amp[:, band_index]):
                o.write(f" {freq:>10.5f}  {amp:>14.4f}\n")
        o.write("\n")


def write_scattering_rate_vs_frequency_notemp(filepath, label, frequency, gamma):
    """Write temperature-independent scattering rate vs frequency.

    Used for isotope scattering (--isotope flag), which does not
    depend on temperature and has shape (nQpt, nBand).

    Parameters
    ----------
    filepath  : str
        Output file path.
    label     : str
        Description written as the first comment line.
    frequency : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    gamma     : ndarray, shape (nQpt, nBand)
        Scattering rates in ps⁻¹ (no temperature axis).
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GAMMA_HEADER + "\n")
            for freq, g in zip(frequency[:, band_index], gamma[:, band_index]):
                o.write(f" {freq:>10.5f}   {g:>14.4e}\n")
        o.write("\n")


def write_lifetime_vs_frequency_notemp(filepath, label, frequency, tau):
    """Write temperature-independent phonon lifetime vs frequency.

    Used for isotope scattering (--isotope flag), which does not
    depend on temperature and has shape (nQpt, nBand).

    Parameters
    ----------
    filepath  : str
        Output file path.
    label     : str
        Description written as the first comment line.
    frequency : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    tau       : ndarray, shape (nQpt, nBand)
        Phonon lifetimes in ps (no temperature axis).
    """
    with open(filepath, 'w') as o:
        o.write(f"# {label}\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TAU_HEADER + "\n")
            for freq, t in zip(frequency[:, band_index], tau[:, band_index]):
                o.write(f" {freq:>10.5f}  {t:>14.4f}\n")
        o.write("\n")


def write_heat_capacity_vs_temperature(filepath, temperature, heat_capacity_total):
    """Write total heat capacity Cv vs temperature.

    Parameters
    ----------
    filepath            : str
        Output file path.
    temperature         : ndarray, shape (nT,)
        Temperature values in K.
    heat_capacity_total : ndarray, shape (nT,)
        Total heat capacity summed over all q-points and bands (eV/K).
    """
    with open(filepath, 'w') as o:
        o.write("# Heat capacity vs Temperature\n")
        o.write(CV_HEADER + "\n")
        for temp, cv in zip(temperature, heat_capacity_total):
            o.write(f"  {temp:>7.1f}   {cv:>8.3f}\n")
        o.write("\n")


def write_gruneisen_vs_frequency(filepath, frequency, gruneisen):
    """Write Grüneisen parameter vs frequency.

    Temperature-independent quantity computed separately via phono3py-load.

    Parameters
    ----------
    filepath  : str
        Output file path.
    frequency : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    gruneisen : ndarray, shape (nQpt, nBand)
        Mode Grüneisen parameters (dimensionless).
    """
    with open(filepath, 'w') as o:
        o.write("# Gruneisen parameter vs Frequency\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GRUN_HEADER + "\n")
            for freq, gr in zip(frequency[:, band_index], gruneisen[:, band_index]):
                o.write(f" {freq:>10.5f}   {gr:>14.8f}\n")
        o.write("\n")


def write_tau_CRTA_vs_temperature(filepath, temperature, tau_CRTA):
    """Write constant relaxation time approximation (CRTA) lifetime tensor vs temperature.

    tau_CRTA = kappa_RTA / sum_qj(C_qj * v_qj ⊗ v_qj) is the effective
    single lifetime that reproduces the RTA kappa when used in the CRTA formula.

    Parameters
    ----------
    filepath    : str
        Output file path.
    temperature : ndarray, shape (nT,)
        Temperature values in K.
    tau_CRTA    : ndarray, shape (nT, 6)
        CRTA lifetime tensor components (xx yy zz yz xz xy) in ps.
    """
    with open(filepath, 'w') as o:
        o.write("# Lifetime(ps) vs Temperature\n")
        o.write(TAU_TENSOR_HEADER + "\n")
        for temp, t in zip(temperature, tau_CRTA):
            o.write(f"  {temp:>7.1f}  {t[0]:>14.4f}{t[1]:>14.4f}{t[2]:>14.4f}{t[3]:>14.4f}{t[4]:>14.4f}{t[5]:>14.4f}\n")
        o.write("\n")


def write_tau_avg_vs_temperature(filepath, temperature, tau_avg):
    """Write average phonon lifetime vs temperature.

    tau_avg = sum_qj(tau_qj) / total_weight, averaged over all q-points and bands.

    Parameters
    ----------
    filepath    : str
        Output file path.
    temperature : ndarray, shape (nT,)
        Temperature values in K.
    tau_avg     : ndarray, shape (nT,)
        Average phonon lifetime in ps.
    """
    with open(filepath, 'w') as o:
        o.write("# Lifetime vs Temperature\n")
        o.write(TAU_TEMP_HEADER + "\n")
        for temp, t in zip(temperature, tau_avg):
            o.write(f"  {temp:>7.1f}  {t:>14.4f}\n")
        o.write("\n")


def write_ave_pp_vs_frequency(filepath, frequency, ave_pp):
    """Write averaged phonon-phonon interaction strength P3 vs frequency.

    Requires phono3py --full-pp flag. P3 is related to the three-phonon
    coupling matrix elements averaged over scattering processes.

    Parameters
    ----------
    filepath  : str
        Output file path.
    frequency : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    ave_pp    : ndarray, shape (nQpt, nBand)
        Averaged phonon-phonon interaction strength in eV².
    """
    with open(filepath, 'w') as o:
        o.write("# Averaged phonon-phonon interaction\n")
        for band_index in range(frequency.shape[1]):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(PP_HEADER + "\n")
            for freq, p in zip(frequency[:, band_index], ave_pp[:, band_index]):
                o.write(f" {freq:>10.5f}      {p:>14.4e}\n")
        o.write("\n")


def outpath(kappa_input_file, tag):
    """Build an output .dat filename from the kappa HDF5 input filename.

    Replaces the 'kappa' token in the filename with the given tag and
    changes the '.hdf5' extension to '.dat'.

    Example
    -------
    outpath('kappa-m111111.hdf5', 'KappaVsT') → 'KappaVsT-m111111.dat'

    Parameters
    ----------
    kappa_input_file : str
        Original kappa HDF5 filename (e.g. 'kappa-m404040.hdf5').
    tag              : str
        Replacement token for 'kappa' (e.g. 'KappaVsT', 'GvVsFrequency').

    Returns
    -------
    str
        Output filename with the tag substituted and extension changed to .dat.
    """
    return kappa_input_file.replace('kappa', tag).replace('hdf5', 'dat')


def compute_kappa_variant(mode_kappa, weight, temperature, frequency, mean_freepath, kappa_ref):
    """Compute all derived quantities for one kappa variant (LBTE, RTA, Wigner, etc.).

    Produces the weight-normalized modal kappa, acoustic/optical branch decomposition,
    cumulative kappa sorted by frequency and mean free path, their spectral derivatives,
    and the per-mode percentage contribution to the total kappa.

    Parameters
    ----------
    mode_kappa    : ndarray, shape (nT, nQpt, nBand, 6)
        Modal kappa tensor for this variant.
    weight        : ndarray, shape (nQpt,)
        q-point integration weights.
    temperature   : ndarray, shape (nT,)
        Temperature values in K.
    frequency     : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    mean_freepath : ndarray, shape (nT, nQpt, nBand)
        Phonon mean free path in Angstrom.
    kappa_ref     : ndarray, shape (nT, 6)
        Total kappa tensor used as denominator for percentage contribution.

    Returns
    -------
    mk_weight    : ndarray, shape (nT, nQpt, nBand, 6)
        Weight-normalized modal kappa  (mode_kappa / weight.sum()).
    branch       : ndarray, shape (nT, 4, 6)
        Kappa summed into 3 acoustic branches + 1 optical-sum branch.
    cum_freq     : ndarray, shape (nT, nQpt*nBand, 6)
        Cumulative kappa sorted by ascending frequency.
    deriv_freq   : ndarray, shape (nT, nQpt*nBand, 6)
        Spectral kappa density d(kappa) / d(frequency).
    cum_mfp      : ndarray, shape (nT, nQpt*nBand, 6)
        Cumulative kappa sorted by ascending mean free path.
    deriv_mfp    : ndarray, shape (nT, nQpt*nBand, 6)
        Spectral kappa density d(kappa) / d(MFP).
    contrib_all  : ndarray, shape (nT, nBand, 6)
        Per-mode percentage contribution to kappa (all bands individually).
    contrib      : ndarray, shape (nT, 4, 6)
        Per-mode percentage contribution (3 acoustic + optical sum).
    """
    w  = weight.sum()
    nT = len(temperature)

    mk_weight = mode_kappa / w

    branch = np.concatenate((
        mk_weight.sum(axis=1)[:, :3, :],
        mk_weight.sum(axis=1)[:, 3:, :].sum(axis=1)[:, np.newaxis, :]
    ), axis=1)

    freq_order = np.argsort(frequency.flatten())
    cum_freq = np.cumsum(
        mk_weight.reshape(nT, -1, 6)[:, freq_order, :], axis=1)
    deriv_freq = np.gradient(cum_freq, axis=1)

    mfp_order = np.argsort(mean_freepath.reshape(nT, -1), axis=1)
    cum_mfp = np.cumsum(
        np.take_along_axis(
            mk_weight.reshape(nT, -1, 6),
            mfp_order[:, :, np.newaxis], axis=1),
        axis=1)
    deriv_mfp = np.gradient(cum_mfp, axis=1)

    safe_kappa = np.where(np.abs(kappa_ref) < 1e-12, np.nan, kappa_ref)
    contrib_all = np.nan_to_num(
        mk_weight.sum(axis=1) / safe_kappa[:, np.newaxis, :] * 100.0,
        nan=0.0)
    contrib = np.concatenate((
        contrib_all[:, :3, :],
        contrib_all[:, 3:, :].sum(axis=1)[:, np.newaxis, :]
    ), axis=1)

    return mk_weight, branch, cum_freq, deriv_freq, cum_mfp, deriv_mfp, contrib_all, contrib


def write_variant_per_temperature(temp_dir, kappa_input_file, tag_freq, tag_mfp, tag_cum_freq,
                                  tag_deriv_freq, tag_cum_mfp, tag_deriv_mfp, frequency,
                                  mean_freepath, mode_kappa, cum_freq, deriv_freq, cum_mfp,
                                  deriv_mfp, sorted_frequency, sorted_mean_freepath, temp_index):
    """Write all six per-temperature output files for one kappa variant.

    Produces: modal kappa vs frequency, modal kappa vs MFP,
    cumulative kappa vs frequency, spectral derivative vs frequency,
    cumulative kappa vs MFP, spectral derivative vs MFP.

    Parameters
    ----------
    temp_dir              : str
        Subdirectory for this temperature (e.g. 'T300.0K').
    kappa_input_file      : str
        Original kappa HDF5 filename used to build output filenames via outpath().
    tag_freq              : str
        outpath tag for modal kappa vs frequency file.
    tag_mfp               : str
        outpath tag for modal kappa vs MFP file.
    tag_cum_freq          : str
        outpath tag for cumulative kappa vs frequency file.
    tag_deriv_freq        : str
        outpath tag for spectral derivative vs frequency file.
    tag_cum_mfp           : str
        outpath tag for cumulative kappa vs MFP file.
    tag_deriv_mfp         : str
        outpath tag for spectral derivative vs MFP file.
    frequency             : ndarray, shape (nQpt, nBand)
        Phonon frequencies in THz.
    mean_freepath         : ndarray, shape (nT, nQpt, nBand)
        Phonon mean free path in Angstrom.
    mode_kappa            : ndarray, shape (nT, nQpt, nBand, 6)
        Weight-normalized modal kappa tensor.
    cum_freq              : ndarray, shape (nT, nQpt*nBand, 6)
        Cumulative kappa sorted by frequency.
    deriv_freq            : ndarray, shape (nT, nQpt*nBand, 6)
        Spectral kappa derivative vs frequency.
    cum_mfp               : ndarray, shape (nT, nQpt*nBand, 6)
        Cumulative kappa sorted by mean free path.
    deriv_mfp             : ndarray, shape (nT, nQpt*nBand, 6)
        Spectral kappa derivative vs mean free path.
    sorted_frequency      : ndarray, shape (nQpt*nBand,)
        All frequencies flattened and sorted ascending (THz).
    sorted_mean_freepath  : ndarray, shape (nT, nQpt*nBand)
        Mean free paths flattened and sorted ascending per temperature (Å).
    temp_index            : int
        Index into the temperature axis to extract.
    """
    label_freq = "Thermal conductivity(W/m-K) vs Frequency"
    label_mfp  = "Thermal conductivity(W/m-K) vs Mean free path"

    write_mode_vs_frequency(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_freq)),
        label_freq, frequency, mode_kappa, temp_index)

    write_mode_vs_mfp(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_mfp)),
        label_mfp, mean_freepath, mode_kappa, temp_index)

    write_cumulative_vs_frequency(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_cum_freq)),
        label_freq, sorted_frequency, cum_freq, temp_index)

    write_cumulative_vs_frequency(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_deriv_freq)),
        label_freq, sorted_frequency, deriv_freq, temp_index)

    write_cumulative_vs_mfp(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_cum_mfp)),
        label_mfp, sorted_mean_freepath, cum_mfp, temp_index)

    write_cumulative_vs_mfp(
        os.path.join(temp_dir, outpath(kappa_input_file, tag_deriv_mfp)),
        label_mfp, sorted_mean_freepath, deriv_mfp, temp_index)


def main():
    """Read HDF5 inputs, compute derived quantities, and write all output files."""
    if '-h' in argv or len(argv) < 2 or len(argv) > 3:
        usage()

    kappa_input_file = argv[1]

    if not os.path.exists(kappa_input_file):
        print(f"ERROR!\nFile: {kappa_input_file} does not exist.")
        exit(0)

    # ── Read kappa HDF5 ───────────────────────────────────────────────────────
    with h5.File(kappa_input_file, 'r') as f:
        frequency             = read_kappa(f, "frequency")
        gamma                 = read_kappa(f, "gamma")
        group_velocity        = read_kappa(f, "group_velocity")
        gv_by_gv              = read_kappa(f, "gv_by_gv")
        heat_capacity         = read_kappa(f, "heat_capacity")
        mesh                  = read_kappa(f, "mesh")
        # qpoint                = read_kappa(f, "qpoint")
        temperature           = read_kappa(f, "temperature")
        weight                = read_kappa(f, "weight")
        kappa_unit_conversion = read_kappa(f, "kappa_unit_conversion")

        # --br and --lbte
        kappa          = read_kappa(f, "kappa",          threshold=1e-12)
        mode_kappa     = read_kappa(f, "mode_kappa",     threshold=1e-12, abs_value=True)
        kappa_RTA      = read_kappa(f, "kappa_RTA",      threshold=1e-12)
        mode_kappa_RTA = read_kappa(f, "mode_kappa_RTA", threshold=1e-12, abs_value=True)
        # --isotope
        gamma_isotope  = read_kappa(f, "gamma_isotope")
        # --nu
        gamma_N        = read_kappa(f, "gamma_N")
        gamma_U        = read_kappa(f, "gamma_U")
        # --full-pp
        ave_pp         = read_kappa(f, "ave_pp")
        # --wigner --br or --wigner --lbte
        kappa_C        = read_kappa(f, "kappa_C",        threshold=1e-12)
        mode_kappa_C   = read_kappa(f, "mode_kappa_C",   threshold=1e-12, abs_value=True)
        # --wigner --br
        kappa_P_RTA      = read_kappa(f, "kappa_P_RTA",      threshold=1e-12)
        mode_kappa_P_RTA = read_kappa(f, "mode_kappa_P_RTA", threshold=1e-12, abs_value=True)
        kappa_TOT_RTA    = read_kappa(f, "kappa_TOT_RTA",    threshold=1e-12)
        # --wigner --lbte
        kappa_P_exact      = read_kappa(f, "kappa_P_exact",      threshold=1e-12)
        mode_kappa_P_exact = read_kappa(f, "mode_kappa_P_exact", threshold=1e-12, abs_value=True)
        kappa_TOT_exact    = read_kappa(f, "kappa_TOT_exact",    threshold=1e-12)

    if not (kappa is not None and mode_kappa is not None) and \
       not (kappa_C is not None and mode_kappa_C is not None and
            kappa_P_RTA is not None and mode_kappa_P_RTA is not None and
            kappa_TOT_RTA is not None):
        print("Error! Essential variables are not exist.")
        exit(0)

    # ── Grüneisen parameter ───────────────────────────────────────────────────
    gruneisen_run = False
    if len(argv) == 3:
        gruneisen_input_file = argv[2]
        if not os.path.exists(gruneisen_input_file):
            gruneisen_run = True
    else:
        gruneisen_run = True

    if gruneisen_run:
        subprocess.run(
            "phono3py-load --mesh " + " ".join(map(str, mesh)) + " --gruneisen",
            shell=True)
        gruneisen_input_file = 'gruneisen-m' + ''.join(map(str, mesh)) + '.hdf5'
        os.rename('gruneisen.hdf5', gruneisen_input_file)

    with h5.File(gruneisen_input_file, 'r') as g:
        gruneisen = g["gruneisen"][()]

    # ── Derived quantities ────────────────────────────────────────────────────
    sorted_frequency = frequency.flatten()[np.argsort(frequency.flatten())]

    with np.errstate(divide='ignore'):
        tau = np.where(gamma > 0.0, 1.0 / (2.0 * 2.0 * np.pi * gamma), 0.0)

    group_velocity_amp = np.linalg.norm(group_velocity, axis=-1)
    mean_freepath = group_velocity_amp * tau
    sorted_mean_freepath = np.take_along_axis(
        mean_freepath.reshape(len(temperature), -1),
        np.argsort(mean_freepath.reshape(len(temperature), -1), axis=1),
        axis=1)

    heat_capacity_total = heat_capacity.sum(axis=(1, 2))

    # Base variant (--lbte or --br)
    (mode_kappa_weight, kappa_branch,
     cumulative_kappa_frequency, derivative_kappa_frequency,
     cumulative_kappa_mfp, derivative_kappa_mfp,
     contribute_kappa_all, _) = compute_kappa_variant(
        mode_kappa, weight, temperature, frequency, mean_freepath, kappa)

    # --lbte: RTA companion
    if kappa_RTA is not None and mode_kappa_RTA is not None:
        (mode_kappa_RTA_weight, kappa_RTA_branch,
         cumulative_kappa_RTA_frequency, derivative_kappa_RTA_frequency,
         cumulative_kappa_RTA_mfp, derivative_kappa_RTA_mfp,
         _, _) = compute_kappa_variant(
            mode_kappa_RTA, weight, temperature, frequency, mean_freepath, kappa_RTA)

    # --wigner coherence term
    if kappa_C is not None and mode_kappa_C is not None:
        (mode_kappa_C_weight, kappa_C_branch,
         cumulative_kappa_C_frequency, derivative_kappa_C_frequency,
         cumulative_kappa_C_mfp, derivative_kappa_C_mfp,
         _, _) = compute_kappa_variant(
            mode_kappa_C, weight, temperature, frequency, mean_freepath, kappa_C)

    # --wigner --br particle term
    if kappa_P_RTA is not None and mode_kappa_P_RTA is not None:
        (mode_kappa_P_RTA_weight, kappa_P_RTA_branch,
         cumulative_kappa_P_RTA_frequency, derivative_kappa_P_RTA_frequency,
         cumulative_kappa_P_RTA_mfp, derivative_kappa_P_RTA_mfp,
         _, _) = compute_kappa_variant(
            mode_kappa_P_RTA, weight, temperature, frequency, mean_freepath, kappa_P_RTA)

    # --wigner --lbte particle term
    if kappa_P_exact is not None and mode_kappa_P_exact is not None:
        (mode_kappa_P_exact_weight, kappa_P_exact_branch,
         cumulative_kappa_P_exact_frequency, derivative_kappa_P_exact_frequency,
         cumulative_kappa_P_exact_mfp, derivative_kappa_P_exact_mfp,
         _, _) = compute_kappa_variant(
            mode_kappa_P_exact, weight, temperature, frequency, mean_freepath, kappa_P_exact)

    # Constant RTA lifetime: tau_CRTA = kappa / sum_qj(C_qj * v_qj ⊗ v_qj)
    kappa_per_tau = (
        kappa_unit_conversion * (2 * np.pi) *
        heat_capacity[:, :, :, np.newaxis] *
        gv_by_gv[np.newaxis, :, :, :]
    ).sum(axis=1).sum(axis=1) / weight.sum()

    if kappa_RTA is not None:
        tau_CRTA = kappa_RTA / kappa_per_tau
    elif kappa_P_RTA is not None:
        tau_CRTA = kappa_P_RTA / kappa_per_tau
    else:
        tau_CRTA = kappa / kappa_per_tau

    # Average lifetime over all q-points and bands
    tau_avg = tau.sum(axis=1).sum(axis=1) / weight.sum()

    # Isotope scattering lifetime (temperature-independent)
    if gamma_isotope is not None:
        with np.errstate(divide='ignore'):
            tau_isotope = np.where(gamma_isotope > 0.0,
                                   1.0 / (2.0 * 2.0 * np.pi * gamma_isotope), 0.0)

    # Normal and Umklapp process lifetimes
    if gamma_N is not None and gamma_U is not None:
        with np.errstate(divide='ignore'):
            tau_N = np.where(gamma_N > 0.0,
                             1.0 / (2.0 * 2.0 * np.pi * gamma_N), 0.0)
            tau_U = np.where(gamma_U > 0.0,
                             1.0 / (2.0 * 2.0 * np.pi * gamma_U), 0.0)

    # ════════════════════════════════════════════════════════════════════════
    #  Temperature-independent output files
    # ════════════════════════════════════════════════════════════════════════

    label_kappa = "Thermal conductivity(W/m-K) vs Temperature"

    write_tensor_vs_temperature(
        outpath(kappa_input_file, 'KappaVsT'),
        label_kappa, kappa, temperature)

    write_band_tensor_vs_temperature(
        outpath(kappa_input_file, 'Kappa_bandVsT'),
        label_kappa, kappa_branch, temperature,
        note="Sum all optical branch to one")

    write_contribution_vs_temperature(
        outpath(kappa_input_file, 'ContributeKappaVsT'),
        label_kappa, contribute_kappa_all, temperature, frequency)

    write_group_velocity(
        outpath(kappa_input_file, 'GvVsFrequency'),
        frequency, group_velocity)

    write_group_velocity_amplitude(
        outpath(kappa_input_file, 'Gv_amplitudeVsFrequency'),
        frequency, group_velocity_amp)

    write_heat_capacity_vs_temperature(
        outpath(kappa_input_file, 'CvVsT'),
        temperature, heat_capacity_total)

    write_gruneisen_vs_frequency(
        outpath(kappa_input_file, 'GruneisenVsFrequency'),
        frequency, gruneisen)

    write_tau_CRTA_vs_temperature(
        outpath(kappa_input_file, 'Tau_CRTAVsT'),
        temperature, tau_CRTA)

    write_tau_avg_vs_temperature(
        outpath(kappa_input_file, 'Tau_AvgVsT'),
        temperature, tau_avg)

    # ── Optional variant: --lbte kappa_RTA ───────────────────────────────────
    if kappa_RTA is not None and mode_kappa_RTA is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_RTAVsT'),
            label_kappa, kappa_RTA, temperature)
        write_band_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_RTA_bandVsT'),
            label_kappa, kappa_RTA_branch, temperature)

    # ── Optional variant: --wigner kappa_C ───────────────────────────────────
    if kappa_C is not None and mode_kappa_C is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_CVsT'),
            label_kappa, kappa_C, temperature)
        write_band_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_C_bandVsT'),
            label_kappa, kappa_C_branch, temperature)

    # ── Optional variant: --wigner --br kappa_P_RTA ──────────────────────────
    if kappa_P_RTA is not None and mode_kappa_P_RTA is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_P_RTAVsT'),
            label_kappa, kappa_P_RTA, temperature)
        write_band_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_P_RTA_bandVsT'),
            label_kappa, kappa_P_RTA_branch, temperature)
    if kappa_TOT_RTA is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_TOT_RTAVsT'),
            label_kappa, kappa_TOT_RTA, temperature)

    # ── Optional variant: --wigner --lbte kappa_P_exact ──────────────────────
    if kappa_P_exact is not None and mode_kappa_P_exact is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_P_exactVsT'),
            label_kappa, kappa_P_exact, temperature)
        write_band_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_P_exact_bandVsT'),
            label_kappa, kappa_P_exact_branch, temperature)
    if kappa_TOT_exact is not None:
        write_tensor_vs_temperature(
            outpath(kappa_input_file, 'Kappa_TOT_exactVsT'),
            label_kappa, kappa_TOT_exact, temperature)

    # ── isotope scattering ───────────────────────────────────────────────────
    if gamma_isotope is not None:
        write_scattering_rate_vs_frequency_notemp(
            outpath(kappa_input_file, 'Gamma_isotopeVsFrequency'),
            "Scattering rate vs Frequency", frequency, gamma_isotope)
        write_lifetime_vs_frequency_notemp(
            outpath(kappa_input_file, 'Tau_isotopeVsFrequency'),
            "Lifetime vs Frequency", frequency, tau_isotope)

    # ── phonon-phonon interaction strength ───────────────────────────────────
    if ave_pp is not None:
        write_ave_pp_vs_frequency(
            outpath(kappa_input_file, 'PqjVsFrequency'),
            frequency, ave_pp)

    # ════════════════════════════════════════════════════════════════════════
    #  Per-temperature output files
    # ════════════════════════════════════════════════════════════════════════

    for temp_index in range(len(temperature)):
        temp_dir = f'T{temperature[temp_index]}K'
        os.makedirs(temp_dir, exist_ok=True)

        # ── base kappa ───────────────────────────────────────────────────────
        write_variant_per_temperature(
            temp_dir, kappa_input_file, 'KappaVsFrequency', 'KappaVsMfp', 'cumulative_KappaVsFrequency',
            'derivative_KappaVsFrequency', 'cumulative_KappaVsMfp', 'derivative_KappaVsMfp',frequency,
            mean_freepath, mode_kappa_weight, cumulative_kappa_frequency, derivative_kappa_frequency,
            cumulative_kappa_mfp, derivative_kappa_mfp, sorted_frequency, sorted_mean_freepath, temp_index)

        # ── gamma and tau ────────────────────────────────────────────────────
        write_scattering_rate_vs_frequency(
            os.path.join(temp_dir, outpath(kappa_input_file, 'GammaVsFrequency')),
            "Scattering rate vs Frequency", frequency, gamma, temp_index)
        write_lifetime_vs_frequency(
            os.path.join(temp_dir, outpath(kappa_input_file, 'TauVsFrequency')),
            "Lifetime vs Frequency", frequency, tau, temp_index)

        # ── kappa_RTA ────────────────────────────────────────────────────────
        if kappa_RTA is not None and mode_kappa_RTA is not None:
            write_variant_per_temperature(
                temp_dir, kappa_input_file, 'Kappa_RTAVsFrequency', 'Kappa_RTAVsMfp', 'cumulative_Kappa_RTAVsFrequency',
                'derivative_Kappa_RTAVsFrequency', 'cumulative_Kappa_RTAVsMfp', 'derivative_Kappa_RTAVsMfp',frequency,
                mean_freepath, mode_kappa_RTA_weight, cumulative_kappa_RTA_frequency, derivative_kappa_RTA_frequency,
                cumulative_kappa_RTA_mfp, derivative_kappa_RTA_mfp, sorted_frequency, sorted_mean_freepath, temp_index)
            
        # ── kappa_C ──────────────────────────────────────────────────────────
        if kappa_C is not None and mode_kappa_C is not None:
            write_variant_per_temperature(
                temp_dir, kappa_input_file, 'Kappa_CVsFrequency', 'Kappa_CVsMfp', 'cumulative_Kappa_CVsFrequency',
                'derivative_Kappa_CVsFrequency', 'cumulative_Kappa_CVsMfp', 'derivative_Kappa_CVsMfp',frequency,
                mean_freepath, mode_kappa_C_weight, cumulative_kappa_C_frequency, derivative_kappa_C_frequency,
                cumulative_kappa_C_mfp, derivative_kappa_C_mfp, sorted_frequency, sorted_mean_freepath, temp_index)
           
        # ── kappa_P_RTA ──────────────────────────────────────────────────────
        if kappa_P_RTA is not None and mode_kappa_P_RTA is not None:
            write_variant_per_temperature(
                temp_dir, kappa_input_file, 'Kappa_P_RTAVsFrequency', 'Kappa_P_RTAVsMfp', 'cumulative_Kappa_P_RTAVsFrequency',
                'derivative_Kappa_P_RTAVsFrequency', 'cumulative_Kappa_P_RTAVsMfp', 'derivative_Kappa_P_RTAVsMfp',frequency,
                mean_freepath, mode_kappa_P_RTA_weight, cumulative_kappa_P_RTA_frequency, derivative_kappa_P_RTA_frequency,
                cumulative_kappa_P_RTA_mfp, derivative_kappa_P_RTA_mfp, sorted_frequency, sorted_mean_freepath, temp_index)

        # ── kappa_P_exact ────────────────────────────────────────────────────
        if kappa_P_exact is not None and mode_kappa_P_exact is not None:
            write_variant_per_temperature(
                temp_dir, kappa_input_file, 'Kappa_P_exactVsFrequency', 'Kappa_P_exactVsMfp', 'cumulative_Kappa_P_exactVsFrequency',
                'derivative_Kappa_P_exactVsFrequency', 'cumulative_Kappa_P_exactVsMfp', 'derivative_Kappa_P_exactVsMfp',frequency,
                mean_freepath, mode_kappa_P_exact_weight, cumulative_kappa_P_exact_frequency, derivative_kappa_P_exact_frequency,
                cumulative_kappa_P_exact_mfp, derivative_kappa_P_exact_mfp, sorted_frequency, sorted_mean_freepath, temp_index)
            
        # ── Normal / Umklapp processes ───────────────────────────────────────
        if gamma_N is not None and gamma_U is not None:
            write_scattering_rate_vs_frequency(
                os.path.join(temp_dir, outpath(kappa_input_file, 'Gamma_NVsFrequency')),
                "Scattering rate vs Frequency", frequency, gamma_N, temp_index)
            write_lifetime_vs_frequency(
                os.path.join(temp_dir, outpath(kappa_input_file, 'Tau_NVsFrequency')),
                "Lifetime vs Frequency", frequency, tau_N, temp_index)
            write_scattering_rate_vs_frequency(
                os.path.join(temp_dir, outpath(kappa_input_file, 'Gamma_UVsFrequency')),
                "Scattering rate vs Frequency", frequency, gamma_U, temp_index)
            write_lifetime_vs_frequency(
                os.path.join(temp_dir, outpath(kappa_input_file, 'Tau_UVsFrequency')),
                "Lifetime vs Frequency", frequency, tau_U, temp_index)

    print("Convert HDF5 file to DAT files ... Finished!")


if __name__ == "__main__":
    main()
