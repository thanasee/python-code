#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    """Print usage information and exit."""
    text = """
Usage: analyzeShengBTE.py <3ph/4ph>

This script extracts thermal transport data from ShengBTE (and FourPhonon) output files.
Run from the ShengBTE output directory.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

TENSOR_HEADER     = "#  T(K)        xx          xy          xz          yx          yy          yz          zx          zy          zz"
FREQ_HEADER       = "#  Frequency(THz) xx          xy          xz          yx          yy          yz          zx          zy          zz"
MFP_HEADER        = "#        MFP(A)       xx          xy          xz          yx          yy          yz          zx          zy          zz"
GV_HEADER         = "#  Frequency(THz)    x               y               z"
GV_AMP_HEADER     = "#  Frequency(THz)    |v|"
CV_HEADER         = "#  T(K)    Cv(eV/K)"
GAMMA_HEADER      = "#  Frequency(THz) Gamma(ps-1)"
TAU_HEADER        = "#  Frequency(THz)    Tau(ps)"
GRUN_HEADER       = "#  Frequency(THz) Gruneisen"
P_HEADER          = "#  Frequency(THz) Phase space"
CV_HEADER         = "#  T(K)    Cv(eV/K)"


def _read_file(filepath):
    
    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)
    
    lines = np.loadtxt(filepath)

    return lines.item() if lines.size == 1 else lines


def _to_THz(omega):
    
    return omega / (2 * np.pi)


def read_kappa_vs_temperature(filepath):
    
    lines = _read_file(filepath)
    temperature = lines[:, 0]
    kappa = lines[:, 1:10]
    
    return temperature, kappa


def write_kappa_vs_temperature(filepath, temperature, kappa):
    
    with open(filepath, 'w') as o:
        o.write("# Thermal conductivity(W/m-K) vs Temperature\n")
        o.write(TENSOR_HEADER + "\n")
        for temp, k in zip(temperature, kappa):
            o.write(f"{temp:>7.1f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}"
                    f"  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}"
                    f"  {k[6]:>10.3f}  {k[7]:>10.3f}  {k[8]:>10.3f}\n")
        o.write("\n")


def read_frequency(filepath):
    
    omega = _read_file(filepath)
    n_qpoints, n_bands = omega.shape
    frequency = _to_THz(omega)
    
    return frequency, n_qpoints, n_bands


def read_qpoints(filepath):
    
    lines = _read_file(filepath)
    n_qpoints = lines.shape[0]
    
    return n_qpoints


def read_gruneisen(filepath):
    
    gruneisen = _read_file(filepath)
    n_qpoints, n_bands = gruneisen.shape
    
    return gruneisen, n_qpoints, n_bands


def write_gruneisen_vs_frequency(filepath, frequency, n_bands, gruneisen):
    
    with open(filepath, 'w') as o:
        o.write("# Gruneisen parameter vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GRUN_HEADER + "\n")
            for freq, gr in zip(frequency[:, band_index], gruneisen[:, band_index]):
                o.write(f" {freq:>10.5f}   {gr:>14.8f}\n")
        o.write("\n")


def read_group_velocity(filepath, n_qpoints, n_bands):
    
    lines = _read_file(filepath)
    group_velocity = lines.reshape(n_qpoints, n_bands, 3)
    group_velocity_amp = np.linalg.norm(group_velocity, axis=-1)
    
    return group_velocity, group_velocity_amp


def write_group_velocity_vs_frequency(filepath, frequency, n_bands, group_velocity):
    
    with open(filepath, 'w') as o:
        o.write("# Group Velocity(km/s) vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GV_HEADER + "\n")
            for freq, gv in zip(frequency[:, band_index],
                                group_velocity[:, band_index, :]):
                o.write(f" {freq:>10.5f}  {gv[0]:>14.4f}  {gv[1]:>14.4f}  {gv[2]:>14.4f}\n")
        o.write("\n")


def write_group_velocity_amplitude_vs_frequency(filepath, frequency, n_bands, group_velocity_amp):
    
    with open(filepath, 'w') as o:
        o.write("# Group Velocity amplitude(km/s) vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GV_AMP_HEADER + "\n")
            for freq, amp in zip(frequency[:, band_index],
                                 group_velocity_amp[:, band_index]):
                o.write(f" {freq:>10.5f}  {amp:>14.4f}\n")
        o.write("\n")


def read_phase_space(filepath):
    
    phase_space = _read_file(filepath)
    total_phase_space = _read_file(filepath + "_total")
    n_qpoints, n_bands = phase_space.shape
    
    return phase_space, total_phase_space, n_qpoints, n_bands


def write_phase_space_vs_frequency(filepath, frequency, n_bands, phase_space, total_phase_space):
    
    with open(filepath, 'w') as o:
        o.write(f"# Phase space vs Frequency\n")
        o.write(f"# Total phase space is {total_phase_space:>14.4e}\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(P_HEADER + "\n")
            for freq, ps in zip(frequency[:, band_index], phase_space[:, band_index]):
                o.write(f" {freq:>10.5f}   {ps:>14.4e}\n")
        o.write("\n")


def read_weighted_phase_space(filepath, n_qpoints, n_bands):
    
    lines = _read_file(filepath)
    omega = lines[:, 0].reshape(n_qpoints, n_bands)
    weighted_phase_space = lines[:, 1].reshape(n_qpoints, n_bands)
    frequency = _to_THz(omega)
    
    return frequency, weighted_phase_space


def write_weighted_phase_space_vs_frequency(filepath, frequency, n_bands, weighted_phase_space):
    
    with open(filepath, 'w') as o:
        o.write(f"# Weighted phase space vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(P_HEADER + "\n")
            for freq, wps in zip(frequency[:, band_index], weighted_phase_space[:, band_index]):
                o.write(f" {freq:>10.5f}   {wps:>14.4e}\n")
        o.write("\n")


def read_scattering_rate(filepath, n_qpoints, n_bands):
    
    lines = _read_file(filepath)
    omega = lines[:, 0].reshape(n_qpoints, n_bands)
    gamma = lines[:, 1].reshape(n_qpoints, n_bands)
    frequency = _to_THz(omega)
    
    return frequency, gamma


def write_scattering_rate_vs_frequency(filepath, frequency, n_bands, gamma):
    
    with open(filepath, 'w') as o:
        o.write(f"# Scattering rate vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(GAMMA_HEADER + "\n")
            for freq, g in zip(frequency[:, band_index], gamma[:, band_index]):
                o.write(f" {freq:>10.5f}   {g:>14.4e}\n")
        o.write("\n")


def compute_lifetime(gamma):
    
    with np.errstate(divide='ignore'):
        tau = np.where(gamma > 0.0, 1.0 / (2.0 * 2.0 * np.pi * gamma), 0.0)
    
    return tau


def write_lifetime_vs_frequency(filepath, frequency, n_bands, tau):
    
    with open(filepath, 'w') as o:
        o.write(f"# Lifetime vs Frequency\n")
        for band_index in range(n_bands):
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TAU_HEADER + "\n")
            for freq, t in zip(frequency[:, band_index], tau[:, band_index]):
                o.write(f" {freq:>10.5f}   {t:>14.4f}\n")
        o.write("\n")


def read_heat_capacity(filepath):
    
    lines = _read_file(filepath)
    temperature = lines[:, 0]
    heat_capacity = lines[:, 1]
    
    return temperature, heat_capacity


def write_heat_capacity_vs_temperature(filepath, temperature, heat_capacity):
    
    with open(filepath, 'w') as o:
        o.write("# Heat capacity vs Temperature\n")
        o.write(CV_HEADER + "\n")
        for temp, cv in zip(temperature, heat_capacity):
            o.write(f"  {temp:>7.1f}   {cv:>8.3f}\n")
        o.write("\n")


def read_mode_kappa(filepath, n_bands):

    lines = _read_file(filepath)
    mode_kappa = lines[-1, 1:].reshape(n_bands, 9)
    mode_kappa_sum = np.concatenate((
        mode_kappa[:3, :],
        mode_kappa[3:, :].sum(axis=0, keepdims=True)
    ), axis=0)

    return mode_kappa_sum


def collect_mode_kappa(temp_dirs, n_bands):
    
    mode_kappa_all = []
    for temppath in temp_dirs:
        filepath = os.path.join(temppath, "BTE.kappa")
        mode_kappa_sum = read_mode_kappa(filepath, n_bands)
        mode_kappa_all.append(mode_kappa_sum)

    return np.array(mode_kappa_all)


def write_kappa_band_vs_temperature(filepath, temperature, mode_kappa_sum):

    with open(filepath, 'w') as o:
        o.write("# Thermal conductivity(W/m-K) vs Temperature\n")
        o.write("# Sum all optical branch to one\n")
        for band_index in range(mode_kappa_sum.shape[1]):  # 3 acoustic + 1 optical
            o.write(f"\n# Band-Index: {band_index + 1}\n")
            o.write(TENSOR_HEADER + "\n")
            for temp, k in zip(temperature, mode_kappa_sum[:, band_index]):
                o.write(f"{temp:>7.1f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}"
                        f"  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}"
                        f"  {k[6]:>10.3f}  {k[7]:>10.3f}  {k[8]:>10.3f}\n")


def read_cumulative_kappa_vs_mfp(filepath):

    lines = _read_file(filepath)
    mfp = lines[:, 0]
    cumulative_kappa = lines[:, 1:10]

    return mfp, cumulative_kappa


def write_cumulative_kappa_vs_mfp(filepath, mfp, cumulative_kappa):

    with open(filepath, 'w') as o:
        o.write("# Cumulative kappa vs MFP\n")
        o.write(MFP_HEADER + "\n")
        for m, k in zip(mfp, cumulative_kappa):
            o.write(f"{m:>14.4f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}"
                    f"  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}"
                    f"  {k[6]:>10.3f}  {k[7]:>10.3f}  {k[8]:>10.3f}\n")
        o.write("\n")


def read_cumulative_kappa_vs_frequency(filepath):

    lines = _read_file(filepath)
    omega = lines[:, 0]
    cumulative_kappa = lines[:, 1:10]
    frequency = _to_THz(omega)

    return frequency, cumulative_kappa


def write_cumulative_kappa_vs_frequency(filepath, frequency, cumulative_kappa):

    with open(filepath, 'w') as o:
        o.write("# Cumulative kappa vs Frequency\n")
        o.write(FREQ_HEADER + "\n")
        for freq, k in zip(frequency, cumulative_kappa):
            o.write(f"{freq:>10.5f}   {k[0]:>10.3f}  {k[1]:>10.3f}  {k[2]:>10.3f}"
                    f"  {k[3]:>10.3f}  {k[4]:>10.3f}  {k[5]:>10.3f}"
                    f"  {k[6]:>10.3f}  {k[7]:>10.3f}  {k[8]:>10.3f}\n")
        o.write("\n")


def detect_temp_dirs(dirpath):
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

    for dir in os.listdir(dirpath):
        full_path = os.path.join(dirpath, dir)

        if not os.path.isdir(full_path):
            continue

        if not (dir.startswith("T") and dir.endswith("K")):
            continue

        temp_str = dir[1:-1]

        if temp_str.isdigit():  # strict integer only
            dirs[int(temp_str)] = full_path

    if not dirs:
        raise RuntimeError(
            f"No T<temp>K subdirectories found in '{dirpath}'. "
            "Check that this is a valid ShengBTE output directory."
        )
    
    sorted_dirs = sorted(dirs.items())
    temperatures = [temp for temp, _ in sorted_dirs]
    temp_dirs = [path for _, path in sorted_dirs]

    return np.array(temperatures), temp_dirs


def main():
    if '-h' in argv or len(argv) != 2:
        usage()
    
    work_dir   = os.path.abspath(os.getcwd())
    try:
        fourphonon = int(argv[1].strip()[0]) == 4
    except (ValueError, IndexError):
        fourphonon = False
    mode_label = "4ph" if fourphonon else "3ph"
    print(f"Mode: {mode_label}")

    temperatures, temp_dirs = detect_temp_dirs(work_dir)
    print(f"Found {len(temp_dirs)} temperature point(s): "
          f"{[temp for temp in temperatures]} K")
    
    print("\nReading temperature-independent files...")
    freq_path = os.path.join(work_dir, "BTE.omega")
    frequency, n_qpoints, n_bands = read_frequency(freq_path)
    qpoints_path = os.path.join(work_dir, "BTE.qpoints")
    n_qpoints_check = read_qpoints(qpoints_path)
    if n_qpoints != n_qpoints_check:
        print(f"Warning: The number of q-points from BTE.omega ({n_qpoints}) does not match the number of q-points from BTE.qpoints ({n_qpoints_check}).")
        print(f" Using {n_qpoints} from BTE.omega.")
    print(f"  BTE.omega  →  {n_qpoints} q-points, {n_bands} branches")

    gv_path = os.path.join(work_dir, "BTE.v")
    group_velocity, group_velocity_amp = read_group_velocity(gv_path, n_qpoints, n_bands)
    write_group_velocity_vs_frequency("GroupVelocityVsFrequency.dat", frequency, n_bands, group_velocity)
    write_group_velocity_amplitude_vs_frequency("GroupVelocityAmplitudeVsFrequency.dat", frequency, n_bands, group_velocity_amp)

    grun_path = os.path.join(work_dir, "BTE.gruneisen")
    gruneisen, _, gr_n_bands = read_gruneisen(grun_path)
    if gr_n_bands != n_bands:
        print(f"Warning: The number of bands from BTE.gruneisen ({gr_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
        print(f"Using {gr_n_bands} from BTE.gruneisen.")
    write_gruneisen_vs_frequency("GruneisenVsFrequency.dat", frequency, gr_n_bands, gruneisen)

    cv_path = os.path.join(work_dir, "BTE.cvVsT")
    cv_temperature, heat_capacity = read_heat_capacity(cv_path)
    if not np.array_equal(temperatures, cv_temperature):
        print(f"Warning: The temperature points from BTE.cvVsT do not match the temperature points from the T<temp>K directories.")
        print(f" Using temperature points from BTE.cvVsT.")
    write_heat_capacity_vs_temperature("HeatCapacityVsT.dat", cv_temperature, heat_capacity)

    kappa_rta_path  = os.path.join(work_dir, "BTE.KappaTensorVsT_RTA")
    T_rta,  kappa_rta  = read_kappa_vs_temperature(kappa_rta_path)
    if not np.array_equal(temperatures, T_rta):
        print(f"Warning: The temperature points from BTE.KappaTensorVsT_RTA do not match the temperature points from the T<temp>K directories.")
        print(f" Using temperature points from BTE.KappaTensorVsT_RTA.")
    write_kappa_vs_temperature("Kappa_RTAVsT.dat", T_rta, kappa_rta)

    kappa_conv_path = os.path.join(work_dir, "BTE.KappaTensorVsT_CONV")
    T_conv, kappa_conv = read_kappa_vs_temperature(kappa_conv_path)
    if not np.array_equal(temperatures, T_conv):
        print(f"Warning: The temperature points from BTE.KappaTensorVsT_CONV do not match the temperature points from the T<temp>K directories.")
        print(f" Using temperature points from BTE.KappaTensorVsT_CONV.")
    write_kappa_vs_temperature("Kappa_CONVVsT.dat", T_conv, kappa_conv)

    p3_path = os.path.join(work_dir, "BTE.P3")
    p3, total_p3, _, p3_n_bands = read_phase_space(p3_path)
    if p3_n_bands != n_bands:
        print(f"Warning: The number of bands from BTE.P3 ({p3_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
        print(f"Using {p3_n_bands} from BTE.P3.")
    write_phase_space_vs_frequency("P3VsFrequency.dat", frequency, p3_n_bands, p3, total_p3)

    p3_plus_path = os.path.join(work_dir, "BTE.P3_plus")
    p3_plus, total_p3_plus, _, p3_plus_n_bands = read_phase_space(p3_plus_path)
    if p3_plus_n_bands != n_bands:
        print(f"Warning: The number of bands from BTE.P3_plus ({p3_plus_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
        print(f"Using {p3_plus_n_bands} from BTE.P3_plus.")
    write_phase_space_vs_frequency("P3_AdsorptionVsFrequency.dat", frequency, p3_plus_n_bands, p3_plus, total_p3_plus)

    p3_minus_path = os.path.join(work_dir, "BTE.P3_minus")
    p3_minus, total_p3_minus, _, p3_minus_n_bands = read_phase_space(p3_minus_path)
    if p3_minus_n_bands != n_bands:
        print(f"Warning: The number of bands from BTE.P3_minus ({p3_minus_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
        print(f"Using {p3_minus_n_bands} from BTE.P3_minus.")
    write_phase_space_vs_frequency("P3_EmissionVsFrequency.dat", frequency, p3_minus_n_bands, p3_minus, total_p3_minus)

    if fourphonon:
        p4_path = os.path.join(work_dir, "BTE.P4")
        p4, total_p4, _, p4_n_bands = read_phase_space(p4_path)
        if p4_n_bands != n_bands:
            print(f"Warning: The number of bands from BTE.P4 ({p4_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
            print(f"Using {p4_n_bands} from BTE.P4.")
        write_phase_space_vs_frequency("P4VsFrequency.dat", frequency, p4_n_bands, p4, total_p4)

        p4_plusplus_path = os.path.join(work_dir, "BTE.P4_plusplus")
        p4_plusplus, total_p4_plusplus, _, p4_plusplus_n_bands = read_phase_space(p4_plusplus_path)
        if p4_plusplus_n_bands != n_bands:
            print(f"Warning: The number of bands from BTE.P4_plusplus ({p4_plusplus_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
            print(f"Using {p4_plusplus_n_bands} from BTE.P4_plusplus.")
        write_phase_space_vs_frequency("P4_RecombinationVsFrequency.dat", frequency, p4_plusplus_n_bands, p4_plusplus, total_p4_plusplus)

        p4_plusminus_path = os.path.join(work_dir, "BTE.P4_plusminus")
        p4_plusminus, total_p4_plusminus, _, p4_plusminus_n_bands = read_phase_space(p4_plusminus_path)
        if p4_plusminus_n_bands != n_bands:
            print(f"Warning: The number of bands from BTE.P4_plusminus ({p4_plusminus_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
            print(f"Using {p4_plusminus_n_bands} from BTE.P4_plusminus.")
        write_phase_space_vs_frequency("P4_RedistributionVsFrequency.dat", frequency, p4_plusminus_n_bands, p4_plusminus, total_p4_plusminus)

        p4_minusminus_path = os.path.join(work_dir, "BTE.P4_minusminus")
        p4_minusminus, total_p4_minusminus, _, p4_minusminus_n_bands = read_phase_space(p4_minusminus_path)
        if p4_minusminus_n_bands != n_bands:
            print(f"Warning: The number of bands from BTE.P4_minusminus ({p4_minusminus_n_bands}) does not match the number of bands from BTE.omega ({n_bands}).")
            print(f"Using {p4_minusminus_n_bands} from BTE.P4_minusminus.")
        write_phase_space_vs_frequency("P4_SplittingVsFrequency.dat", frequency, p4_minusminus_n_bands, p4_minusminus, total_p4_minusminus)

    w_isotopic_path = os.path.join(work_dir, "BTE.w_isotopic")
    frequency_w_isotopic, gamma_isotopic = read_scattering_rate(w_isotopic_path, n_qpoints, n_bands)
    write_scattering_rate_vs_frequency("ScatteringRate_IsotopicVsFrequency.dat", frequency_w_isotopic, n_bands, gamma_isotopic)
    tau_isotopic = compute_lifetime(gamma_isotopic)
    write_lifetime_vs_frequency("Lifetime_IsotopicVsFrequency.dat", frequency_w_isotopic, n_bands, tau_isotopic)

    mode_kappa_sum = collect_mode_kappa(temp_dirs, n_bands)
    write_kappa_band_vs_temperature("Kappa_bandVsT.dat", temperatures, mode_kappa_sum)

    for temppath in temp_dirs:
        cumulative_kappa_mfp_path = os.path.join(temppath, "BTE.cumulative_kappa_tensor")
        mfp, cumulative_kappa_mfp = read_cumulative_kappa_vs_mfp(cumulative_kappa_mfp_path)
        write_cumulative_kappa_vs_mfp(os.path.join(temppath, "CumulativeKappaVsMFP.dat"), mfp, cumulative_kappa_mfp)

        cumulative_kappa_freq_path = os.path.join(temppath, "BTE.cumulative_kappaVsOmega_tensor")
        frequency_cum, cumulative_kappa_freq = read_cumulative_kappa_vs_frequency(cumulative_kappa_freq_path)
        write_cumulative_kappa_vs_frequency(os.path.join(temppath, "CumulativeKappaVsFrequency.dat"), frequency_cum, cumulative_kappa_freq)

        w_3ph_path = os.path.join(temppath, "BTE.w_3ph") if fourphonon else os.path.join(temppath, "BTE.w_anharmonic")
        frequency_w_3ph, gamma_3ph = read_scattering_rate(w_3ph_path, n_qpoints, n_bands)
        write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_3phVsFrequency.dat"), frequency_w_3ph, n_bands, gamma_3ph)
        tau_3ph = compute_lifetime(gamma_3ph)
        write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_3phVsFrequency.dat"), frequency_w_3ph, n_bands, tau_3ph)

        w_3ph_plus_path = os.path.join(temppath, "BTE.w_3ph_plus") if fourphonon else os.path.join(temppath, "BTE.w_anharmonic_plus")
        frequency_w_3ph_plus, gamma_3ph_plus = read_scattering_rate(w_3ph_plus_path, n_qpoints, n_bands)
        write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_3ph_AdsorptionVsFrequency.dat"), frequency_w_3ph_plus, n_bands, gamma_3ph_plus)
        tau_3ph_plus = compute_lifetime(gamma_3ph_plus)
        write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_3ph_AdsorptionVsFrequency.dat"), frequency_w_3ph_plus, n_bands, tau_3ph_plus)

        w_3ph_minus_path = os.path.join(temppath, "BTE.w_3ph_minus") if fourphonon else os.path.join(temppath, "BTE.w_anharmonic_minus")
        frequency_w_3ph_minus, gamma_3ph_minus = read_scattering_rate(w_3ph_minus_path, n_qpoints, n_bands)
        write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_3ph_EmissionVsFrequency.dat"), frequency_w_3ph_minus, n_bands, gamma_3ph_minus)
        tau_3ph_minus = compute_lifetime(gamma_3ph_minus)
        write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_3ph_EmissionVsFrequency.dat"), frequency_w_3ph_minus, n_bands, tau_3ph_minus)

        if fourphonon:
            w_4ph_path = os.path.join(temppath, "BTE.w_4ph")
            frequency_w_4ph, gamma_4ph = read_scattering_rate(w_4ph_path, n_qpoints, n_bands)
            write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_4phVsFrequency.dat"), frequency_w_4ph, n_bands, gamma_4ph)
            tau_4ph = compute_lifetime(gamma_4ph)
            write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_4phVsFrequency.dat"), frequency_w_4ph, n_bands, tau_4ph)

            w_4ph_plusplus_path = os.path.join(temppath, "BTE.w_4ph_plusplus")
            frequency_w_4ph_plusplus, gamma_4ph_plusplus = read_scattering_rate(w_4ph_plusplus_path, n_qpoints, n_bands)
            write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_4ph_RecombinationVsFrequency.dat"), frequency_w_4ph_plusplus, n_bands, gamma_4ph_plusplus)
            tau_4ph_plusplus = compute_lifetime(gamma_4ph_plusplus)
            write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_4ph_RecombinationVsFrequency.dat"), frequency_w_4ph_plusplus, n_bands, tau_4ph_plusplus)

            w_4ph_plusminus_path = os.path.join(temppath, "BTE.w_4ph_plusminus")
            frequency_w_4ph_plusminus, gamma_4ph_plusminus = read_scattering_rate(w_4ph_plusminus_path, n_qpoints, n_bands)
            write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_4ph_RedistributionVsFrequency.dat"), frequency_w_4ph_plusminus, n_bands, gamma_4ph_plusminus)
            tau_4ph_plusminus = compute_lifetime(gamma_4ph_plusminus)
            write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_4ph_RedistributionVsFrequency.dat"), frequency_w_4ph_plusminus, n_bands, tau_4ph_plusminus)

            w_4ph_minusminus_path = os.path.join(temppath, "BTE.w_4ph_minusminus")
            frequency_w_4ph_minusminus, gamma_4ph_minusminus = read_scattering_rate(w_4ph_minusminus_path, n_qpoints, n_bands)
            write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRate_4ph_SplittingVsFrequency.dat"), frequency_w_4ph_minusminus, n_bands, gamma_4ph_minusminus)
            tau_4ph_minusminus = compute_lifetime(gamma_4ph_minusminus)
            write_lifetime_vs_frequency(os.path.join(temppath, "Lifetime_4ph_SplittingVsFrequency.dat"), frequency_w_4ph_minusminus, n_bands, tau_4ph_minusminus)

        w_path = os.path.join(temppath, "BTE.w")
        frequency_w, gamma_w = read_scattering_rate(w_path, n_qpoints, n_bands)
        write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRateVsFrequency.dat"), frequency_w, n_bands, gamma_w)
        tau_w = compute_lifetime(gamma_w)
        write_lifetime_vs_frequency(os.path.join(temppath, "LifetimeVsFrequency.dat"), frequency_w, n_bands, tau_w)

        w_final_path = os.path.join(temppath, "BTE.w_final")
        frequency_w_final, gamma_w_final = read_scattering_rate(w_final_path, n_qpoints, n_bands)
        write_scattering_rate_vs_frequency(os.path.join(temppath, "ScatteringRateFinalVsFrequency.dat"), frequency_w_final, n_bands, gamma_w_final)
        tau_w_final = compute_lifetime(gamma_w_final)
        write_lifetime_vs_frequency(os.path.join(temppath, "LifetimeFinalVsFrequency.dat"), frequency_w_final, n_bands, tau_w_final)

        wp3_path = os.path.join(temppath, "BTE.WP3")
        frequency_wp3, gamma_wp3 = read_weighted_phase_space(wp3_path, n_qpoints, n_bands)
        write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_3phVsFrequency.dat"), frequency_wp3, n_bands, gamma_wp3)

        wp3_plus_path = os.path.join(temppath, "BTE.WP3_plus")
        frequency_wp3_plus, gamma_wp3_plus = read_weighted_phase_space(wp3_plus_path, n_qpoints, n_bands)
        write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_3ph_AdsorptionVsFrequency.dat"), frequency_wp3_plus, n_bands, gamma_wp3_plus)

        wp3_minus_path = os.path.join(temppath, "BTE.WP3_minus")
        frequency_wp3_minus, gamma_wp3_minus = read_weighted_phase_space(wp3_minus_path, n_qpoints, n_bands)
        write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_3ph_EmissionVsFrequency.dat"), frequency_wp3_minus, n_bands, gamma_wp3_minus)

        if fourphonon:
            wp4_path = os.path.join(temppath, "BTE.WP4")
            frequency_wp4, gamma_wp4 = read_weighted_phase_space(wp4_path, n_qpoints, n_bands)
            write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_4phVsFrequency.dat"), frequency_wp4, n_bands, gamma_wp4)

            wp4_plusplus_path = os.path.join(temppath, "BTE.WP4_plusplus")
            frequency_wp4_plusplus, gamma_wp4_plusplus = read_weighted_phase_space(wp4_plusplus_path, n_qpoints, n_bands)
            write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_4ph_RecombinationVsFrequency.dat"), frequency_wp4_plusplus, n_bands, gamma_wp4_plusplus)

            wp4_plusminus_path = os.path.join(temppath, "BTE.WP4_plusminus")
            frequency_wp4_plusminus, gamma_wp4_plusminus = read_weighted_phase_space(wp4_plusminus_path, n_qpoints, n_bands)
            write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_4ph_RedistributionVsFrequency.dat"), frequency_wp4_plusminus, n_bands, gamma_wp4_plusminus)

            wp4_minusminus_path = os.path.join(temppath, "BTE.WP4_minusminus")
            frequency_wp4_minusminus, gamma_wp4_minusminus = read_weighted_phase_space(wp4_minusminus_path, n_qpoints, n_bands)
            write_weighted_phase_space_vs_frequency(os.path.join(temppath, "WeightedPhaseSpace_4ph_SplittingVsFrequency.dat"), frequency_wp4_minusminus, n_bands, gamma_wp4_minusminus)


if __name__ == "__main__":
    main()
