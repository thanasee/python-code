#!/usr/bin/env python

import os
import subprocess
from sys import argv, exit
from glob import glob
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, enforce_rotational_sum_rules
from hiphive.force_constants import ForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters


def usage():
    """Print usage information and exit."""
    text = """
Usage: enforceFC.py [INPUT_FC_FILE] [OUTPUT_FC_FILE]

This script enforces the rotational sum rules on the second-order force constants (FC2) of a supercell structure.
It reads the primitive and supercell structures, extracts the FC2 from a specified file, applies the rotational sum rules, and writes the modified FC2 back to a file.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


def read_structure(filename):
    """Read the structure from a file."""
    return read(filename)


def build_cluster_space(primitive, supercell, cutoff_margin=0.01):
    max_cutoff = estimate_maximum_cutoff(supercell) - cutoff_margin
    cutoffs = [max_cutoff]
    cluster_space = ClusterSpace(primitive, cutoffs)
    return cluster_space

def read_phonopy_fc2(supercell, fc_file="FORCE_CONSTANTS", fc_format="text"):
    return ForceConstants.read_phonopy(
        supercell,
        fc_file,
        format=fc_format,
    )

def project_fc2_to_parameters(force_constants, cluster_space):
    return extract_parameters(force_constants, cluster_space)


def apply_rotational_sum_rules(
    cluster_space,
    parameters,
    sum_rules=("Huang", "Born-Huang"),
):
    return enforce_rotational_sum_rules(
        cluster_space,
        parameters,
        list(sum_rules),
    )


def parameters_to_force_constants(cluster_space, parameters, supercell):
    fcp = ForceConstantPotential(cluster_space, parameters)
    return fcp.get_force_constants(supercell)


def write_phonopy_fc2(force_constants, fc_file, fc_format="text"):
    force_constants.write_to_phonopy(
        fc_file,
        format=fc_format,
    )


def main():
    
    if '-h' in argv or '--help' in argv or len(argv) > 3:
        usage()
    
    primitive_file = "POSCAR"
    if not os.path.isfile(primitive_file):
        print(f"Error: Primitive structure file '{primitive_file}' not found.")
        exit(1)

    supercell_file = "SPOSCAR"
    if not os.path.isfile(supercell_file):
        print(f"Error: Supercell structure file '{supercell_file}' not found.")
        exit(1)

    input_fc_file = argv[1] if len(argv) > 1 else "FORCE_CONSTANTS"
    if not os.path.isfile(input_fc_file):
        if not os.path.isfile("phonopy_params.yaml"):
            vasprun_files = sorted(f for f in glob("vasprun.xml-*") if f != "vasprun.xml-sposcar")
            subcommand = f"phonopy --fz vasprun.xml-sposcar {' '.join(vasprun_files)} --sp"
            subprocess.run(subcommand, shell=True, check=True)
    
        command = "phonopy-load phonopy_params.yaml --writefc --full-fc"
        subprocess.run(command, shell=True, check=True)
        input_fc_file = "FORCE_CONSTANTS"
    input_format = "hdf5" if input_fc_file.endswith(".hdf5") else "text"

    output_fc_file = argv[2] if len(argv) > 2 else (
    os.path.splitext(input_fc_file)[0] + "_rot" +
    (".hdf5" if input_format == "hdf5" else "")
    )
    output_format = "hdf5" if output_fc_file.endswith(".hdf5") else "text"

    cutoff_margin = 1e-5
    sum_rules = ("Huang", "Born-Huang")
    # ---------------------

    primitive = read_structure(primitive_file)
    supercell = read_structure(supercell_file)

    cluster_space = build_cluster_space(
        primitive,
        supercell,
        cutoff_margin,
    )

    force_constants = read_phonopy_fc2(
        supercell,
        input_fc_file,
        input_format,
    )

    parameters = project_fc2_to_parameters(
        force_constants,
        cluster_space,
    )

    parameters_rot = apply_rotational_sum_rules(
        cluster_space,
        parameters,
        sum_rules,
    )

    force_constants_rot = parameters_to_force_constants(
        cluster_space,
        parameters_rot,
        supercell,
    )

    write_phonopy_fc2(
        force_constants_rot,
        output_fc_file,
        output_format,
    )


if __name__ == "__main__":
    main()
