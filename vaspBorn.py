#!/usr/bin/env python

from sys import argv, exit
import os
import xml.etree.ElementTree as ET
import numpy as np


def usage():
    """Print usage information and exit."""

    print("""
Usage: vaspBornCharge.py <OUTCAR or vasprun.xml>

This script reads a VASP OUTCAR or vasprun.xml file and extracts:
  - Static (ion-clamped) dielectric tensor
  - Born effective charge tensors for each ion

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)


def read_OUTCAR(filepath):
    """Read Born effective charges and static dielectric tensor from OUTCAR.

    Parses the OUTCAR written by a VASP DFPT run (IBRION=8,
    LEPSILON=.TRUE.).  The ion-clamped (electronic) dielectric tensor is
    taken from the ``MACROSCOPIC STATIC DIELECTRIC TENSOR`` block that
    appears *before* the ionic contributions.  Born effective charges are
    read from the ``BORN EFFECTIVE CHARGES`` block.

    Parameters
    ----------
    filepath : str
        Path to the OUTCAR file.

    Returns
    -------
    dielectric : numpy.ndarray, shape (3, 3)
        Static (ion-clamped + ionic) dielectric tensor, dimensionless.
    born_charges : numpy.ndarray, shape (NIONS, 3, 3)
        Born effective charge tensor for each ion, in units of
        elementary charge (e).
    nions : int
        Number of ions in the cell.
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # ── Collect all dielectric tensor blocks ─────────────────────────────────
    # VASP writes two: (1) ion-clamped, (2) ion-clamped + ionic.
    # Phono3py expects the full static tensor (including ionic contribution),
    # which is the LAST occurrence.
    dielectric_blocks = []
    i = 0
    while i < len(lines):
        if 'MACROSCOPIC STATIC DIELECTRIC TENSOR' in lines[i]:
            # Skip the header line and the separator line
            j = i + 2
            block = []
            while j < len(lines) and len(block) < 3:
                parts = lines[j].split()
                if len(parts) == 3:
                    try:
                        block.append([float(x) for x in parts])
                    except ValueError:
                        pass
                j += 1
            if len(block) == 3:
                dielectric_blocks.append(np.array(block))
        i += 1

    if not dielectric_blocks:
        print("ERROR!\nNo dielectric tensor found in OUTCAR.\n"
              "Make sure the calculation used IBRION=8 and LEPSILON=.TRUE.")
        exit(1)

    # Use the last occurrence (includes ionic contribution)
    dielectric = dielectric_blocks[-1]

    # ── Born effective charges ────────────────────────────────────────────────
    born_charges = []
    i = 0
    while i < len(lines):
        if 'BORN EFFECTIVE CHARGES' in lines[i] and 'ion' not in lines[i].lower():
            # Format: header, blank/separator, then repeated:
            #   "ion   N"
            #   "   1  Zxx  Zxy  Zxz"
            #   "   2  Zyx  Zyy  Zyz"
            #   "   3  Zzx  Zzy  Zzz"
            j = i + 1
            # Skip separator lines
            while j < len(lines) and lines[j].strip().startswith('-'):
                j += 1
            while j < len(lines):
                # Check for "ion   N" marker
                if lines[j].strip().lower().startswith('ion'):
                    # Read the 3×3 tensor for this ion
                    tensor = []
                    k = j + 1
                    while k < len(lines) and len(tensor) < 3:
                        parts = lines[k].split()
                        if len(parts) == 4:
                            try:
                                tensor.append([float(x) for x in parts[1:4]])
                            except ValueError:
                                pass
                        k += 1
                    if len(tensor) == 3:
                        born_charges.append(np.array(tensor))
                    j = k
                elif lines[j].strip().startswith('-') or not lines[j].strip():
                    j += 1
                else:
                    break
            break
        i += 1

    if not born_charges:
        print("ERROR!\nNo Born effective charges found in OUTCAR.\n"
              "Make sure the calculation used IBRION=8 and LEPSILON=.TRUE.")
        exit(1)

    nions = len(born_charges)
    born_charges = np.array(born_charges)  # shape (NIONS, 3, 3)

    return dielectric, born_charges, nions


def read_vasprun(filepath):
    """Read Born effective charges and static dielectric tensor from vasprun.xml.

    Parses the ``<dielectricfunction>`` and ``<born_charges>`` sections of
    the vasprun.xml produced by a VASP DFPT run (IBRION=8, LEPSILON=.TRUE.).

    Parameters
    ----------
    filepath : str
        Path to the vasprun.xml file.

    Returns
    -------
    dielectric : numpy.ndarray, shape (3, 3)
        Static dielectric tensor (ion-clamped + ionic), dimensionless.
    born_charges : numpy.ndarray, shape (NIONS, 3, 3)
        Born effective charge tensor for each ion, in units of
        elementary charge (e).
    nions : int
        Number of ions in the cell.
    """

    try:
        tree = ET.parse(filepath)
    except ET.ParseError as err:
        print(f"ERROR!\nFailed to parse vasprun.xml: {err}")
        exit(1)

    root = tree.getroot()

    # ── Dielectric tensor ─────────────────────────────────────────────────────
    # vasprun.xml contains <dielectricfunction> with children <epsilon> (real)
    # and possibly <epsilon_ion>.  The full static tensor is the sum of the
    # electronic part and the ionic contribution.  We look for the array named
    # "density" inside the <dielectricfunction> block.
    # Alternatively, the <array name="epsilon"> under <calculation> holds the
    # ion-clamped tensor and <array name="epsilon_ion"> the ionic part.
    dielectric = None

    # Strategy: find the last <varray name="epsilon"> and <varray name="epsilon_ion">
    # and sum them.  If only one exists, use that.
    eps_electronic = None
    eps_ionic = None

    for varray in root.iter('varray'):
        name = varray.get('name', '')
        if name == 'epsilon':
            rows = []
            for v in varray.findall('v'):
                rows.append([float(x) for x in v.text.split()])
            if len(rows) == 3:
                eps_electronic = np.array(rows)
        elif name == 'epsilon_ion':
            rows = []
            for v in varray.findall('v'):
                rows.append([float(x) for x in v.text.split()])
            if len(rows) == 3:
                eps_ionic = np.array(rows)

    if eps_electronic is not None and eps_ionic is not None:
        dielectric = eps_electronic + eps_ionic
    elif eps_electronic is not None:
        dielectric = eps_electronic
        print("Note: Only electronic dielectric tensor found (no ionic part).")
    else:
        print("ERROR!\nNo dielectric tensor found in vasprun.xml.\n"
              "Make sure the calculation used LEPSILON=.TRUE.")
        exit(1)

    # ── Born effective charges ────────────────────────────────────────────────
    # Located in <array name="born_charges"> under <calculation>
    born_charges = []

    for array in root.iter('array'):
        if array.get('name') == 'born_charges':
            for set_elem in array.iter('set'):
                tensor = []
                for v in set_elem.findall('v'):
                    tensor.append([float(x) for x in v.text.split()])
                if len(tensor) == 3:
                    born_charges.append(np.array(tensor))

    if not born_charges:
        print("ERROR!\nNo Born effective charges found in vasprun.xml.\n"
              "Make sure the calculation used LEPSILON=.TRUE.")
        exit(1)

    nions = len(born_charges)
    born_charges = np.array(born_charges)  # shape (NIONS, 3, 3)

    return dielectric, born_charges, nions


def format_val(x):
    """Format a single float value for INCAR.LR output.

    Parameters
    ----------
    x : float
        Value to format.

    Returns
    -------
    str
        String representation with 8 decimal places.
    """

    return f'{x:>12.8f}'


def write_INCAR_LR(dielectric, born_charges, nions, output_file='INCAR.LR'):
    """Write PHON_DIELECTRIC and PHON_BORN_CHARGES tags to INCAR.LR.

    The output format follows the Phono3py INCAR backslash-continuation
    convention.  Each tag is written as::

        PHON_DIELECTRIC = \\
        e00  e10  e20 \\
        e01  e11  e21 \\
        e02  e12  e22

        PHON_BORN_CHARGES = \\
        Z1_00  Z1_10  Z1_20 \\
        ...
        ZN_02  ZN_12  ZN_22

    where columns of each tensor are written as successive lines
    (i.e. line i contains column i: ``tensor[:, i]``), matching the
    format expected by Phono3py.

    Parameters
    ----------
    dielectric : numpy.ndarray, shape (3, 3)
        Static dielectric tensor, dimensionless.
    born_charges : numpy.ndarray, shape (NIONS, 3, 3)
        Born effective charge tensors, in units of elementary charge (e).
    nions : int
        Number of ions in the cell.
    output_file : str, optional
        Path to the output file.  Default is ``'INCAR.LR'``.
    """

    lines = []

    # ── PHON_DIELECTRIC ───────────────────────────────────────────────────────
    lines.append('PHON_DIELECTRIC = \\')
    dielectric_strings = []
    for i in range(3):
        dielectric_strings.append('  '.join(map(format_val, dielectric[:, i])))
    lines.append('\\\n'.join(dielectric_strings))
    lines.append('')

    # ── PHON_BORN_CHARGES ─────────────────────────────────────────────────────
    lines.append('PHON_BORN_CHARGES = \\')
    born_strings = []
    for b in born_charges:
        for i in range(3):
            born_strings.append('  '.join(map(format_val, b[:, i])))
    lines.append('\\\n'.join(born_strings))
    lines.append('')

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def main():
    """
    Parse arguments, read the specified VASP output file, extract the dielectric tensor and Born effective charges, and write output to INCAR.LR.
    """

    if '-h' in argv or len(argv) != 2:
        usage()

    filepath = argv[1]

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    filename = os.path.basename(filepath)

    # Dispatch by filename
    if filename == 'vasprun.xml':
        dielectric, born_charges, nions = read_vasprun(filepath)
    elif filename == 'OUTCAR':
        dielectric, born_charges, nions = read_OUTCAR(filepath)
    else:
        # Try to detect by content
        with open(filepath, 'r') as f:
            first_line = f.readline()
        if first_line.strip().startswith('<'):
            dielectric, born_charges, nions = read_vasprun(filepath)
        else:
            dielectric, born_charges, nions = read_OUTCAR(filepath)

    write_INCAR_LR(dielectric, born_charges, nions)


if __name__ == '__main__':
    main()
