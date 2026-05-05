# VASP Python Utility Scripts

A collection of Python scripts for VASP output analysis and related tasks, developed for computational materials science research on the LANTA HPC cluster.

**Author:** [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en)

---

## Overview

This repository is organized into four functional categories:

1. **Thermal transport analysis** — post-process force constants, extract and analyze lattice thermal conductivity variables from Phono3py HDF5 output files and ShengBTE output files
2. **Structural analysis** — calculate structural properties (e.g., bond distances) from VASP POSCAR/CONTCAR files
3. **Mechanical properties** — extract and plot elastic tensors, piezoelectric tensors, and related quantities from VASP output files
4. **Structure preparation** — generate and manipulate POSCAR files for various VASP calculations
5. **MLFF utilities** — monitor training errors, evaluate MLFF accuracy against DFT references, and convert or merge VASP `ML_AB` training data files

All scripts are standalone CLI tools written in Python using NumPy as the primary dependency. Each follows a consistent modular design with a `main()` entry point and NumPy-style docstrings.

---

## Requirements

- Python 3.8+
- NumPy
- h5py
- matplotlib
- ASE — Atomic Simulation Environment
- hiPhive (for `enforceFC.py`)
- Phonopy/Phono3py (as data source; not imported directly)

---

## Script Reference

### 1. Thermal Transport Analysis

Scripts in this category read HDF5 output files from [Phono3py](https://phonopy.github.io/phono3py/) or output files from [ShengBTE](https://www.shengbte.org/) and analyze lattice thermal conductivity data.

---

#### `enforceFC.py`

Enforces rotational sum rules (Huang and Born-Huang) on second-order force constants (FC2) using [hiPhive](https://hiphive.materialsmodeling.org/), and writes the corrected FC2 in Phonopy-compatible format. Reads `POSCAR` (primitive cell) and `SPOSCAR` (supercell) from the working directory. If no input FC file is found, the script auto-generates one by calling Phonopy on any `vasprun.xml-*` displacement files present in the working directory.

```
Usage: enforceFC.py [INPUT_FC_FILE] [OUTPUT_FC_FILE]
```

File format is auto-detected from the extension: `.hdf5` → HDF5; any other extension → Phonopy text format. Both input and output independently follow this rule. The cutoff radius for the hiPhive cluster space is set to the maximum cutoff supported by the supercell geometry minus a small margin (0.00001 Å).

**Defaults (when arguments are omitted):**
- `INPUT_FC_FILE` — `FORCE_CONSTANTS` (Phonopy text format)
- `OUTPUT_FC_FILE` — `<input_basename>_rot` (same format as input; e.g., `FORCE_CONSTANTS_rot` or `FORCE_CONSTANTS_rot.hdf5`)

**Auto-generation of FC file (when `INPUT_FC_FILE` is absent):**
1. If `phonopy_params.yaml` is not present, runs `phonopy --fz vasprun.xml-sposcar <vasprun.xml-*> --sp` to generate it
2. Runs `phonopy-load phonopy_params.yaml --writefc --full-fc` to write `FORCE_CONSTANTS`

**Output:** One FC2 file with rotational sum rules enforced, ready for use with Phonopy or Phono3py.

---


#### `convergePhono3py.py`

Checks the convergence of lattice thermal conductivity (κ) as a function of q-mesh density by reading multiple `kappa-mXXX.hdf5` files from the current directory.

```
Usage: convergePhono3py.py
```

Automatically scans for all `kappa-m*.hdf5` files, sorts them by mesh number, and writes convergence data. Supports all Phono3py calculation modes: `--br`, `--lbte`, `--wigner`, and their combinations (`kappa`, `kappa_RTA`, `kappa_C`, `kappa_P_RTA`, `kappa_TOT_RTA`, `kappa_P_exact`, `kappa_TOT_exact`).

**2D renormalization:** After loading the HDF5 files, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, the vacuum direction is assumed to be c. A dimensionless renormalization factor derived from the c-axis length is applied to all κ values, correcting Phono3py's bulk-convention κ to the 2D-referenced value. Units remain W/m-K throughout.

---

#### `analyzePhono3py.py`

Extracts mode-resolved thermal transport properties from a single Phono3py `kappa-mXXX.hdf5` file and writes output files suitable for plotting in xmgrace or matplotlib. Supports all Phono3py calculation modes (`--br`, `--lbte`, `--wigner`). If a Grüneisen HDF5 file is provided, Grüneisen parameters and group velocities are also extracted.

```
Usage: analyzePhono3py.py <kappa HDF5 file> <gruneisen HDF5 file (optional)>
```

Output filenames follow the pattern `<tag>-mXXXXXX.dat`, where the mesh token is preserved from the input filename (e.g., `kappa-m111111.hdf5` → `KappaVsT-m111111.dat`). All κ tensor components are written in Voigt notation (xx, yy, zz, yz, xz, xy) in W/m-K.

**2D renormalization:** After loading the HDF5 file, the script interactively prompts for dimensionality (1 = 3D, 2 = 2D). For 2D materials, the vacuum direction is assumed to be c. A dimensionless renormalization factor derived from the c-axis length is applied to all κ arrays before any output is written, correcting Phono3py's bulk-convention κ to the 2D-referenced value. Units remain W/m-K throughout. The renormalization applies to all output file groups below.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `KappaVsT.dat` / `Kappa_bandVsT.dat` — total κ tensor and band decomposition (3 acoustic + 1 summed optical) vs. temperature
- `ContributeKappaVsT.dat` — per-mode percentage contribution to total κ vs. temperature
- `CvVsT.dat` — total heat capacity Cv (eV/K) vs. temperature
- `Tau_CRTAVsT.dat` / `Tau_AvgVsT.dat` — CRTA and average phonon lifetime τ (ps) vs. temperature
- `Kappa_RTAVsT.dat` / `Kappa_RTA_bandVsT.dat` — RTA κ tensor and band decomposition vs. temperature *(--lbte only)*
- `Kappa_C*VsT.dat` — wave-like (coherence) Wigner κ tensor and band decomposition vs. temperature *(--wigner only)*
- `Kappa_P_RTA*VsT.dat` / `Kappa_TOT_RTA*VsT.dat` — particle-like and total Wigner κ (RTA) vs. temperature *(--wigner --br only)*
- `Kappa_P_exact*VsT.dat` / `Kappa_TOT_exact*VsT.dat` — particle-like and total Wigner κ (exact) vs. temperature *(--wigner --lbte only)*

**Temperature-independent files** (written to the working directory):
- `GvVsFrequency.dat` / `Gv_amplitudeVsFrequency.dat` — group velocity vector (vx, vy, vz) and amplitude |v| vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz) *(if Grüneisen HDF5 provided)*
- `Gamma_isotopeVsFrequency.dat` — isotope scattering rate vs. frequency *(if available)*

**Per-temperature spectral files** (one file per temperature, written to subdirectories `T<value>K/`):
- `KappaVsFrequency.dat` / `KappaVsMfp.dat` — mode κ vs. phonon frequency (THz) and vs. mean free path (Å)
- `cumulative_KappaVsFrequency.dat` / `cumulative_KappaVsMfp.dat` — cumulative κ sorted by ascending frequency and MFP
- `derivative_KappaVsFrequency.dat` / `derivative_KappaVsMfp.dat` — spectral κ density d(κ)/d(frequency) and d(κ)/d(MFP)

---

#### `compareIFCs.py`

Compares interatomic force constants (IFCs) between DFT and MLFF calculations by reading Phono3py HDF5 files and writing the residual (MLFF − DFT) to `.dat` files. Auto-detects whether the file contains 2nd-order (`force_constants`) or 3rd-order (`fc3`) IFCs.

```
Usage: compareIFCs.py <DFT's force constants HDF5 input> <MLFF's force constants HDF5 input>
```

**Output files:**
- `2ndIFCs.dat` — 2nd-order IFC comparison in eV/Å^2
- `3rdIFCs.dat` — 3rd-order IFC comparison in eV/Å^3

---

#### `getQPATH.py`

Reads the high-symmetry q-point path positions from a `band.dat` file produced by `phonopy-bandplot --gnuplot` and writes `QLINES.dat` — a boundary-line file in the same format as `KLINES.dat` from VASPKIT, suitable for overlaying q-path tick marks and the frequency window on a phonon band structure plot in xmgrace or gnuplot.

```
Usage: getQPATH.py <band.dat input>
```

Q-point path distances (1/Å) are read from the second line of the input file. The frequency range is determined automatically as floor(f_min) to ceil(f_max) from all frequency values in the file. For each interior high-symmetry q-point, three coordinate pairs are written to trace a vertical tick from `fmin` to `fmax` and back. The outer box boundaries and the zero-frequency axis are appended at the end.

**Output:** `QLINES.dat` — columns: q-path distance (1/Å), frequency boundary (THz).

---

#### `analyzeShengBTE.py`

Extracts thermal transport properties from [ShengBTE](https://www.shengbte.org/) output files and writes output files suitable for plotting in xmgrace or matplotlib. If `4ph` is specified, all [FourPhonon](https://github.com/FourPhonon/FourPhonon) four-phonon scattering quantities are also extracted. Run from the ShengBTE output directory.

```
Usage: analyzeShengBTE.py <3ph/4ph>
```

Temperature subdirectories (`T<value>K/`) are detected automatically from the working directory. All κ tensor components are written as the full 3×3 tensor (xx, xy, xz, yx, yy, yz, zx, zy, zz) in W/(m·K). All scattering rate and lifetime files are written per phonon branch. Phonon lifetimes are computed as τ = 1/(2 × 2π × Γ) (ps); modes with Γ ≤ 0 are assigned τ = 0.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `Kappa_RTAVsT.dat` / `Kappa_CONVVsT.dat` — total κ tensor vs. temperature, RTA and iterative (CONV) solutions
- `Kappa_bandVsT.dat` — κ tensor decomposed into 3 acoustic branches + 1 summed optical branch vs. temperature
- `HeatCapacityVsT.dat` — total heat capacity Cv (eV/K) vs. temperature

**Temperature-independent files** (written to the working directory):
- `GroupVelocityVsFrequency.dat` / `GroupVelocityAmplitudeVsFrequency.dat` — group velocity vector (vx, vy, vz) and amplitude |v| in km/s vs. frequency (THz)
- `GruneisenVsFrequency.dat` — Grüneisen parameter vs. frequency (THz)
- `ScatteringRate_IsotopicVsFrequency.dat` / `Lifetime_IsotopicVsFrequency.dat` — isotope Γ (ps^-1) and τ (ps) vs. frequency
- `P3VsFrequency.dat`, `P3_AdsorptionVsFrequency.dat`, `P3_EmissionVsFrequency.dat` — total, absorption (+), and emission (−) 3-phonon phase space vs. frequency; each header records the corresponding scalar total
- `P4*.dat` — same set for 4-phonon phase space (total, recombination ++, redistribution +-, splitting −−) *[FourPhonon only]*

**Per-temperature files** (written into each `T<value>K/` subdirectory):
- `CumulativeKappaVsMFP.dat` / `CumulativeKappaVsFrequency.dat` — cumulative κ tensor vs. mean free path (Å) and vs. frequency (THz)
- `ScatteringRate_3ph*.dat` / `Lifetime_3ph*.dat` — 3ph scattering rate Γ and lifetime τ vs. frequency; process variants: total, `_Adsorption`, `_Emission`
- `ScatteringRateVsFrequency.dat` / `LifetimeVsFrequency.dat` — total combined (3ph + isotope) Γ and τ vs. frequency
- `ScatteringRateFinalVsFrequency.dat` / `LifetimeFinalVsFrequency.dat` — final iterative Γ and τ vs. frequency
- `WeightedPhaseSpace_3ph*.dat` — weighted 3-phonon phase space vs. frequency; process variants: total, `_Adsorption`, `_Emission`
- `ScatteringRate_4ph*.dat` / `Lifetime_4ph*.dat` — 4ph Γ and τ vs. frequency; process variants: total, `_Recombination`, `_Redistribution`, `_Splitting` *[FourPhonon only]*
- `WeightedPhaseSpace_4ph*.dat` — weighted 4-phonon phase space vs. frequency; process variants: total, `_Recombination`, `_Redistribution`, `_Splitting` *[FourPhonon only]*

---

### 2. Structural Analysis

Scripts that read VASP POSCAR/CONTCAR structure files and compute structural properties.

---

#### `vaspVibration.py`

Extracts vibrational normal modes from a VASP `OUTCAR` or Phonopy `YAML` file and writes each mode as an XSF file for visualization in VESTA or XCrySDen.

```
Usage: vaspVibration.py <structure file> <OUTCAR or phonopy YAML> [scaling factor]
```

- VASP OUTCAR: modes written in descending frequency order (VASP convention)
- Phonopy YAML: modes written in ascending frequency order (Phonopy convention)

**Output:** One `.xsf` file per normal mode.

---

### 3. Mechanical Properties

Scripts for extracting, computing, and visualizing elastic and piezoelectric properties from VASP output files.

---

#### `vaspMechanics.py`

Reads a VASP `POSCAR` and `OUTCAR` to compute mechanical properties for either 2D or 3D materials.

```
Usage: vaspMechanics.py <POSCAR> <OUTCAR>
```

**2D mode** (N/m units):
- Detects lattice type: hexagonal, square, rectangular, or oblique
- Computes angle-dependent Young's modulus E(θ), Poisson's ratio ν(θ), and shear modulus G(θ) using compliance tensor rotation
- **Output:** `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat`

**3D mode** (GPa units):
- Identifies crystal system from spacegroup number via ASE
- Computes Voigt, Reuss, and Hill (VRH) averages for bulk and shear moduli
- Derives Young's modulus, Poisson's ratio, P-wave modulus, Lamé parameter, Pugh's ratio
- Computes sound velocities (transverse, longitudinal, mean) and Debye temperature
- Computes anisotropy indices: universal (A_U), bulk (A_B), shear (A_G), and planar (A_1, A_2, A_3)
- **Output:** `Elastic.dat`, `Mechanics.dat`, `Anisotropy.dat`

---

#### `ElasticTensor2D.py`

Calculates the 2D elastic tensor from VASP DFT calculations using the strain-energy method, with two operating modes.

```
Usage: ElasticTensor2D.py pre  <structure file>   # Generate strained POSCARs
       ElasticTensor2D.py post                    # Fit energies and extract constants
```

**`pre` mode:** Applies a set of strain tensors to the input structure and writes strained POSCAR files to individual directories. Detects crystal system (oblique vs. non-oblique) and applies the appropriate strain set.

**`post` mode:** Reads total energies from each strain directory's `OUTCAR`, fits energy vs. strain to a quadratic, extracts elastic constants (C11, C22, C12, C66, and C16/C26 for oblique), checks mechanical stability via eigenvalue positivity, and computes angle-dependent mechanical properties.

**Output:** `Elastic.dat`, `Young.dat`, `Poisson.dat`, `Shear.dat`

---

#### `vaspPiezoelectric.py`

Extracts the piezoelectric stress tensor (e, C/m^2) and elastic stiffness tensor (C, GPa or N/m) from a VASP `OUTCAR` and computes the piezoelectric strain tensor (d = e · S, pm/V) via the compliance tensor S = inverse(C).

```
Usage: vaspPiezoelectric.py <OUTCAR>
```

Supports both 2D materials (with vacuum-layer thickness correction) and 3D bulk materials. Uses a three-level fallback chain for the elastic tensor (ionic + electronic → total → user input).

**Output:** Piezoelectric tensor files in Voigt notation.

---

#### `plotMechanics.py`

Plots polar diagrams of angle-dependent mechanical properties (Young's modulus, Poisson's ratio, Shear modulus) for up to 6 materials simultaneously for visual comparison.

```
Usage: plotMechanics.py <file1> [file2 ... file6]
```

Input files are the `.dat` output files from `vaspMechanics.py` or `ElasticTensor2D.py`. Auxetic materials (negative Poisson's ratio) are handled by plotting |ν| as a dashed envelope. Output is saved as a 300 dpi PNG.

---

### 4. Structure Preparation

Scripts for generating, transforming, and manipulating VASP POSCAR files for various DFT calculations.

---

#### `vaspSupercell.py`

Generates a supercell POSCAR from a unit cell input using a 3×3 expansion matrix.

```
Usage: vaspSupercell.py <POSCAR> <output POSCAR>
```

Uses an integer-exact (adjugate-matrix-based) grid point generation to avoid floating-point rounding errors. Supports anisotropic scale factors and Selective Dynamics.

---

#### `vaspStrain.py`

Applies a strain tensor to a crystal structure POSCAR for DFT elastic constant calculations.

```
Usage: vaspStrain.py <POSCAR> <output POSCAR>
```

Accepts 3 values (diagonal strain) or 9 values (full 3×3 tensor). Off-diagonal inputs are symmetrized. Applies the deformation gradient F = I + ε to the lattice matrix, preserving fractional atomic coordinates.

---

#### `vaspStack.py`

Generates bilayer and heterostructure POSCAR files from one or two input POSCAR files by stacking along the c-axis.

```
Usage: vaspStack.py <POSCAR> [POSCAR2]
```

Features: lattice compatibility checking (`check_lattice`), 2D Bravais lattice type detection, high-symmetry stacking shift grids per lattice type, mirror-flip of the second layer, and a summary `STACK_LIST.txt` of all generated POSCARs.

---

#### `vaspTwist.py`
**Inspired by:** [CellMatch](https://doi.org/10.1016/j.cpc.2015.08.038)

Generates moiré twisted bilayer POSCAR files by searching for commensurate supercells across twist angles. The workflow is split into two modes run sequentially.

```
Usage: vaspTwist.py match    <bottom POSCAR> [top POSCAR]   # Step 1: search & write TWIST_LIST.dat
       vaspTwist.py generate <bottom POSCAR> [top POSCAR]   # Step 2: read TWIST_LIST.dat & write POSCARs
```

**`match` mode:** Searches twist angles from 0° to 180° (step: 0.1°) for commensurate supercell pairs using the CellMatch symmetric relative distance metric. Results are written to `TWIST_LIST.dat`, sorted by (θ, strain). No POSCAR is written at this stage.

**`generate` mode:** Reads `TWIST_LIST.dat`, displays the candidate table interactively, and writes one POSCAR per selected stacking configuration. The post-search prompt accepts individual indices, `'all'`, or `'none'`.

Providing a single POSCAR uses it for both layers (homobilayer). Providing two different POSCARs enables heterobilayer mode. Key parameters: `MAX_ATOMS = 200` (hard cap on total supercell atoms); `N_MAX` derived as `ceil(sqrt(MAX_ATOMS / primitive_atoms))`; `THETA_STEP = 0.1°` (suitable for angles above ~2°; use `0.01°` for magic-angle regimes below ~2°). Lattice-type detection and high-symmetry stacking shift grids are implemented for hexagonal, square, rectangular, and oblique supercells.

---

#### `vaspShift.py`

Shifts atomic positions in a crystal structure to a desired reference frame, with modes for 0D molecules, 1D nanowires, 2D sheets, and 3D bulk materials.

```
Usage: vaspShift.py <POSCAR> <output POSCAR>
```

---

#### `vaspRotate.py`

Rotates atoms in a VASP POSCAR/CONTCAR file about a user-specified pivot point and axis using Rodrigues' rotation formula.

```
Usage: vaspRotate.py <POSCAR> <output POSCAR>
```

Supports rotation about arbitrary axes; pivot can be set to a specific atom, the center of mass, or a custom Cartesian point.

---

#### `vaspMirror.py`

Reflects a VASP POSCAR structure across a chosen Cartesian plane (XY, XZ, or YZ).

```
Usage: vaspMirror.py <POSCAR> <output POSCAR>
```

---

#### `vaspFix.py`

Applies Selective Dynamics constraints to a VASP POSCAR, fixing atoms in specified Cartesian directions.

```
Usage: vaspFix.py <POSCAR> <output POSCAR>
```

Three atom-selection modes: by index/label, by cutoff radius (PBC-aware), or from an existing `SELECTED_FIX_ATOMS_LIST` file. Writes a `SELECTED_FIX_ATOMS_LIST` log for reference and reuse.

---

#### `vaspAdsorb.py`

Combines a substrate and an adsorbent POSCAR file into a single POSCAR suitable for adsorption DFT calculations.

```
Usage: vaspAdsorb.py <substrate POSCAR> <adsorbent POSCAR> <output POSCAR>
```

Two placement modes: on top of a specific substrate site (Mode 1) or arranged symmetrically around a target atom in a ring (Mode 2). Handles Selective Dynamics merging and element reordering for VASP compatibility.

---

#### `vaspReformat.py`

Converts a VASP POSCAR/CONTCAR to standardized VASP5 format, with optional Selective Dynamics support and per-atom label comments.

```
Usage: vaspReformat.py <POSCAR> <output POSCAR>
```

---

#### `poscar2control.py`

Converts a VASP POSCAR file into a CONTROL input file for the [ShengBTE](https://www.shengbte.org/) lattice thermal conductivity code (Fortran BTE solver).

```
Usage: poscar2control.py <POSCAR>
```

Interactively prompts for supercell matrix and phonon process order (3-phonon or 4-phonon, with CPU/GPU branching). Sets `lfactor=0.1` (Å → nm) as required by ShengBTE. **Output:** `CONTROL.initial`

---

### 5. MLFF Utilities

Scripts for working with VASP Machine Learning Force Fields (MLFF).

---

#### `mlError.py`

Extracts Bayesian Error Estimation in Forces (BEEF) and Root Mean Square Errors (RMSE) from a VASP `ML_LOGFILE` during MLFF on-the-fly training.

```
Usage: mlError.py <ML_LOGFILE>
```

**Output:** `BEEF.dat`, `ERR.dat` (formatted for xmgrace)

---

#### `mlRegression.py`

Evaluates MLFF accuracy against DFT reference data from the VASP `ML_REG` file, computing RMSE, MAE, and R-square for energies (meV/atom), forces (eV/Å), and stresses (kbar).

```
Usage: mlRegression.py <ML_REG>
```

**Output:** `Energy.dat`, `Force.dat`, `Stress.dat`, `ERROR.dat`

---

#### `mlab2extxyz.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Converts VASP's `ML_AB` binary training data file to extended XYZ (`.extxyz`) format for use with external MLFF training frameworks such as MACE, NequIP, and GPUMD.

```
Usage: mlab2extxyz.py <ML_AB input> <output.extxyz>
```

Each configuration block is mapped to one extxyz frame with lattice, positions, energy, forces, and stress. Stress is converted from kbar to eV/Å^3.

---

#### `mergeMLAB.py`
**Inspired by:** [utf/pymlff](https://github.com/utf/pymlff)

Merges multiple VASP `ML_AB` training data files into a single unified file, unifying element lists, basis sets, and renumbering configurations.

```
Usage: mergeMLAB.py <ML_AB 1> <ML_AB 2> [ML_AB 3 ...] <output ML_AB>
```

Resolves header metadata conflicts (reference energies, atomic masses) by first-file-wins with a warning for mismatches.

---

## Design Conventions

All scripts follow the same conventions modeled on `vaspSupercell.py`:

- Single-responsibility functions with NumPy-style docstrings (`Parameters`, `Returns`, inline notes for unit conversions and formulas)
- `main()` entry point with `if __name__ == '__main__'` guard
- Interactive input loops with validation and retry on invalid input
- Handles both VASP4 (no element line), VASP5, and VASP6 (with Hash code) POSCAR formats
- Handles Selective Dynamics, anisotropic scale factors, and non-orthogonal cells
- Atom-label comments (e.g., `Mo001`, `S002`) in output POSCARs for VESTA/XCrySDen identification
- Output files formatted for xmgrace (`.dat` with column headers) unless otherwise noted

---

## License

Developed by [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en) and [Cluade](https://claude.ai/). For research and academic use.
