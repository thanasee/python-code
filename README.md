# VASP Python Utility Scripts

A collection of Python scripts for VASP output analysis and related tasks, developed for computational materials science research on the LANTA HPC cluster.

**Author:** [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en)

---

## Overview

This repository is organized into four functional categories:

1. **Thermal transport analysis** — extract and analyze lattice thermal conductivity variables from Phono3py HDF5 output files
2. **Structural analysis** — calculate structural properties (e.g., bond distances) from VASP POSCAR/CONTCAR files
3. **Mechanical properties** — extract and plot elastic tensors, piezoelectric tensors, and related quantities from VASP output files
4. **Structure preparation** — generate and manipulate POSCAR files for various VASP calculations

All scripts are standalone CLI tools written in Python using NumPy as the primary dependency. Each follows a consistent modular design with a `main()` entry point and NumPy-style docstrings.

---

## Requirements

- Python 3.8+
- NumPy
- h5py
- matplotlib
- ASE — Atomic Simulation Environment
- Phonopy/Phono3py (as data source; not imported directly)

---

## Script Reference

### 1. Thermal Transport Analysis

Scripts in this category read HDF5 output files from [Phono3py](https://phonopy.github.io/phono3py/) or output files from [ShengBTE](https://www.shengbte.org/) and analyze lattice thermal conductivity data.

---

#### `convergePhono3py.py`

Checks the convergence of lattice thermal conductivity (κ) as a function of q-mesh density by reading multiple `kappa-mXXX.hdf5` files from the current directory.

```
Usage: convergePhono3py.py
```

Automatically scans for all `kappa-m*.hdf5` files, sorts them by mesh number, and writes convergence data. Supports all Phono3py calculation modes: `--br`, `--lbte`, `--wigner`, and their combinations (`kappa`, `kappa_RTA`, `kappa_C`, `kappa_P_RTA`, `kappa_TOT_RTA`, `kappa_P_exact`, `kappa_TOT_exact`).

---

#### `analyzePhono3py.py`

Extracts mode-resolved thermal transport properties from a single Phono3py `kappa-mXXX.hdf5` file and writes output files suitable for plotting in xmgrace or matplotlib. Supports all Phono3py calculation modes (`--br`, `--lbte`, `--wigner`). If a Grüneisen HDF5 file is provided, Grüneisen parameters and group velocities are also extracted.

```
Usage: analyzePhono3py.py <kappa HDF5 file> <gruneisen HDF5 file (optional)>
```

Output filenames follow the pattern `<tag>-mXXXXXX.dat`, where the mesh token is preserved from the input filename. All κ tensor components are written in Voigt notation (xx, yy, zz, yz, xz, xy) in W/m-K.

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `KappaVsT` — total κ tensor vs. temperature
- `Kappa_bandVsT` — κ tensor decomposed into 3 acoustic branches + 1 summed optical branch vs. temperature
- `ContributeKappaVsT` — per-mode percentage contribution to total κ vs. temperature
- `Tau_CRTAVsT` — phonon lifetime (CRTA) vs. temperature *(if available)*
- Equivalent files for RTA, Wigner-C, Wigner-P (RTA), and Wigner-P (exact) variants *(if available)*

**Per-temperature spectral files** (one file per temperature, written to subdirectories `T{value}K/`):
- `KappaVsFrequency` — mode κ vs. phonon frequency (THz)
- `KappaVsMfp` — mode κ vs. mean free path (Å)
- `cumulative_KappaVsFrequency` — cumulative κ sorted by ascending frequency
- `derivative_KappaVsFrequency` — spectral κ density d(κ)/d(frequency)
- `cumulative_KappaVsMfp` — cumulative κ sorted by ascending mean free path
- `derivative_KappaVsMfp` — spectral κ density d(κ)/d(MFP)

**Grüneisen / group velocity files** *(written only when the Grüneisen HDF5 is provided)*:
- `GvVsFrequency` — group velocity magnitude vs. frequency
- `Gamma_isotopeVsFrequency` — isotope scattering rate vs. frequency *(if available)*

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

#### `analyzeShengBTE.py`

Extracts thermal transport properties from [ShengBTE](https://www.shengbte.org/) output files and writes them to `.dat` files suitable for plotting in xmgrace or matplotlib. [FourPhonon](https://github.com/FourPhonon/FourPhonon) four-phonon scattering quantities are auto-detected and extracted when present. Run from the ShengBTE output directory; no arguments are required.

```
Usage: analyzeShengBTE.py
```

Temperature subdirectories (`T-<int>K/`) are detected automatically from the working directory. If more than one temperature is found, an interactive prompt asks which temperature(s) to process. FourPhonon is detected by the presence of `BTE.w4_*` or `BTE.P4` files inside any `T-<int>K/` subdirectory.

All κ tensor components are written in Voigt notation (xx, yy, zz, yz, xz, xy) in W/(m·K).

**Temperature-dependent files** (one value per temperature row, written to the working directory):
- `kappa_tensor_RTA.dat` — total κ tensor vs. temperature (3-phonon RTA)
- `kappa_tensor_CONV.dat` — total κ tensor vs. temperature (iterative solution, if available)
- `kappa_tensor_4ph.dat` — total κ tensor vs. temperature (3ph+4ph RTA) *[FourPhonon only]*

**Per-temperature spectral files** (written into each selected `T-<int>K/` subdirectory):
- `kappa_mode.dat` — mode κ vs. phonon frequency (THz), 3-phonon RTA
- `kappa_mode_4ph.dat` — mode κ vs. frequency, 3ph+4ph RTA *[FourPhonon only]*
- `gruneisen.dat` — Grüneisen parameter vs. frequency
- `scattering_rate_3ph.dat` — 3-phonon scattering rate Γ (ps⁻¹) vs. frequency
- `lifetime_3ph.dat` — 3-phonon phonon lifetime τ (ps) vs. frequency
- `scattering_rate_4ph.dat` — 4-phonon scattering rate Γ (ps⁻¹) vs. frequency *[FourPhonon only]*
- `lifetime_4ph.dat` — 4-phonon phonon lifetime τ (ps) vs. frequency *[FourPhonon only]*
- `phase_space_3ph.dat` — 3-phonon phase space P₃ (absorption + emission) vs. frequency; file header records the scalar totals P3\_total, P3\_plus\_total, P3\_minus\_total
- `phase_space_4ph.dat` — 4-phonon phase space P₄ (++, +−, −−) vs. frequency; header records P4\_total and each process-type total *[FourPhonon only]*
- `cumulative_kappa_mfp.dat` — cumulative κ vs. mean free path (nm)
- `cumulative_kappa_freq.dat` — cumulative κ vs. phonon frequency (THz)

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

Developed by [Thanasee Thanasarnsurapong](https://scholar.google.com/citations?user=4KHXv9gAAAAJ&hl=en). For research and academic use.
