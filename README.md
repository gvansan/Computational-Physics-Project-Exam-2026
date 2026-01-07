# Hydrogen Molecule Dissociation via Configuration Interaction (CI)

This repository contains a Jupyter notebook (`FILL IN NAME OF NOTEBOOK`) that explores the dissociation curve of the hydrogen molecule ($H_2$) using Restricted Hartree-Fock (RHF), Unrestricted Hartree-Fock (UHF), and CI (Configuration Interaction). The notebook shows how CI improves upon standard Hartree-Fock results.

## Overview

### Problem Statement

The dissociation curve of $H_2$ reveals how its energy changes as the internuclear distance varies. Hartree-Fock theory provides a mean-field approximation, but fails to account for electron correlation, especially at large distances where correct dissociation is crucial. Configuration Interaction (CI) includes additional excited-state determinants, improving correlation and providing more accurate results across all geometries.

### Techniques

1. **RHF (Restricted Hartree-Fock):**
   - Both electrons occupy the same spatial orbital.
   - Suitable for short bond distances (equilibrium).
   - Incorrectly predicts energy at dissociation.

2. **UHF (Unrestricted Hartree-Fock):**
   - Alpha and beta electrons can occupy different orbitals.
   - Improves results at large bond distances by enabling electron localization.

3. **CI (Configuration Interaction):**
   - Uses a linear combination of Slater determinants (SDs).
   - Introduces single and double electron excitations to compute the energy.
   - Diagonalizes the CI matrix to account for both static and dynamic correlation.
   - Approaches the exact solution (within the chosen basis set).

## Structure of the Notebook

- **Imports and Setup:** Loads required Python libraries (NumPy, SciPy, matplotlib, tqdm) and prepares basis sets and grids for the calculation.
- **Integrals Calculation:** Computes one- and two-electron integrals using a chosen basis (aug-cc-pvdz.gbs).
- **Hartree-Fock Calculations:** Performs RHF and UHF calculations to yield energies and electron densities across a range of H–H distances.
- **Plots:**
  - Molecular orbitals $\sigma_g$ and $\sigma_u$ 
  - Dissociation curves for RHF, UHF, CI and experimental results.
  - Binding energy curve for RHF and CI
  - CI composition configuration as a function of R
  - Electron densities at equilibrium and large separations.
- **CI Calculations:** Converts atomic orbital basis to molecular orbitals, restricting to the 2 lowest molecular orbitals. It sets up and diagonalizes the CI Hamiltonian using the correct Slater-Condon rules and parity and spin selection rules. It analyzes the resulting energies and densities.
- **Comparison to Experiment:** Includes experimental data for $H_2$ energy based on a parametrization with 3 variables.
- **Analysis:** Examines correlation energies, CI composition weights (bonding vs antibonding orbital contributions), binding energies, and equilibrium properties.

## Key Results

- **CI Corrects partially HF static correlation:** At dissociation, CI mixes determinants such that electrons correctly localize on separate atoms.
- **Correlation Energy:** CI recovers part of the correlation energy missing in HF. Due to restricing the CI space to the 2 lowest MO's the error is not completely fixed.
                          Dynamic correlation is not fixed with our restriced CI model, due to the contributions of a lot of Slater Determinants.
- **Comparison to Experiment:** Shows close agreement at equilibrium and highlights residual errors due to the finite basis set.

## How to Run

1. **Prerequisites:** Ensure you have a Python 3 environment with the following packages:
   - numpy
   - scipy
   - matplotlib
   - tqdm

2. **Custom Modules:** The notebook requires a function called `compute_integrals` from `utils.py` and an appropriate basis set file (`basis_sets/aug-cc-pvdz.gbs`). Also used are also functions in the `utils.py` file, namely, `solve_rhf`, `solve_uhf`, `analyze_equilibrium_orbitals`, `ao_to_mo_2orb`, `ci_block_singlet_2orb`, `compute_ci_energy`.
3. Make sure these are present in your repository.

4. **Jupyter Notebook:** Open the notebook in Jupyter or JupyterLab and run the cells sequentially. Plots will be saved in a directory called `Images/`.

## Citations


- [Szabo, A., Ostlund, N.S. "Modern Quantum Chemistry"](https://chemistlibrary.wordpress.com/wp-content/uploads/2015/02/modern-quantum-chemistry.pdf)

For experimental results:
- Vanderslice et al. (1962), Journal of Molecular Spectroscopy

---

**Author:** Miel Mathys, Grégoire Van San, Mathijs DeKeyser, Dries Kallaert
**Date:** Jan 2026  
