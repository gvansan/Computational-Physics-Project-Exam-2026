"""Simple wrapper around gbasis library to get all inputs for a basic Hartree Fock calculation.

See <https://gbasis.qcdevs.org/> for more info.
"""

import os
from collections import Counter

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.integrals.electron_repulsion import electron_repulsion_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.nuclear_electron_attraction import nuclear_electron_attraction_integral
from gbasis.integrals.overlap import overlap_integral
from gbasis.parsers import make_contractions, parse_gbs
from numpy.typing import ArrayLike

__all__ = ("compute_integrals",)

ANGMOM_CHARACTERS = "spdfghiklmnoqrtuvwxyzabce"
SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
SYM2NUM = {symbol: i + 1 for i, symbol in enumerate(SYMBOLS)}


def compute_integrals(
    symbols: list[str], atcoords: ArrayLike, path_basis: str, points: ArrayLike | None = None
) -> dict[str]:
    """Compute Gaussian integrals for a given molecular geometry.

    This code assumes that you are using an all-electron basis set.

    Parameters
    ----------
    symbols
        Atomic elements, e.g. ["H", "H"]
    atcoords
        Positions of the atomic nuclei in atomic units.
        Floating point array with shape (natom, 3).
    path_basis
        The name or path of a file defining the basis set.
    points
        If given, a NumPy array of grid points where the basis functions are evaluated.
        Floating point array with shape (npoint, 3).

    Returns
    -------
    results
        A dictionary with string keys and array values,
        using the following conventions:

        - `oi`: overlap integrals, shape (nbasis, nbasis),
        - `kei`: kinetic energy integrals, shape (nbasis, nbasis).
        - `nai`: nuclear attraction integrals, shape (nbasis, nbasis).
        - `eri`: electron repulsion integrals, shape (nbasis, nbasis, nbasis, nbasis).
        - `bfs`: values of basis functions at the given grid points, shape (npoint, nbasis).
        - `labels`: human-readable labels for the individual basis functions.
        - `atcorenums`: nuclear charges, shape (natom,)
        - `atcoords`: the given atomic coordinates, shape (natom, 3)
        - `symbols`: the given atom symbols, shape (natom,)
        - `points`: the given grid points, shape (npoint, 3)
    """
    # Argument checking and conversion
    atcoords = np.asarray(atcoords, dtype=float)
    if atcoords.ndim != 2 or atcoords.shape[1] != 3:
        raise TypeError("Argument atcoords must be a 2-index array with 3 columns.")
    if not isinstance(path_basis, str):
        raise TypeError("path_basis must be a string.")
    if not os.path.isfile(path_basis):
        raise ValueError(f"The file {path_basis} (path_basis) does not exist.")
    if points is not None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise TypeError("Argument points must be a 2-index array with 3 columns.")

    # Load basis and apply it to the given geometry.
    basis_information = parse_gbs(path_basis)
    basis = make_contractions(basis_information, symbols, atcoords, coord_types="cartesian")

    # Derive the nuclear charges
    nuclear_charges = np.array([SYM2NUM[symbol] for symbol in symbols], dtype=float)

    # Construct the labels.
    labels = []
    counter = Counter()
    for shell in basis:
        # Reverse-engineer the atom index of the shell.
        iatom = np.linalg.norm(atcoords - shell.coord, axis=1).argmin()
        angmomchar = "spdfgh"[shell.angmom]
        key = (iatom, angmomchar)
        ishell = counter[key]
        for px, py, pz in shell.angmom_components_cart:
            label = f"at{iatom}:{angmomchar}{ishell}:" + "x" * px + "y" * py + "z" * pz
            if label.endswith(":"):
                label = label[:-1]
            labels.append(label)
        counter[key] += 1

    # Compute stuff
    results = {
        "oi": overlap_integral(basis),
        "kei": kinetic_energy_integral(basis),
        "nai": nuclear_electron_attraction_integral(basis, atcoords, nuclear_charges),
        "eri": electron_repulsion_integral(basis),
        "atcorenums": nuclear_charges,
        "symbols": np.array(symbols),
        "atcoords": atcoords.copy(),
        "labels": np.array(labels),
    }

    if points is not None:
        results["bfs"] = evaluate_basis(basis, points)
        results["points"] = points.copy()

    return results