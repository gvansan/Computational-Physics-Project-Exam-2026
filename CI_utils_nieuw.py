from tqdm import tqdm
import numpy as np
import scipy as sp

def solve_rhf(dists,datas):
    for dist, data in zip(tqdm(dists, desc="Solving RHF"), datas, strict=False):
        # Get a few operators.
        coreham = data["kei"] + data["nai"]
        eri = data["eri"]
        olp = data["oi"]
        data["hcore"] = coreham
        
        # The initial guess.
        eigvals, eigvecs = sp.linalg.eigh(coreham, olp)
        dm = np.dot(eigvecs[:, :1], eigvecs[:, :1].T) #RHF (1 alpha, 1 beta)

        # The scf cycle (See 02_helium.ipynb for details.)
        for _scf_counter in range(1000):
            hartree = np.einsum("kmln,nm->kl", eri, dm)
            exchange = np.einsum("kmnl,nm->kl", eri, dm)
            fock = coreham + 2 * hartree - exchange  # Specific for RHF

            errors_rh = np.dot(fock, eigvecs) - np.einsum("ij,jk,k->ik", olp, eigvecs, eigvals)
            error_rh = np.linalg.norm(errors_rh)  # Frobenius norm
            if error_rh < 1e-7:
                break

            eigvals, eigvecs = sp.linalg.eigh(fock, olp)
            dm = np.dot(eigvecs[:, :1], eigvecs[:, :1].T)
        else:
            raise RuntimeError("SCF convergence failed")

        # Compute the _electronic_ energy.
        ham = coreham + hartree - 0.5 * exchange
        electronic_energy = 2 * np.einsum("ij,ji", ham, dm)

        # Compute the occupied orbital on the grid.
        psi0 = np.dot(eigvecs[:, 0], data["bfs"])
        psi1 = np.dot(eigvecs[:, 1], data["bfs"])

        # Store results back into the data dictionary.
        # (Mind the two electrons.)
        data["eigvals_rhf"] = eigvals
        data['eigvecs_rhf'] = eigvecs
        data["energy_rhf"] = electronic_energy + 1 / dist
        data["density0_rhf"] = psi0**2
        data["density1_rhf"] = psi1 ** 2
        data["density_rhf"] = psi0**2 + psi1**2



def solve_uhf(dists,datas):
    for dist, data in zip(tqdm(dists, desc="Solving UHF"), datas, strict=False):
        # Get a few operators.
        coreham = data["kei"] + data["nai"]
        eri = data["eri"]
        olp = data["oi"]

        # The initial guess.
        eigvals, eigvecs = sp.linalg.eigh(coreham, olp)
        dm = np.dot(eigvecs[:, :2], eigvecs[:, :2].T)  # UHF (2 alphas)

        # The SCF cycle (See 02_helium.ipynb for details.)
        for _scf_counter in range(500):
            hartree = np.einsum("kmln,nm->kl", eri, dm)
            exchange = np.einsum("kmnl,nm->kl", eri, dm)
            fock = coreham + hartree - exchange

            errors_rh = np.dot(fock, eigvecs) - np.einsum("ij,jk,k->ik", olp, eigvecs, eigvals)
            error_rh = np.linalg.norm(errors_rh)  # Frobenius norm
            if error_rh < 1e-7:
                break

            eigvals, eigvecs = sp.linalg.eigh(fock, olp)
            dm = np.dot(eigvecs[:, :2], eigvecs[:, :2].T)
        else:
            raise RuntimeError("SCF convergence failed")

        # Compute the _electronic_ energy.
        ham = coreham + 0.5 * hartree - 0.5 * exchange
        electronic_energy = np.einsum("ij,ji", ham, dm)

        # Compute the occupied orbitals on the grid.
        psi0 = np.dot(eigvecs[:, 0], data["bfs"])
        psi1 = np.dot(eigvecs[:, 1], data["bfs"])

        # Store the energy. The second term in the energy is the nucleus-nucleus
        # repulsion.
        data["eigvecs_uhf"] = eigvecs
        data["eigvals_uhf"] = eigvals
        data["energy_uhf"] = electronic_energy + 1 / dist
        data["density_uhf"] = psi0**2 + psi1**2
        data["density0_uhf"] = psi0**2
        data["density1_uhf"] = psi1**2


def analyze_equilibrium_orbitals(datas):
    """
    Find equilibrium geometry and analyze bonding/antibonding MOs.
    Returns dict with MO data and prints qualitative checks. 
    ----------
    Parameters
    ----------
    datas : list
        list containing all data (integrals, ...) for different internuclear distances
    ----------
    returns
    ----------
    datas : dicttionary containing data needed to plot the densities
    """
    eq_id = int(np.argmin([d["energy_rhf"] for d in datas])) #index of distance where energy is minimal
    data = datas[eq_id] #corresponding data where energy is minimal
    
    psi0 = data["density0_rhf"]  # bonding (σg)
    psi1 = data["density1_rhf"]  # antibonding (σu)
    
    mid_idx = np. argmin(np. abs(data["points"][: , 2]))
    
    print(f"Equilibrium index: {eq_id}")
    print(f"psi0 (σg) midpoint value: {psi0[mid_idx]:.6f}")
    print(f"psi1 (σu) midpoint value: {psi1[mid_idx]:.6f}")
    
    return {
        "eq_id": eq_id,
        "data": data,
        "psi0": psi0,
        "psi1": psi1,
        "mid_idx": mid_idx,
    }


def ao_to_mo_2orb(data):
    """
    Function to change between the basis consisting of basis functions (b_i) and the basis consisting
    of molecular orbitals (MO). We only keep the 2 MO with lowest energies, corresponding to the two
    sigma orbitals
    ----------
    Parameters
    ----------
    data : dict
        Dictionary containing the data we need (integrals, basis functions, ...)
    ----------
    returns
    ----------
    h, g : np.array(2,2), np.aray(18,18,18,18)
        h is the core Hamiltonian H_kl = T_kl + V_ext_kl
        g is the electron repulsion integral V_ee_ijkl
    """
    C = data["eigvecs_rhf"][:, :2]   # keep 2 lowest MOs (bonding and antibonding MO)

    h_ao = data["hcore"] #core Hamiltonian in atomic orbital basis
    eri_ao = data["eri"] #electronic repulsion integral in atomic orbital basis

    #Transform to molecular orbital basis
    h = C.T @ h_ao @ C
    g = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
    
    return h, g


def ci_block_singlet_2orb(h, g, verbose=False):
    """
    Function to calculate the CI Hamiltonian that we diagonalize to get the CI orbitals and energies.
    ----------
    Parameters
    ----------
    h : np.array
        Core Hamiltonian of the system. In our case this has shape (2,2) due to the approximation we make
        of only keeping σg and σu orbitals
    g : npNarray
        Electron repulsion integral for the system. This has shape (2,2,2,2) for the same reason as the
        core Hamiltonian h
    verbose : bool
        True if we want to print text to check if the right terms of the eri are 0
    ----------
    returns
    ----------
    H : np.array
        The CI matrix that we have to diagonalize to find the CI energies for H2
    """

    #One particle operator (core Hamiltonian)
    h00, h11 = h[0,0], h[1,1]
    h01 = h[0,1]

    #Two particle operator (electron repulsion integral)
    g0000 = g[0,0,0,0] #both SD with both electrons in sigma_g (<sigma_g^2|eri|sigma_g^2>)
    g1111 = g[1,1,1,1] #both SD with both electrons in sigma_u (<sigma_u^2|eri|sigma_u^2>)
    g0011 = g[0,0,1,1] #1 SD with 2 electrons in GS, other SD with 2 electrons in excited state
    g0110 = g[0,1,1,0] #both SD with 1 electron in GS and 1 electron in excited state
    g0101 = g[0,1,0,1] #both SD with 1 electron in GS and 1 electron in excited state
    g0001 = g[0,0,0,1] #1 SD with 2 electrons in GS, other SD with 1 electron in excited state
    g0010 = g[0,0,1,0] #1 SD with 2 electrons in GS, other SD with 1 electron in excited state
    g0111 = g[0,1,1,1] #1 SD with 1 electron in GS, other SD with 2 electrons in excited state

    if verbose:
        print(f"h01 = {h01:.6f}, g0001 = {g0001:.6f}, sum = {h01 + g0001:.6f}")
        print(f"(Should be ~0 by Brillouin's theorem for canonical HF orbitals)")

    #diagonal elements of CI hamiltonian
    H = np.zeros((3, 3))
    H[0,0] = 2*h00 + g0000                              #Ground config energy (both electrons in GS), no exchange term
    H[1,1] = h00 + h11 + g0101                          #Singlet single-excitation energy
    H[2,2] = 2*h11 + g1111                              #Doubly-excited config energy (both electrons in excited state) 

    #off-diagonal elements of CI hamiltonian
    H[0,1] = H[1,0] = np. sqrt(2) * (h01 + g0001)       #Coupling between ground and single-excited config
    H[1,2] = H[2,1] = np. sqrt(2) * (h01 + g0111)       #Coupling between single- and doubly-excited config
    H[0,2] = H[2,0] = g0011                             #Coupling between ground and doubly-excited config


    return H

def compute_ci_energy(h, g, dist, data, verbose=True):
    """
    Function that diagonalizes the Hamiltonian computed in ci_block_singlet_2orb to find the
    energies of H2
    ----------
    Parameters
    ----------
    h : np.array
        Core Hamiltonian of the system. In our case this has shape (2,2) due to the approximation we make
        of only keeping σg and σu orbitals
    g : npNarray
        Electron repulsion integral for the system. This has shape (2,2,2,2) for the same reason as the
        core Hamiltonian h
    dist : float
        Value for the internuclear distance
    data : dict 
        Data containing integrals,... for the given internuclear distance
    verbose : bool
        True if we want to print text to check if the right terms of the eri
        are 0 in ci_block_singlet_2orb
    ----------
    returns
    ----------
    dict
        Dictionary containing information such as the Hamiltonian, the CI energies, eigenvectors etc.
    """
    H = ci_block_singlet_2orb(h, g, verbose=verbose)
    eigvals, eigvecs = np.linalg. eigh(H)
    E_elec = eigvals[0]
    E_nuc = 1.0 / dist
    E_tot = E_elec + E_nuc  
    
    # Compare CI reference energy (no correlation) to RHF
    E_ci_ref = H[0,0] + E_nuc  # CI |0,0⟩ state energy
    if verbose:
        print("CI Hamiltonian matrix (Ha):\n", H)
        print(f"H[0,0] + V_nuc (CI ref):    {E_ci_ref:.8f}")
        print(f"RHF total energy:            {data['energy_rhf']:.8f}")
        print(f"Difference (should be ~0):  {E_ci_ref - data['energy_rhf']:.2e}")
        print(f"CI ground-state energy:     {E_tot:.8f}")
    
    return {"H": H, "eigvals": eigvals, "eigvecs": eigvecs, "E_elec": E_elec, "E_tot": E_tot}