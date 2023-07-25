"""
A file containing the parallelized MP2 codes sped up by numba.
"""
#import numpy and numba
import numpy as np
import numba
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

@numba.njit(parallel=True)
def MP2_energy_split(nocc, e, eri):
    """Calculating the split MP2 correlation energy in parallel.

    Args:
        nocc (integer): number of occupied orbitals.
        e (ndarray): 1d array containing the orbital energies of both the occupied and virtual orbitals.
        eri (_type_): 4d array containing the four orbital integrals in the MO basis.

    Returns:
        float, float: the same spin and opposite spin MP2 correlation energies.
    """
    energy_SS = 0. #- EMP2 SS
    energy_OS = 0. #- MP2 OS
    norb = e.shape[0] #total number of orbitals
    e_occ = e[:nocc] #occupied energies
    e_virt = e[nocc:] #virtual energies
    nvirt = norb-nocc #number of virtuals
    #loop over ijab, numba automatically figures out how to parallelize, probably only does so over i, but it can reorder
    for i in numba.prange(nocc):
        for j in numba.prange(nocc):
            for a in numba.prange(nvirt):
                for b in numba.prange(nvirt):
                    energy_SS += eri[i,a,j,b]*(eri[i,a,j,b]-eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])
                    energy_OS += eri[i,a,j,b]*(eri[i,a,j,b])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])
    return - energy_SS, - energy_OS


@numba.njit(parallel=True)
def MP2_energy_kappa_p_SS(nocc, e, eri, kappa_SS, p_SS):
    """Computes the SS MP2 correlation energies with \kappa and p in parallel.

    Args:
        nocc (integer): number of occupied orbitals.
        e (ndarray): 1d array containing the orbital energies of both the occupied and virtual orbitals.
        eri (_type_): 4d array containing the four orbital integrals in the MO basis.
        kappa_SS (float): the same spin \kappa regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097
        p_SS (float): the same spin p regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097

    Returns:
        float: the same spin correlation MP2 correlation energy for a specific \kappa and p.
    """
    energy_SS = 0. #- EMP2 SS
    #energy_OS = 0. #- MP2 OS
    norb = e.shape[0] #total number of orbitals
    e_occ = e[:nocc] #occupied energies
    e_virt = e[nocc:] #virtual energies
    nvirt = norb-nocc #number of virtuals
    for i in numba.prange(nocc):
        for j in numba.prange(nocc):
            for a in numba.prange(nvirt):
                for b in numba.prange(nvirt):
                    energy_SS += eri[i,a,j,b]*(eri[i,a,j,b]-eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_SS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_SS))**2
    return - energy_SS 


@numba.njit(parallel=True)
def MP2_energy_kappa_p_SS_parallel(nocc, e, eri, kappa_SS, p_SS):
    """Computes the SS MP2 correlation energies with p and for different \kappa in parallel.

    Args:
        nocc (integer): number of occupied orbitals.
        e (ndarray): 1d array containing the orbital energies of both the occupied and virtual orbitals.
        eri (_type_): 4d array containing the four orbital integrals in the MO basis.
        kappa_SS (array): 1d array containing the same spin \kappa regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097
        p_SS (float): the same spin p regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097

    Returns:
        ndarray: 1d array containing the same spin MP2 correlation energies for a specific p and a range of different \kappa's.
    """
    energy_SS = np.zeros(kappa_SS.shape[0])
    for k in numba.prange(kappa_SS.shape[0]):
        energy_SS[k] = MP2_energy_kappa_p_SS(nocc, e, eri, kappa_SS[k], p_SS)
    return energy_SS


@numba.njit(parallel=True)
def MP2_energy_kappa_p_OS(nocc, e, eri, kappa_OS, p_OS):
    """Computes the OS MP2 correlation energies with \kappa and p in parallel.

    Args:
        nocc (integer): number of occupied orbitals.
        e (ndarray): 1d array containing the orbital energies of both the occupied and virtual orbitals.
        eri (_type_): 4d array containing the four orbital integrals in the MO basis.
        kappa_OS (float): the opposite spin \kappa regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097
        p_OS (float): the opposite spin p regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097

    Returns:
        float: the opposite spin correlation MP2 correlation energy for a specific \kappa and p.
    """
    # energy_SS = 0. #- EMP2 SS
    energy_OS = 0.
    norb = e.shape[0]
    e_occ = e[:nocc] #occupied energies
    e_virt = e[nocc:] #virtual energies
    nvirt = norb-nocc #number of virtuals
    for i in numba.prange(nocc):
        for j in numba.prange(nocc):
            for a in numba.prange(nvirt):
                for b in numba.prange(nvirt):
                    energy_OS += eri[i,a,j,b]*(eri[i,a,j,b])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_OS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_OS))**2
    return - energy_OS

@numba.njit(parallel=True)
def MP2_energy_kappa_p_OS_parallel(nocc, e, eri, kappa_OS, p_OS):
    """Computes the OS MP2 correlation energies with p and for different \kappa in parallel.

    Args:
        nocc (integer): number of occupied orbitals.
        e (ndarray): 1d array containing the orbital energies of both the occupied and virtual orbitals.
        eri (_type_): 4d array containing the four orbital integrals in the MO basis.
        kappa_SS (array): 1d array containing the opposite spin \kappa regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097
        p_SS (float): the opposite spin p regularizer derived from Shee, J.; Loipersberger, M.; Rettig, A.; Lee, J.; Head-Gordon, M. JPCL 2021, 12, 12084–12097

    Returns:
        ndarray: 1d array containing the opposite spin MP2 correlation energies for a specific p and a range of different \kappa's.
    """
    energy_OS = np.zeros(kappa_OS.shape[0])
    for k in numba.prange(kappa_OS.shape[0]):
        energy_OS[k] = MP2_energy_kappa_p_OS(nocc, e, eri, kappa_OS[k], p_OS)
    return energy_OS
