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
    '''Compute MP2 energy in parallel'''
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
    '''Compute SS MP2 energy with kapp and p in parallel'''
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
    energy_SS = np.zeros(kappa_SS.shape[0])
    for k in numba.prange(kappa_SS.shape[0]):
        energy_SS[k] = MP2_energy_kappa_p_SS(nocc, e, eri, kappa_SS[k], p_SS)
    return energy_SS


@numba.njit(parallel=True)
def MP2_energy_kappa_p_OS(nocc, e, eri, kappa_OS, p_OS):
    '''Compute MP2 energy with kapp and p in parallel'''
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
    energy_OS = np.zeros(kappa_OS.shape[0])
    for k in numba.prange(kappa_OS.shape[0]):
        energy_OS[k] = MP2_energy_kappa_p_OS(nocc, e, eri, kappa_OS[k], p_OS)
    return energy_OS
