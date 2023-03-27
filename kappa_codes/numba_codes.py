#import pyscf, numpy and numba
from pyscf import gto, mp, ao2mo, dft, scf
import numpy as np
import numba
from numba.experimental import jitclass
from numba import int32, double
import glob
import os
import scipy
spec = [
    ('nocc', int32),
    ('e',double[:]),
    ('eri',double[:,:,:,:]),
]
@jitclass(spec)
class reg_sos_mp2: #the class containing all the MP2 integrals

    def __init__(self,nocc,e,eri):
        self.nocc=nocc
        self.e=e
        self.eri=eri
    
    def MP2_energy(self):
        '''Compute standard MP2 energy in parallel'''
        energy = 0. #- EMP2
        norb = self.e.shape[0] #total number of orbitals
        e_occ =self.e[:self.nocc] #occupied energies
        e_virt =self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        #loop over ijab, numba automatically figures out how to parallelize, probably only does so over i, but it can reorder
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy += self.eri[i,a,j,b]*(2*self.eri[i,a,j,b]-self.eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])
        return - energy

    #@numba.njit(parallel=True)
    def MP2_energy_split(self):
        '''Compute MP2 energy in parallel, split in a same spind and opposite spin part'''
        energy_SS = 0. #- EMP2 SS
        energy_OS = 0. #- MP2 OS
        norb = self.e.shape[0] #total number of orbitals
        e_occ = self.e[:self.nocc] #occupied energies
        e_virt = self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        #loop over ijab, numba automatically figures out how to parallelize, probably only does so over i, but it can reorder
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy_SS += self.eri[i,a,j,b]*(self.eri[i,a,j,b]-self.eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])
                        energy_OS += self.eri[i,a,j,b]*(self.eri[i,a,j,b])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])
        return - energy_SS, - energy_OS


    def MP2_energy_kappa_p(self, kappa, p):
        '''Compute MP2 energy with kapp and p in parallel'''
        energy = 0. #- EMP2
        norb = self.e.shape[0] #total number of orbitals
        e_occ =self.e[:self.nocc] #occupied energies
        e_virt =self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy += self.eri[i,a,j,b]*(2*self.eri[i,a,j,b]-self.eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p))**2
        return - energy

    def MP2_energy_kappa_p_split(self, kappa_SS, kappa_OS, p_SS, p_OS):
        '''Compute MP2 energy with kapp and p in parallel with split same spin and opposite spin integrals'''
        energy_SS = 0. #- EMP2 SS
        energy_OS = 0. #- MP2 OS
        norb = self.e.shape[0] #total number of orbitals
        e_occ =self.e[:self.nocc] #occupied energies
        e_virt =self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy_SS += self.eri[i,a,j,b]*(self.eri[i,a,j,b]-self.eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_SS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_SS))**2
                        energy_OS += self.eri[i,a,j,b]*(self.eri[i,a,j,b])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_OS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_OS))**2
        return - energy_SS, -energy_OS

    #SV: I split the above expression into two, one for OS and the other for SS (in case we need only one of the two, no need to do both)

    def MP2_energy_kappa_p_SS(self,kappa_SS, p_SS):
        '''Compute same spin MP2 energy with kapp and p in parallel'''
        energy_SS = 0. #- EMP2 SS
        #energy_OS = 0. #- MP2 OS
        norb = self.e.shape[0] #total number of orbitals
        e_occ =self.e[:self.nocc] #occupied energies
        e_virt =self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy_SS += self.eri[i,a,j,b]*(self.eri[i,a,j,b]-self.eri[i,b,j,a])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_SS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_SS))**2
        return - energy_SS 

    def MP2_energy_kappa_p_SS_parallel(self, kappa_SS, p_SS):
        '''compute same spin MP2 energy for all kapp's'''
        energy_SS = np.zeros(kappa_SS.shape[0])
        for k in numba.prange(kappa_SS.shape[0]):
            energy_SS[k] = self.MP2_energy_kappa_p_SS(kappa_SS[k], p_SS)
        return energy_SS

    def MP2_energy_kappa_p_OS(self, kappa_OS, p_OS):
        '''Compute MP2 energy with kapp and p in parallel'''
        # energy_SS = 0. #- EMP2 SS
        energy_OS = 0. #- MP2 OS
        norb = self.e.shape[0] #total number of orbitals
        e_occ =self.e[:self.nocc] #occupied energies
        e_virt =self.e[self.nocc:] #virtual energies
        nvirt = norb-self.nocc #number of virtuals
        for i in numba.prange(self.nocc):
            for j in numba.prange(self.nocc):
                for a in numba.prange(nvirt):
                    for b in numba.prange(nvirt):
                        energy_OS += self.eri[i,a,j,b]*(self.eri[i,a,j,b])/(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])*(1-np.exp(-kappa_OS*(e_virt[a]+e_virt[b]-e_occ[i]-e_occ[j])**p_OS))**2
        return - energy_OS

    def MP2_energy_kappa_p_OS_parallel(self, kappa_OS, p_OS):
        '''compute same spin MP2 energy for all kapp's'''
        energy_OS = np.zeros(kappa_OS.shape[0])
        for k in numba.prange(kappa_OS.shape[0]):
            energy_OS[k] = self.MP2_energy_kappa_p_OS(kappa_OS[k], p_OS)
        return energy_OS



