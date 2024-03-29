"""
A file that runs the pyscf RHF calculation or loads the chk-file in.
"""

import os
import shutil
import numpy as np
from tempfile import NamedTemporaryFile
from pyscf import gto, ao2mo, dft

class run_pyscf():

    def __init__(
        self,
        atom: str = "",
        charge: int = 0,
        spin: int = 0,
        rho_trunc: float = 1e-10, #to avoid dividing by 0 in the GEA integrals.
        basis: str = "aug-cc-pvqz",
    ):
        """A class that runs a RHF calculation using pyscf and extracts all the important quantities from it.

        Args:
            atom (str, optional): The atom, molecule or complex stored in a .xyz file. Defaults to "".
            charge (int, optional): The charge of the atom, molecule or complex. Defaults to 0.
            spin (int, optional): The spin of the atom, molecule or complex.  Defaults to 0.
            rho_trunc (float, optional): The truncation threshold on the density for integration. Defaults to 1e-10.

        Raises:
            ValueError: gives an error when input other than a .xyz file is given to atom.
        """
        if atom == "":
            raise ValueError("m.xyz file must be given")
        self.mol=atom
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.rho_trunc = rho_trunc
        self.mol = gto.M(atom=self.mol,basis=basis,charge=charge)
        self.mol.max_memory = 2000000
        nel = sum(self.mol.nelec) #number of electrons
        self.nocc = nel//2  #number of occupied orbitals

    def run_mol(self, chkfile_name: str = "", chkfile_dir: str =""):
        """Performs the HF SCF calculations or loads the chkfile if the calculations has already been run.

        Args:
            chkfile_name (str, optional): the name of the chkfile that either will be used to store the calculation or already contains a finished RHF SCF calculation. Defaults to "".
            chkfile_dir (str, optional): the directory that contains said chkfile. Defaults to "".

        Raises:
            ValueError: gives an error if no chkfile has been given as input.

        Returns:
            ndarray,float,ndarray,ndarray,ndarray,ndarray,float,float: outputs a 2d array containing the one particle density matrix, the HF energy, 1d array with the HF orbital energies, 2d array with the orbital coefficients, 1d array containing the grid, 1d array containing the integration weights, the Hartree Energy and Exchange Energy.
        """
        mf = dft.RKS(self.mol).density_fit()
        mf.xc='HF' #do RKS, but xc functional is 100% HF
        mf.grids.level = 4 #highest level of grids is 9; seems too large
        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir
        chkfile = os.path.join(chkfile_dir, chkfile_name)
        if os.path.isfile(chkfile):
            # If chkfile exists, load it and update the scf object
            mf.chkfile = chkfile
            mf.update()
            mf.initialize_grids(self.mol) #make a grid to use for the Winf integrations
        else:
            # If chkfile does not exist, run the calculation and save the chkfile
            with NamedTemporaryFile() as tempchk:
                mf.chkfile = tempchk.name
                mf.kernel()
                shutil.copyfile(tempchk.name, chkfile) #save chkfile

        # Build the Hartree-Fock density matrix in the AO basis
        self.dm = mf.make_rdm1()
        self.ehf = mf.e_tot #HF energy
        self.e = mf.mo_energy #mo-energies
        self.U = np.trace(self.dm.dot(mf.get_j())) / 2 #Hartree Energy
        self.Ex = -1/4*np.einsum('ij,ij',self.dm,mf.get_k()) #Exchange
        self.coords = mf.grids.coords # coordinates of the grids
        self.weights = mf.grids.weights #weights of the grid
        self.mo_coeff = mf.mo_coeff #mo-coefficients
        return self.dm, self.ehf, self.e, self.mo_coeff, self.coords, self.weights, self.U, self.Ex

    def run_eris(self, chkfile_name: str = "", chkfile_dir: str =""):
        """calculates the four orbital integrals and the LDA and GEA integrals of W_{c,\infty} and W_{1/2}. 

        Args:
            chkfile_name (str, optional): the name of the chkfile that either will be used to store the calculation or already contains a finished RHF SCF calculation. Defaults to "".
            chkfile_dir (str, optional): the directory that contains said chkfile. Defaults to "".

        Returns:
            list,list: outputs a list containing the HF energy, Hartree energy, Exchange energy, LDA integral W_{c,\infty}, GEA integral W_{c,\infty}, LDA integral W_{1/2} and GEA integral W_{1/2} and a list containing the number of occupied orbitals, HF orbital energies and the 4 orbital integrals.
        """
        self.chkfile_name=chkfile_name
        self.chkfile_dir=chkfile_dir
        dm, ehf, e, mo_coeff, coords, weights, Uh, Ex = self.run_mol(self.chkfile_name,self.chkfile_dir) #run the HF calculations
        aovals = dft.numint.eval_ao(self.mol, coords, deriv=1) #atomic orbital values
        rho = dft.numint.eval_rho(self.mol, aovals, dm, xctype='GGA') #calculate the density
        self.rho_4_3 = np.sum(weights*rho[0]**(4/3)) #calculate the LDA integral
        grad_square = np.sum(np.square(rho[1:4]),axis=0) #define the square of the laplacian
        non_zero = np.where(rho[0] > self.rho_trunc) #remove the points where the density is 0, to avoid dividing by 0.
        self.grad_square_over_rho_4_3 = np.sum(weights[non_zero]*grad_square[non_zero]/(rho[0][non_zero]**(4/3))) #calculate the GEA integral
        self.rho_3_2 = np.sum(weights*rho[0]**(3/2)) #the LDA integral for the next order term W_{1/2}
        self.grad_square_over_rho_7_6 = np.sum(weights[non_zero]*grad_square[non_zero]/(rho[0][non_zero]**(7/6))) #the GEA integral for the next order term W_{1/2}
        self.Ex=Ex
        self.Uh=Uh
        tab=[self.ehf, self.Uh, self.Ex, self.rho_4_3, self.grad_square_over_rho_4_3, self.rho_3_2, self.grad_square_over_rho_7_6] #the tab file that will be printed
        self.e=e
        norb = self.e.shape[0] #number of orbitals #number of orbitals
        nvirt = norb-self.nocc #number of virtuals #number of virtual orbitals
        self.eri = ao2mo.outcore.general_iofree(self.mol, (mo_coeff[:,:self.nocc],mo_coeff[:,self.nocc:], mo_coeff[:,:self.nocc],mo_coeff[:,self.nocc:]),compact=False).reshape(self.nocc,nvirt,self.nocc,nvirt)
        eris=[self.nocc,self.e,self.eri] 
        return tab, eris
