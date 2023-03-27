import os
import shutil
import numpy as np
from tempfile import NamedTemporaryFile
from pyscf import gto, mp, ao2mo, dft, scf
import numpy as np


class run_pyscf():

    def __init__(
        self,
        atom: str = "",
        charge: int = 0,
        spin: int = 0,
        rho_trunc: float = 1e-10,
        basis: str = "aug-cc-pvqz",
    ):
        if atom == "":
            raise ValueError("m.xyz file must be given")
        self.mol=atom
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.rho_trunc = rho_trunc
        self.mol = gto.M(atom=self.mol,basis=basis,charge=charge)
        self.mol.max_memory = 2000000
        self.mol.verbose = 4
        nel = sum(self.mol.nelec)
        self.nocc = nel//2 

    def run_mol(self, chkfile_name: str = "", chkfile_dir: str =""):
                # Check if chkfile_name is given
        mf = dft.RKS(self.mol).density_fit()
        mf.verbose = 4
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
            mf.initialize_grids(self.mol)
        else:
            # If chkfile does not exist, run the calculation and save the chkfile
            with NamedTemporaryFile() as tempchk:
                mf.chkfile = tempchk.name
                mf.kernel()
                shutil.copyfile(tempchk.name, chkfile)

        # Build the Hartree-Fock density matrix in the AO basis
        self.dm = mf.make_rdm1()
        self.ehf = mf.e_tot #HF energy
        self.e = mf.mo_energy #mo-energies
        self.U = np.trace(self.dm.dot(mf.get_j())) / 2
        self.Ex = -1/4*np.einsum('ij,ij',self.dm,mf.get_k()) #Exchange
        self.coords = mf.grids.coords
        self.weights = mf.grids.weights
        self.mo_coeff = mf.mo_coeff
        return self.dm, self.ehf, self.e, self.mo_coeff, self.coords, self.weights, self.U, self.Ex

    def run_eris(self, chkfile_name: str = "", chkfile_dir: str =""):
        self.chkfile_name=chkfile_name
        self.chkfile_dir=chkfile_dir
        dm, ehf, e, mo_coeff, coords, weights, Uh, Ex = self.run_mol(self.chkfile_name,self.chkfile_dir)
        aovals = dft.numint.eval_ao(self.mol, coords, deriv=1)
        rho = dft.numint.eval_rho(self.mol, aovals, dm, xctype='GGA')
        self.rho_4_3 = np.sum(weights*rho[0]**(4/3))
        grad_square = np.sum(np.square(rho[1:4]),axis=0)
        non_zero = np.where(rho[0] > self.rho_trunc)
        self.grad_square_over_rho_4_3 = np.sum(weights[non_zero]*grad_square[non_zero]/(rho[0][non_zero]**(4/3)))
        self.rho_3_2 = np.sum(weights*rho[0]**(3/2))
        self.grad_square_over_rho_7_6 = np.sum(weights[non_zero]*grad_square[non_zero]/(rho[0][non_zero]**(7/6)))
        self.Ex=Ex
        self.Uh=Uh
        tab=[self.ehf, self.Uh, self.Ex, self.rho_4_3, self.grad_square_over_rho_4_3, self.rho_3_2, self.grad_square_over_rho_7_6]
        self.e=e
        norb = self.e.shape[0] #number of orbitals
        nvirt = norb-self.nocc #number of virtuals
        self.eri = ao2mo.outcore.general_iofree(self.mol, (mo_coeff[:,:self.nocc],mo_coeff[:,self.nocc:], mo_coeff[:,:self.nocc],mo_coeff[:,self.nocc:]),compact=False).reshape(self.nocc,nvirt,self.nocc,nvirt)
        eris=[self.nocc,self.e,self.eri]
        return tab, eris


