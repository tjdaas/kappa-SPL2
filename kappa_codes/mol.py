#basis=bse.get_basis('jorge-AQZP',fmt='nwchem')
basis='aug-cc-pVQZ'
mol = gto.M(atom="m.xyz",basis=basis,charge=0)
mol.max_memory = 2000000
mol.verbose = 4
nel = sum(mol.nelec)
nocc = nel//2 #singlet, so number of occupied orbitals is N/2
#mf = dft.RKS(mol)
#to turn on Density-Fitting [see pyscf.org/user/df.html],
#comment 'mf' above
#uncomment below
mf = dft.RKS(mol).density_fit()
mf.verbose = 4
mf.xc='HF' #do RKS, but xc functional is 100% HF
mf.grids.level = 4 #highest level of grids is 9; seems too large
