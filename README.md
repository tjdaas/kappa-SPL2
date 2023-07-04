# kappa-SPL2: Running the new MPAC functionals using different MP2 implementations

Depends on the following packages:
- [pyscf]
- [numba]
- [numpy]
- [json]

## To run the calculations correctly one needs to have the following directory structure in the working directory:
* A/m.xyz
* B/m.xyz
* AB/m.xyz

for fragment A, fragment B and the full complex AB.

## Input parameters to give on the command line:
1. --charge: the charge of the molecule (int) (default=0)
2. --spin: the spin of the molecule (int) (default=0)
3. --basis: the basisset (str) (default="aug-cc-pvqz") (only supports basissets implemented in pyscf) 
4. --func: the functional that you want to run (str) (default="coskos-SPL2)
For func use mp2, spl2, f1 or f1ab as base and add a prefix as coskos-, cos-, ksskos-, k- or no prefix.

## Other Support:
run_all.py can be found in the run_all directory, which runs all the 20 functionals and outputs a .json file.

### Input parameters of this are:
1. --charge: the charge of the molecule (int) (default=0)
2. --spin: the spin of the molecule (int) (default=0)
3. --basis: the basisset (str) (default="aug-cc-pvqz") (only supports basissets implemented in pyscf) 

## Known Issues:
There is currently a workaround to fix an issue that numba has.
To solve any issue install openmp, then conda install numba cffi -c drtodd13 -c conda-forge --override-channel

## Future implementations:
1. add optimization scheme on S22 to allow all combinations of \kappa's, spin scaling and mpac functionals.
