# kappa-SPL2: Running the new MPAC functionals using different MP2 implementations

Depends on the following packages:
- [pyscf]
- [numba]
- [numpy]
- [scipy]

## To run the calculations correctly one need to have the following directory structure in the working directory:
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

## Issues:
None

## Other Support:
run_all.py can be found in the run_all directory, which runs all the 20 functionals and outputs a .json file.

### Input parameters of this are:
1. --charge: the charge of the molecule (int) (default=0)
2. --spin: the spin of the molecule (int) (default=0)
3. --basis: the basisset (str) (default="aug-cc-pvqz") (only supports basissets implemented in pyscf) 

## To do list:
1. Try to add parallalization while using jitclass
2. Make the output of all_run.py be a.json file.

## Future implementations:
1. add optimization scheme on S22 to allow all combinations of \kappa's, spin scaling and mpac functionals.
2. A smarter method to calculate all the functionals in all_run.py (Derk's method)
