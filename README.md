# kappa-SPL2: Running the new MPAC functionals using different MP2 implementations

Depends on the following packages:
- [pyscf]
- [numba]
- [numpy]
- [scipy]

To run the calculations correctly one need to have the following directory structure in the working directory:
A/m.xyz
B/m.xyz
AB/m.xyz
for fragment A, fragment B and the full complex AB.

Input parameters to give on the command line:
1. the charge of the molecule (int)
2. the spin of the molecule (int)
3. the basisset (str) (only supports basissets implemented in pyscf)
4. \kappa regularizer (bool) (True=True, no argument=False)
5. spin-opposite-scaling (bool) (True=True, no argument=False)
6. original \kappa-mp2 (bool) (True=True, no argument=False)

Issues:

To do list:
1. Try to add parallalization while using jitclass

Future implementations:
1. add optimization scheme on S22 to allow all combinations of \kappa's, spin scaling and mpac functionals.
2. file that runs all calculations at once and outputs 
