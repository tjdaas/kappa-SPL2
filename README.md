# kappa-SPL2: Running the new MPAC functionals using different MP2 implementations

Depends on the following packages:
- [pyscf]
- [numba]
- [numpy]
- [argparse]
- [openmp]
- [json]

## Requirements:
* Python 3.9 or newer (earlier versions might still work)
* numba 0.49.0 (earlier versions might still work)

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
To solve any issue install openmp, then:
- conda install numba cffi -c drtodd13 -c conda-forge --override-channel

## Future implementations:
1. add optimization scheme on S22 to allow all combinations of \kappa's, spin scaling and mpac functionals.

## References:
1. K. J. Daas, D.P. Kooi, N.M. Peters, E. Fabiano, F. Della Sala, P. Gori-Giorgi, S. Vuckovic, Regularized and scaled Opposite-spin Functionals in Møller-Plesset Adiabatic Connection: Higher Accuracy at a Lower Cost, Arxiv 2023 https://doi.org/10.48550/arXiv.2307.02715

## License
MIT License
Copyright (c) 2023 Kimberly J. Daas, Derk P. Kooi, Stefan Vuckovic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECT
