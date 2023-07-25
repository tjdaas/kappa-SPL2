"""
A file containing constants and dictionary containing the parameters of the MP AC functionals
"""

A = -1.451 #The constant in front of the LDA integral coming from the PC Model.
B = 5.317e-3 #The constant in front of the GEA integral coming from the PC Model.
kcal = 627.51 #The conversion factor from Hartree to kcal/mol.

params={
    ###A dictionary containing the parameters of all the different MP AC functionals.
    ###The key is the name of the variant of the MP2 correlation energy, whereas the value contains the list of parameters of MP2, SPL2, MPACF1 and F1 respectively.
    "MP2": [[],[0.117,10.68,1.1472,-0.7397],[0.294,0.934,1,1],[2.151, 0.413,3.837,-6.620]],
    "k-MP2": [[],[-0.433, 5.775,1.843,-1.750],[-0.3660,0.4677,1,1],[1.147,-0.6191,2.279,-4.989]],
    "coskos-MP2": [[],[0.287, 148.982,1.674,-1.973],[0.9965,0.6799,1,1],[1.380,-0.5590,2.902,-7.836]],
    "ksskos-MP2": [[],[-0.690, 3.831,3.382,-4.026],[0.0001615,-0.0151,1,1],[0.398,0.663,2.715,-3.982]],
    "cos-MP2": [[],[0.527, 58.850,1.278,-1.059],[2.206,0.7068,1,1],[2.769, -0.3665,8.3970,-14.2015]],
}
