"""
A file containing constants and dictionary containing the parameters of the MP AC functionals
"""

A = -1.451 #The constant in front of the LDA integral coming from the PC Model.
B = 5.317e-3 #the constant in front of the GEA integral coming from the PC Model
kcal = 627.51 #@

params={
    ###A dictionary containing the parameters of all the different MP AC functionals.
    ###The key is build as follows, the 1st argument is to turn on the \kappa regularizer, the 2nd turn on opposite-spin scaling, the 3rd to set k_ss=k_os and the last one the name of the MP AC functional.
    ###The value contains the list of parameters with and the name of the MP AC functional
    (True,False,True,"spl2"): [[-0.433, 5.775,1.843,-1.750,1.7,1.7],"k-SPL2"],
    (True,False,True,"f1"): [[-0.3660,0.4677,1,1,1.3,1.3],"k-F1"],
    (True,False,True,"f1ab"): [[1.147,-0.6191,2.279,-4.989,1.5,1.5], "k-F1ab"],
    (True,False,True,"mp2"): [[1.1,1.1], "k-MP2"],
    (True,True,False,"spl2"): [[0.287, 148.982,1.674,-1.973,2.1,1.3], "coskos-SPL2"],
    (True,True,False,"f1"): [[0.9965,0.6799,1,1,2.3,1.1], "coskos-F1"],
    (True,True,False,"f1ab"): [[1.380,-0.5590,2.902,-7.836,2.4,1], "coskos-F1ab"],
    (True,True,False,"mp2"): [[0.9965,0.6799,1,1,2.3,1.1], "coskos-MP2"],
    (True,False,False,"spl2"): [[-0.690, 3.831,3.382,-4.026,1.3,1.7], "ksskos-SPL2"],
    (True,False,False,"f1"): [[0.0001615,-0.0151,1,1,1,1.4], "ksskos-F1"],
    (True,False,False,"f1ab"): [[0.398,0.663,2.715,-3.982,1.6,1.3], "ksskos-F1ab"],
    (True,False,False,"mp2"): [[0.9,1.4], "ksskos-mp2"],
    (False,False,False,"spl2"): [[0.117,10.68,1.1472,-0.7397], "SPL2"],
    (False,False,False,"f1"): [[0.294,0.934,1,1], "F1"],
    (False,False,False,"f1ab"): [[2.151, 0.413,3.837,-6.620], "F1ab"],
    (False,False,False,"mp2"): [[], "MP2"],
    (False,True,False,"spl2"): [[0.527, 58.850,1.278,-1.059,1.8], "cos-SPL2"],
    (False,True,False,"f1"): [[2.206,0.7068,1,1,2.2], "cos-F1"],
    (False,True,False,"f1ab"): [[2.769, -0.3665,8.3970,-14.2015,2], "cos-F1ab"],
    (False,True,False,"mp2"): [[1.7], "cos-MP2"],
}
