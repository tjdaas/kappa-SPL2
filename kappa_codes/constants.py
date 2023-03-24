"""
a file containing constants
"""

A = -1.451
B = 5.317e-3
kcal = 627.51
para1=[]
para2=[1.1,1.1]
para3=[0.9,1.4]
para4=[2.1,0.9]
para5=[1.7]
para6 = [0.117,10.68,1.1472,-0.7397]
para7 = [0.294,0.934,1,1]
para8 = [-0.690, 3.831,3.382,-4.026,1.3,1.7]
para9 = [0.287, 148.982,1.674,-1.973,2.1,1.3]
para10 = [0.527, 58.850,1.278,-1.059,1.8]
para11 = [-0.433, 5.775,1.843,-1.750,1.7,1.7]
para12 = [-0.3660,0.4677,1,1,1.3,1.3]
para13 = [0.0001615,-0.0151,1,1,1,1.4]
para14 = [0.9965,0.6799,1,1,2.3,1.1]
para15 = [2.206,0.7068,1,1,2.2]
para16 = [2.151, 0.413,3.837,-6.620]
para17 = [1.147,-0.6191,2.279,-4.989,1.5,1.5]
para18 = [0.398,0.663,2.715,-3.982,1.6,1.3]
para19 = [1.380,-0.5590,2.902,-7.836,2.4,1]
para20 = [2.769, -0.3665,8.3970,-14.2015,2]
def params(kappa,cos,mpac,ksam):
    if kappa==True and cos==True:
        if mpac=="spl2":
            para=para9
        elif mpac=="f1":
            para=para14
        elif mpac=="f1ab":
            para=para19
        elif mpac=="mp2":
            para=para4
    elif kappa==False and cos==True:
        if mpac=="spl2":
            para=para10
        elif mpac=="f1":
            para=para15
        elif mpac=="f1ab":
            para=para20
        elif mpac=="mp2":
            para=para5
    elif kappa==True and cos==False and ksam==False:
        if mpac=="spl2":
            para=para8
        elif mpac=="f1":
            para=para13
        elif mpac=="f1ab":
            para=para18
        elif mpac=="mp2":
            para=para3
    elif kappa==False and cos==False:
        if mpac=="spl2":
            para=para6
        elif mpac=="f1":
            para=para7
        elif mpac=="f1ab":
            para=para16
        elif mpac=="mp2":
            para=para1
    elif ksam == True:
        if mpac=="spl2":
            para=para11
        elif mpac=="f1":
            para=para12
        elif mpac=="f1ab":
            para=para17
        elif mpac=="mp2":
            para=para2
    return para

