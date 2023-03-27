import numpy as np
from constants import A,B

class MPAC_functionals: #the class of all MPAC functionals
    def __init__(self,Ex,Ec_mp2,rho_4_3,gea_4_3):
        self.w0=Ex
        self.wd0=2*Ec_mp2
        self.pc=A*rho_4_3+B*gea_4_3

    def spl2(self,params): #calculating the E_c_int for the SPL2 form
        b2,m2,alp,bet=params
        self.winf=alp*self.pc + bet*self.w0
        spl2_form=self.winf-(2*(-1+np.sqrt(1+b2))*m2)/(b2)+(2*(-1+np.sqrt((m2+b2*m2+self.w0-2*self.wd0-self.winf)/(m2+self.w0-self.winf)))*(m2+self.w0-self.winf)**2)/(b2*m2-2*self.wd0)-self.w0
        return spl2_form

    def f1(self,params): #calculating the E_c_int for the F1 forms
        g,h,alp,bet=params
        self.winf=alp*self.pc + bet*self.w0
        f1_form=self.winf-(self.winf*(1+(2*(self.wd0-self.winf*g**2))/(-2*self.wd0+self.winf*h**4)))/(np.sqrt(1+g**2)+(2*(1+h**4)**(1/4)*(self.wd0-self.winf*g**2))/(-2*self.wd0+self.winf*h**4))
        return f1_form

    def mp2(self,params): #calculating E_c_int for regular MP2
        return self.wd0/2
