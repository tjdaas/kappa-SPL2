"""
Calculating the interaction energy for all MPAC functionals at once
"""

#import pyscf, numpy and numba
import numpy as np
import argparse
import os
from kappa_codes.numba_all import *
from all_codes.mol_all import run_pyscf
from all_codes.mpac_all import MPAC_functionals
from all_codes.constants_all import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--charge", type=int, default=0, help="the charge of the system")
    parser.add_argument("--spin", type=int, default=0, help="the spin of the system")
    parser.add_argument("--basis", type=str, default="aug-cc-pvqz",help="the basisset used in the calculations")

args = parser.parse_args()
mols=["A","B","AB"] #A and B are fragments, AB is the complex
mpacf=["spl2","f1","f1ab","mp2"]

Ex=[]
ehf=[]
Uh=[]
rho_4_3=[]
gea_4_3=[]
rho_3_2=[]
gea_7_6=[]
E_c_mp2=[]
E_c_spl2=[]
E_c_f1=[]
E_c_f1ab=[]
E_c_SS_k=[]
E_c_OS=[]
E_c_OS_k=[]
E_c_mp2tot=[]

for i in range(3): #run over the fragements and complex
    run_mol=mols[i]
    #add here path to frag m.xyz file
    chkfile="chkfile_"+run_mol+".chk"
    old_pwd=os.getcwd()
    datadir=old_pwd+"/"+run_mol
    os.chdir(datadir)
    ###runs HF
    py_run=run_pyscf(atom="m.xyz",charge=args.charge,spin=args.spin,basis=args.basis)
    tab, eris = py_run.run_eris(chkfile_name=chkfile,chkfile_dir=datadir)
    #this prints and extracts all of the ingredients except MP2
    np.savetxt("tab.csv", tab, delimiter=",", fmt='%s')
    ehf.append(tab[0])
    Uh.append(tab[1])
    Ex.append(tab[2])
    rho_4_3.append(tab[3])
    gea_4_3.append(tab[4])
    rho_3_2.append(tab[5])
    gea_7_6.append(tab[6])
    #k1 is for same spin
    #k2 is for the opposite spin
    k1ss = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    k2ss = k1ss
    np.savetxt("k1.csv", k1ss, delimiter=",", fmt='%s')
    np.savetxt("k2.csv", k2ss, delimiter=",", fmt='%s')
    k1s=np.array(k1ss,dtype=float)
    k2s=np.array(k2ss,dtype=float)
    mp2OS = MP2_energy_kappa_p_OS_parallel(*eris,k2s, 1) #calculate the opposite spin integral with kappa
    E_c_OS_k.append(mp2OS)
    np.savetxt("os.csv", mp2OS, delimiter=",", fmt='%s')
    #Spin scaled \kappa's
    mp2SS = MP2_energy_kappa_p_SS_parallel(*eris,k1s, 1) #calculate the same spin integral with kappa
    E_c_SS_k.append(mp2SS)
    np.savetxt("ss.csv", mp2SS, delimiter=",", fmt='%s')
    e_mp2_split = MP2_energy_split(*eris,) #calculates the same an opposite mp2 integrals
    np.savetxt("mp2.csv", e_mp2_split, delimiter=",", fmt='%s')
    E_c_SS.append(e_mp2_split[0])
    E_c_OS.append(e_mp2_split[1])
    E_c_mp2tot.append(sum(e_mp2_split))
    os.chdir(old_pwd)

#getting the arrays into the correct shape
E_c_SS_k=np.array(E_c_SS_k).T
E_c_OS_k=np.array(E_c_OS_k).T
E_c_OS=np.array(E_c_OS)

#initializing the MPAC functionals and calculating the HF energy difference
form_frags=MPAC_functionals(Ex[0]+Ex[1],rho_4_3[0]+rho_4_3[1],gea_4_3[0]+gea_4_3[1])
form_com=MPAC_functionals(Ex[2],rho_4_3[2],gea_4_3[2]) 
ehfdiv=ehf[2]-ehf[1]-ehf[0]
funcs=["MP2","SPL2","F1","F1ab","k-MP2","k-SPL2","k-F1","k-F1ab","ksskos-MP2","ksskos-SPL2","ksskos-F1","ksskos-F1ab","coskos-MP2","coskos-SPL2","coskos-F1","coskos-F1ab","cos-MP2","cos-SPL2","cos-F1","cos-F1ab"]

#storing all the EMP2 data
EMP2vals={
    "MP2": [E_c_mp2tot]*4,
    "k-MP2": [E_c_SS_k[5] + E_c_OS_k[5],E_c_SS_k[11] + E_c_OS_k[11],E_c_SS_k[7] + E_c_OS_k[7],E_c_SS_k[9] +E_c_OS_k[9]],
    "ksskos-MP2": [E_c_SS_k[3] + E_c_OS_k[8],E_c_SS_k[5] + E_c_OS_k[11],E_c_SS_k[4] + E_c_OS_k[8],E_c_SS_k[10] + E_c_OS_k[7]],
    "coskos-MP2":[2.1*E_c_OS_k[3],2.1*E_c_OS_k[7],2.3*E_c_OS_k[5],2.5*E_c_OS_k[4]],
    "cos-MP2": [1.7*E_c_OS,1.8*E_c_OS,2.2*E_c_OS,2*E_c_OS]
}

#calculating all the 20 functionals
for name,emp2 in EMP2vals.items():
    E_c_int.append(form_com.mp2(params[name][0],emp2[0][2])-form_frags.mp2(params[name][0],emp2[0][1]+emp2[0][0]))
    E_c_int.append(form_com.spl2(params[name][1],emp2[1][2])-form_frags.spl2(params[name][1],emp2[1][1]+emp2[1][0]))
    E_c_int.append(form_com.f1(params[name][2],emp2[2][2])-form_frags.f1(params[name][2],emp2[2][1]+emp2[2][0]))
    E_c_int.append(form_com.f1(params[name][3],emp2[3][2])-form_frags.f1(params[name][3],emp2[3][1]+emp2[3][0]))

#print json file
E_c_ints=dict(zip(funcs,kcal*(ehfdiv+np.array(E_c_int))))
print(E_c_ints) #prints out the correct E_c_int
with open("E_c_all.json","w",encoding="utf-8") as f:
    json.dump(E_c_ints,f)
