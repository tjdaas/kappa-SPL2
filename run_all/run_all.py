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
E_c_SS=[]
E_c_SS_k=[]
E_c_OS=[]
E_c_OS_k=[]
E_c_mp2tot=[]
E_c_mp2_com=[]
E_c_mp2_frag=[]

for i in range(3): #run over the fragements and complex
    run_mol=mols[i]
    #add here path to frag m.xyz file
    chkfile="chkfile31_"+run_mol+".chk"
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

#calculating the normal MP2 for functionals
form_frags=MPAC_functionals(Ex[0]+Ex[1],rho_4_3[0]+rho_4_3[1],gea_4_3[0]+gea_4_3[1]) #initialize the MPAC functionals
form_com=MPAC_functionals(Ex[2],rho_4_3[2],gea_4_3[2]) 
ehfdiv=ehf[2]-ehf[1]-ehf[0]
for i in range(len(mpacf)):
    E_c_mp2_com.append(E_c_mp2tot[2])
    E_c_mp2_frag.append(E_c_mp2tot[0]+E_c_mp2tot[1])
c_os=[1.8,2.2,2,1.7]
for i in range(len(mpacf)):
    E_c_mp2_com.append(c_os[i]*E_c_OS[2])
    E_c_mp2_frag.append(c_os[i]*(E_c_OS[0]+E_c_OS[1]))
#calculating the coskos-MP2 for all functionals
coskos=[[2.1,1.3],[2.3,1.1],[2.4,1],[2.1,0.9]]
for i in range(len(mpacf)):
    kos=k2ss.index(coskos[i][1])
    E_c_mp2_com.append(coskos[i][0]*E_c_OS_k[2][kos])
    E_c_mp2_frag.append(coskos[i][0]*(E_c_OS_k[0][kos]+E_c_OS_k[1][kos]))
#calculating the ksskos-MP2 for all functionals
ksskos=[[1.1,1.7],[1,1.4],[1.6,1.3],[0.9,1.4]]
for i in range(len(mpacf)):
    kss=k1ss.index(ksskos[i][0])
    kos=k2ss.index(ksskos[i][1])
    E_c_mp2_com.append(E_c_SS_k[2][kss]+E_c_OS_k[2][kos])
    E_c_mp2_frag.append(E_c_SS_k[0][kss]+E_c_OS_k[0][kos]+E_c_SS_k[1][kss]+E_c_OS_k[1][kos])
#calculating the k-MP2 for all functionals
ktot=[1.7,1.3,1.5,1.1]
for i in range(len(mpacf)):
    k=k1ss.index(ktot[i])
    E_c_mp2_com.append(E_c_SS_k[2][k]+E_c_OS_k[2][k])
    E_c_mp2_frag.append(E_c_SS_k[0][k]+E_c_OS_k[0][k]+E_c_SS_k[1][k]+E_c_OS_k[1][k])
funcs=["SPL2","cos-SPL2","coskos-SPL2","ksskos-SPL2","k-SPL2","F1","cos-F1","coskos-F1","ksskos-F1","k-F1","F1ab","cos-F1ab","coskos-F1ab","ksskos-F1ab","k-F1ab","MP2","cos-MP2","coskos-MP2","ksskos-MP2","k-MP2"]

#running all the functionals for all E_c_mp2 methods
E_c_spl2.append((ehfdiv+form_com.spl2(params[funcs[0]],E_c_mp2_com[0])-form_frags.spl2(params[funcs[0]],E_c_mp2_frag[0]))*kcal)
E_c_f1.append((ehfdiv+form_com.f1(params[funcs[5]],E_c_mp2_com[1])-form_frags.f1(params[funcs[5]],E_c_mp2_frag[1]))*kcal)
E_c_f1ab.append((ehfdiv+form_com.f1(params[funcs[10]],E_c_mp2_com[2])-form_frags.f1(params[funcs[10]],E_c_mp2_frag[2]))*kcal)
E_c_mp2.append((ehfdiv+form_com.mp2(params[funcs[15]],E_c_mp2_com[3])-form_frags.mp2(params[funcs[15]],E_c_mp2_frag[3]))*kcal)
E_c_spl2.append((ehfdiv+form_com.spl2(params[funcs[1]],E_c_mp2_com[4])-form_frags.spl2(params[funcs[1]],E_c_mp2_frag[4]))*kcal)
E_c_f1.append((ehfdiv+form_com.f1(params[funcs[6]],E_c_mp2_com[5])-form_frags.f1(params[funcs[6]],E_c_mp2_frag[5]))*kcal)
E_c_f1ab.append((ehfdiv+form_com.f1(params[funcs[11]],E_c_mp2_com[6])-form_frags.f1(params[funcs[11]],E_c_mp2_frag[6]))*kcal)
E_c_mp2.append((ehfdiv+form_com.mp2(params[funcs[16]],E_c_mp2_com[7])-form_frags.mp2(params[funcs[16]],E_c_mp2_frag[7]))*kcal)
E_c_spl2.append((ehfdiv+form_com.spl2(params[funcs[2]],E_c_mp2_com[8])-form_frags.spl2(params[funcs[2]],E_c_mp2_frag[8]))*kcal)
E_c_f1.append((ehfdiv+form_com.f1(params[funcs[7]],E_c_mp2_com[9])-form_frags.f1(params[funcs[7]],E_c_mp2_frag[9]))*kcal)
E_c_f1ab.append((ehfdiv+form_com.f1(params[funcs[12]],E_c_mp2_com[10])-form_frags.f1(params[funcs[12]],E_c_mp2_frag[10]))*kcal)
E_c_mp2.append((ehfdiv+form_com.mp2(params[funcs[17]],E_c_mp2_com[11])-form_frags.mp2(params[funcs[17]],E_c_mp2_frag[11]))*kcal)
E_c_spl2.append((ehfdiv+form_com.spl2(params[funcs[3]],E_c_mp2_com[12])-form_frags.spl2(params[funcs[3]],E_c_mp2_frag[12]))*kcal)
E_c_f1.append((ehfdiv+form_com.f1(params[funcs[8]],E_c_mp2_com[13])-form_frags.f1(params[funcs[8]],E_c_mp2_frag[13]))*kcal)
E_c_f1ab.append((ehfdiv+form_com.f1(params[funcs[13]],E_c_mp2_com[14])-form_frags.f1(params[funcs[13]],E_c_mp2_frag[14]))*kcal)
E_c_mp2.append((ehfdiv+form_com.mp2(params[funcs[18]],E_c_mp2_com[15])-form_frags.mp2(params[funcs[18]],E_c_mp2_frag[15]))*kcal)
E_c_spl2.append((ehfdiv+form_com.spl2(params[funcs[4]],E_c_mp2_com[16])-form_frags.spl2(params[funcs[4]],E_c_mp2_frag[16]))*kcal)
E_c_f1.append((ehfdiv+form_com.f1(params[funcs[9]],E_c_mp2_com[17])-form_frags.f1(params[funcs[9]],E_c_mp2_frag[17]))*kcal)
E_c_f1ab.append((ehfdiv+form_com.f1(params[funcs[14]],E_c_mp2_com[18])-form_frags.f1(params[funcs[14]],E_c_mp2_frag[18]))*kcal)
E_c_mp2.append((ehfdiv+form_com.mp2(params[funcs[19]],E_c_mp2_com[19])-form_frags.mp2(params[funcs[19]],E_c_mp2_frag[19]))*kcal)

#print json file
E_c_ints=E_c_spl2+E_c_f1+E_c_f1ab+E_c_mp2
E_c_int=dict(zip(funcs,E_c_ints))
print(E_c_int) #prints out the correct E_c_int
with open("E_c_all.json","w",encoding="utf-8") as f:
    json.dump(E_c_int,f)
