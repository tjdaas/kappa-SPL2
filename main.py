#import pyscf, numpy and numba
from pyscf import gto, mp, ao2mo, dft, scf
import numpy as np
import argparse
import os
from numba_codes import reg_sos_mp2
from mol import run_pyscf
from kappa_codes.mpac_fun import MPAC_functionals
from kappa_codes.constants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--basis", type=str, default="aug-cc-pvqz")
    parser.add_argument("--kappa", type=bool, default=False)
    parser.add_argument("--cos", type=bool, default=False)
    parser.add_argument("--mpacf",type=str,default="mp2")
    parser.add_argument("--ksam",type=bool,default=False)

args = parser.parse_args()
mols=["A","B","AB"]
Ex=[]
ehf=[]
Uh=[]
rho_4_3=[]
gea_4_3=[]
rho_3_2=[]
gea_7_6=[]
E_c_mp2=[]
para=params(args.kappa,args.cos,args.mpacf,args.ksam)
kapcoslist=[]
while len(para)>4 or (len(para)<3 and len(para)>0):
    kapcoslist.append(para.pop())
for i in range(3):
    run_mol=mols[i]
    #add here path to frag m.xyz file
    #open m.xyz fike
    chkfile="chkfile31_"+run_mol+".chk"
    ###runs HF
    old_pwd=os.getcwd()
    datadir=old_pwd+"/Ne/"+run_mol
    os.chdir(datadir)
    py_run=run_pyscf(atom="m.xyz",charge=args.charge,spin=args.spin,basis=args.basis)
    tab, eris = py_run.run_eris(chkfile_name=chkfile,chkfile_dir=datadir)
    nocc, e, eri = eris
    #this prints and extracts what we need
    np.savetxt("tab.csv", tab, delimiter=",", fmt='%s')
    ehf.append(tab[0])
    Uh.append(tab[1])
    Ex.append(tab[2])
    rho_4_3.append(tab[3])
    gea_4_3.append(tab[4])
    rho_3_2.append(tab[5])
    gea_7_6.append(tab[6])
    defs=reg_sos_mp2(nocc,e,eri)
    if args.kappa==True:
        ### Runs normal \kappa
        #e_mp2_k_split = defs.MP2_energy_kappa_p_split(1.2, 1.2, 1, 1) #do MP2 energy with kappa=1.2 and p=1
        #np.savetxt("mp2_k.csv", e_mp2_k_split, delimiter=",", fmt='%s')
        #k1 is for same spin
        #k2 is for the opposite spin
        k1ss = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        k2ss = k1ss
        #np.savetxt("k1.csv", k1s, delimiter=",", fmt='%s')
        #np.savetxt("k2.csv", k2s, delimiter=",", fmt='%s')

        k1s=np.array(k1ss,dtype=float)
        k2s=np.array(k2ss,dtype=float)
        k_os=kapcoslist[0]
        mp2OS = defs.MP2_energy_kappa_p_OS_parallel(k2s, 1)
        np.savetxt("os.csv", mp2OS, delimiter=",", fmt='%s')
        #Spin scaled \kappa's
        if args.cos==False:
            mp2SS = defs.MP2_energy_kappa_p_SS_parallel(k1s, 1)
            np.savetxt("ss.csv", mp2SS, delimiter=",", fmt='%s')
            k_ss=kapcoslist[1]
            E_c_kmp2_tot= mp2SS[k1ss.index(k_ss)] + mp2OS[k2ss.index(k_os)]
            E_c_mp2.append(E_c_kmp2_tot)
        else:
            c_os=kapcoslist[1]
            E_c_kmp2_cos= c_os*mp2OS[k2ss.index(k_os)]
            E_c_mp2.append(E_c_kmp2_cos)
    else:
        ###Runs E_c^MP2(ss) and E_c^MP2(os)
        e_mp2_split = defs.MP2_energy_split() #cal
        np.savetxt("mp2.csv", e_mp2_split, delimiter=",", fmt='%s')
        if args.cos==False:
            E_c_mp2_tot= sum(e_mp2_split)
            E_c_mp2.append(E_c_mp2_tot)
        else:
            c_os=kapcoslist[0]
            E_c_mp2_cos= c_os*e_mp2_split[1]
            E_c_mp2.append(E_c_mp2_cos)

    os.chdir(old_pwd)
form_frags=MPAC_functionals(Ex[0]+Ex[1],(E_c_mp2[0]+E_c_mp2[1]),rho_4_3[0]+rho_4_3[1],gea_4_3[0]+gea_4_3[1])
form_com=MPAC_functionals(Ex[2],E_c_mp2[2],rho_4_3[2],gea_4_3[2])
if args.mpacf == "spl2":
    E_c_int=(ehf[2]-ehf[1]-ehf[0]+form_com.spl2(para)-form_frags.spl2(para))*kcal

elif args.mpacf == "f1":
    E_c_int=(ehf[2]-ehf[1]-ehf[0]+form_com.f1(para)-form_frags.f1(para))*kcal

elif args.mpacf == "f1ab":
    E_c_int=(ehf[2]-ehf[1]-ehf[0]+form_com.f1(para)-form_frags.f1(para))*kcal

elif args.mpacf == "mp2":
    E_c_int=(ehf[2]-ehf[1]-ehf[0]+form_com.mp2(para)-form_frags.mp2(para))*kcal

else:
    print("only supports: spl2, f1, f1ab and mp2")
print(E_c_int)
