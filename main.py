#import pyscf and numpy
from pyscf import gto, mp, ao2mo, dft, scf
import numpy as np
import argparse
import os
from numba_codes import reg_sos_mp2
from mol import run_pyscf
from mpac_fun import MPAC_functionals
from constants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--charge", type=int, default=0, help="the charge of the system")
    parser.add_argument("--spin", type=int, default=0, help="the spin of the system")
    parser.add_argument("--basis", type=str, default="aug-cc-pvqz",help="the basisset used in the calculations")
    parser.add_argument("--func",type=str,default="coskos-SPL2")

args = parser.parse_args()
#obtaining the attributes of the inputted functional:
if "kos" in args.func: #checking if it uses the \kappa regularizer
    kappa=True
else:
    kappa=False
if "cos" in args.func: #checking if it uses the spin opposite scaling
    cos=True
else:
    cos=False
if "k-" in args.func: #checking if it is the original \kappa-method
    ksam=True
    kappa=True
else:
    ksam=False
if "mp2" in args.func: #checking which base functional is used
    mpacf="mp2"
elif "spl2" in args.func:
    mpacf="spl2"
elif "f1ab" in args.func:
    mpacf="f1ab"
elif "f1" in args.func:
    mpacf="f1"
else:
    raise ValueError("no valid functional provided, please use mp2, spl2, f1 or f1ab") #gives error if the wrong functional is used

if kappa==False and cos==False and args.func.split(mpacf)[0]!="":
    raise ValueError("Unknown prefix use coskos-, ksskos-, k- or no prefix") #gives error if the wrong prefix is used
    
mols=["A","B","AB"] #A and B are fragments, AB is the complex
Ex=[]
ehf=[]
Uh=[]
rho_4_3=[]
gea_4_3=[]
rho_3_2=[]
gea_7_6=[]
E_c_mp2=[]
para,name=params[(kappa,cos,ksam,mpacf)] #obtain parameters for the chosen MPAC functional
print(f"the functional that will be run is: {name}")
kapcoslist=[]

while len(para)>4 or (len(para)<3 and len(para)>0): #removes the non-functional specific paremeters (i.e. removes \kappa_ss, \kappa_os and c_os)
    kapcoslist.append(para.pop())

for i in range(3): #run over the fragments and complex
    run_mol=mols[i]
    #add here path to frag m.xyz file
    chkfile="chkfile_"+run_mol+".chk"
    old_pwd=os.getcwd()
    datadir=old_pwd+"/"+run_mol
    os.chdir(datadir)
    ###runs HF
    py_run=run_pyscf(atom="m.xyz",charge=args.charge,spin=args.spin,basis=args.basis)
    tab, eris = py_run.run_eris(chkfile_name=chkfile,chkfile_dir=datadir)
    nocc, e, eri = eris
    #this prints and extracts all of the ingredients except MP2
    np.savetxt("tab.csv", tab, delimiter=",", fmt='%s')
    ehf.append(tab[0])
    Uh.append(tab[1])
    Ex.append(tab[2])
    rho_4_3.append(tab[3])
    gea_4_3.append(tab[4])
    rho_3_2.append(tab[5])
    gea_7_6.append(tab[6])
    defs=reg_sos_mp2(nocc,e,eri) #define MP2
    if kappa==True: #if \kappa is turned on
        #k1 is for same spin
        #k2 is for the opposite spin
        k1ss = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        k2ss = k1ss
        np.savetxt("k1.csv", k1s, delimiter=",", fmt='%s')
        np.savetxt("k2.csv", k2s, delimiter=",", fmt='%s')

        k1s=np.array(k1ss,dtype=float)
        k2s=np.array(k2ss,dtype=float)
        k_os=kapcoslist[0]
        mp2OS = defs.MP2_energy_kappa_p_OS_parallel(k2s, 1) #calculate the opposite spin integral
        np.savetxt("os.csv", mp2OS, delimiter=",", fmt='%s')
        #Spin scaled \kappa's
        if cos==False:
            mp2SS = defs.MP2_energy_kappa_p_SS_parallel(k1s, 1) #calculate the same spin integral
            np.savetxt("ss.csv", mp2SS, delimiter=",", fmt='%s')
            k_ss=kapcoslist[1] 
            E_c_kmp2_tot= mp2SS[k1ss.index(k_ss)] + mp2OS[k2ss.index(k_os)] #take only the value that corresponds to the optimal k_ss and k_os
            E_c_mp2.append(E_c_kmp2_tot)
        else:
            c_os=kapcoslist[1]
            E_c_kmp2_cos= c_os*mp2OS[k2ss.index(k_os)] #take only the value that corresponds to the optimal c_os and k_os values
            E_c_mp2.append(E_c_kmp2_cos)
    else: #run MP2 without \kappa
        ###Runs E_c^MP2(ss) and E_c^MP2(os)
        e_mp2_split = defs.MP2_energy_split() #cal
        np.savetxt("mp2.csv", e_mp2_split, delimiter=",", fmt='%s')
        if cos==False: #run regular MP2
            E_c_mp2_tot= sum(e_mp2_split)
            E_c_mp2.append(E_c_mp2_tot)
        else: # run spin opposite scaled mp2
            c_os=kapcoslist[0]
            E_c_mp2_cos= c_os*e_mp2_split[1]
            E_c_mp2.append(E_c_mp2_cos)

    os.chdir(old_pwd)
form_frags=MPAC_functionals(Ex[0]+Ex[1],(E_c_mp2[0]+E_c_mp2[1]),rho_4_3[0]+rho_4_3[1],gea_4_3[0]+gea_4_3[1]) #initialize the MPAC functionals
form_com=MPAC_functionals(Ex[2],E_c_mp2[2],rho_4_3[2],gea_4_3[2])
ehfdiv=ehf[2]-ehf[1]-ehf[0]
if mpacf == "spl2": #calculate the interaction energy of SPL2
    E_c_int=(ehfdiv+form_com.spl2(para)-form_frags.spl2(para))*kcal

elif mpacf == "f1": #calculate the interaction energy of F1
    E_c_int=(ehfdiv+form_com.f1(para)-form_frags.f1(para))*kcal

elif mpacf == "f1ab": #calculate the interaction energy of F1[\alpha,\beta]
    E_c_int=(ehfdiv+form_com.f1(para)-form_frags.f1(para))*kcal

elif mpacf == "mp2": #calcullate the interaction energy of MP2
    E_c_int=(ehfdiv]+form_com.mp2(para)-form_frags.mp2(para))*kcal

print(f"The {name} interaction energy: {E_c_int}") #prints out the correct E_c_int
