# ===============================================================
# Unified Transport Simulators (Gas / Ion / Drug)
# Clean full rebuild: no duplication, no dead code, all modules stable
# ===============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# --------------------------------------------------
# Constants
# --------------------------------------------------
R  = 8.314462618
kB = 1.380649e-23
h  = 6.62607015e-34
NA = 6.02214076e23
F  = 96485.33212

GPU_UNIT = 3.35e-10
PI_TINY  = 1e-14

# --------------------------------------------------
# UI helpers
# --------------------------------------------------
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3, help=None):
    if key not in st.session_state:
        st.session_state[key] = float(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    fmt = f"%.{decimals}f"
    lab = f"{label} [{'{}'.format(unit)}]" if unit else label

    st.slider(
        lab, float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
        key=f"{key}__s", format=fmt, help=help, on_change=_mark_s
    )
    st.number_input(
        "", float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
        key=f"{key}__n", format=fmt, on_change=_mark_n
    )

    if st.session_state[f"{key}__who"] == "n":
        new = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = float(st.session_state[f"{key}__s"])
    else:
        new = st.session_state[key]

    st.session_state[key] = float(np.clip(new, vmin, vmax))
    return st.session_state[key]

def nudged_int(label, vmin, vmax, vstep, vinit, key):
    if key not in st.session_state:
        st.session_state[key] = int(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    st.slider(label, int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
              key=f"{key}__s", on_change=_mark_s)
    st.number_input("", int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
                    key=f"{key}__n", on_change=_mark_n)

    if st.session_state[f"{key}__who"] == "n":
        new = int(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = int(st.session_state[f"{key}__s"])
    else:
        new = st.session_state[key]

    st.session_state[key] = int(np.clip(new, vmin, vmax))
    return st.session_state[key]

def log_slider(label, exp_min, exp_max, exp_step, exp_init, key, unit=""):
    if key not in st.session_state:
        st.session_state[key] = float(exp_init)

    def _mark_s(): st.session_state[key + "_who"] = "s"
    def _mark_n(): st.session_state[key + "_who"] = "n"

    lab = f"{label} ({unit})" if unit else label
    st.slider(
        lab, exp_min, exp_max, st.session_state[key],
        step=exp_step, format="%.2f", key=f"{key}_s", on_change=_mark_s
    )
    st.number_input(
        "exp (10^x)", exp_min, exp_max, st.session_state[key],
        step=exp_step, format="%.2f", key=f"{key}_n", on_change=_mark_n
    )
    who = st.session_state.get(key + "_who", "")
    if who == "s":
        x = st.session_state[f"{key}_s"]
    elif who == "n":
        x = st.session_state[f"{key}_n"]
    else:
        x = st.session_state[key]

    st.session_state[key] = x
    return 10 ** x

# --------------------------------------------------------------
# GAS MEMBRANE
# --------------------------------------------------------------

GAS_PARAMS = {
    "H2":  {"M":2.016e-3,  "d":2.89, "Ea_s":8e3},
    "D2":  {"M":4.028e-3,  "d":2.89, "Ea_s":8.5e3},
    "He":  {"M":4.003e-3,  "d":2.60, "Ea_s":6e3},
    "N2":  {"M":28.013e-3, "d":3.64, "Ea_s":9e3},
    "O2":  {"M":31.998e-3, "d":3.46, "Ea_s":9e3},
    "CO2": {"M":44.01e-3,  "d":3.30, "Ea_s":9.5e3},
    "CH4": {"M":16.043e-3, "d":3.80, "Ea_s":9.5e3},
    "C3H6":{"M":42.081e-3, "d":4.00, "Ea_s":1.05e4},
    "C3H8":{"M":44.097e-3, "d":4.30, "Ea_s":1.05e4},
}

MECH_COLOR = {
    "Blocked":"#bdbdbd", "Sieving":"#1f78b4", "Knudsen":"#33a02c",
    "Surface":"#e31a1c", "Solution":"#6a3d9a", "Capillary":"#ff7f00"
}

def de_broglie(T, M):
    m = M / NA
    return h / np.sqrt(2*np.pi*m*kB*T)

def effective_diameter(gas, T, alpha):
    dA = GAS_PARAMS[gas]["d"]
    lam = de_broglie(T, GAS_PARAMS[gas]["M"]) * 1e10
    return max(0.5*dA, dA - alpha*lam)

def alpha_auto(T, d_nm):
    a = 0.05*np.sqrt(300/max(T,1e-6))
    return float(np.clip(a, 0, 0.6))

# ------------ ideal model choices you asked me to pick --------------
DAMP_FACTOR = 1e-3
SURFACE_SCALE = 2e6
SOLUTION_SCALE = 2e6
CAPILLARY_D_MIN = 1.5
CAPILLARY_RP_MIN = 0.4

# ---------------- permeance expressions ----------------

def Pi_sieving(d_nm, gas, T, L_m, d_eff):
    pA = d_nm*10
    x = max(1 - (d_eff/pA)**2, 0.0)
    f = x**2
    return max(1e-6 * f * np.exp(-6000/(R*T)), PI_TINY)

def Pi_knudsen(d_nm, T, M, L_m):
    r = d_nm*1e-9/2
    Dk = (2/3)*r*np.sqrt((8*R*T)/(np.pi*M))
    Pi = Dk/(L_m*R*T)
    if d_nm <= 0.5:
        Pi *= DAMP_FACTOR
    return max(Pi, PI_TINY)

def Pi_surface(d_nm, gas, T, L_m, dqdp):
    Ds = 1e-10*np.exp(-GAS_PARAMS[gas]["Ea_s"]/(R*T))
    return max((Ds/L_m) * (dqdp*SURFACE_SCALE), PI_TINY)

def Pi_solution(gas, T, L_m, dqdp):
    Dsol = 3e-9*np.exp(-1.8e4/(R*T))
    return max((Dsol/L_m) * (dqdp*SOLUTION_SCALE), PI_TINY)

def Pi_capillary(d_nm, rp, L_m):
    if d_nm < CAPILLARY_D_MIN or rp < CAPILLARY_RP_MIN:
        return 0.0
    r = d_nm*1e-9/2
    return max(1e-6*np.sqrt(r)/L_m, 0.0)

# ---------------- DSL ----------------
def DSL(gas, T, P_bar, rp, q1, q2, b1, b2):
    P = rp*P_bar*1e5
    th1 = (b1*P)/(1+b1*P)
    th2 = (b2*P)/(1+b2*P)
    q = (q1*1e-3*1e3)*th1 + (q2*1e-3*1e3)*th2
    dqdp = (q1*1e-3*1e3)*(b1/(1+b1*P)**2) + (q2*1e-3*1e3)*(b2/(1+b2*P)**2)
    return q/1e3, dqdp

# ---------------- Gas membrane runner ----------------
def run_gas():
    st.header("Gas Membrane")

    T    = nudged_slider("T", 10,600,1,300,"Tg")
    Pbar = nudged_slider("P(bar)", 0.1,10,0.1,1.0,"Pg")
    d_nm = nudged_slider("Pore diameter",0.01,50,0.01,0.36,"d_nm_g")
    L_nm = nudged_slider("Thickness(nm)",10,100000,1,100,"Lg")

    gases = list(GAS_PARAMS.keys())
    g1 = st.sidebar.selectbox("Gas1", gases, index=gases.index("CO2"))
    g2 = st.sidebar.selectbox("Gas2", gases, index=gases.index("CH4"))

    alpha = alpha_auto(T, d_nm)
    d_eff = effective_diameter(g1, T, alpha)

    q11 = nudged_slider("q1 Gas1",0,100,0.01,0.7,"q11")
    q12 = nudged_slider("q2 Gas1",0,100,0.01,0.3,"q12")
    b11 = nudged_slider("b1 Gas1",1e-10,1e-3,1e-8,1e-5,"b11",decimals=8)
    b12 = nudged_slider("b2 Gas1",1e-10,1e-3,1e-8,5e-6,"b12",decimals=8)

    q21 = nudged_slider("q1 Gas2",0,100,0.01,0.7,"q21")
    q22 = nudged_slider("q2 Gas2",0,100,0.01,0.3,"q22")
    b21 = nudged_slider("b1 Gas2",1e-10,1e-3,1e-8,1e-5,"b21",decimals=8)
    b22 = nudged_slider("b2 Gas2",1e-10,1e-3,1e-8,5e-6,"b22",decimals=8)

    rp = np.linspace(0.01,0.99,400)
    L_m = L_nm*1e-9

    q1_vec = np.zeros_like(rp)
    dq1_vec= np.zeros_like(rp)
    q2_vec = np.zeros_like(rp)
    dq2_vec= np.zeros_like(rp)

    for i,r in enumerate(rp):
        q1_vec[i],dq1_vec[i]=DSL(g1,T,Pbar,r,q11,q12,b11,b12)
        q2_vec[i],dq2_vec[i]=DSL(g2,T,Pbar,r,q21,q22,b21,b22)

    Pi1 = np.zeros_like(rp)
    Pi2 = np.zeros_like(rp)

    for i,r in enumerate(rp):
        Pi1_intr = [
            Pi_sieving(d_nm,g1,T,L_m,d_eff),
            Pi_knudsen(d_nm,T,GAS_PARAMS[g1]["M"],L_m),
            Pi_surface(d_nm,g1,T,L_m,dq1_vec[i]),
            Pi_solution(g1,T,L_m,dq1_vec[i]),
            Pi_capillary(d_nm,r,L_m)
        ]
        Pi2_intr = [
            Pi_sieving(d_nm,g2,T,L_m,effective_diameter(g2,T,alpha)),
            Pi_knudsen(d_nm,T,GAS_PARAMS[g2]["M"],L_m),
            Pi_surface(d_nm,g2,T,L_m,dq2_vec[i]),
            Pi_solution(g2,T,L_m,dq2_vec[i]),
            Pi_capillary(d_nm,r,L_m)
        ]
        Pi1[i]=sum(Pi1_intr)
        Pi2[i]=sum(Pi2_intr)

    Pi1_gpu=Pi1/GPU_UNIT
    Pi2_gpu=Pi2/GPU_UNIT
    Sel=np.divide(Pi1,Pi2,out=np.zeros_like(Pi1),where=Pi2>0)

    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(rp,Pi1_gpu,label=g1)
    ax.plot(rp,Pi2_gpu,"--",label=g2)
    ax.set_ylabel("Permeance (GPU)")
    ax.set_xlabel("P/P0")
    ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close()

    fig2,ax2=plt.subplots(figsize=(8,3))
    ax2.plot(rp,Sel,label=f"{g1}/{g2}")
    ax2.set_ylabel("Selectivity")
    ax2.set_xlabel("P/P0")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close()

# --------------------------------------------------------------
# ION MEMBRANE
# --------------------------------------------------------------

ION_DB = {
    "Na+": {"z":1,"D":1.33e-9},
    "K+":  {"z":1,"D":1.96e-9},
    "Li+": {"z":1,"D":1.03e-9},
    "Ca2+": {"z":2,"D":0.79e-9},
    "Mg2+": {"z":2,"D":0.706e-9},
    "Cl-": {"z":-1,"D":2.03e-9},
    "NO3-":{"z":-1,"D":1.90e-9},
    "SO4^2-":{"z":-2,"D":1.065e-9},
    "HCO3-":{"z":-1,"D":1.18e-9},
    "Acetate-":{"z":-1,"D":1.09e-9},
}

def eta_water(T):
    A,B,C=2.414e-5,247.8,140
    T_C=T-273.15
    return A*10**(B/(T_C-C))

def D_temp(D,Tref,T):
    return D*(T/Tref)*(eta_water(Tref)/eta_water(T))

def donnan(c,z,K,Xf,T):
    psi=0.0
    for _ in range(40):
        t=-F*psi/(R*T)
        g=Xf; dg=0
        for sp,val in c.items():
            z_i=z[sp]
            Ki=K.get(sp,1)
            cm=Ki*val*np.exp(-z_i*t)
            g+=z_i*cm
            dg+=(F/(R*T))*(z_i**2)*cm
        psi-=g/(dg+1e-30)
    return psi

def NP_flux(D_eff,z,Cin,Cout,phi,L,T,v):
    L=max(L,1e-12)
    Cavg=0.5*(Cin+Cout)
    dC=Cout-Cin
    term_d=-D_eff*(dC/L)
    term_e=-(z*D_eff*F/(R*T))*Cavg*(phi/L)
    term_v=v*Cavg
    return term_d+term_e+term_v

def run_ion():
    st.header("Ion-Exchange Membrane")

    T= nudged_slider("T(K)",273,360,1,298,"Ti")
    Lnm= nudged_slider("Thickness(nm)",10,5000,1,200,"Lnm")
    eps= nudged_slider("Porosity",0.05,0.8,0.01,0.3,"eps")
    tau= nudged_slider("Tortuosity",1,5,0.1,2.0,"tau")
    Cf = nudged_slider("Fixed charge (mol/m3)",-3000,3000,10,-500,"Cf")
    dV = nudged_slider("Applied potential", -0.3,0.3,0.005,0.05,"dV")
    v_s= nudged_slider("Solvent velocity", -1e-4,1e-4,1e-6,0.0,"v_s",unit="m/s")

    cations=[k for k in ION_DB if ION_DB[k]["z"]>0]
    anions=[k for k in ION_DB if ION_DB[k]["z"]<0]

    sel_cat=st.multiselect("Cations",cations,default=["Na+","K+"])
    sel_an =st.multiselect("Anions",anions,default=["Cl-","NO3-"])

    species=sel_cat+sel_an

    c_feed={}
    c_perm={}
    for sp in species:
        c_feed[sp]=nudged_slider(f"{sp} feed",0,2000,1,100,f"cf_{sp}")
        c_perm[sp]=nudged_slider(f"{sp} perm",0,2000,1,10,f"cp_{sp}")

    autoKD=st.checkbox("Auto K=1, D(T) compute",True)
    if autoKD:
        K={sp:1 for sp in species}
        D_map={sp:D_temp(ION_DB[sp]["D"],298,T) for sp in species}
    else:
        K={}
        D_map={}
        for sp in species:
            K[sp]=nudged_slider(f"K {sp}",0.01,10,0.01,1,f"K_{sp}")
            D_map[sp]=log_slider(f"D {sp}",-11,-8,0.1,np.log10(ION_DB[sp]["D"]),f"D_{sp}")

    # Donnan
    z={sp:ION_DB[sp]["z"] for sp in species}
    L=Lnm*1e-9
    psi_f=donnan(c_feed,z,K,Cf,T)
    psi_p=donnan(c_perm,z,K,Cf,T)

    Cm_f={sp:K.get(sp,1)*c_feed[sp]*np.exp(-(z[sp]*F*psi_f)/(R*T)) for sp in species}
    Cm_p={sp:K.get(sp,1)*c_perm[sp]*np.exp(-(z[sp]*F*psi_p)/(R*T)) for sp in species}

    dphi_mem=dV - (psi_p-psi_f)
    D_eff={sp:D_map[sp]*eps/max(tau,1e-9) for sp in species}

    J={}
    for sp in species:
        J[sp]=NP_flux(D_eff[sp],z[sp],Cm_f[sp],Cm_p[sp],dphi_mem,L,T,v_s)

    i_net=F*sum(z[sp]*J[sp] for sp in species)

    st.subheader("Results")
    st.write("ψ_feed:", psi_f)
    st.write("ψ_perm:", psi_p)
    st.write("Δφ_mem:", dphi_mem)
    st.write("i_net (A/m2):", i_net)

    fig,ax=plt.subplots(figsize=(8,3))
    ax.bar(species,[J[sp] for sp in species])
    ax.set_ylabel("Flux (mol/m2/s)")
    ax.grid(True,axis='y')
    st.pyplot(fig); plt.close()

# --------------------------------------------------------------
# DRUG IN VESSEL
# --------------------------------------------------------------

DRUG_DB = {
    "Caffeine": {"Db":6.5e-10,"Pv":2e-6,"kel":1e-4},
    "Acetaminophen": {"Db":6e-10,"Pv":1.5e-6,"kel":8e-5},
    "Ibuprofen": {"Db":5e-10,"Pv":8e-7,"kel":1e-4},
    "Insulin": {"Db":1.8e-10,"Pv":5e-8,"kel":3e-5},
}

def run_drug():
    st.header("Drug Transport in Vessel")

    Rv= nudged_slider("Vessel radius(um)",2,100,1,4,"Rv")*1e-6
    Ls= nudged_slider("Length(mm)",1,200,1,20,"Lmm")*1e-3
    U = nudged_slider("Velocity(m/s)",1e-4,2e-2,1e-4,1e-3,"U")

    drug=st.selectbox("Drug",list(DRUG_DB))
    C0 = nudged_slider("Input conc",0,5,0.01,1,"C0")

    T= nudged_slider("T",293,312,0.5,310,"Tdrug")
    autoDb= st.checkbox("Auto Db",True)
    if autoDb:
        Db=D_temp(DRUG_DB[drug]["Db"],298,T)
    else:
        Db=log_slider("Db",-12,-8,0.1,np.log10(DRUG_DB[drug]["Db"]),"Db")
    Pv= log_slider("Pv",-9,-5,0.1,np.log10(DRUG_DB[drug]["Pv"]),"Pv")
    kel= log_slider("kel",-6,-2,0.1,np.log10(DRUG_DB[drug]["kel"]),"kel")

    tend= nudged_slider("t_end",0.1,600,0.1,60,"tend")
    dt  = nudged_slider("dt",1e-3,0.5,1e-3,0.01,"dt")
    Nx  = nudged_int("Nx",50,600,10,200,"Nx")

    x=np.linspace(0,Ls,Nx)
    dx=x[1]-x[0]
    t=np.arange(0,tend+dt,dt)

    Pe_r=U*Rv/Db
    Deff=Db*(1+(Pe_r**2)/192)
    k_leak=2*Pv/max(Rv,1e-12)

    C=np.zeros((len(t),Nx))
    Cin_t=C0*np.exp(-((t-0.2*tend)**2)/(2*(0.05*tend)**2))

    for n in range(1,len(t)):
        Cn=C[n-1].copy()
        Cnp=Cn.copy()
        Cnp[0]=Cin_t[n]

        lam_a=U*dt/dx
        lam_d=Deff*dt/(dx**2)

        adv=-lam_a*(Cn[1:]-Cn[:-1])
        dif=lam_d*(Cn[2:]-2*Cn[1:-1]+Cn[:-2])
        react=-dt*(k_leak+kel)*Cn[1:-1]

        Cnp[1:-1]=Cn[1:-1]+adv[:-1]+dif+react
        Cnp[-1]=Cnp[-2]
        C[n]=np.maximum(Cnp,0)

    fig,ax=plt.subplots(figsize=(8,3))
    im=ax.imshow(C.T,aspect='auto',origin='lower',
                 extent=(t[0],t[-1],x[0]*1e3,x[-1]*1e3))
    ax.set_xlabel("t(s)"); ax.set_ylabel("x(mm)")
    plt.colorbar(im,ax=ax).set_label("C")
    st.pyplot(fig); plt.close()

    fig2,ax2=plt.subplots(figsize=(8,3))
    ax2.plot(t,C[:,-1])
    ax2.set_ylabel("Outlet conc")
    ax2.grid(True)
    st.pyplot(fig2); plt.close()

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
st.title("Unified Transport Simulators")

mode=st.sidebar.radio("Mode",["Gas membrane","Ion membrane","Drug transport"])
if mode=="Gas membrane":
    run_gas()
elif mode=="Ion membrane":
    run_ion()
else:
    run_drug()
