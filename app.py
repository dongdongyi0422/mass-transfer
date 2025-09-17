# app.py — Membrane Transport Simulator (SI)
# - Mechanism band: weight-based (softmax)
# - Time 모드: 메커니즘 바 x축 = Time (s), 압력 모드: x축 = P/P0
# - DSL: (q1,q2,b1,b2) 직접 입력
# - Quantum-size correction α: T 기반 자동 설정 옵션 포함
# - Π는 GPU로 표시

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# -------------------- constants / globals --------------------
R = 8.314                      # J/mol/K
kB = 1.380649e-23              # J/K
h  = 6.62607015e-34            # J·s
GPU_UNIT = 3.35e-10            # mol m^-2 s^-1 Pa^-1
PI_TINY = 1e-14

# porous transport scales
RHO_EFF   = 500.0
D0_SURF   = 1e-9
D0_SOL    = 1e-10
E_D_SOL   = 1.8e4
K_CAP     = 1e-7
E_SIEVE   = 6.0e3
SOL_TH_NM = 0.30

# sieving smoothing
SIEVE_BAND_A = 0.15
DELTA_A      = 0.4
DELTA_SOFT_A = 0.50
PI_SOFT_REF  = 1e-6

# weighting
WEIGHT_MODE   = "softmax"
SOFTMAX_GAMMA = 0.8
DAMP_KNUDSEN  = True
DAMP_FACTOR   = 1e-3

PARAMS = {
    "H2":  {"M":2.016e-3,  "d":2.89, "Ea_s":8.0e3},
    "D2":  {"M":4.028e-3,  "d":2.89, "Ea_s":8.5e3},
    "He":  {"M":4.003e-3,  "d":2.60, "Ea_s":6.0e3},
    "N2":  {"M":28.0134e-3,"d":3.64, "Ea_s":9.0e3},
    "O2":  {"M":31.998e-3, "d":3.46, "Ea_s":9.0e3},
    "CO2": {"M":44.01e-3,  "d":3.30, "Ea_s":9.5e3},
    "CH4": {"M":16.043e-3, "d":3.80, "Ea_s":9.5e3},
    "C2H6":{"M":30.070e-3, "d":4.44, "Ea_s":1.0e4},
    "C3H6":{"M":42.081e-3, "d":4.00, "Ea_s":1.05e4},
    "C3H8":{"M":44.097e-3, "d":4.30, "Ea_s":1.05e4},
}

MECH_ORDER = ["Blocked","Sieving","Knudsen","Surface","Capillary","Solution"]
MECH_COLOR = {
    "Blocked":"#bdbdbd","Sieving":"#1f78b4","Knudsen":"#33a02c",
    "Surface":"#e31a1c","Capillary":"#ff7f00","Solution":"#6a3d9a"
}

# -------------------- quantum / helpers --------------------
def de_broglie_lambda_m(T, M):
    """Thermal de Broglie wavelength, λ = h / sqrt(2π m k_B T). Returns [m]."""
    m = M            # kg/mol? -> per molecule mass:
    m = M / 6.02214076e23
    return h / np.sqrt(2.0*np.pi*m*kB*max(float(T),1e-9))

def alpha_auto_by_temperature(T, a0=0.05, T_ref=300.0, a_min=0.0, a_max=0.60, d_nm=None):
    """온도만으로 α 계산 + (선택) 좁은 기공에서 보정 강화"""
    a = a0 * np.sqrt(T_ref / max(float(T), 1e-9))
    if d_nm is not None:
        a *= (1.0 + 0.3*np.exp(-(max(d_nm,1e-6)/0.5)**2))
    return float(np.clip(a, a_min, a_max))

def effective_diameter_A(gas, T, alpha):
    """양자 보정된 유효 분자 직경 [Å]; d_eff = d - α*λ(Å), 하한 0.5*d."""
    dA = PARAMS[gas]["d"]
    lamA = de_broglie_lambda_m(T, PARAMS[gas]["M"]) * 1e10  # m->Å
    d_eff = dA - float(alpha)*lamA
    return float(max(0.5*dA, d_eff))

# -------------------- physics --------------------
def mean_free_path_nm(T, P_bar, d_ang):
    P = P_bar*1e5; d = d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def pintr_knudsen_SI(d_nm, T, M, L_m):
    r = max(d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0)*r*np.sqrt((8.0*R*T)/(np.pi*M))
    return Dk/(L_m*R*T)

def pintr_sieving_SI(d_nm, gas, T, L_m, d_eff_A):
    dA_eff = d_eff_A
    pA = d_nm*10.0
    delta = dA_eff - pA          # >0: 기공이 더 작음(차단)
    if delta > 0:
        return max(PI_SOFT_REF*np.exp(-(delta/DELTA_SOFT_A)**2)*np.exp(-E_SIEVE/(R*T)), PI_TINY)
    x = max(1.0 - (dA_eff/pA)**2, 0.0)
    f = x**2
    return max(3e-4*f*np.exp(-E_SIEVE/(R*T)), PI_TINY)

def pintr_surface_SI(d_nm, gas, T, L_m, dqdp):
    Ds = D0_SURF*np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    return max((Ds/L_m)*(dqdp*RHO_EFF),0.0)

def pintr_capillary_SI(d_nm, rp, L_m):
    r_m = max(d_nm*1e-9/2.0, 1e-12)
    thresh = np.exp(-120.0/(((d_nm/2.0)*rp*300.0)+1e-12))
    if rp <= thresh: return 0.0
    return K_CAP*np.sqrt(r_m)/L_m

def pintr_solution_SI(gas, T, L_m, dqdp):
    Dsol = D0_SOL*np.exp(-E_D_SOL/(R*T))/np.sqrt(PARAMS[gas]["M"]/1e-3)
    return max((Dsol/L_m)*(dqdp*RHO_EFF),0.0)

def _series_parallel(Pp,Pd,eps=1e-30):
    if not np.isfinite(Pp): Pp=0.0
    if not np.isfinite(Pd): Pd=0.0
    if Pp<=eps and Pd<=eps: return PI_TINY
    if Pp<=eps: return Pd
    if Pd<=eps: return Pp
    return 1.0/((1.0/(Pp+eps))+(1.0/(Pd+eps)))

# -------------------- DSL --------------------
def dsl_loading_and_slope_b(gas,T,P_bar,relP,q1,q2,b1,b2):
    P0=P_bar*1e5
    b1=max(float(b1),0.0); b2=max(float(b2),0.0)
    q1_molkg=q1*1e-3*1e3; q2_molkg=q2*1e-3*1e3
    q_vec=np.zeros_like(relP,float); dqdp=np.zeros_like(relP,float)
    for i,rp in enumerate(relP):
        P=max(float(rp),1e-9)*P0
        th1=(b1*P)/(1.0+b1*P); th2=(b2*P)/(1.0+b2*P)
        q_molkg=q1_molkg*th1 + q2_molkg*th2
        q_vec[i]=q_molkg/1e3
        dqdp[i]=(q1_molkg*b1)/(1.0+b1*P)**2 + (q2_molkg*b2)/(1.0+b2*P)**2
    return q_vec,dqdp

# -------------------- weights --------------------
def weights_from_intrinsic(Pi_intr, gamma=SOFTMAX_GAMMA):
    keys = MECH_ORDER
    x=np.array([np.log(max(float(Pi_intr.get(k,0.0)),1e-30)) for k in keys])
    e=np.exp(gamma*x); s=float(e.sum())
    w=e/s if s>0 and np.isfinite(s) else np.array([0,0,0,1,0,0],float)
    return {k:float(w[i]) for i,k in enumerate(keys)}

def damp_knudsen_if_needed(Pi_intr, d_nm, rp):
    if WEIGHT_MODE=="softmax" and DAMP_KNUDSEN and (d_nm>=2.0 and rp>=0.55):
        Pi_intr["Knudsen"]*=DAMP_FACTOR
    return Pi_intr

# -------------------- mechanism band --------------------
def mechanism_band_rgba(g1,g2,T,P_bar,d_nm,relP,L_nm,q11,q12,b11,b12,alpha):
    _,dv = dsl_loading_and_slope_b(g1,T,P_bar,relP,q11,q12,b11,b12)
    L_m=max(float(L_nm),1e-3)*1e-9; M1=PARAMS[g1]["M"]
    d_eff_A = effective_diameter_A(g1,T,alpha)
    names=[]
    for i,rp in enumerate(relP):
        dq=float(dv[i])
        Pi_intr={
            "Blocked": PI_TINY,
            "Sieving": pintr_sieving_SI(d_nm,g1,T,L_m,d_eff_A),
            "Knudsen": pintr_knudsen_SI(d_nm,T,M1,L_m),
            "Surface": pintr_surface_SI(d_nm,g1,T,L_m,dq),
            "Capillary":pintr_capillary_SI(d_nm,float(rp),L_m),
            "Solution": pintr_solution_SI(g1,T,L_m,dq),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,float(rp))
        w=weights_from_intrinsic(Pi_intr)
        names.append(max(w,key=w.get))
    rgba=np.array([to_rgba(MECH_COLOR[n]) for n in names])[None,:,:]
    return rgba, names

def mechanism_band_rgba_time(g1,g2,T,P_bar,d_nm,L_nm,t_vec,P_bar_t,dqdp1,P0bar,alpha):
    L_m=max(float(L_nm),1e-3)*1e-9; M1=PARAMS[g1]["M"]
    rp_t=np.clip(P_bar_t/float(P0bar),1e-6,0.9999)
    d_eff_A = effective_diameter_A(g1,T,alpha)
    names=[]
    for i in range(len(t_vec)):
        rp=float(rp_t[i]); dq=float(dqdp1[i])
        Pi_intr={
            "Blocked": PI_TINY,
            "Sieving": pintr_sieving_SI(d_nm,g1,T,L_m,d_eff_A),
            "Knudsen": pintr_knudsen_SI(d_nm,T,M1,L_m),
            "Surface": pintr_surface_SI(d_nm,g1,T,L_m,dq),
            "Capillary":pintr_capillary_SI(d_nm,rp,L_m),
            "Solution": pintr_solution_SI(g1,T,L_m,dq),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,rp)
        w=weights_from_intrinsic(Pi_intr)
        names.append(max(w,key=w.get))
    rgba=np.array([to_rgba(MECH_COLOR[n]) for n in names])[None,:,:]
    return rgba, names

# -------------------- transient LDF --------------------
def pressure_schedule_series(t,P0_bar,ramp,tau):
    if isinstance(ramp,str) and ramp.lower().startswith("step"):
        return np.full_like(t,float(P0_bar),dtype=float)
    tau=max(float(tau),1e-9)
    return float(P0_bar)*(1.0-np.exp(-t/tau))

def ldf_evolve_q(t,P_bar_t,qeq_fn,kLDF,q0=0.0):
    q_dyn=np.zeros_like(t,float); dqdp=np.zeros_like(t,float); q=float(q0)
    for i in range(len(t)):
        Pbar=float(P_bar_t[i]); qeq, slope = qeq_fn(Pbar)
        dqdp[i]=slope
        if i>0:
            dt=float(t[i]-t[i-1]); q += dt*float(kLDF)*(qeq - q)
        q_dyn[i]=q
    return q_dyn, dqdp

# -------------------- permeance mix model --------------------
def permeance_series_SI(d_nm, gas, other, T, P_bar, relP, L_nm, q_mmolg, dqdp, q_other, alpha):
    L_m=max(L_nm,1e-3)*1e-9; M=PARAMS[gas]["M"]
    d_eff_A = effective_diameter_A(gas,T,alpha)
    Pi=np.zeros_like(relP,float)
    for i,rp in enumerate(relP):
        Pi_intr={
            "Blocked": PI_TINY,
            "Sieving": pintr_sieving_SI(d_nm,gas,T,L_m,d_eff_A),
            "Knudsen": pintr_knudsen_SI(d_nm,T,M,L_m),
            "Surface": pintr_surface_SI(d_nm,gas,T,L_m,dqdp[i]),
            "Capillary":pintr_capillary_SI(d_nm,rp,L_m),
            "Solution": pintr_solution_SI(gas,T,L_m,dqdp[i]),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,rp)
        w=weights_from_intrinsic(Pi_intr)
        Pi_pore = w["Sieving"]*Pi_intr["Sieving"] + w["Knudsen"]*Pi_intr["Knudsen"] + w["Capillary"]*Pi_intr["Capillary"]
        Pi_diff = w["Surface"]*Pi_intr["Surface"] + w["Solution"]*Pi_intr["Solution"]
        Pi0=_series_parallel(Pi_pore,Pi_diff)
        theta = (q_mmolg[i]/(q_mmolg[i]+q_other[i])) if (q_mmolg[i]+q_other[i])>0 else 0.0
        Pi[i]=Pi0*theta
    return Pi

# -------------------- UI --------------------
st.set_page_config(page_title="Membrane Permeance (SI)", layout="wide")
st.title("Membrane Transport Simulator (SI units)")

with st.sidebar:
    st.header("Global Conditions")

    def nudged_slider(label,vmin,vmax,vstep,vinit,key,unit="",decimals=3):
        if key not in st.session_state: st.session_state[key]=float(vinit)
        cur=float(st.session_state[key])
        lab=f"{label}{(' ['+unit+']') if unit else ''}"
        fmt=f"%.{int(decimals)}f"
        sld=st.slider(lab,float(vmin),float(vmax),float(cur),float(vstep),key=f"{key}_s")
        num=st.number_input("",float(vmin),float(vmax),float(cur),float(vstep),format=fmt,key=f"{key}_n")
        new=float(num) if num!=cur else float(sld)
        new=float(np.clip(new,vmin,vmax))
        if new!=cur: st.session_state[key]=new
        return st.session_state[key]

    T    = nudged_slider("Temperature",10.0,600.0,1.0,300.0,key="T",unit="K")
    Pbar = nudged_slider("Total Pressure",0.1,10.0,0.1,1.0,key="Pbar",unit="bar")
    d_nm = nudged_slider("Pore diameter",0.01,50.0,0.01,0.36,key="d_nm",unit="nm")
    L_nm = nudged_slider("Membrane thickness",10.0,100000.0,1.0,100.0,key="L_nm",unit="nm")

    gases=list(PARAMS.keys())
    gas1=st.selectbox("Gas1 (numerator)",gases,index=gases.index("CO2"))
    gas2=st.selectbox("Gas2 (denominator)",gases,index=gases.index("CH4"))

    st.header("DSL parameters (q1,q2,b1,b2)")
    st.subheader("Gas1")
    q11=nudged_slider("q1 Gas1",0.0,100.0,0.01,0.70,key="q11",unit="mmol/g")
    q12=nudged_slider("q2 Gas1",0.0,100.0,0.01,0.30,key="q12",unit="mmol/g")
    b11=nudged_slider("b1 Gas1",1e-10,1e-1,1e-8,1e-5,key="b11",unit="Pa⁻¹",decimals=8)
    b12=nudged_slider("b2 Gas1",1e-10,1e-1,1e-8,5e-6,key="b12",unit="Pa⁻¹",decimals=8)

    st.subheader("Gas2")
    q21=nudged_slider("q1 Gas2",0.0,100.0,0.01,0.70,key="q21",unit="mmol/g")
    q22=nudged_slider("q2 Gas2",0.0,100.0,0.01,0.30,key="q22",unit="mmol/g")
    b21=nudged_slider("b1 Gas2",1e-10,1e-1,1e-8,1e-5,key="b21",unit="Pa⁻¹",decimals=8)
    b22=nudged_slider("b2 Gas2",1e-10,1e-1,1e-8,5e-6,key="b22",unit="Pa⁻¹",decimals=8)

    mode = st.radio("X-axis / Simulation mode",["Relative pressure (P/P0)","Time (transient LDF)"],index=0)

    if mode=="Time (transient LDF)":
        st.subheader("Transient (LDF) settings")
        t_end=nudged_slider("Total time",0.1,3600.0,0.1,120.0,key="t_end",unit="s")
        dt   =nudged_slider("Time step",1e-3,10.0,1e-3,0.1,key="dt",unit="s")
        kLDF =nudged_slider("k_LDF",1e-4,10.0,1e-4,0.05,key="kLDF",unit="s⁻¹")
        P0bar=nudged_slider("Feed P₀",0.1,10.0,0.1,Pbar,key="P0bar",unit="bar")
        ramp =st.selectbox("Pressure schedule P(t)",["Step (P=P₀)","Exp ramp: P₀(1-exp(-t/τ))"],index=1)
        tau  =nudged_slider("τ (only for exp ramp)",1e-3,1000.0,1e-3,5.0,key="tau",unit="s")

    # --- Quantum size correction (α) ---
    st.markdown("---")
    st.subheader("Quantum size correction (α)")
    auto_alpha = st.checkbox("Auto-set α from temperature", value=True,
                             help="α(T)=a₀·√(T₀/T), 좁은 기공에서 약간 강화")
    a0 = st.slider("Auto α scale (a₀)", 0.00, 0.20, 0.05, 0.01)
    if "alpha" not in st.session_state:
        st.session_state["alpha"] = 0.05

    if auto_alpha:
        alpha = alpha_auto_by_temperature(T, a0=a0, T_ref=300.0, a_min=0.0, a_max=0.60, d_nm=d_nm)
        st.session_state["alpha"] = alpha
        st.info(f"α auto-set → {alpha:.4f}")
    else:
        st.session_state["alpha"] = st.slider("Manual α", 0.0, 0.60, float(st.session_state["alpha"]), 0.01)

alpha = float(st.session_state["alpha"])

# -------------------- compute --------------------
time_mode = (mode == "Time (transient LDF)")

if time_mode:
    t=np.arange(0.0, st.session_state["t_end"]+st.session_state["dt"], st.session_state["dt"])
    P_bar_t=pressure_schedule_series(t, st.session_state["P0bar"], ramp, st.session_state["tau"])
    relP=np.clip(P_bar_t/float(st.session_state["P0bar"]),1e-6,0.9999)

    def qeq_g1(Pbar_scalar):
        rp=float(np.clip(Pbar_scalar/float(Pbar),1e-6,0.9999))
        qv,dv=dsl_loading_and_slope_b(gas1,T,Pbar,np.array([rp]),q11,q12,b11,b12)
        return float(qv[0]),float(dv[0])

    def qeq_g2(Pbar_scalar):
        rp=float(np.clip(Pbar_scalar/float(Pbar),1e-6,0.9999))
        qv,dv=dsl_loading_and_slope_b(gas2,T,Pbar,np.array([rp]),q21,q22,b21,b22)
        return float(qv[0]),float(dv[0])

    q1_dyn,dqdp1=ldf_evolve_q(t,P_bar_t,qeq_g1,st.session_state["kLDF"],0.0)
    q2_dyn,dqdp2=ldf_evolve_q(t,P_bar_t,qeq_g2,st.session_state["kLDF"],0.0)

    Pi1=permeance_series_SI(d_nm,gas1,gas2,T,Pbar,relP,L_nm,q1_dyn,dqdp1,q2_dyn,alpha)
    Pi2=permeance_series_SI(d_nm,gas2,gas1,T,Pbar,relP,L_nm,q2_dyn,dqdp2,q1_dyn,alpha)

    X_vals=t; X_label="Time (s)"
else:
    relP=np.linspace(0.01,0.99,500)
    q1_mg,dqdp1=dsl_loading_and_slope_b(gas1,T,Pbar,relP,q11,q12,b11,b12)
    q2_mg,dqdp2=dsl_loading_and_slope_b(gas2,T,Pbar,relP,q21,q22,b21,b22)
    Pi1=permeance_series_SI(d_nm,gas1,gas2,T,Pbar,relP,L_nm,q1_mg,dqdp1,q2_mg,alpha)
    Pi2=permeance_series_SI(d_nm,gas2,gas1,T,Pbar,relP,L_nm,q2_mg,dqdp2,q1_mg,alpha)
    X_vals=relP; X_label=r"Relative pressure, $P/P_0$ (–)"

Sel = np.divide(Pi1,Pi2,out=np.zeros_like(Pi1),where=(Pi2>0))
Pi1_gpu = Pi1/GPU_UNIT; Pi2_gpu = Pi2/GPU_UNIT

# -------------------- layout --------------------
colA, colB = st.columns([1, 2])

with colB:
    # Mechanism map
    figBand, axBand = plt.subplots(figsize=(9, 0.7))
    if time_mode:
        rgba,_ = mechanism_band_rgba_time(gas1,gas2,T,Pbar,d_nm,L_nm,t,P_bar_t,dqdp1,st.session_state["P0bar"],alpha)
        x_min, x_max = float(t[0]), float(t[-1]); x_ticks = np.linspace(x_min,x_max,6)
    else:
        rgba,_ = mechanism_band_rgba(gas1,gas2,T,Pbar,d_nm,relP,L_nm,q11,q12,b11,b12,alpha)
        x_min, x_max = 0.0, 1.0; x_ticks = [0,0.2,0.4,0.6,0.8,1.0]
    axBand.imshow(rgba, extent=(x_min,x_max,0,1), aspect="auto", origin="lower")
    axBand.set_xlim(x_min,x_max); axBand.set_xticks(x_ticks); axBand.set_yticks([])
    axBand.set_xlabel(X_label)
    handles=[plt.Rectangle((0,0),1,1,fc=MECH_COLOR[n],ec='none',label=n) for n in MECH_ORDER]
    leg=axBand.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.7),ncol=6,frameon=True)
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True); plt.close(figBand)

    # Permeance
    fig1, ax1 = plt.subplots(figsize=(9,3))
    ax1.plot(X_vals, Pi1_gpu, label=f"{gas1}")
    ax1.plot(X_vals, Pi2_gpu, '--', label=f"{gas2}")
    ax1.set_xlabel(X_label); ax1.set_ylabel(r"$\Pi$ (GPU)")
    ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.get_yaxis().get_offset_text().set_visible(False)
    ax1.grid(True); ax1.legend(title="Permeance (GPU)")
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

    # Selectivity
    fig2, ax2 = plt.subplots(figsize=(9,3))
    ax2.plot(X_vals, Sel, label=f"{gas1}/{gas2}")
    ax2.set_xlabel(X_label); ax2.set_ylabel("Selectivity (–)")
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.get_yaxis().get_offset_text().set_visible(False)
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with colA:
    st.subheader("Mechanism (rule) vs intrinsic (Gas1) — reference")
    # 간단 룰(참조용)
    def classify_mech(d_nm,g1,g2,T,P_bar,rp):
        d_eff = effective_diameter_A(g1,T,alpha)  # α 반영
        dmin = min(d_eff, PARAMS[g2]["d"])
        pA=d_nm*10.0; lam=mean_free_path_nm(T,P_bar,0.5*(d_eff+PARAMS[g2]["d"]))
        if pA<=dmin-SIEVE_BAND_A: return "Blocked"
        if d_nm<=SOL_TH_NM: return "Solution"
        if (pA>=dmin+DELTA_A) and (d_nm<0.5*lam): return "Knudsen"
        if d_nm>=2.0 and rp>0.5: return "Capillary"
        if pA<=dmin+SIEVE_BAND_A: return "Sieving"
        return "Surface"

    rp_mid = (np.clip(P_bar_t/float(st.session_state["P0bar"]),1e-6,0.9999)[len(t)//2] if time_mode
              else float(relP[len(relP)//2]))
    L_m=L_nm*1e-9; M1=PARAMS[gas1]["M"]
    dq_mid = float(dqdp1[len(dqdp1)//2]) if np.ndim(dqdp1)>0 and len(dqdp1)>0 else 0.0
    d_eff_A = effective_diameter_A(gas1,T,alpha)
    cand={
        "Blocked":  PI_TINY,
        "Sieving":  pintr_sieving_SI(d_nm,gas1,T,L_m,d_eff_A),
        "Knudsen":  pintr_knudsen_SI(d_nm,T,M1,L_m),
        "Surface":  pintr_surface_SI(d_nm,gas1,T,L_m,dq_mid),
        "Capillary":pintr_capillary_SI(d_nm,rp_mid,L_m),
        "Solution": pintr_solution_SI(gas1,T,L_m,dq_mid),
    }
    st.markdown(
        f"**Mechanism (rule):** `{classify_mech(d_nm,gas1,gas2,T,Pbar,rp_mid)}`  "
        f"|  **Best intrinsic:** `{max(cand,key=cand.get)}`"
    )
    st.caption(
        "Values shown in the band are weight-based winners per x-position. "
        "α uses de Broglie scaling α(T)=a₀√(T₀/T) (with optional pore-size boost)."
    )
