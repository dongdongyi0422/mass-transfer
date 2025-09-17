# app.py — Membrane Transport Simulator (SI)
# - Mechanism band: weight-based (softmax/heuristic 선택)
# - Time 모드일 때 메커니즘 바 x축 = Time (s)
# - DSL: (q1,q2,b1,b2) 직접 입력
# - Π는 GPU로 표시

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# -------------------- constants / globals --------------------
R = 8.314
GPU_UNIT = 3.35e-10
PI_TINY = 1e-14

RHO_EFF   = 500.0
D0_SURF   = 1e-9
D0_SOL    = 1e-10
E_D_SOL   = 1.8e4
K_CAP     = 1e-7
E_SIEVE   = 6.0e3
SOL_TH_NM = 0.30

SIEVE_BAND_A = 0.15
DELTA_A      = 0.4
DELTA_SOFT_A = 0.50
PI_SOFT_REF  = 1e-6

# weighting
WEIGHT_MODE   = "softmax"   # "softmax" or "heuristic"
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

# -------------------- physics helpers --------------------
def mean_free_path_nm(T, P_bar, d_ang):
    kB=1.380649e-23; P=P_bar*1e5; d=d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def pintr_knudsen_SI(d_nm, T, M, L_m):
    r = max(d_nm*1e-9/2, 1e-12)
    Dk = (2/3)*r*np.sqrt((8*R*T)/(np.pi*M))
    return Dk/(L_m*R*T)

def pintr_sieving_SI(d_nm, gas, T, L_m):
    dA = PARAMS[gas]["d"]; pA = d_nm*10.0; delta = dA - pA
    if delta > 0:
        return max(PI_SOFT_REF*np.exp(-(delta/DELTA_SOFT_A)**2)*np.exp(-E_SIEVE/(R*T)), PI_TINY)
    x=max(1-(dA/pA)**2,0.0); f=x**2
    return max(3e-4*f*np.exp(-E_SIEVE/(R*T)), PI_TINY)

def pintr_surface_SI(d_nm, gas, T, L_m, dqdp):
    Ds = D0_SURF*np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    return max((Ds/L_m)*(dqdp*RHO_EFF),0.0)

def pintr_capillary_SI(d_nm, rp, L_m):
    r_m=max(d_nm*1e-9/2,1e-12)
    thresh=np.exp(-120.0/(((d_nm/2)*rp*300.0)+1e-12))
    if rp<=thresh: return 0.0
    return K_CAP*np.sqrt(r_m)/L_m

def pintr_solution_SI(gas,T,L_m,dqdp):
    Dsol = D0_SOL*np.exp(-E_D_SOL/(R*T))/np.sqrt(PARAMS[gas]["M"]/1e-3)
    return max((Dsol/L_m)*(dqdp*RHO_EFF),0.0)

def _series_parallel(Pp,Pd,eps=1e-30):
    if not np.isfinite(Pp): Pp=0.0
    if not np.isfinite(Pd): Pd=0.0
    if Pp<=eps and Pd<=eps: return PI_TINY
    if Pp<=eps: return Pd
    if Pd<=eps: return Pp
    return 1.0/((1.0/(Pp+eps))+(1.0/(Pd+eps)))

# -------------------- DSL (b 직접 입력) --------------------
def dsl_loading_and_slope_b(gas,T,P_bar,relP,q1,q2,b1,b2):
    P0=P_bar*1e5; b1=max(float(b1),0.0); b2=max(float(b2),0.0)
    q1_molkg=q1*1e-3*1e3; q2_molkg=q2*1e-3*1e3
    q_vec=np.zeros_like(relP,float); dqdp=np.zeros_like(relP,float)
    for i,rp in enumerate(relP):
        P=max(float(rp),1e-9)*P0
        th1=(b1*P)/(1+b1*P); th2=(b2*P)/(1+b2*P)
        q_molkg=q1_molkg*th1+q2_molkg*th2
        q_vec[i]=q_molkg/1e3
        dqdp[i]=(q1_molkg*b1)/(1+b1*P)**2+(q2_molkg*b2)/(1+b2*P)**2
    return q_vec,dqdp

# -------------------- weights --------------------
def weights_from_intrinsic(Pi_intr, gamma=SOFTMAX_GAMMA):
    keys = MECH_ORDER
    x=np.array([np.log(max(float(Pi_intr.get(k,0.0)),1e-30)) for k in keys])
    e=np.exp(gamma*x); s=float(e.sum())
    w=e/s if s>0 and np.isfinite(s) else np.array([0,0,0,1,0,0],float)
    return {k:float(w[i]) for i,k in enumerate(keys)}

def mechanism_weights(gas,other,T,P_bar,d_nm,rp,dqdp):
    d1=PARAMS[gas]["d"]; d2=PARAMS[other]["d"]; dmin=min(d1,d2)
    pA=d_nm*10.0; lam=mean_free_path_nm(T,P_bar,0.5*(d1+d2))
    if pA<=dmin-SIEVE_BAND_A:
        return {"Blocked":1.0,"Sieving":0.0,"Knudsen":0.0,"Surface":0.0,"Capillary":0.0,"Solution":0.0}
    if d_nm<=SOL_TH_NM:
        return {"Blocked":0.0,"Sieving":0.0,"Knudsen":0.0,"Surface":0.0,"Capillary":0.0,"Solution":1.0}
    def sig(x,s=1.0): return 1/(1+np.exp(-x/s))
    w_sieve=np.exp(-((pA-dmin)/max(SIEVE_BAND_A,1e-6))**2)
    r=d_nm/max(lam,1e-9); w_kn=1/(1+(r/0.5)**2)
    w_cap_raw=sig((d_nm-2.0)/0.25)*sig((rp-0.60)/0.06)
    alpha=5e4; s_base=1-np.exp(-alpha*max(float(dqdp),0.0))
    w_surf=s_base*(1-0.9*w_cap_raw); w_cap=w_cap_raw*(1-0.3*s_base)
    w_sol=sig((SOL_TH_NM-d_nm)/0.02); w_blk=np.exp(-((dmin-pA)/max(SIEVE_BAND_A,1e-6))**2) if pA<dmin else 0.0
    w={"Blocked":w_blk,"Sieving":w_sieve,"Knudsen":w_kn,"Surface":w_surf,"Capillary":w_cap,"Solution":w_sol}
    s=sum(w.values())
    return {k:v/s for k,v in w.items()} if s>1e-12 else {"Blocked":0,"Sieving":0,"Knudsen":0,"Surface":1,"Capillary":0,"Solution":0}

def damp_knudsen_if_needed(Pi_intr, d_nm, rp):
    if WEIGHT_MODE=="softmax" and DAMP_KNUDSEN and (d_nm>=2.0 and rp>=0.55):
        Pi_intr["Knudsen"]*=DAMP_FACTOR
    return Pi_intr

# -------------------- bands --------------------
def mechanism_band_rgba(g1,g2,T,P_bar,d_nm,relP,L_nm,q11,q12,b11,b12):
    _,dv = dsl_loading_and_slope_b(g1,T,P_bar,relP,q11,q12,b11,b12)
    L_m=max(float(L_nm),1e-3)*1e-9; M1=PARAMS[g1]["M"]
    names=[]
    for i,rp in enumerate(relP):
        dq=float(dv[i])
        Pi_intr={
            "Blocked":PI_TINY,
            "Sieving":pintr_sieving_SI(d_nm,g1,T,L_m),
            "Knudsen":pintr_knudsen_SI(d_nm,T,M1,L_m),
            "Surface":pintr_surface_SI(d_nm,g1,T,L_m,dq),
            "Capillary":pintr_capillary_SI(d_nm,float(rp),L_m),
            "Solution":pintr_solution_SI(g1,T,L_m,dq),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,float(rp))
        w=weights_from_intrinsic(Pi_intr) if WEIGHT_MODE=="softmax" else mechanism_weights(g1,g2,T,P_bar,d_nm,float(rp),dq)
        names.append(max(w,key=w.get))
    rgba=np.array([to_rgba(MECH_COLOR[n]) for n in names])[None,:,:]
    return rgba, names

def mechanism_band_rgba_time(g1,g2,T,P_bar,d_nm,L_nm,t_vec,P_bar_t,dqdp_series_g1,P0bar):
    L_m=max(float(L_nm),1e-3)*1e-9; M1=PARAMS[g1]["M"]
    rp_t=np.clip(P_bar_t/float(P0bar),1e-6,0.9999)
    names=[]
    for i in range(len(t_vec)):
        rp=float(rp_t[i]); dq=float(dqdp_series_g1[i])
        Pi_intr={
            "Blocked":PI_TINY,
            "Sieving":pintr_sieving_SI(d_nm,g1,T,L_m),
            "Knudsen":pintr_knudsen_SI(d_nm,T,M1,L_m),
            "Surface":pintr_surface_SI(d_nm,g1,T,L_m,dq),
            "Capillary":pintr_capillary_SI(d_nm,rp,L_m),
            "Solution":pintr_solution_SI(g1,T,L_m,dq),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,rp)
        w=weights_from_intrinsic(Pi_intr) if WEIGHT_MODE=="softmax" else mechanism_weights(g1,g2,T,P_bar,d_nm,rp,dq)
        names.append(max(w,key=w.get))
    rgba=np.array([to_rgba(MECH_COLOR[n]) for n in names])[None,:,:]
    return rgba, names

# -------------------- transient LDF --------------------
def pressure_schedule_series(t,P0_bar,ramp,tau):
    if isinstance(ramp,str) and ramp.lower().startswith("step"):
        return np.full_like(t,float(P0_bar),dtype=float)
    tau=max(float(tau),1e-9)
    return float(P0_bar)*(1-np.exp(-t/tau))

def ldf_evolve_q(t,P_bar_t,qeq_fn,kLDF,q0=0.0):
    q_dyn=np.zeros_like(t,float); dqdp=np.zeros_like(t,float); q=float(q0)
    for i in range(len(t)):
        Pbar=float(P_bar_t[i]); qeq, slope = qeq_fn(Pbar)
        dqdp[i]=slope
        if i>0:
            dt=float(t[i]-t[i-1]); q += dt*float(kLDF)*(qeq - q)
        q_dyn[i]=q
    return q_dyn, dqdp

# -------------------- mix model --------------------
def permeance_series_SI(d_nm, gas, other, T, P_bar, relP, L_nm, q_mmolg, dqdp, q_other):
    L_m=max(L_nm,1e-3)*1e-9; M=PARAMS[gas]["M"]
    Pi=np.zeros_like(relP,float)
    for i,rp in enumerate(relP):
        Pi_intr={
            "Blocked":PI_TINY,
            "Sieving":pintr_sieving_SI(d_nm,gas,T,L_m),
            "Knudsen":pintr_knudsen_SI(d_nm,T,M,L_m),
            "Surface":pintr_surface_SI(d_nm,gas,T,L_m,dqdp[i]),
            "Capillary":pintr_capillary_SI(d_nm,rp,L_m),
            "Solution":pintr_solution_SI(gas,T,L_m,dqdp[i]),
        }
        Pi_intr=damp_knudsen_if_needed(Pi_intr,d_nm,rp)
        w=weights_from_intrinsic(Pi_intr) if WEIGHT_MODE=="softmax" else mechanism_weights(gas,other,T,P_bar,d_nm,rp,dqdp[i])
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
        cur=float(st.session_state[key]); lab=f"{label}{(' ['+unit+']') if unit else ''}"
        fmt=f"%.{int(decimals)}f"
        sld=st.slider(lab,float(vmin),float(vmax),float(cur),float(vstep),key=f"{key}_s")
        num=st.number_input("",float(vmin),float(vmax),float(cur),float(vstep),format=fmt,key=f"{key}_n")
        new=float(num) if num!=cur else float(sld); new=float(np.clip(new,vmin,vmax))
        if new!=cur: st.session_state[key]=new
        return st.session_state[key]

    T    = nudged_slider("Temperature",10.0,600.0,1.0,450.0,key="T",unit="K")
    Pbar = nudged_slider("Total Pressure",0.1,10.0,0.1,5.0,key="Pbar",unit="bar")
    d_nm = nudged_slider("Pore diameter",0.01,50.0,0.01,2.45,key="d_nm",unit="nm")
    L_nm = nudged_slider("Membrane thickness",10.0,100000.0,1.0,100.0,key="L_nm",unit="nm")

    gases=list(PARAMS.keys())
    gas1=st.selectbox("Gas1 (numerator)",gases,index=gases.index("CO2"))
    gas2=st.selectbox("Gas2 (denominator)",gases,index=gases.index("CH4"))

    st.header("DSL parameters (q1,q2,b1,b2)")
    st.subheader("Gas1")
    q11=nudged_slider("q1 Gas1",0.0,100.0,0.01,0.80,key="q11",unit="mmol/g")
    q12=nudged_slider("q2 Gas1",0.0,100.0,0.01,0.30,key="q12",unit="mmol/g")
    b11=nudged_slider("b1 Gas1",1e-10,1e-1,1e-8,1e-6,key="b11",unit="Pa⁻¹",decimals=8)
    b12=nudged_slider("b2 Gas1",1e-10,1e-1,1e-8,5e-7,key="b12",unit="Pa⁻¹",decimals=8)

    st.subheader("Gas2")
    q21=nudged_slider("q1 Gas2",0.0,100.0,0.01,0.50,key="q21",unit="mmol/g")
    q22=nudged_slider("q2 Gas2",0.0,100.0,0.01,0.20,key="q22",unit="mmol/g")
    b21=nudged_slider("b1 Gas2",1e-10,1e-1,1e-8,5e-7,key="b21",unit="Pa⁻¹",decimals=8)
    b22=nudged_slider("b2 Gas2",1e-10,1e-1,1e-8,2e-7,key="b22",unit="Pa⁻¹",decimals=8)

    mode = st.radio("X-axis / Simulation mode",["Relative pressure (P/P0)","Time (transient LDF)"],index=1)

    if mode=="Time (transient LDF)":
        st.subheader("Transient (LDF) settings")
        t_end=nudged_slider("Total time",0.1,3600.0,0.1,120.0,key="t_end",unit="s")
        dt   =nudged_slider("Time step",1e-3,10.0,1e-3,0.1,key="dt",unit="s")
        kLDF =nudged_slider("k_LDF",1e-4,10.0,1e-4,0.05,key="kLDF",unit="s⁻¹")
        P0bar=nudged_slider("Feed P₀",0.1,10.0,0.1,Pbar,key="P0bar",unit="bar")
        ramp =st.selectbox("Pressure schedule P(t)",["Step (P=P₀)","Exp ramp: P₀(1-exp(-t/τ))"],index=1)
        tau  =nudged_slider("τ (only for exp ramp)",1e-3,1000.0,1e-3,5.0,key="tau",unit="s")

        # === SINGLE SOURCE OF TRUTH for X-axis ===
        if time_mode:
            X_vals  = t
            X_label = "Time (s)"
            X_min, X_max = float(t[0]), float(t[-1])
            X_ticks = np.linspace(X_min, X_max, 6)
        else:
            X_vals  = relP
            X_label = r"Relative pressure, $P/P_0$ (–)"
            X_min, X_max = 0.0, 1.0
            X_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # 디버그(잠깐만 켜두세요)
        st.write({"DBG_time_mode": time_mode, "X_min": X_min, "X_max": X_max})

time_mode = (mode == "Time (transient LDF)")

# === Common X axis (단 하나의 진실) ===
if time_mode:
    X_vals  = t
    X_label = "Time (s)"
    X_min, X_max = float(t[0]), float(t[-1])
    X_ticks = np.linspace(X_min, X_max, 6)
else:
    X_vals  = relP
    X_label = r"Relative pressure, $P/P_0$ (–)"
    X_min, X_max = 0.0, 1.0
    X_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# -------------------- compute --------------------
if time_mode:
    t=np.arange(0.0,t_end+dt,dt)
    P_bar_t=pressure_schedule_series(t,P0bar,ramp,tau)
    relP=np.clip(P_bar_t/float(P0bar),1e-6,0.9999)

    def qeq_g1(Pbar_scalar):
        rp=float(np.clip(Pbar_scalar/float(Pbar),1e-6,0.9999))
        qv,dv=dsl_loading_and_slope_b(gas1,T,Pbar,np.array([rp]),q11,q12,b11,b12)
        return float(qv[0]),float(dv[0])

    def qeq_g2(Pbar_scalar):
        rp=float(np.clip(Pbar_scalar/float(Pbar),1e-6,0.9999))
        qv,dv=dsl_loading_and_slope_b(gas2,T,Pbar,np.array([rp]),q21,q22,b21,b22)
        return float(qv[0]),float(dv[0])

    q1_dyn,dqdp1=ldf_evolve_q(t,P_bar_t,qeq_g1,kLDF,0.0)
    q2_dyn,dqdp2=ldf_evolve_q(t,P_bar_t,qeq_g2,kLDF,0.0)

    Pi1=permeance_series_SI(d_nm,gas1,gas2,T,Pbar,relP,L_nm,q1_dyn,dqdp1,q2_dyn)
    Pi2=permeance_series_SI(d_nm,gas2,gas1,T,Pbar,relP,L_nm,q2_dyn,dqdp2,q1_dyn)

    x_axis=t; x_label="Time (s)"
else:
    relP=np.linspace(0.01,0.99,500)
    q1_mg,dqdp1=dsl_loading_and_slope_b(gas1,T,Pbar,relP,q11,q12,b11,b12)
    q2_mg,dqdp2=dsl_loading_and_slope_b(gas2,T,Pbar,relP,q21,q22,b21,b22)
    Pi1=permeance_series_SI(d_nm,gas1,gas2,T,Pbar,relP,L_nm,q1_mg,dqdp1,q2_mg)
    Pi2=permeance_series_SI(d_nm,gas2,gas1,T,Pbar,relP,L_nm,q2_mg,dqdp2,q1_mg)
    x_axis=relP; x_label=r"Relative pressure, $P/P_0$ (–)"

Sel = np.divide(Pi1,Pi2,out=np.zeros_like(Pi1),where=(Pi2>0))
Pi1_gpu=Pi1/GPU_UNIT; Pi2_gpu=Pi2/GPU_UNIT

# ==== 공통 X축 정의 (둘 다 여기만 봅니다) ====
if time_mode:
    x_axis_common = t
    x_label_common = "Time (s)"
    x_min, x_max = float(t[0]), float(t[-1])
    x_ticks_common = np.linspace(x_min, x_max, 6)
else:
    x_axis_common = relP
    x_label_common = r"Relative pressure, $P/P_0$ (–)"
    x_min, x_max = 0.0, 1.0
    x_ticks_common = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


# -------------------- layout --------------------
colA, colB = st.columns([1, 2])

# === 공통 X축 정의 (이미 있음) ===
if time_mode:
    X_vals  = t
    X_label = "Time (s)"
    X_min, X_max = float(t[0]), float(t[-1])
    X_ticks = np.linspace(X_min, X_max, 6)
else:
    X_vals  = relP
    X_label = r"Relative pressure, $P/P_0$ (–)"
    X_min, X_max = 0.0, 1.0
    X_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 디버그용: 지금 x축이 어떻게 잡혀 있는지 확인
st.write({"DBG": {"time_mode": time_mode, "X_min": X_min, "X_max": X_max}})

colA, colB = st.columns([1, 2])

with colB:
    # === Mechanism map (weighted) ===
    figBand, axBand = plt.subplots(figsize=(9, 0.7))

    if time_mode:
        rgba, _ = mechanism_band_rgba_time(
            gas1, gas2, T, Pbar, d_nm, L_nm, t, P_bar_t, dqdp1, P0bar
        )
    else:
        rgba, _ = mechanism_band_rgba(
            gas1, gas2, T, Pbar, d_nm, relP, L_nm, q11, q12, b11, b12
        )

    # 공통 X축 강제 적용 (시간이면 초, 압력이면 0~1)
    axBand.imshow(rgba, extent=(X_min, X_max, 0, 1), aspect="auto", origin="lower")
    axBand.set_xlabel(X_label)
    axBand.set_xlim(X_min, X_max)
    axBand.set_xticks(X_ticks)
    axBand.set_yticks([])

    handles = [plt.Rectangle((0,0),1,1, fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
    leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.7),
                        ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True)
    plt.close(figBand)


    # === Permeance (GPU) ===
    fig1, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(X_vals, Pi1 / GPU_UNIT, label=f"{gas1}")
    ax1.plot(X_vals, Pi2 / GPU_UNIT, '--', label=f"{gas2}")
    ax1.set_xlim(X_min, X_max)
    ax1.set_xticks(X_ticks)
    ax1.set_xlabel(X_label)
    ax1.set_ylabel(r"$\Pi$ (GPU)")

    from matplotlib.ticker import ScalarFormatter
    ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.get_yaxis().get_offset_text().set_visible(False)
    ax1.grid(True); ax1.legend(title="Permeance (GPU)")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # === Selectivity ===
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(X_vals, Sel, label=f"{gas1}/{gas2}")
    ax2.set_xlim(X_min, X_max)
    ax2.set_xticks(X_ticks)
    ax2.set_xlabel(X_label)
    ax2.set_ylabel("Selectivity (–)")
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.get_yaxis().get_offset_text().set_visible(False)
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


with colA:
    st.subheader("Mechanism (rule) vs intrinsic (Gas1) — reference")
    def classify_mech(d_nm,g1,g2,T,P_bar,rp):
        d1,d2=PARAMS[g1]["d"],PARAMS[g2]["d"]; dmin=min(d1,d2); pA=d_nm*10.0
        lam=mean_free_path_nm(T,P_bar,0.5*(d1+d2))
        if pA<=dmin-SIEVE_BAND_A: return "Blocked"
        if d_nm<=SOL_TH_NM: return "Solution"
        if (pA>=dmin+DELTA_A) and (d_nm<0.5*lam): return "Knudsen"
        if d_nm>=2.0 and rp>0.5: return "Capillary"
        if pA<=dmin+SIEVE_BAND_A: return "Sieving"
        return "Surface"

    if time_mode:
        rp_mid = float(np.clip(P_bar_t/float(P0bar), 1e-6, 0.9999)[len(t)//2])
    else:
        rp_mid = float(relP[len(relP)//2])

    L_m = L_nm*1e-9; M1 = PARAMS[gas1]["M"]
    # dqdp1은 위에서 이미 계산되어 있음. x축 길이 기준 인덱스 사용
    mid_idx = len(X_vals)//2
    dq_mid = float(dqdp1[mid_idx]) if np.ndim(dqdp1)>0 and len(dqdp1)>mid_idx else 0.0

    cand = {
        "Blocked":  PI_TINY,
        "Sieving":  pintr_sieving_SI(d_nm, gas1, T, L_m),
        "Knudsen":  pintr_knudsen_SI(d_nm, T, M1, L_m),
        "Surface":  pintr_surface_SI(d_nm, gas1, T, L_m, dq_mid),
        "Capillary":pintr_capillary_SI(d_nm, rp_mid, L_m),
        "Solution": pintr_solution_SI(gas1, T, L_m, dq_mid),
    }
    st.markdown(
        f"**Mechanism (rule):** `{classify_mech(d_nm,gas1,gas2,T,Pbar,rp_mid)}`  "
        f"|  **Best intrinsic:** `{max(cand,key=cand.get)}`"
    )
