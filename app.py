# app.py — Unified Transport Simulators (Gas / Ion / Vascular)
# - 모든 슬라이더에 대응하는 입력칸(number_input) 제공
# - Gas membrane: α(드브로이) 자동/수동 + λ,d_eff 표시, 메커니즘 밴드/퍼미언스/선택도
# - Ion membrane: Bulk conc sliders + K&D 자동(T/η) + 수동 오버라이드 옵션
#                 + (NEW) crack 유도 열폭주 시뮬레이션(Transient)
# - Drug in vessel: Db 자동 보정(water/plasma) 옵션 + Pv, k_elim 로그 슬라이더

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# -------------------- page config --------------------
st.set_page_config(page_title="Unified Transport Simulators (Gas / Ion / Vascular)", layout="wide")

# -------------------- constants --------------------
R  = 8.314462618            # J/mol/K
kB = 1.380649e-23           # J/K
h  = 6.62607015e-34         # J·s
NA = 6.02214076e23          # 1/mol
F  = 96485.33212            # C/mol

# =====================================================================
# Shared UI helpers  — 슬라이더 + 입력칸 동기화(안정 플래그 방식)
# =====================================================================
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3, help=None):
    if key not in st.session_state:
        st.session_state[key] = float(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    fmt = f"%.{int(decimals)}f"

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    st.slider(lab, float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
              help=help, key=f"{key}__s", format=fmt, on_change=_mark_s)
    st.number_input("", float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
                    key=f"{key}__n", format=fmt, on_change=_mark_n)

    if st.session_state[f"{key}__who"] == "n":
        new = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = float(st.session_state[f"{key}__s"])
    else:
        new = float(st.session_state[key])

    st.session_state[key] = float(np.clip(new, vmin, vmax))
    return float(st.session_state[key])

def nudged_int(label, vmin, vmax, vstep, vinit, key, help=None):
    if key not in st.session_state:
        st.session_state[key] = int(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    st.slider(label, int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
              help=help, key=f"{key}__s", on_change=_mark_s)
    st.number_input("", int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
                    key=f"{key}__n", on_change=_mark_n)

    if st.session_state[f"{key}__who"] == "n":
        new = int(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = int(st.session_state[f"{key}__s"])
    else:
        new = int(st.session_state[key])

    st.session_state[key] = int(np.clip(new, vmin, vmax))
    return int(st.session_state[key])

def log_slider(label, exp_min, exp_max, exp_step, exp_init, key, unit="", help=None):
    """
    반환: 10^x (실값). 세션에는 지수 x를 저장.
    → 이 반환값을 '그대로' 사용하고, 추가로 10** 하지 마세요!
    """
    if key not in st.session_state:
        st.session_state[key] = float(exp_init)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    lab = f"{label}{(' ['+unit+']') if unit else ''}"

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    st.slider(lab, float(exp_min), float(exp_max), float(st.session_state[key]),
              float(exp_step), help=help, key=f"{key}__s", format="%.2f", on_change=_mark_s)
    st.number_input("exp (10^x)", float(exp_min), float(exp_max), float(st.session_state[key]),
                    float(exp_step), key=f"{key}__n", format="%.2f", on_change=_mark_n)

    if st.session_state[f"{key}__who"] == "n":
        exp_val = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        exp_val = float(st.session_state[f"{key}__s"])
    else:
        exp_val = float(st.session_state[key])

    st.session_state[key] = float(exp_val)
    return 10.0 ** float(st.session_state[key])

# ---------- viscosity & D(T) helpers ----------
def eta_water_PaS(T_K: float) -> float:
    """물 점도(μ, Pa·s) 근사: μ = A * 10^(B/(T_C - C))"""
    T_C = float(T_K) - 273.15
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * (10.0 ** (B / (T_C - C)))

def D_temp_correction(D_ref: float, T_ref: float, T: float, eta_ref: float, eta: float) -> float:
    """Stokes–Einstein: D(T) ≈ D_ref * (T/T_ref) * (η_ref/η)"""
    return float(D_ref) * (float(T)/float(T_ref)) * (float(eta_ref)/float(eta))

# =====================================================================
# MODE 1 — GAS MEMBRANE
# =====================================================================
GPU_UNIT = 3.35e-10        # mol m^-2 s^-1 Pa^-1
PI_TINY  = 1e-14

# porous transport scales
RHO_EFF   = 500.0
D0_SURF   = 1e-10
D0_SOL    = 3e-9
E_D_SOL   = 1.2e4
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
DAMP_FACTOR   = 1e-3  # ← 상수 통일(아래 함수에서도 이 값 사용)

GAS_PARAMS = {
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

def de_broglie_lambda_m(T, M):
    m = M / NA
    return h / np.sqrt(2.0*np.pi*m*kB*max(float(T),1e-9))

def alpha_auto_by_temperature(T, a0=0.05, T_ref=300.0, a_min=0.0, a_max=0.60, d_nm=None):
    a = a0 * np.sqrt(T_ref / max(float(T), 1e-9))
    if d_nm is not None:
        a *= (1.0 + 0.3*np.exp(-(max(d_nm,1e-6)/0.5)**2))
    return float(np.clip(a, a_min, a_max))

def effective_diameter_A(gas, T, alpha):
    dA = GAS_PARAMS[gas]["d"]
    lamA = de_broglie_lambda_m(T, GAS_PARAMS[gas]["M"]) * 1e10  # m->Å
    d_eff = dA - float(alpha)*lamA
    return float(max(0.5*dA, d_eff))

def mean_free_path_nm(T, P_bar, d_ang):
    P = P_bar*1e5; d = d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def pintr_knudsen_SI(d_nm, T, M, L_m):
    r = max(d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0)*r*np.sqrt((8.0*R*T)/(np.pi*M))
    return Dk/(L_m*R*T)

# ---------------- 수정 내용 ----------------

# ---------------- physics-balanced permeance formulas ----------------
def pintr_sieving_SI(d_nm, gas, T, L_m, d_eff_A):
    """분자체 확산 — 표준 스케일 기준"""
    dA_eff = d_eff_A
    pA = d_nm * 10.0
    delta = dA_eff - pA
    # Sieving 약화: 고온/대기압 기준 10^-6~10^-8 사이
    if delta > 0:
        return max(5e-7*np.exp(-(delta/0.5)**2)*np.exp(-E_SIEVE/(R*T)), PI_TINY)
    x = max(1.0 - (dA_eff/pA)**2, 0.0)
    f = x**2
    return max(5e-7*f*np.exp(-E_SIEVE/(R*T)), PI_TINY)

def pintr_knudsen_SI(d_nm, T, M, L_m):
    """Knudsen 확산 — 중간 기공 기준"""
    r = max(d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0)*r*np.sqrt((8.0*R*T)/(np.pi*M))
    return 0.3 * Dk/(L_m*R*T)  # 상대 약화

def pintr_surface_SI(d_nm, gas, T, L_m, dqdp):
    """Surface diffusion — dq/dp에 비례"""
    Ds = D0_SURF * np.exp(-GAS_PARAMS[gas]["Ea_s"]/(R*T))
    mult = st.session_state.get("surf_mult", 1.0)
    return max(mult * (Ds/L_m) * (dqdp * 5e5), 0.0)  # 500 → 5e5 (물리 스케일 반영)

def pintr_solution_SI(gas, T, L_m, dqdp):
    """Solution diffusion — 낮은 dq/dp에서도 완만"""
    Dsol = D0_SOL * np.exp(-1.8e4/(R*T)) / np.sqrt(GAS_PARAMS[gas]["M"]/1e-3)
    mult = st.session_state.get("sol_mult", 1.0)
    return max(mult * (Dsol/L_m) * (dqdp * 1e5), 0.0)  # 완화

def pintr_capillary_SI(d_nm, rp, L_m):
    """모세관 응축 — 큰 기공에서만"""
    if not (d_nm >= 1.5 and rp > 0.4):
        return 0.0
    r_m = max(d_nm*1e-9/2.0, 1e-12)
    base = 1e-6*np.sqrt(r_m)/L_m
    # rₚ가 높을수록 강하게
    return base * np.clip((rp - 0.4)/0.6, 0, 1)**0.5


def intrinsic_permeances(
    gas, T, Pbar, d_nm, rp, L_nm, dqdp, alpha,
    apply_gates=True, small_pore_knudsen_factor=1e-4
):
    """
    라벨/밴드 공용: 동일 규칙/감쇠를 적용한 내재 퍼미언스 dict 반환
    + BLOCK 게이트: (i) 입체 차단, (ii) 확산 불가 조건에서 다른 항을 0으로
    """
    L_m = max(float(L_nm), 1e-3) * 1e-9
    M   = GAS_PARAMS[gas]["M"]
    d_eff_A = effective_diameter_A(gas, T, alpha)

    Pi = {
        "Blocked":   PI_TINY,
        "Sieving":   pintr_sieving_SI(d_nm, gas, T, L_m, d_eff_A),
        "Knudsen":   pintr_knudsen_SI(d_nm, T, M, L_m),
        "Surface":   pintr_surface_SI(d_nm, gas, T, L_m, dqdp),
        "Capillary": pintr_capillary_SI(d_nm, rp, L_m),
        "Solution":  pintr_solution_SI(gas, T, L_m, dqdp),
    }

    if apply_gates:
        # (A) Capillary 게이트
        if not (d_nm >= 2.0 and rp > 0.5):
            Pi["Capillary"] = 0.0

        # (B) Knudsen 감쇠: 초소기공에서 추가 감쇠
        if d_nm <= 0.5:
            Pi["Knudsen"] *= small_pore_knudsen_factor

        # (C) 기존 큰-기공/rp 영역 감쇠(사용 중이면)
        Pi = damp_knudsen_if_needed(Pi, d_nm, rp)

    # ---------------- BLOCK 게이트 ----------------
    if st.session_state.get("block_gate_on", False):
        # 1) 입체 차단
        pA = d_nm * 10.0  # nm → Å
        steric_block = (pA <= (d_eff_A - SIEVE_BAND_A)) or (d_nm <= 0.08)  # 0.08 nm ~ 극소기공
        # 2) 확산 불가
        dqdp_cut = float(st.session_state.get("dqdp_cut_val", 1e-10))
        diffusion_off = (abs(dqdp) < dqdp_cut) and (rp <= 0.05)

        if steric_block or diffusion_off:
            for k in ("Sieving", "Knudsen", "Surface", "Capillary", "Solution"):
                Pi[k] = 0.0
    # ------------------------------------------------

    return Pi

def weights_from_intrinsic(Pi_intr, gamma=0.8):
    keys = MECH_ORDER
    x=np.array([np.log(max(float(Pi_intr.get(k,0.0)),1e-30)) for k in keys])
    e=np.exp(gamma*x); s=float(e.sum())
    w=e/s if s>0 and np.isfinite(s) else np.array([0,0,0,1,0,0],float)
    return {k:float(w[i]) for i,k in enumerate(keys)}

def damp_knudsen_if_needed(Pi_intr, d_nm, rp):
    # ← 수정: 상수 DAMP_FACTOR 사용
    if WEIGHT_MODE=="softmax" and DAMP_KNUDSEN and (d_nm<=0.5):
        Pi_intr["Knudsen"]*=DAMP_FACTOR
    return Pi_intr

def mechanism_band_rgba(g1,g2,T,P_bar,d_nm,relP,L_nm,q11,q12,b11,b12,alpha):
    _, dv = dsl_loading_and_slope_b(g1, T, P_bar, relP, q11, q12, b11, b12)
    names = []
    for i, rp in enumerate(relP):
        Pi_intr = intrinsic_permeances(
            gas=g1, T=T, Pbar=P_bar, d_nm=d_nm, rp=float(rp),
            L_nm=L_nm, dqdp=float(dv[i]), alpha=alpha,
            apply_gates=True, small_pore_knudsen_factor=1e-4
        )
        winner = max(Pi_intr, key=Pi_intr.get)
        names.append(winner)
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :]
    return rgba, names

def mechanism_band_rgba_time(g1,g2,T,P_bar,d_nm,L_nm,t_vec,P_bar_t,dqdp1,P0bar,alpha):
    rp_t = np.clip(P_bar_t/float(P0bar), 1e-6, 0.9999)
    names = []
    for i in range(len(t_vec)):
        Pi_intr = intrinsic_permeances(
            gas=g1, T=T, Pbar=P_bar, d_nm=d_nm, rp=float(rp_t[i]),
            L_nm=L_nm, dqdp=float(dqdp1[i]), alpha=alpha,
            apply_gates=True, small_pore_knudsen_factor=1e-4
        )
        winner = max(Pi_intr, key=Pi_intr.get)
        names.append(winner)
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :]
    return rgba, names

def _series_parallel(Pp,Pd,eps=1e-30):
    if not np.isfinite(Pp): Pp=0.0
    if not np.isfinite(Pd): Pd=0.0
    if Pp<=eps and Pd<=eps: return PI_TINY
    if Pp<=eps: return Pd
    if Pd<=eps: return Pp
    return 1.0/((1.0/(Pp+eps))+(1.0/(Pd+eps)))

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

def permeance_series_SI(d_nm, gas, other, T, P_bar, relP, L_nm, q_mmolg, dqdp, q_other, alpha):
    Pi = np.zeros_like(relP, float)
    for i, rp in enumerate(relP):
        Pi_intr = intrinsic_permeances(
            gas=gas, T=T, Pbar=P_bar, d_nm=d_nm, rp=float(rp),
            L_nm=L_nm, dqdp=float(dqdp[i]), alpha=alpha,
            apply_gates=True, small_pore_knudsen_factor=1e-4
        )
        # 밴드와 동일한 softmax 가중 + 직렬/병렬 결합
        w = weights_from_intrinsic(Pi_intr)
        Pi_pore = (w["Sieving"]*Pi_intr["Sieving"] +
                   w["Knudsen"]*Pi_intr["Knudsen"] +
                   w["Capillary"]*Pi_intr["Capillary"])
        Pi_diff = (w["Surface"]*Pi_intr["Surface"] +
                   w["Solution"]*Pi_intr["Solution"])
        Pi0 = _series_parallel(Pi_pore, Pi_diff)
        theta = (q_mmolg[i]/(q_mmolg[i]+q_other[i])) if (q_mmolg[i]+q_other[i])>0 else 0.0
        Pi[i] = Pi0 * theta
    return Pi

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

def run_gas_membrane():
    st.header("Gas Membrane Simulator")

    # =================== Sidebar ===================
    with st.sidebar:
        st.subheader("Global Conditions")
        T    = nudged_slider("Temperature", 10.0, 600.0, 1.0, 300.0, key="T_g",    unit="K")
        Pbar = nudged_slider("Total Pressure", 0.1, 10.0, 0.1, 1.0,  key="Pbar_g", unit="bar")
        d_nm = nudged_slider("Pore diameter", 0.01, 50.0, 0.01, 0.36, key="d_nm_g", unit="nm")
        L_nm = nudged_slider("Membrane thickness", 10.0, 100000.0, 1.0, 100.0, key="L_nm_g", unit="nm")

        gases = list(GAS_PARAMS.keys())
        gas1  = st.selectbox("Gas1 (numerator)",   gases, index=gases.index("CO2"), key="gas1_g")
        gas2  = st.selectbox("Gas2 (denominator)", gases, index=gases.index("CH4"), key="gas2_g")

        # ---- Quantum size (α): auto/manual + λ,d_eff 표시 ----
        st.subheader("Quantum size effect (α)")
        alpha_mode = st.radio("α mode", ["Auto (by T & pore)", "Manual"], index=0, key="alpha_mode_g")

        lam_m = de_broglie_lambda_m(T, GAS_PARAMS[gas1]["M"])  # [m]
        lam_A = lam_m * 1e10                                    # [Å]

        if alpha_mode == "Auto (by T & pore)":
            alpha_val = alpha_auto_by_temperature(T, a0=0.05, T_ref=300.0, a_min=0.0, a_max=0.60, d_nm=d_nm)
            st.session_state["alpha_g"] = float(alpha_val)
            st.caption(f"λ({gas1},{T:.1f}K) ≈ {lam_A:.2f} Å")
            d_effA = effective_diameter_A(gas1, T, st.session_state["alpha_g"])
            st.caption(f"d_eff({gas1}) ≈ {d_effA:.2f} Å  (α={st.session_state['alpha_g']:.3f}, d_vdw={GAS_PARAMS[gas1]['d']:.2f} Å)")
        else:
            alpha_in = nudged_slider("α (manual)", 0.0, 0.60, 0.005, 0.05, key="alpha_g", unit="–",
                                     help="값이 클수록 유효 지름 d_eff = d_vdw − α·λ 가 작아짐")
            st.caption(f"λ({gas1},{T:.1f}K) ≈ {lam_A:.2f} Å")
            d_effA = effective_diameter_A(gas1, T, alpha_in)
            st.caption(f"d_eff({gas1}) ≈ {d_effA:.2f} Å")

        st.subheader("DSL parameters (q1,q2,b1,b2)")
        st.caption("Units: q — mmol/g, b — Pa⁻¹")
        q11 = nudged_slider("q1 Gas1", 0.0, 100.0, 0.01, 0.70, key="q11_g", unit="mmol/g")
        q12 = nudged_slider("q2 Gas1", 0.0, 100.0, 0.01, 0.30, key="q12_g", unit="mmol/g")
        b11 = nudged_slider("b1 Gas1", 1e-10, 1e-1, 1e-8, 1e-5, key="b11_g", unit="Pa⁻¹", decimals=8)
        b12 = nudged_slider("b2 Gas1", 1e-10, 1e-1, 1e-8, 5e-6, key="b12_g", unit="Pa⁻¹", decimals=8)

        q21 = nudged_slider("q1 Gas2", 0.0, 100.0, 0.01, 0.70, key="q21_g", unit="mmol/g")
        q22 = nudged_slider("q2 Gas2", 0.0, 100.0, 0.01, 0.30, key="q22_g", unit="mmol/g")
        b21 = nudged_slider("b1 Gas2", 1e-10, 1e-1, 1e-8, 1e-5, key="b21_g", unit="Pa⁻¹", decimals=8)
        b22 = nudged_slider("b2 Gas2", 1e-10, 1e-1, 1e-8, 5e-6, key="b22_g", unit="Pa⁻¹", decimals=8)

        mode = st.radio("X-axis / Simulation mode",
                        ["Relative pressure (P/P0)", "Time (transient LDF)"],
                        index=0, key="mode_g")

        # time-mode settings
        if mode == "Time (transient LDF)":
            st.subheader("Transient (LDF) settings")
            t_end = nudged_slider("Total time", 0.1, 3600.0, 0.1, 120.0, key="t_end_g", unit="s")
            dt    = nudged_slider("Time step", 1e-3, 10.0, 1e-3, 0.1, key="dt_g", unit="s")
            kLDF  = nudged_slider("k_LDF", 1e-4, 10.0, 1e-4, 0.05, key="kLDF_g", unit="s⁻¹")
            P0bar = nudged_slider("Feed P₀", 0.1, 10.0, 0.1, Pbar, key="P0bar_g", unit="bar")
            ramp  = st.selectbox("Pressure schedule P(t)",
                                 ["Step (P=P₀)", "Exp ramp: P₀(1-exp(-t/τ))"], index=1, key="ramp_g")
            tau   = nudged_slider("τ (only for exp ramp)", 1e-3, 1000.0, 1e-3, 5.0, key="tau_g", unit="s")

        # r_p focus for labels & optional band zoom (relative-pressure mode only)
        rp_focus = None
        if mode == "Relative pressure (P/P0)":
            rp_focus = nudged_slider("rₚ focus for labels", 0.01, 0.99, 0.01, 0.35,
                                     key="rp_focus_g", unit="–",
                                     help="라벨(승자)을 평가할 rₚ. Capillary 게이트에도 이 rₚ이 사용됩니다.")
            zoom_band = st.checkbox("Zoom band around rₚ focus", value=False, key="zoom_band_g")
            zoom_half = nudged_slider("Zoom half-width Δrₚ", 0.01, 0.49, 0.01, 0.15,
                                      key="zoom_half_g", unit="–",
                                      help="밴드를 [rₚ-Δ, rₚ+Δ] 범위만 표시")

        # ---------- (optional) tuning & block gates ----------
        st.markdown("---")
        st.subheader("Mechanism tuning (advanced)")
        st.session_state["sol_mult"]  = log_slider("Solution multiplier", -2.0, 3.0, 0.1, 0.0,
                                                   key="sol_mult_g",  unit="×")
        st.session_state["surf_mult"] = log_slider("Surface multiplier",  -2.0, 3.0, 0.1, 0.0,
                                                   key="surf_mult_g", unit="×")

        st.markdown("---")
        st.subheader("Blocking gate (advanced)")
        st.checkbox("Enable strict BLOCK gate", value=True, key="block_gate_on")
        st.session_state["dqdp_cut_val"] = log_slider(
            "dq/dp cutoff for diffusion", -14.0, -6.0, 0.5, -10.0,
            key="dqdp_cut_g", unit="(mol/kg)/Pa",
            help="|dq/dp|<cutoff 이고 rₚ≤0.05면 Surface/Solution을 0으로 잘라 Blocked 우세 유도"
        )

    # =================== Compute ===================
    alpha = float(st.session_state.get("alpha_g", alpha_auto_by_temperature(T, d_nm=d_nm)))
    time_mode = (mode == "Time (transient LDF)")

    if time_mode:
        t = np.arange(0.0, st.session_state["t_end_g"] + st.session_state["dt_g"], st.session_state["dt_g"])
        P_bar_t = pressure_schedule_series(t, st.session_state["P0bar_g"],
                                           st.session_state["ramp_g"], st.session_state["tau_g"])
        relP = np.clip(P_bar_t / float(st.session_state["P0bar_g"]), 1e-6, 0.9999)

        def qeq_g1(Pbar_scalar):
            rp = float(np.clip(Pbar_scalar/float(Pbar), 1e-6, 0.9999))
            qv, dv = dsl_loading_and_slope_b(gas1, T, Pbar, np.array([rp]), q11, q12, b11, b12)
            return float(qv[0]), float(dv[0])

        def qeq_g2(Pbar_scalar):
            rp = float(np.clip(Pbar_scalar/float(Pbar), 1e-6, 0.9999))
            qv, dv = dsl_loading_and_slope_b(gas2, T, Pbar, np.array([rp]), q21, q22, b21, b22)
            return float(qv[0]), float(dv[0])

        q1_dyn, dqdp1 = ldf_evolve_q(t, P_bar_t, qeq_g1, st.session_state["kLDF_g"], 0.0)
        q2_dyn, dqdp2 = ldf_evolve_q(t, P_bar_t, qeq_g2, st.session_state["kLDF_g"], 0.0)

        Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, Pbar, relP, L_nm, q1_dyn, dqdp1, q2_dyn, alpha)
        Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, Pbar, relP, L_nm, q2_dyn, dqdp2, q1_dyn, alpha)

        X_vals, X_label = t, "Time (s)"

    else:
        # relative-pressure mode
        if st.session_state.get("zoom_band_g", False):
            c = float(st.session_state.get("rp_focus_g", 0.35))
            w = float(st.session_state.get("zoom_half_g", 0.15))
            rmin, rmax = max(0.01, c - w), min(0.99, c + w)
            relP = np.linspace(rmin, rmax, 400)
        else:
            relP = np.linspace(0.01, 0.99, 500)

        q1_mg, dqdp1 = dsl_loading_and_slope_b(gas1, T, Pbar, relP, q11, q12, b11, b12)
        q2_mg, dqdp2 = dsl_loading_and_slope_b(gas2, T, Pbar, relP, q21, q22, b21, b22)
        Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, Pbar, relP, L_nm, q1_mg, dqdp1, q2_mg, alpha)
        Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, Pbar, relP, L_nm, q2_mg, dqdp2, q1_mg, alpha)

        X_vals, X_label = relP, r"Relative pressure, $P/P_0$ (–)"

    Sel     = np.divide(Pi1, Pi2, out=np.zeros_like(Pi1), where=(Pi2 > 0))
    Pi1_gpu = Pi1 / GPU_UNIT
    Pi2_gpu = Pi2 / GPU_UNIT

    # =================== Plots ===================
    colA, colB = st.columns([1, 2])

    with colB:
        # Mechanism band
        figBand, axBand = plt.subplots(figsize=(9, 0.7))
        if time_mode:
            rgba, _ = mechanism_band_rgba_time(gas1, gas2, T, Pbar, d_nm, L_nm, t, P_bar_t, dqdp1,
                                               st.session_state["P0bar_g"], alpha)
            x_min, x_max = float(t[0]), float(t[-1])
            x_ticks = np.linspace(x_min, x_max, 6)
        else:
            rgba, _ = mechanism_band_rgba(gas1, gas2, T, Pbar, d_nm, relP, L_nm, q11, q12, b11, b12, alpha)
            x_min, x_max = X_vals[0], X_vals[-1]
            x_ticks = np.linspace(x_min, x_max, 6)
        axBand.imshow(rgba, extent=(x_min, x_max, 0, 1), aspect="auto", origin="lower")
        axBand.set_xlim(x_min, x_max); axBand.set_xticks(x_ticks); axBand.set_yticks([])
        axBand.set_xlabel(X_label)
        handles = [plt.Rectangle((0,0),1,1,fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
        leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.7),
                            ncol=6, frameon=True)
        leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white")
        st.pyplot(figBand, use_container_width=True); plt.close(figBand)

        # Permeance (GPU)
        fig1, ax1 = plt.subplots(figsize=(9, 3))
        ax1.plot(X_vals, Pi1_gpu, label=f"{gas1}")
        ax1.plot(X_vals, Pi2_gpu, "--", label=f"{gas2}")
        ax1.set_xlabel(X_label); ax1.set_ylabel(r"$\Pi$ (GPU)")
        ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax1.get_yaxis().get_offset_text().set_visible(False)
        ax1.grid(True); ax1.legend(title="Permeance (GPU)")
        st.pyplot(fig1, use_container_width=True); plt.close(fig1)

        # Selectivity
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        ax2.plot(X_vals, Sel, label=f"{gas1}/{gas2}")
        ax2.set_xlabel(X_label); ax2.set_ylabel("Selectivity (–)")
        ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax2.get_yaxis().get_offset_text().set_visible(False)
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    with colA:
        st.subheader("Mechanism (synced with band)")
        if time_mode:
            rp_mid = float(np.clip(P_bar_t/float(st.session_state["P0bar_g"]), 1e-6, 0.9999)[len(t)//2])
            dq_mid = float(dqdp1[len(dqdp1)//2]) if len(dqdp1)>0 else 0.0
        else:
            rp_mid = float(st.session_state.get("rp_focus_g", np.median(X_vals)))
            idx = int(np.argmin(np.abs(X_vals - rp_mid)))
            dq_mid = float(dqdp1[idx]) if len(dqdp1)>0 else 0.0

        cand = intrinsic_permeances(
            gas=gas1, T=T, Pbar=Pbar, d_nm=d_nm, rp=rp_mid,
            L_nm=L_nm, dqdp=dq_mid, alpha=alpha,
            apply_gates=True, small_pore_knudsen_factor=1e-4
        )
        winner = max(cand, key=cand.get)
        st.markdown(f"**Mechanism (synced):** `{winner}`  |  **Best intrinsic:** `{winner}`")
        st.caption("라벨/밴드는 동일한 내재식, 동일 게이트/감쇠로 계산됩니다.")

# =====================================================================
# MODE 2 — ION MEMBRANE (steady + crack 열폭주 transient)
# =====================================================================
ION_DB = {
    "Na+":       {"z": +1, "D": 1.33e-9},
    "K+":        {"z": +1, "D": 1.96e-9},
    "Li+":       {"z": +1, "D": 1.03e-9},
    "Ca2+":      {"z": +2, "D": 0.79e-9},
    "Mg2+":      {"z": +2, "D": 0.706e-9},
    "Cl-":       {"z": -1, "D": 2.03e-9},
    "NO3-":      {"z": -1, "D": 1.90e-9},
    "SO4^2-":    {"z": -2, "D": 1.065e-9},
    "HCO3-":     {"z": -1, "D": 1.18e-9},
    "Acetate-":  {"z": -1, "D": 1.09e-9},
}
T_REF_ION = 298.15  # ION_DB 기준온도 (25°C)

def donnan_potential_general(c_bulk, z_map, K_map, Cf, T):
    psi = 0.0
    for _ in range(80):
        t = -F*psi/(R*T)
        s=0.0; ds=0.0
        for sp, cb in c_bulk.items():
            z = z_map[sp]; K = K_map.get(sp, 1.0)
            val = K*cb*np.exp(-z*t)
            s  += z * val
            ds += (F/(R*T))*(z**2) * val
        g = s + Cf
        step = -g/(ds + 1e-30)
        psi += step
        if abs(step) < 1e-12: break
    return psi

def ghk_flux(Pi, z, Cin, Cout, dV, T):
    a = -z*F*dV/(R*T)
    if abs(a) < 1e-6:
        return Pi*((z*z)*(F**2)*dV/(R*T))*((Cin - Cout) - 0.5*a*(Cin + Cout))
    num = Cin - Cout*np.exp(a)
    den = 1.0 - np.exp(a)
    return Pi*((z*z)*(F**2)*dV/(R*T))*(num/(den+1e-30))

def run_ion_membrane():
    st.header("Ion-Exchange / NF Membrane (Multi-ion)")

    with st.sidebar:
        st.subheader("Membrane & Operation")
        T = nudged_slider("Temperature", 273.15, 350.0, 1.0, 298.15, key="T_i", unit="K")
        Lnm = nudged_slider("Active layer thickness", 10.0, 5000.0, 1.0, 200.0, key="Lnm_i", unit="nm")
        eps = nudged_slider("Porosity ε", 0.05, 0.8, 0.01, 0.30, key="eps_i", unit="–")
        tau = nudged_slider("Tortuosity τ", 1.0, 5.0, 0.1, 2.0, key="tau_i", unit="–")
        Cf  = nudged_slider("Fixed charge C_f", -3000.0, 3000.0, 10.0, -500.0, key="Cf_i")
        dV  = nudged_slider("Membrane potential dV", -0.2, 0.2, 0.005, 0.0, key="dV_i")
        v_s = nudged_slider("Solvent velocity v", -1e-6, 1e-6, 1e-7, 0.0, key="vsolv_i",
                            help="Convective term v·C_avg added to each ion flux")

        st.subheader("Select ions (≤5 cations, ≤5 anions)")
        all_cations = [k for k in ION_DB if ION_DB[k]["z"]>0]
        all_anions  = [k for k in ION_DB if ION_DB[k]["z"]<0]
        sel_cat = st.multiselect("Cations", all_cations, default=["Na+","K+","Ca2+","Li+","Mg2+"], key="sel_cat_i")
        sel_an  = st.multiselect("Anions",  all_anions,  default=["Cl-","NO3-","SO4^2-","HCO3-","Acetate-"], key="sel_an_i")
        if len(sel_cat)>5: st.warning("Cations >5 → 앞 5개만 사용합니다."); sel_cat = sel_cat[:5]
        if len(sel_an)>5:  st.warning("Anions >5 → 앞 5개만 사용합니다.");  sel_an  = sel_an[:5]

        st.subheader("Bulk concentrations (mol/m³)")
        st.caption("종마다 feed/permeate 농도를 슬라이더로 조정 (0~1000 mol/m³)")
        c_feed = {}; c_perm = {}
        for sp in sel_cat+sel_an:
            c_feed[sp] = nudged_slider(f"{sp} feed", 0.0, 1000.0, 1.0, 100.0, key=f"cf_{sp}", unit="mol/m³")
            c_perm[sp] = nudged_slider(f"{sp} permeate", 0.0, 1000.0, 1.0, 10.0,  key=f"cp_{sp}", unit="mol/m³")

        st.subheader("K & D 설정")
        auto_kd = st.checkbox("Auto-compute K & D (권장)", value=True, key="autoKD_i")

        if auto_kd:
            K_map = {sp: 1.0 for sp in (sel_cat + sel_an)}
            T_ref = T_REF_ION
            eta_ref = eta_water_PaS(T_ref)
            eta_now = eta_water_PaS(T)
            D_map = {sp: D_temp_correction(ION_DB[sp]["D"], T_ref, T, eta_ref, eta_now)
                     for sp in (sel_cat + sel_an)}
            with st.expander("Auto values (read-only)"):
                st.caption("D는 Stokes–Einstein으로 T/η 보정. K는 화학적 분배=1.0 (전하 분배는 Donnan에서 처리).")
                for sp in (sel_cat + sel_an):
                    st.write(f"{sp}:  K={K_map[sp]:.2f},  D(T)={D_map[sp]:.3e} m²/s  (from {ION_DB[sp]['D']:.3e} @25°C)")
        else:
            st.caption("수동 모드: DB 기본값에서 벗어나 실험 피팅/감도분석용")
            K_map = {sp: nudged_slider(f"K {sp}", 0.01, 10.0, 0.01, 1.0, key=f"K_{sp}") for sp in (sel_cat+sel_an)}
            D_map = {sp: log_slider(f"D {sp}", -11.0, -8.0, 0.1, np.log10(ION_DB[sp]['D']), key=f"D_{sp}", unit="m²/s")
                     for sp in (sel_cat+sel_an)}

    # ---- Steady 계산 ----
    L = Lnm*1e-9
    P_map_base = {sp: (D_map[sp]*eps/max(tau,1e-9))/max(L,1e-12) for sp in (sel_cat+sel_an)}
    z_map = {sp: int(ION_DB[sp]["z"]) for sp in (sel_cat+sel_an)}

    psi_f = donnan_potential_general(c_feed, z_map, K_map, Cf, T)
    psi_p = donnan_potential_general(c_perm, z_map, K_map, Cf, T)

    Cm_f = {}; Cm_p = {}
    for sp in (sel_cat+sel_an):
        z = z_map[sp]
        Cm_f[sp] = K_map[sp]*c_feed[sp]*np.exp(-(z*F*psi_f)/(R*T))
        Cm_p[sp] = K_map[sp]*c_perm[sp]*np.exp(-(z*F*psi_p)/(R*T))

    def compute_flux_and_current(P_map_local, T_local):
        J = {}
        for sp in (sel_cat+sel_an):
            z = z_map[sp]
            J_ghk = ghk_flux(P_map_local[sp], z, Cm_f[sp], Cm_p[sp], dV, T_local)
            J_conv = v_s*0.5*(Cm_f[sp]+Cm_p[sp])
            J[sp] = J_ghk + J_conv
        i_net_local = F*sum(z_map[sp]*J[sp] for sp in (sel_cat+sel_an))  # A/m²
        return J, i_net_local

    J0, i_net0 = compute_flux_and_current(P_map_base, T)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Global (steady)")
        st.metric("Net current density", f"{i_net0:.3e} A m⁻²")
        st.caption("전류가 0에 가까울수록 전하수송 균형. dV, C_f, K_i, D_i로 조절해 보세요.")
        st.subheader("Membrane-side concentrations")
        for sp in (sel_cat+sel_an):
            st.write(f"{sp}: feed-side {Cm_f[sp]:.1f}, perm-side {Cm_p[sp]:.1f}  (mol/m³)")
    with col2:
        species = sel_cat + sel_an
        vals = [J0[s] for s in species]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(range(len(species)), vals)
        ax.set_xticks(range(len(species))); ax.set_xticklabels(species)
        ax.set_ylabel("Flux J (mol m⁻² s⁻¹)")
        ax.grid(True, axis='y')
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ---- NEW: Crack 유도 열폭주(Transient) ----
    st.markdown("---")
    st.subheader("Crack-induced thermal runaway (Transient) — Optional")
    enable_runaway = st.checkbox("Enable crack/thermal runaway simulation", value=False, key="runaway_on")

    if enable_runaway:
        colL, colR = st.columns(2)
        with colL:
            t_end = nudged_slider("Sim time", 0.05, 200.0, 0.05, 20.0, key="tr_tend", unit="s")
            dt    = nudged_slider("Δt", 1e-3, 0.5, 1e-3, 0.02, key="tr_dt", unit="s")
            T_amb = nudged_slider("Ambient T", 273.15, 330.0, 0.5, 298.15, key="tr_Tamb", unit="K")
            C_th  = log_slider("Thermal capacity C_th", -1.0, 4.0, 0.1, 1.0, key="tr_Cth", unit="J m⁻² K⁻¹")
            tau_c = log_slider("Cooling time τ_cool", -1.0, 4.0, 0.1, 1.3, key="tr_tau", unit="s")
        with colR:
            a_crk = nudged_slider("Crack area fraction a_crack", 0.0, 0.5, 0.01, 0.05, key="tr_acrack")
            G_crk = log_slider("Crack conductance gain G_crack", 0.0, 3.0, 0.05, 2.0, key="tr_Gcrack", unit="×")
            f0    = nudged_slider("Initial crack growth f0", 0.0, 1.0, 0.01, 0.10, key="tr_f0")
            Tcrit = nudged_slider("Runaway threshold T_crit", 290.0, 360.0, 0.5, 320.0, key="tr_Tcrit", unit="K")
            k_g   = log_slider("Crack growth rate k_g", -4.0, 1.0, 0.1, -1.0, key="tr_kg", unit="s⁻¹")

        nstep = int(np.ceil(t_end/max(dt,1e-9)))+1
        tvec  = np.linspace(0.0, t_end, nstep)
        Tv    = np.zeros_like(tvec); Tv[0] = T
        iv    = np.zeros_like(tvec)
        fv    = np.zeros_like(tvec); fv[0] = f0

        eta_ref = eta_water_PaS(T_REF_ION)

        L = Lnm*1e-9
        for k in range(1, nstep):
            T_old = Tv[k-1]
            f_old = fv[k-1]

            eta_now = eta_water_PaS(T_old)
            D_T = {sp: D_temp_correction(ION_DB[sp]["D"], T_REF_ION, T_old, eta_ref, eta_now)
                   for sp in (sel_cat+sel_an)}
            P_crack = {sp: (D_T[sp]*eps/max(tau,1e-9))/max(L,1e-12) * (1.0 + G_crk * f_old) for sp in (sel_cat+sel_an)}
            P_bulk  = {sp: (D_T[sp]*eps/max(tau,1e-9))/max(L,1e-12) for sp in (sel_cat+sel_an)}
            P_eff   = {sp: (1.0 - a_crk)*P_bulk[sp] + a_crk*P_crack[sp] for sp in (sel_cat+sel_an)}

            _, i_now = compute_flux_and_current(P_eff, T_old)
            iv[k] = i_now

            # 전력(발열/흡열 방향성 유지; 필요시 clip으로 양의 값만 반영 가능)
            Pj = i_now * dV
            dTdt = Pj/max(C_th,1e-9) - (T_old - T_amb)/max(tau_c, 1e-6)

            kg = k_g  # log_slider는 실값 반환
            drive = max(T_old - Tcrit, 0.0)
            dfdt = kg * drive * (1.0 - f_old)
            fv[k] = float(np.clip(f_old + dt*dfdt, 0.0, 1.0))
            Tv[k] = T_old + dt*dTdt

        c1, c2 = st.columns(2)
        with c1:
            figT, axT = plt.subplots(figsize=(6,3))
            axT.plot(tvec, Tv, label="Temperature")
            axT.axhline(Tcrit, linestyle="--", label="T_crit")
            axT.set_xlabel("t (s)"); axT.set_ylabel("T (K)")
            axT.grid(True); axT.legend()
            st.pyplot(figT, use_container_width=True); plt.close(figT)

        with c2:
            figI, axI = plt.subplots(figsize=(6,3))
            axI.plot(tvec, iv, label="i (A/m²)")
            axI.set_xlabel("t (s)"); axI.set_ylabel("Current density")
            axI.grid(True); axI.legend()
            st.pyplot(figI, use_container_width=True); plt.close(figI)

        figF, axF = plt.subplots(figsize=(12,2.4))
        axF.plot(tvec, fv, label="f_crack")
        axF.set_xlabel("t (s)"); axF.set_ylabel("f")
        axF.grid(True); axF.legend(loc="upper left")
        st.pyplot(figF, use_container_width=True); plt.close(figF)

        st.info(f"Final T = {Tv[-1]:.2f} K,  Final i = {iv[-1]:.3e} A/m²,  Final f_crack = {fv[-1]:.3f}")

# =====================================================================
# MODE 3 — DRUG IN VESSEL (1D ADR, transient)
# =====================================================================
DRUG_DB = {
    "Caffeine":        {"Db": 6.5e-10, "Pv": 2.0e-6, "kel": 1.0e-4},
    "Acetaminophen":   {"Db": 6.0e-10, "Pv": 1.5e-6, "kel": 8.0e-5},
    "Ibuprofen":       {"Db": 5.0e-10, "Pv": 8.0e-7, "kel": 1.0e-4},
    "Dopamine":        {"Db": 6.0e-10, "Pv": 1.2e-6, "kel": 1.2e-4},
    "Fluorescein":     {"Db": 4.5e-10, "Pv": 6.0e-7, "kel": 6.0e-5},
    "Glucose":         {"Db": 6.7e-10, "Pv": 1.0e-6, "kel": 7.0e-5},
    "Vancomycin":      {"Db": 2.5e-10, "Pv": 1.0e-7, "kel": 5.0e-5},
    "Doxorubicin":     {"Db": 3.0e-10, "Pv": 2.0e-7, "kel": 8.0e-5},
    "Insulin":         {"Db": 1.8e-10, "Pv": 5.0e-8, "kel": 3.0e-5},
    "Albumin":         {"Db": 0.7e-10, "Pv": 1.0e-8, "kel": 2.0e-5},
}

def run_vascular_drug():
    st.header("Drug Transport in a Vessel (1D ADR, transient)")

    with st.sidebar:
        st.subheader("Geometry & Flow")
        Rv_um = nudged_slider("Vessel radius", 2.0, 100.0, 1.0, 4.0, key="Rv_um_v", unit="μm")
        L_mm  = nudged_slider("Segment length", 1.0, 200.0, 1.0, 20.0, key="Lv_mm_v", unit="mm")
        U     = nudged_slider("Mean velocity U", 0.1e-3, 20e-3, 0.1e-3, 1.0e-3, key="U_v", unit="m/s",
                              help="~1 mm/s in capillaries")

        st.subheader("Drug & Mass Transfer")
        drug = st.selectbox("Drug (defaults loaded)", list(DRUG_DB.keys()), index=0, key="drug_v")

        st.subheader("Thermophysical (for Db auto)")
        T_body = nudged_slider("Temperature", 293.15, 312.15, 0.5, 310.15, key="Tdrug_v", unit="K",
                               help="Db 자동 보정에 사용 (기본 310K ≈ 37°C)")
        medium = st.selectbox("Solvent", ["water-like","plasma-like"], index=1, key="med_v",
                              help="점도 모델 선택; 근사값")

        auto_Db = st.checkbox("Auto-compute D_b from T & viscosity", value=True, key="autoDb_v")
        if auto_Db:
            T_ref = 298.15
            eta_ref = eta_water_PaS(T_ref)
            if medium == "water-like":
                eta_now = eta_water_PaS(T_body)
            else:
                eta_now = 1.7 * eta_water_PaS(T_body)  # plasma ~1.7×water(37°C) 근사
            Db = D_temp_correction(DRUG_DB[drug]["Db"], T_ref, T_body, eta_ref, eta_now)
            st.caption(f"D_b auto: {Db:.3e} m²/s  (from {DRUG_DB[drug]['Db']:.3e} @25°C, medium={medium})")
        else:
            Db = log_slider("D_b",  -12.0, -8.0, 0.1, np.log10(DRUG_DB[drug]["Db"]), key="Db_v",  unit="m²/s")

        Pv   = log_slider("P_v",   -9.0,  -5.0, 0.1, np.log10(DRUG_DB[drug]["Pv"]), key="Pv_v",  unit="m/s")
        kel  = log_slider("k_elim",-6.0,  -2.0, 0.1, np.log10(DRUG_DB[drug]["kel"]), key="kel_v", unit="s⁻¹")

        st.subheader("Inlet profile")
        C0   = nudged_slider("Reference conc. C₀", 0.0, 5.0, 0.01, 1.0, key="C0_v", unit="mol/m³")
        profile = st.selectbox("Profile", ["Bolus (Gaussian)","Constant infusion"], index=0, key="pulse_v")
        t_end = nudged_slider("Sim time", 0.1, 600.0, 0.1, 60.0, key="tend_v", unit="s")
        dt    = nudged_slider("Δt", 1e-3, 0.5, 1e-3, 0.01, key="dt_v", unit="s")
        Nx    = nudged_int("Grid Nx", 50, 600, 10, 200, key="Nx_v")

    Rv = Rv_um*1e-6
    L  = L_mm*1e-3
    x  = np.linspace(0, L, Nx)
    if len(x) < 2:
        st.error("Nx too small. Increase Grid Nx."); st.stop()
    dx = x[1]-x[0]
    t  = np.arange(0.0, t_end+dt, dt)
    if len(t) < 2:
        st.error("Simulation time too short. Increase t_end or reduce Δt."); st.stop()

    Pe_r = U*Rv/Db
    Deff = Db*(1.0 + (Pe_r**2)/192.0)
    k_leak = 2.0*Pv/max(Rv,1e-12)

    if profile == "Bolus (Gaussian)":
        t0 = 0.2*t_end
        sig = 0.05*t_end if t_end>0 else 1.0
        Cin_t = C0*np.exp(-((t-t0)**2)/(2*sig**2))
    else:
        Cin_t = C0*np.ones_like(t)

    C = np.zeros((len(t), Nx), dtype=float)
    C[0,:] = 0.0

    lam_a = U*dt/dx
    lam_d = Deff*dt/(dx*dx)
    if lam_a>1.0: st.warning(f"Advection CFL>1 (UΔt/Δx={lam_a:.2f}). Reduce Δt or increase Nx.")
    if lam_d>0.5: st.warning(f"Diffusion number>0.5 (D_effΔt/Δx²={lam_d:.2f}). Use smaller Δt.")

    for n in range(1, len(t)):
        Cn = C[n-1,:].copy()
        Cnp = Cn.copy()
        Cnp[0] = Cin_t[n]
        adv = -lam_a*(Cn[1:] - Cn[:-1])
        dif = lam_d*(np.roll(Cn,-1)[1:-1] - 2*Cn[1:-1] + Cn[0:-2])
        react = -dt*(k_leak + kel)*Cn[1:-1]
        Cnp[1:-1] = Cn[1:-1] + adv[:-1] + dif + react
        Cnp[-1] = Cnp[-2]
        C[n,:] = np.maximum(Cnp, 0.0)

    Cout_t = C[:, -1]

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Key numbers")
        st.metric("Pe_r = U·R/D_b", f"{Pe_r:.2f}")
        st.metric("D_eff", f"{Deff:.3e} m²/s")
        st.metric("k_leak = 2P_v/R", f"{k_leak:.3e} s⁻¹")

    with col2:
        fig, ax = plt.subplots(figsize=(8,3))
        vmin, vmax = float(np.min(C)), float(np.max(C))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax) or abs(vmax-vmin) < 1e-15: vmax = vmin + 1e-12
        im = ax.imshow(C.T, aspect='auto', origin='lower',
                       extent=(t[0], t[-1], x[0]*1e3, x[-1]*1e3),
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("t (s)"); ax.set_ylabel("x (mm)")
        cb = plt.colorbar(im, ax=ax); cb.set_label("C (mol/m³)")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(t, Cout_t, label="Outlet")
        ax2.set_xlabel("t (s)"); ax2.set_ylabel("C_out (mol/m³)")
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

# =====================================================================
# App shell — Mode selection
# =====================================================================
st.title("Unified Transport Simulators")
mode_main = st.sidebar.radio("Select simulation",
                             ["Gas membrane", "Ion membrane", "Drug in vessel"],
                             index=0)

if mode_main == "Gas membrane":
    run_gas_membrane()
elif mode_main == "Ion membrane":
    run_ion_membrane()
else:
    run_vascular_drug()
