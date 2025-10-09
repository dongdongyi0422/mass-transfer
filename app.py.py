# app.py — Unified Transport Simulators (Gas / Ion / Vascular)
# - 모든 슬라이더에 대응하는 입력칸(number_input) 제공
# - Ion membrane: Bulk conc sliders + K&D 자동(T/η) + 수동 오버라이드 옵션
# - Drug in vessel: Db 자동 보정(water/plasma) 옵션 + Pv, k_elim 로그 슬라이더
# - NEW: α(드브로이) 슬라이더 아래 숫자 입력칸 + λ,d_eff 표시
# - NEW: Ion membrane — crack 유도 열폭주(Transient) 시각화

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# -------------------- page config (권장: 최상단) --------------------
st.set_page_config(page_title="Unified Transport Simulators (Gas / Ion / Vascular)", layout="wide")

# -------------------- constants --------------------
R  = 8.314462618            # J/mol/K
kB = 1.380649e-23           # J/K
h  = 6.62607015e-34         # J·s
NA = 6.02214076e23          # 1/mol
F  = 96485.33212            # C/mol

# =====================================================================
# Shared UI helpers  — 슬라이더 + 입력칸 동기화(플래그 방식, 안정)
# =====================================================================
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3, help=None):
    """
    슬라이더 + 숫자 입력칸(아래). on_change 플래그로 '어느 쪽이 수정됐는지' 안정 판정.
    """
    if key not in st.session_state:
        st.session_state[key] = float(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    fmt = f"%.{int(decimals)}f"

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    sld = st.slider(lab, float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
                    help=help, key=f"{key}__s", format=fmt, on_change=_mark_s)
    num = st.number_input("", float(vmin), float(vmax), float(st.session_state[key]), float(vstep),
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

    sld = st.slider(label, int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
                    help=help, key=f"{key}__s", on_change=_mark_s)
    num = st.number_input("", int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
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
    로그 스케일 슬라이더(10^x). 슬라이더 아래 exp 입력칸 제공.
    """
    if key not in st.session_state:
        st.session_state[key] = float(exp_init)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    lab = f"{label}{(' ['+unit+']') if unit else ''}"

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    exp_s = st.slider(lab, float(exp_min), float(exp_max), float(st.session_state[key]),
                      float(exp_step), help=help, key=f"{key}__s", format="%.2f", on_change=_mark_s)
    exp_n = st.number_input("exp (10^x)", float(exp_min), float(exp_max), float(st.session_state[key]),
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

# porous transport scales (고정 파라미터)
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

# ... (중략: 가스 모드의 pintr_* / DSL / weights / band / time-evolve 함수는 기존 그대로 유지)
# 아래 run_gas_membrane()에서 α UI 부분만 확장됨

def run_gas_membrane():
    st.header("Gas Membrane Simulator")

    with st.sidebar:
        st.subheader("Global Conditions")
        T    = nudged_slider("Temperature",10.0,600.0,1.0,300.0,key="T_g",unit="K")
        Pbar = nudged_slider("Total Pressure",0.1,10.0,0.1,1.0,key="Pbar_g",unit="bar")
        d_nm = nudged_slider("Pore diameter",0.01,50.0,0.01,0.36,key="d_nm_g",unit="nm")
        L_nm = nudged_slider("Membrane thickness",10.0,100000.0,1.0,100.0,key="L_nm_g",unit="nm")

        gases=list(GAS_PARAMS.keys())
        gas1=st.selectbox("Gas1 (numerator)",gases,index=gases.index("CO2"), key="gas1_g")
        gas2=st.selectbox("Gas2 (denominator)",gases,index=gases.index("CH4"), key="gas2_g")

        st.subheader("DSL parameters (q1,q2,b1,b2)")
        st.caption("Units: q — mmol/g, b — Pa⁻¹")
        q11=nudged_slider("q1 Gas1",0.0,100.0,0.01,0.70,key="q11_g",unit="mmol/g")
        q12=nudged_slider("q2 Gas1",0.0,100.0,0.01,0.30,key="q12_g",unit="mmol/g")
        b11=nudged_slider("b1 Gas1",1e-10,1e-1,1e-8,1e-5,key="b11_g",unit="Pa⁻¹",decimals=8)
        b12=nudged_slider("b2 Gas1",1e-10,1e-1,1e-8,5e-6,key="b12_g",unit="Pa⁻¹",decimals=8)

        q21=nudged_slider("q1 Gas2",0.0,100.0,0.01,0.70,key="q21_g",unit="mmol/g")
        q22=nudged_slider("q2 Gas2",0.0,100.0,0.01,0.30,key="q22_g",unit="mmol/g")
        b21=nudged_slider("b1 Gas2",1e-10,1e-1,1e-8,1e-5,key="b21_g",unit="Pa⁻¹",decimals=8)
        b22=nudged_slider("b2 Gas2",1e-10,1e-1,1e-8,5e-6,key="b22_g",unit="Pa⁻¹",decimals=8)

        mode = st.radio("X-axis / Simulation mode",
                        ["Relative pressure (P/P0)","Time (transient LDF)"],
                        index=0, key="mode_g")

        if mode=="Time (transient LDF)":
            st.subheader("Transient (LDF) settings")
            t_end=nudged_slider("Total time",0.1,3600.0,0.1,120.0,key="t_end_g",unit="s")
            dt   =nudged_slider("Time step",1e-3,10.0,1e-3,0.1,key="dt_g",unit="s")
            kLDF =nudged_slider("k_LDF",1e-4,10.0,1e-4,0.05,key="kLDF_g",unit="s⁻¹")
            P0bar=nudged_slider("Feed P₀",0.1,10.0,0.1,Pbar,key="P0bar_g",unit="bar")
            ramp =st.selectbox("Pressure schedule P(t)",
                               ["Step (P=P₀)","Exp ramp: P₀(1-exp(-t/τ))"],index=1, key="ramp_g")
            tau  =nudged_slider("τ (only for exp ramp)",1e-3,1000.0,1e-3,5.0,key="tau_g",unit="s")

        st.markdown("---")
        st.subheader("Quantum size correction α (De Broglie)")

        # NEW — α 자동/수동 + 슬라이더 아래 숫자 입력칸 + λ, d_eff 표시
        auto_alpha = st.checkbox("Auto-set α from temperature", value=True,
                                 help="α(T)=a₀·√(T₀/T), 좁은 기공에서 약간 강화", key="autoalpha_g")
        a0 = nudged_slider("Auto α scale (a₀)", 0.00, 0.40, 0.01, 0.05, key="a0_g")

        if "alpha_g" not in st.session_state:
            st.session_state["alpha_g"] = 0.05

        if auto_alpha:
            alpha_calc = alpha_auto_by_temperature(T, a0=a0, T_ref=300.0, a_min=0.0, a_max=0.60, d_nm=d_nm)
            st.session_state["alpha_g"] = alpha_calc
            # 슬라이더 아래 읽기전용 숫자 입력칸
            st.number_input("α (auto)", value=float(alpha_calc), format="%.4f", disabled=True, key="alpha_auto_display_g")
        else:
            # 수동: 슬라이더 + 숫자입력칸 동기화형
            st.session_state["alpha_g"] = nudged_slider("Manual α", 0.0, 0.60, 0.01, float(st.session_state["alpha_g"]),
                                                        key="alpha_manual_g")

        # 드브로이 λ와 d_eff 즉시 표시
        lamA = de_broglie_lambda_m(T, GAS_PARAMS[gas1]["M"]) * 1e10  # Å
        d_effA = effective_diameter_A(gas1, T, float(st.session_state["alpha_g"]))
        col_l, col_r = st.columns(2)
        with col_l:
            st.number_input("De Broglie λ (Gas1, Å)", value=float(lamA), format="%.3f", disabled=True, key="lam_display")
        with col_r:
            st.number_input("Effective diameter d_eff (Å)", value=float(d_effA), format="%.3f", disabled=True, key="deff_display")

    alpha = float(st.session_state["alpha_g"])

    # --------- compute/plot (기존 로직 그대로) ----------
    # ... (가스 모드 계산/플롯 부분은 기존 코드 그대로 사용)
    # NOTE: 너의 원본 run_gas_membrane() 내부 계산/플롯을 이 위치에 그대로 유지하세요.
    # (너의 파일에 이미 전체 구현이 있으므로, α UI 파트만 위처럼 바꾸면 됩니다.)
    # ------------------ 원본 가스모드 본문 붙여넣기 ------------------

    # ====== ↓↓↓ 여기에 너의 기존 가스모드 'compute/plot' 블록을 그대로 넣어주세요 ↓↓↓
    # (Sel, Pi1_gpu, Pi2_gpu, band plot 등등… 기존 내용과 동일)
    # ====== ↑↑↑ 기존 블록 유지 ↑↑↑


# =====================================================================
# MODE 2 — ION MEMBRANE (multi-ion, steady + NEW: crack 열폭주 transient)
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
        # 소전위 근사: lim a->0 (Cin - Cout*e^a)/(1 - e^a) ≈ Cin - Cout*(1+a) / (a - a^2/2) ≈ (Cin-Cout) - 0.5*a*(Cin+Cout)
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

        # Bulk concentrations
        st.subheader("Bulk concentrations (mol/m³)")
        st.caption("종마다 feed/permeate 농도를 슬라이더로 조정 (0~1000 mol/m³)")
        c_feed = {}; c_perm = {}
        for sp in sel_cat+sel_an:
            c_feed[sp] = nudged_slider(f"{sp} feed", 0.0, 1000.0, 1.0, 100.0, key=f"cf_{sp}", unit="mol/m³")
            c_perm[sp] = nudged_slider(f"{sp} permeate", 0.0, 1000.0, 1.0, 10.0,  key=f"cp_{sp}", unit="mol/m³")

        # K & D 설정 (Auto 권장)
        st.subheader("K & D 설정")
        auto_kd = st.checkbox("Auto-compute K & D (권장)", value=True, key="autoKD_i")

        if auto_kd:
            K_map = {sp: 1.0 for sp in (sel_cat + sel_an)}  # 화학적 분배
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

    # ----------------- Steady 계산 (기존 로직) -----------------
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

    # ---------------- NEW: Crack 유도 열폭주(Transient) ----------------
    st.markdown("---")
    st.subheader("Crack-induced thermal runaway (Transient) — Optional")
    enable_runaway = st.checkbox("Enable crack/thermal runaway simulation", value=False, key="runaway_on")

    if enable_runaway:
        colL, colR = st.columns(2)
        with colL:
            t_end = nudged_slider("Sim time", 0.05, 200.0, 0.05, 20.0, key="tr_tend", unit="s")
            dt    = nudged_slider("Δt", 1e-3, 0.5, 1e-3, 0.02, key="tr_dt", unit="s")
            T_amb = nudged_slider("Ambient T", 273.15, 330.0, 0.5, 298.15, key="tr_Tamb", unit="K")
            C_th  = log_slider("Thermal capacity C_th", -1.0, 4.0, 0.1, 1.0, key="tr_Cth", unit="J m⁻² K⁻¹",
                               help="막 단면 기준 열용량(면적당)")
            tau_c = log_slider("Cooling time τ_cool", -1.0, 4.0, 0.1, 1.3, key="tr_tau", unit="s",
                               help="대류/전도 냉각 등 효과를 등가 시정수로")
        with colR:
            a_crk = nudged_slider("Crack area fraction a_crack", 0.0, 0.5, 0.01, 0.05, key="tr_acrack")
            G_crk = log_slider("Crack conductance gain G_crack", 0.0, 3.0, 0.05, 2.0, key="tr_Gcrack",
                               unit="×", help="crack 쪽 투과/전도 증폭 계수(10^x)")
            f0    = nudged_slider("Initial crack growth f0", 0.0, 1.0, 0.01, 0.10, key="tr_f0")
            Tcrit = nudged_slider("Runaway threshold T_crit", 290.0, 360.0, 0.5, 320.0, key="tr_Tcrit", unit="K")
            k_g   = log_slider("Crack growth rate k_g", -4.0, 1.0, 0.1, -1.0, key="tr_kg",
                               unit="s⁻¹", help="df/dt = k_g·max(T-Tcrit,0)·(1-f)")

        # 시뮬레이션 버퍼
        nstep = int(np.ceil(t_end/max(dt,1e-9)))+1
        tvec  = np.linspace(0.0, t_end, nstep)
        Tv    = np.zeros_like(tvec); Tv[0] = T
        iv    = np.zeros_like(tvec)
        fv    = np.zeros_like(tvec); fv[0] = f0

        # 기준 점도/참조
        eta_ref = eta_water_PaS(T_REF_ION)

        # 루프
        P_base = P_map_base.copy()
        for k in range(1, nstep):
            T_old = Tv[k-1]
            f_old = fv[k-1]

            # 점도/확산 업데이트 (T 의존)
            eta_now = eta_water_PaS(T_old)
            D_T = {sp: D_temp_correction(ION_DB[sp]["D"], T_REF_ION, T_old, eta_ref, eta_now)
                   for sp in (sel_cat+sel_an)}
            # 균열 채널 — 병렬 혼합(면적가중)
            # crack 영역의 P는 (1 + G_crack * f)배 강화
            P_crack = {sp: (D_T[sp]*eps/max(tau,1e-9))/max(L,1e-12) * (1.0 + 10.0**G_crk * f_old) for sp in (sel_cat+sel_an)}
            P_bulk  = {sp: (D_T[sp]*eps/max(tau,1e-9))/max(L,1e-12) for sp in (sel_cat+sel_an)}
            P_eff   = {sp: (1.0 - a_crk)*P_bulk[sp] + a_crk*P_crack[sp] for sp in (sel_cat+sel_an)}

            # 전류 (현재 T 로 Donnan 농도 이미 계산됨; 온도 변화가 크면 Cm도 T 반영하려면 재계산 가능)
            _, i_now = compute_flux_and_current(P_eff, T_old)
            iv[k] = i_now

            # 전력 밀도 (면적당) ~ i*dV  (부호 무시)
            Pj = abs(i_now * dV)  # W/m²

            # 열수지: dT/dt = Pj/C_th - (T - T_amb)/τ_cool
            dTdt = Pj/max(C_th,1e-9) - (T_old - T_amb)/max(10.0**st.session_state["tr_tau"], 1e-6)
            Tv[k] = T_old + dt*dTdt

            # crack 성장: df/dt = k_g·max(T-Tcrit,0)·(1-f)
            kg = 10.0**st.session_state["tr_kg"]
            drive = max(T_old - Tcrit, 0.0)
            dfdt = kg * drive * (1.0 - f_old)
            fv[k] = float(np.clip(f_old + dt*dfdt, 0.0, 1.0))

        # Plot
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

        # Crack fraction
        figF, axF = plt.subplots(figsize=(12,2.4))
        axF.plot(tvec, fv, label="f_crack")
        axF.set_xlabel("t (s)"); axF.set_ylabel("f")
        axF.grid(True); axF.legend(loc="upper left")
        st.pyplot(figF, use_container_width=True); plt.close(figF)

        # 상태 요약
        st.info(f"Final T = {Tv[-1]:.2f} K,  Final i = {iv[-1]:.3e} A/m²,  Final f_crack = {fv[-1]:.3f}")

# =====================================================================
# MODE 3 — DRUG IN VESSEL (1D ADR, transient)  — 기존 그대로
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
    # (너의 기존 구현 그대로)
    pass  # ← 네 파일의 기존 run_vascular_drug() 본문을 그대로 유지하세요.

# =====================================================================
# App shell — Mode selection
# =====================================================================
st.title("Unified Transport Simulators (SI)")

mode_main = st.sidebar.radio("Select simulation",
                             ["Gas membrane", "Ion membrane", "Drug in vessel"],
                             index=0)

if mode_main == "Gas membrane":
    run_gas_membrane()
elif mode_main == "Ion membrane":
    run_ion_membrane()
else:
    run_vascular_drug()
