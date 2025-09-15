# app.py
# Membrane Transport Simulator (SI units)
# - DSL: b1,b2 직접 입력 (Qst 불필요)
# - Sidebar 입력: 2-Row (Slider / NumberInput), 버튼(−/＋) 제거 → "네모 칸" 없음
# - Permeance 플롯: 범례 제목 "Permeance (GPU)", 데이터는 GPU로 환산하여 표시

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba

# ---------------------------- Physical constants ----------------------------
R = 8.314  # J/mol/K

# ---------------------------- Model global knobs ----------------------------
RHO_EFF = 500.0   # kg/m^3  (sorption per membrane volume conversion)
D0_SURF = 1e-9    # m^2/s   (surface diffusion scale)
D0_SOL  = 1e-10   # m^2/s   (solution diffusion scale)
E_D_SOL = 1.8e4   # J/mol   (solution diffusion activation)
K_CAP   = 1e-7    # (mol m^-1.5 s^-1 Pa^-1) capillary proxy scale
E_SIEVE = 6.0e3   # J/mol   (near-threshold sieving barrier)
PI_TINY = 1e-14   # mol m^-2 s^-1 Pa^-1 (numerical floor)
SOL_TH_NM = 0.30  # nm: solution-diffusion favored at/under this pore size (<=)
# 상단 상수
DELTA_SOFT_A = 0.50    # [Å] 근계 폭; 0.4~0.6에서 조정
PI_SOFT_REF  = 1e-6    # [mol m^-2 s^-1 Pa^-1] 근계 투과 기준 (수 μGPU~수십 GPU 수준으로 조절)
# 체거름 경계 완충대 / Knudsen 여유폭(Å)
SIEVE_BAND_A = 0.15
DELTA_A      = 0.4
# --- Auto calibration (Option 2) ---
AUTO_CALIB = True

# 참조점(안정적인 중간 조건)
REF = {"T": 298.15, "Pbar": 1.0, "d_nm": 0.36, "L_nm": 100.0, "rp": 0.5}

# 목표 GPU (가스별 1회 보정치) — 필요에 따라 조정
TARGET_GPU = {
    "CO2": 150.0,
    "CH4": 40.0,
    "N2": 5.0,
    "H2": 200.0,
    "D2": 180.0,
    "He": 300.0,
}

# 보정 배수 저장소
GAS_SCALE = {}       # 예: {"CO2": 1.23, "CH4": 0.87}
GPU_UNIT  = 3.35e-10 # 1 GPU = 3.35e-10 mol m^-2 s^-1 Pa^-1

# ---------------------------- Gas parameters ----------------------------
# M [kg/mol], kinetic diameter d [Å], surface Ea_s [J/mol]
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

# ---------------------------- UI helper: Slider + Number (no buttons) ----------------------------
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3):
    """
    2-Row UI:
      Row1: [ Slider (full width) ]
      Row2: [ NumberInput (wide) ]
    - st.session_state[key] 하나만 '캐논' 값 유지
    - 버튼 제거 → 사이드바에 빈 네모 칸 생기지 않음
    """
    if key not in st.session_state:
        st.session_state[key] = float(vinit)

    cur = float(st.session_state[key])
    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    fmt = f"%.{int(decimals)}f"

    # Row1: Slider
    sld_val = st.slider(
        lab,
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(cur),
        step=float(vstep),
        key=f"{key}_sld_view",
    )

    # Row2: Number
    num_val = st.number_input(
        "",
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(cur),
        step=float(vstep),
        format=fmt,
        key=f"{key}_num_view",
    )

    # 변경 우선순위: 숫자입력 > 슬라이더
    new_val = float(num_val) if num_val != cur else float(sld_val)
    new_val = float(np.clip(new_val, vmin, vmax))
    if new_val != cur:
        st.session_state[key] = new_val

    return st.session_state[key]

# ---------------------------- DSL adsorption (b1,b2 direct) ----------------------------
def dsl_loading_and_slope_b(
    gas, T, P_bar, relP_vec,
    q1_mmolg, q2_mmolg,
    b1, b2
):
    """
    Double-site Langmuir using (q1,q2,b1,b2).
    Returns:
      q_vec [mmol/g]
      dqdp  [mol/kg/Pa]
    """
    P0 = P_bar * 1e5  # Pa
    b1_T = max(float(b1), 0.0)
    b2_T = max(float(b2), 0.0)

    q_vec  = np.zeros_like(relP_vec, float)
    dqdp_MK = np.zeros_like(relP_vec, float)  # mol/kg/Pa

    q1_molkg = q1_mmolg * 1e-3 * 1e3
    q2_molkg = q2_mmolg * 1e-3 * 1e3

    for i, rp in enumerate(relP_vec):
        P = max(rp, 1e-9) * P0
        th1 = (b1_T*P)/(1.0 + b1_T*P)
        th2 = (b2_T*P)/(1.0 + b2_T*P)
        q_molkg = q1_molkg*th1 + q2_molkg*th2
        q_vec[i] = q_molkg / 1e3  # -> mmol/g

        dqdp_MK[i] = (q1_molkg * b1_T)/(1.0 + b1_T*P)**2 + (q2_molkg * b2_T)/(1.0 + b2_T*P)**2

    return q_vec, dqdp_MK

# ---------------------------- Mechanisms & permeance (SI) ----------------------------
MECH_ORDER = ["Blocked", "Sieving", "Knudsen", "Surface", "Capillary", "Solution"]
MECH_COLOR = {
    "Blocked":  "#bdbdbd",
    "Sieving":  "#1f78b4",
    "Knudsen":  "#33a02c",
    "Surface":  "#e31a1c",
    "Capillary":"#ff7f00",
    "Solution": "#6a3d9a",
}

def mean_free_path_nm(T, P_bar, d_ang):
    kB = 1.380649e-23; P = P_bar*1e5; d = d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def classify_mechanism(pore_d_nm, gas1, gas2, T, P_bar, rp):
    d1, d2 = PARAMS[gas1]["d"], PARAMS[gas2]["d"]  # [Å]
    dmin = min(d1, d2)
    p_eff_A = pore_d_nm * 10.0                     # [Å]
    lam = mean_free_path_nm(T, P_bar, 0.5*(d1+d2)) # [nm]

    # 0) 기하학적 차단 최우선
    if p_eff_A <= dmin - SIEVE_BAND_A:
        return "Blocked"

    # 1) 매우 작은 기공은 용액-확산
    if pore_d_nm <= SOL_TH_NM:
        return "Solution"

    # 2) Knudsen: 분자보다 충분히 여유 + pore << λ
    if (p_eff_A >= dmin + DELTA_A) and (pore_d_nm < 0.5 * lam):
        return "Knudsen"

    # 3) 큰 기공 + 높은 상대압 → 모세관
    if pore_d_nm >= 2.0 and rp > 0.5:
        return "Capillary"

    # 4) 분자 직경 근방 → 체거름
    if p_eff_A <= dmin + SIEVE_BAND_A:
        return "Sieving"

    # 5) 그 외 → 표면 확산
    return "Surface"

def pintr_knudsen_SI(pore_d_nm, T, M, L_m):
    r = max(pore_d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0) * r * np.sqrt((8.0*R*T)/(np.pi*M))
    Pi = Dk / (L_m * R * T)
    return Pi

def pintr_sieving_SI(pore_d_nm, gas, T, L_m):
    dA = PARAMS[gas]["d"]       # Å
    pA = pore_d_nm * 10.0       # Å
    delta = dA - pA             # >0 이면 기공이 더 작음(차단 경향)

    if delta > 0:
        # --- 소프트 차단 (gate opening/tunneling proxy) ---
        # delta가 작을수록(경계에 가까울수록) 유한한 투과 발생
        Pi_gate = PI_SOFT_REF * np.exp(- (delta/DELTA_SOFT_A)**2) * np.exp(-E_SIEVE/(R*T))
        return max(Pi_gate, PI_TINY)

    # --- 기공이 더 큰 경우: 기존 체거름 식 ---
    x = max(1.0 - (dA/pA)**2, 0.0)
    f_open = x**2
    Pi_ref = 3e-4   # 문헌 스케일(수십~백 GPU)을 맞추기 위한 보정치
    Pi = Pi_ref * f_open * np.exp(-E_SIEVE/(R*T))
    return max(Pi, PI_TINY)


def pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa):
    Ds = D0_SURF * np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    Pi = (Ds / L_m) * (dqdp_molkgPa * RHO_EFF)
    return max(Pi, 0.0)

def pintr_capillary_SI(pore_d_nm, rp, L_m):
    r_m = max(pore_d_nm*1e-9/2.0, 1e-12)
    thresh = np.exp(-120.0/( (pore_d_nm/2.0)*rp*300.0 + 1e-12 ))
    if rp <= thresh:
        return 0.0
    return K_CAP * np.sqrt(r_m) / L_m

def pintr_solution_SI(gas, T, L_m, dqdp_molkgPa):
    Dsol = D0_SOL * np.exp(-E_D_SOL/(R*T)) / np.sqrt(PARAMS[gas]["M"]/1e-3)
    Pi = (Dsol / L_m) * (dqdp_molkgPa * RHO_EFF)
    return max(Pi, 0.0)

def permeance_series_SI(pore_d_nm, gas, other, T, P_bar, relP, L_nm,
                        q_mmolg, dqdp_molkgPa, q_other_mmolg):
    L_m = max(L_nm, 1e-3) * 1e-9
    M = PARAMS[gas]["M"]
    Pi = np.zeros_like(relP, float)

    for i, rp in enumerate(relP):
        rule = classify_mechanism(pore_d_nm, gas, other, T, P_bar, rp)
        if   rule == "Blocked":  Pi0 = PI_TINY
        elif rule == "Sieving":  Pi0 = pintr_sieving_SI(pore_d_nm, gas, T, L_m)
        elif rule == "Knudsen":  Pi0 = pintr_knudsen_SI(pore_d_nm, T, M, L_m)
        elif rule == "Surface":  Pi0 = pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa[i])
        elif rule == "Capillary":Pi0 = pintr_capillary_SI(pore_d_nm, rp, L_m)
        elif rule == "Solution": Pi0 = pintr_solution_SI(gas, T, L_m, dqdp_molkgPa[i])
        else:                    Pi0 = PI_TINY

        Pi0 *= GAS_SCALE.get(gas, 1.0)

        qi = q_mmolg[i]; qj = q_other_mmolg[i]
        theta = (qi/(qi+qj)) if (qi+qj) > 0 else 0.0
        Pi[i] = Pi0 * theta

    return Pi

def mechanism_band_rgba(g1, g2, T, P_bar, d_nm, relP):
    names = [classify_mechanism(d_nm, g1, g2, T, P_bar, r) for r in relP]
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :]
    return rgba, names

def intrinsic_pi0_SI(gas, other, T, Pbar, d_nm, L_nm, rp, dqdp_mkpa, q_mmolg):
    """현재 모델에서 '경쟁/세타' 빼고 순수 intrinsic Π0(SI) 한 점을 계산"""
    L_m = max(L_nm, 1e-3) * 1e-9
    M = PARAMS[gas]["M"]
    rule = classify_mechanism(d_nm, gas, other, T, Pbar, rp)
    if   rule == "Blocked":   Pi0 = PI_TINY
    elif rule == "Sieving":   Pi0 = pintr_sieving_SI(d_nm, gas, T, L_m)
    elif rule == "Knudsen":   Pi0 = pintr_knudsen_SI(d_nm, T, M, L_m)
    elif rule == "Surface":   Pi0 = pintr_surface_SI(d_nm, gas, T, L_m, dqdp_mkpa)
    elif rule == "Capillary": Pi0 = pintr_capillary_SI(d_nm, rp, L_m)
    else:                     Pi0 = pintr_solution_SI(gas, T, L_m, dqdp_mkpa)
    return Pi0

def ensure_gas_scale_once(gas, other, q1, q2, b1, b2):
    """가스별 보정 배수를 1회 산출 (이미 있으면 건너뜀)"""
    if (not AUTO_CALIB) or (gas in GAS_SCALE) or (gas not in TARGET_GPU):
        return
    # 참조점에서 DSL 기울기 한 점 계산
    relP_ref = np.array([REF["rp"]], float)
    q_vec, dqdp_vec = dsl_loading_and_slope_b(gas, REF["T"], REF["Pbar"], relP_ref, q1, q2, b1, b2)
    dqdp_ref = float(dqdp_vec[0])
    q_ref    = float(q_vec[0])

    # intrinsic Π0(SI) → GPU 환산
    Pi0_SI  = intrinsic_pi0_SI(gas, other, REF["T"], REF["Pbar"], REF["d_nm"], REF["L_nm"], REF["rp"], dqdp_ref, q_ref)
    now_gpu = Pi0_SI / GPU_UNIT
    target  = TARGET_GPU[gas]
    scale   = target / max(now_gpu, 1e-20)

    # 폭주 방지 클립
    GAS_SCALE[gas] = float(np.clip(scale, 1e-3, 1e3))

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Membrane Permeance (SI)", layout="wide")
st.title("Membrane Transport Simulator (SI units)")

with st.sidebar:
    st.header("Global Conditions")
    T    = nudged_slider("Temperature", 10.0, 600.0, 1.0, 300.0, key="T",    unit="K")
    Pbar = nudged_slider("Total Pressure", 0.1, 10.0, 0.1, 1.0,  key="Pbar", unit="bar")
    d_nm = nudged_slider("Pore diameter", 0.01, 50.0, 0.01, 0.34, key="d_nm", unit="nm")
    L_nm = nudged_slider("Membrane thickness", 10.0, 100000.0, 1.0, 100.0, key="L_nm", unit="nm")

    gases = list(PARAMS.keys())
    gas1 = st.selectbox("Gas1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas2 (denominator)", gases, index=gases.index("C3H8"))

    st.header("DSL parameters (double-site, use b directly)")
    st.caption("q: 0–5 mmol/g, b: 1e-8 ~ 1e-2 Pa⁻¹ (예시 범위)")

    st.subheader("Gas1")
    q11 = nudged_slider("q1 Gas1",   0.0, 100.0,    0.01, 0.70, key="q11", unit="mmol/g")
    q12 = nudged_slider("q2 Gas1",   0.0, 100.0,    0.01, 0.30, key="q12", unit="mmol/g")
    b11 = nudged_slider("b1 Gas1",   1e-10, 1e-1,  1e-8, 1e-5, key="b11", unit="Pa⁻¹", decimals=8)
    b12 = nudged_slider("b2 Gas1",   1e-10, 1e-1,  1e-8, 5e-6, key="b12", unit="Pa⁻¹", decimals=8)

    st.subheader("Gas2")
    q21 = nudged_slider("q1 Gas2",   0.0, 100.0,    0.01, 0.70, key="q21", unit="mmol/g")
    q22 = nudged_slider("q2 Gas2",   0.0, 100.0,    0.01, 0.30, key="q22", unit="mmol/g")
    b21 = nudged_slider("b1 Gas2",   1e-10, 1e-1,  1e-8, 1e-5, key="b21", unit="Pa⁻¹", decimals=8)
    b22 = nudged_slider("b2 Gas2",   1e-10, 1e-1,  1e-8, 5e-6, key="b22", unit="Pa⁻¹", decimals=8)

# ---------------------------- Compute ----------------------------
relP = np.linspace(0.01, 0.99, 500)

q1_mmolg, dqdp1 = dsl_loading_and_slope_b(gas1, T, Pbar, relP, q11, q12, b11, b12)
q2_mmolg, dqdp2 = dsl_loading_and_slope_b(gas2, T, Pbar, relP, q21, q22, b21, b22)

# --- 가스별 1회 자동 캘리브레이션 (참조점에서 scale 추정) ---
ensure_gas_scale_once(gas1, gas2, q11, q12, b11, b12)
ensure_gas_scale_once(gas2, gas1, q21, q22, b21, b22)

Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, Pbar, relP, L_nm, q1_mmolg, dqdp1, q2_mmolg)
Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, Pbar, relP, L_nm, q2_mmolg, dqdp2, q1_mmolg)
Sel = np.divide(Pi1, Pi2, out=np.zeros_like(Pi1), where=(Pi2>0))

rgba, mech_names = mechanism_band_rgba(gas1, gas2, T, Pbar, d_nm, relP)

# ---------------------------- Layout: Plots & Info ----------------------------
colA, colB = st.columns([1,2])

with colB:
    st.subheader("Mechanism map (along relative pressure)")
    figBand, axBand = plt.subplots(figsize=(9, 0.7))
    axBand.imshow(rgba, extent=(0,1,0,1), aspect='auto', origin='lower')
    axBand.set_yticks([])
    axBand.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); axBand.set_xlim(0,1)
    handles = [plt.Rectangle((0,0),1,1, fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
    leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.7),
                        ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True); plt.close(figBand)

    # --- Permeance (GPU) ---
    st.subheader("Permeance (GPU)")
    fig1, ax1 = plt.subplots(figsize=(9,3))

    GPU = 3.35e-10  # 1 GPU = 3.35×10^-10 mol m^-2 s^-1 Pa^-1
    y1 = Pi1 / GPU
    y2 = Pi2 / GPU

    ax1.plot(relP, y1, label=f"{gas1}")
    ax1.plot(relP, y2, '--', label=f"{gas2}")

    ax1.set_ylabel(r"$\Pi$  (GPU)")

    # y축에서 1e6 같은 오프셋 표기 제거
    from matplotlib.ticker import ScalarFormatter
    ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.get_yaxis().get_offset_text().set_visible(False)

    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True)
    ax1.legend(title="Permeance (GPU)")
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

    st.subheader("Selectivity")
    fig2, ax2 = plt.subplots(figsize=(9,3))
    ax2.plot(relP, Sel, label=f"{gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (–)")
    ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.get_yaxis().get_offset_text().set_visible(False)

    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with colA:
    st.subheader("Mechanism (rule) vs intrinsic (Gas1)")
    rp_mid = relP[len(relP)//2]
    L_m = L_nm*1e-9
    M1 = PARAMS[gas1]["M"]
    dqdp1_mid = dqdp1[len(relP)//2]
    cand = {
        "Blocked":  PI_TINY,
        "Sieving":  pintr_sieving_SI(d_nm, gas1, T, L_m),
        "Knudsen":  pintr_knudsen_SI(d_nm, T, M1, L_m),
        "Surface":  pintr_surface_SI(d_nm, gas1, T, L_m, dqdp1_mid),
        "Capillary":pintr_capillary_SI(d_nm, rp_mid, L_m),
        "Solution": pintr_solution_SI(gas1, T, L_m, dqdp1_mid),
    }
    best_intrinsic = max(cand, key=cand.get)
    rule_mech = classify_mechanism(d_nm, gas1, gas2, T, Pbar, rp_mid)

    st.markdown(
        f"**Mechanism (rule):** `{rule_mech}` &nbsp;&nbsp; | &nbsp;&nbsp;"
        f"**Best intrinsic (Gas1):** `{best_intrinsic}`"
    )
    st.caption(
        "Intrinsic permeance (no competition) at mid $P/P_0$:  \n"
        + "  • " + "  \n  • ".join([f"{k}: {v:.3e} mol m⁻² s⁻¹ Pa⁻¹" for k,v in cand.items()])
    )

st.markdown("---")
st.caption(
    "Permeance in SI is converted to GPU for visualization. "
    "Capillary/Sieving are calibrated proxies. "
    "Surface/Solution terms use DSL slope (∂q/∂p) via ρ_eff."
)
st.markdown("---")
st.caption(
    f"Auto-calibration scale (per gas): {GAS_SCALE}"
)
