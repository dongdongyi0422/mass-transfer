# app.py
# Streamlit web app: Permeance in SI (mol m^-2 s^-1 Pa^-1), Selectivity, Mechanism band
# - 2-Row nudged slider (겹침 방지): Row1 Slider / Row2 [Number][Spacer][-][+]
# - 숫자입력은 소수점 3자리 표시
# - - / + 버튼 간격 확보, + 아이콘 명확 표시
# - classify_mechanism: 0.30 nm 이하는 Solution로 처리(<=), 유효구경 오프셋 제거

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
E_SIEVE = 9.0e3   # J/mol   (near-threshold sieving barrier)
PI_TINY = 1e-14   # mol m^-2 s^-1 Pa^-1 (numerical floor instead of perfect zero)
SOL_TH_NM = 0.30  # nm: solution-diffusion favored at/under this pore size (<=)

# (선택) 체거름 전이 완충대 [Å] — 너무 날카로운 경계를 완화
SIEVE_BAND_A = 0.15

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

# ---------------------------- UI helper: slider + number + − / + ----------------------------
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3):
    """
    2-Row UI:
      Row1: [ Slider (full width) ]
      Row2: [ NumberInput (wide) ][ spacer ][ − ][ + ]
    - st.session_state[key] 하나만 '캐논' 값으로 유지
    - view 위젯은 *_view 키만 사용 (직접 대입 금지)
    - decimals: 숫자 입력 표시 자릿수 (기본 3)
    """
    if key not in st.session_state:
        st.session_state[key] = float(vinit)

    cur = float(st.session_state[key])
    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    fmt = f"%.{int(decimals)}f"

    # -------- Row 1: slider (겹침 방지 위해 단독 행) --------
    sld_val = st.slider(
        lab,
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(cur),
        step=float(vstep),
        key=f"{key}_sld_view",
    )

    # -------- Row 2: number + spacer + - + --------
    c_num, c_sp, c_minus, c_plus = st.columns([0.70, 0.10, 0.10, 0.10], gap="small")

    num_val = c_num.number_input(
        "",
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(cur),
        step=float(vstep),
        format=fmt,
        key=f"{key}_num_view",
    )

    with c_sp:
        st.write("")  # spacer

    minus = c_minus.button("-", key=f"{key}_minus", help=f"-{vstep:g}")
    plus  = c_plus.button("+", key=f"{key}_plus",  help=f"+{vstep:g}")

    # 변경 우선순위: 버튼 > 숫자입력 > 슬라이더
    new_val = cur
    if minus:
        new_val = max(vmin, cur - vstep)
    elif plus:
        new_val = min(vmax, cur + vstep)
    elif num_val != cur:
        new_val = float(num_val)
    elif sld_val != cur:
        new_val = float(sld_val)

    new_val = float(np.clip(new_val, vmin, vmax))
    if new_val != cur:
        st.session_state[key] = new_val

    return st.session_state[key]

# ---------------------------- DSL adsorption ----------------------------
def dsl_loading_and_slope(gas, T, P_bar, relP_vec, q1_mmolg, q2_mmolg, Qst1_kJ, Qst2_kJ):
    """
    Returns:
      q_vec   [mmol/g]
      dqdp    [mol/kg/Pa]  (slope wrt pressure)
    """
    P0 = P_bar * 1e5  # Pa
    b0 = 1e-4
    b1 = b0 * np.exp(max(Qst1_kJ,0.0)*1e3/(R*T))
    b2 = b0 * np.exp(max(Qst2_kJ,0.0)*1e3/(R*T))

    q_vec  = np.zeros_like(relP_vec, float)
    dqdp_MK = np.zeros_like(relP_vec, float)  # mol/kg/Pa

    q1_molkg = q1_mmolg * 1e-3 * 1e3
    q2_molkg = q2_mmolg * 1e-3 * 1e3

    for i, rp in enumerate(relP_vec):
        P = max(rp, 1e-9) * P0
        th1 = (b1*P)/(1.0 + b1*P)
        th2 = (b2*P)/(1.0 + b2*P)
        q_molkg = q1_molkg*th1 + q2_molkg*th2
        q_vec[i] = q_molkg / 1e3  # -> mmol/g

        dqdp_MK[i] = (q1_molkg * b1)/(1.0 + b1*P)**2 + (q2_molkg * b2)/(1.0 + b2*P)**2

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

# 상단 상수 (원래 값 쓰되, 필요시 조정)
SIEVE_BAND_A = 0.15   # [Å] 차단/체거름 경계 완충
DELTA_A      = 0.4    # [Å] Knudsen로 보기 위한 분자 대비 여유 폭
SOL_TH_NM    = 0.30   # [nm] 용액-확산 우세 임계

def classify_mechanism(pore_d_nm, gas1, gas2, T, P_bar, rp):
    d1, d2 = PARAMS[gas1]["d"], PARAMS[gas2]["d"]  # [Å]
    dmin = min(d1, d2)
    p_eff_A = pore_d_nm * 10.0                     # [Å] (nm→Å)
    lam = mean_free_path_nm(T, P_bar, 0.5*(d1+d2)) # [nm]

    # 0) **기하학적 차단이 최우선**
    if p_eff_A <= dmin - SIEVE_BAND_A:
        return "Blocked"

    # 1) 매우 작은 기공은 용액-확산(단, 위에서 이미 막히지 않은 경우에만)
    if pore_d_nm <= SOL_TH_NM:
        return "Solution"

    # 2) 자유분자(벽-충돌 지배): 분자보다 충분히 여유 + pore << λ
    if (p_eff_A >= dmin + DELTA_A) and (pore_d_nm < 0.5 * lam):
        return "Knudsen"

    # 3) 큰 기공 + 높은 상대압에서는 모세관
    if pore_d_nm >= 2.0 and rp > 0.5:
        return "Capillary"

    # 4) 분자 직경에 근접한 초미세 구경은 체거름
    if p_eff_A <= dmin + SIEVE_BAND_A:
        return "Sieving"

    # 5) 그 외는 표면 확산 지배
    return "Surface"



def pintr_knudsen_SI(pore_d_nm, T, M, L_m):
    r = max(pore_d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0) * r * np.sqrt((8.0*R*T)/(np.pi*M))      # [m^2/s]
    Pi = Dk / (L_m * R * T)                                # [mol m^-2 s^-1 Pa^-1]
    return Pi

def pintr_sieving_SI(pore_d_nm, gas, T, L_m):
    dA = PARAMS[gas]["d"] # Å
    pA = pore_d_nm*10.0   # nm->Å (오프셋 제거)
    if pA <= dA - SIEVE_BAND_A:
        return PI_TINY
    x = max(1.0 - (dA/pA)**2, 0.0)
    f_open = x**2
    Pi_ref = 5e-9  # baseline [mol m^-2 s^-1 Pa^-1]
    Pi = Pi_ref * f_open * np.exp(-E_SIEVE/(R*T))
    return max(Pi, PI_TINY)

def pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa):
    Ds = D0_SURF * np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    Pi = (Ds / L_m) * (dqdp_molkgPa * RHO_EFF)             # [mol m^-2 s^-1 Pa^-1]
    return max(Pi, 0.0)

def pintr_capillary_SI(pore_d_nm, rp, L_m):
    r_m = max(pore_d_nm*1e-9/2.0, 1e-12)
    thresh = np.exp(-120.0/( (pore_d_nm/2.0)*rp*300.0 + 1e-12 ))  # heuristic threshold
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

        qi = q_mmolg[i]; qj = q_other_mmolg[i]
        theta = (qi/(qi+qj)) if (qi+qj) > 0 else 0.0
        Pi[i] = Pi0 * theta

    return Pi

def mechanism_band_rgba(g1, g2, T, P_bar, d_nm, relP):
    names = [classify_mechanism(d_nm, g1, g2, T, P_bar, r) for r in relP]
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :]
    return rgba, names

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Membrane Permeance (SI)", layout="wide")
st.title("Membrane Transport Simulator (SI units)")

with st.sidebar:
    st.header("Global Conditions")
    T    = nudged_slider("Temperature", 10.0, 600.0, 1.0, 300.0, key="T",    unit="K")
    Pbar = nudged_slider("Total Pressure", 0.1, 10.0, 0.1, 1.0,  key="Pbar", unit="bar")
    d_nm = nudged_slider("Pore diameter", 0.01, 50.0, 0.01, 0.34, key="d_nm", unit="nm")
    L_nm = nudged_slider("Membrane thickness", 10.0, 1000.0, 1.0, 100.0, key="L_nm", unit="nm")

    gases = list(PARAMS.keys())
    gas1 = st.selectbox("Gas1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas2 (denominator)", gases, index=gases.index("C3H8"))

    st.header("DSL parameters (double-site, each gas)")
    st.caption("Qst: 0–100 kJ/mol, q: 0–5 mmol/g  (숫자 입력은 소수 3자리)")

    st.subheader("Gas1")
    Q11 = nudged_slider("Qst1 Gas1", 0.0, 100.0, 0.1, 27.0, key="Q11", unit="kJ/mol")
    Q12 = nudged_slider("Qst2 Gas1", 0.0, 100.0, 0.1, 18.0, key="Q12", unit="kJ/mol")
    q11 = nudged_slider("q1 Gas1",   0.0, 5.0,    0.01, 0.70, key="q11", unit="mmol/g")
    q12 = nudged_slider("q2 Gas1",   0.0, 5.0,    0.01, 0.30, key="q12", unit="mmol/g")

    st.subheader("Gas2")
    Q21 = nudged_slider("Qst1 Gas2", 0.0, 100.0, 0.1, 26.5, key="Q21", unit="kJ/mol")
    Q22 = nudged_slider("Qst2 Gas2", 0.0, 100.0, 0.1, 17.0, key="Q22", unit="kJ/mol")
    q21 = nudged_slider("q1 Gas2",   0.0, 5.0,    0.01, 0.70, key="q21", unit="mmol/g")
    q22 = nudged_slider("q2 Gas2",   0.0, 5.0,    0.01, 0.30, key="q22", unit="mmol/g")

# ---------------------------- Compute ----------------------------
relP = np.linspace(0.01, 0.99, 500)

q1_mmolg, dqdp1 = dsl_loading_and_slope(gas1, T, Pbar, relP, q11, q12, Q11, Q12)
q2_mmolg, dqdp2 = dsl_loading_and_slope(gas2, T, Pbar, relP, q21, q22, Q21, Q22)

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

    # --- Permeance (SI) + Legend in GPU ---
    st.subheader("Permeance (SI)")
    fig1, ax1 = plt.subplots(figsize=(9, 3))

    GPU = 3.35e-10  # 1 GPU = 3.35×10^-10 mol m^-2 s^-1 Pa^-1

    # 범례에는 GPU 수치 표시 (최대값 기준)
    gpu1 = Pi1.max() / GPU
    gpu2 = Pi2.max() / GPU

    ax1.plot(relP, Pi1, label=f"{gas1}")
    ax1.plot(relP, Pi2, '--', label=f"{gas2}")

    ax1.set_ylabel(r"$\Pi$  (GPU)")
    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True)
    
    # 범례 제목을 Permeance (GPU)로
    leg1 = ax1.legend(title="Permeance (GPU)")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # --- Selectivity (그대로) ---
    st.subheader("Selectivity")
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.plot(relP, Sel)
    ax2.set_ylabel("Selectivity (–)")
    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True); 
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


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
    "All permeance values in SI: mol·m⁻²·s⁻¹·Pa⁻¹. "
    "Capillary/Sieving are calibrated proxies for visualization. "
    "Surface/Solution terms use DSL slope (∂q/∂p) converted by ρ_eff."
)
