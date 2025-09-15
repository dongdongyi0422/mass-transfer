# app.py — Forced-sieving simulator with Permeance/Selectivity
# Slider + Number Input + ± Buttons (fully synced)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# -------------------- Constants & Gas DB --------------------
R = 8.314  # J/mol/K

# 튜닝 파라미터 (sieving 가시성↑)
FLEX_A = 0.2        # Å, p_eff = d_nm*10 + FLEX_A
SIEV_A = 2.5        # sieving weight
CAP_A  = 0.45       # capillary weight
RELPA_CONST = -150  # capillary onset (더 보수적으로)

THICK_M = 100e-9    # m, arbitrary thickness for permeance scale
E_D_SOL = 1.8e4     # J/mol
E_D_SRF = 9.0e3     # J/mol

# (분자량, 운동지름 Å)
GASES = {
    "H2":  (2.016, 2.89),
    "D2":  (4.028, 2.89),
    "He":  (4.003, 2.60),
    "N2":  (28.0134, 3.64),
    "O2":  (31.998, 3.46),
    "CO2": (44.01, 3.30),
    "CH4": (16.043, 3.80),
    "C2H6":(30.070, 4.44),
    "C3H6":(42.081, 4.68),
    "C3H8":(44.097, 4.65),
}

MECHS = ["Blocked", "Sieving", "Knudsen", "Surface", "Capillary", "Solution"]
MCOLOR = {
    "Blocked":  "#bdbdbd",
    "Sieving":  "#1f78b4",
    "Knudsen":  "#33a02c",
    "Surface":  "#e31a1c",
    "Capillary":"#ff7f00",
    "Solution": "#6a3d9a",
}

# -------------------- UI helper: slider + number + ± buttons --------------------
def control_with_buttons(label, minv, maxv, default, step, key, host=None, fmt="float"):
    """
    한 줄에 [슬라이더 | 숫자입력 | - | +]를 배치하고 모두 동기화.
    - fmt="float" or "int"
    """
    if host is None:
        host = st
    ss = st.session_state
    vkey = f"{key}__val"
    skey = f"{key}__slider"
    nkey = f"{key}__number"
    dkey = f"{key}__dec"
    pkey = f"{key}__inc"

    # 초기값
    if vkey not in ss:
        ss[vkey] = float(default)

    # 증감 계산 함수
    def _clip(v):
        v = float(v)
        v = max(minv, min(maxv, v))
        if fmt == "int":
            # 정수 스텝에 맞게 반올림
            return float(int(round(v)))
        return v

    # 버튼 클릭 처리 (먼저 처리)
    bcol = host.columns([7, 2.2, 0.6, 0.6])  # slider | number | - | +
    with bcol[2]:
        dec = st.button("−", key=dkey)
    with bcol[3]:
        inc = st.button("+", key=pkey)
    if dec:
        ss[vkey] = _clip(ss[vkey] - step)
    if inc:
        ss[vkey] = _clip(ss[vkey] + step)

    # 슬라이더와 숫자입력
    with bcol[0]:
        sval = st.slider(
            label, float(minv), float(maxv), float(ss[vkey]),
            step=float(step), key=skey, label_visibility="visible"
        )
    with bcol[1]:
        if fmt == "int":
            nval = st.number_input(
                " ", min_value=float(minv), max_value=float(maxv),
                value=float(ss[vkey]), step=float(step),
                key=nkey, label_visibility="collapsed", format="%.0f"
            )
        else:
            nval = st.number_input(
                " ", min_value=float(minv), max_value=float(maxv),
                value=float(ss[vkey]), step=float(step),
                key=nkey, label_visibility="collapsed"
            )

    # 동기화: 최근 변경을 반영
    # 우선순위: 버튼 → 숫자입력 → 슬라이더
    new_val = ss[vkey]
    if nval != ss[vkey]:
        new_val = nval
    if sval != ss[vkey]:
        new_val = sval
    ss[vkey] = _clip(new_val)

    # 슬라이더/입력칸을 세션 값에 맞춰 정렬 (Streamlit rerun으로 대개 정렬됨)
    return float(ss[vkey])

# -------------------- Adsorption (Double-site Langmuir) --------------------
def dsl_loading(T, P_bar, relP, Q1_kJ, Q2_kJ, q1, q2, b0=1e-4):
    """
    q(relP) = q1*b1*P/(1+b1*P) + q2*b2*P/(1+b2*P)
    - Q in kJ/mol, q in mmol/g
    - P = relP * (P_bar*1e5) [Pa], 단순 스케일
    """
    P0 = P_bar * 1e5
    Q1 = max(Q1_kJ, 0.0) * 1e3
    Q2 = max(Q2_kJ, 0.0) * 1e3
    b1 = b0 * np.exp(Q1/(R*T))
    b2 = b0 * np.exp(Q2/(R*T))
    P = np.clip(relP, 1e-9, 1.0) * P0
    return q1 * (b1*P)/(1.0 + b1*P) + q2 * (b2*P)/(1.0 + b2*P)  # mmol/g

# -------------------- Mechanism proxies --------------------
def proxy_knudsen(d_nm, M):
    return (d_nm/2.0) / np.sqrt(M)

def proxy_sieving(d_nm, d_ang, T):
    p_eff = d_nm*10.0 + FLEX_A  # nm->Å
    x = 1.0 - (d_ang/p_eff)**2
    return SIEV_A * np.clip(x, 0.0, None)**2 * np.exp(-9.0e3/(R*T))

def proxy_surface(d_nm, T, loading_mmolg):
    return np.exp(-E_D_SRF/(R*T)) * loading_mmolg / np.maximum(d_nm, 1e-9)

def proxy_capillary(d_nm, T, relP, cap_scale):
    r = max(d_nm/2.0, 1e-9)  # nm
    relP_th = np.exp(RELPA_CONST/(r*T))  # conservative
    return CAP_A * cap_scale*np.sqrt(r) * np.clip(relP - relP_th, 0.0, None)

def proxy_solution(T, loading_mmolg, M):
    return np.exp(-E_D_SOL/(R*T)) / np.sqrt(M) * np.maximum(loading_mmolg, 0.0)

# -------------------- Forced-sieving rule per gas --------------------
def proxies_for_gas(gas, other, T, P_bar, d_nm, relP, load_self, load_other):
    """
    각 relP에서 gas에 대한 6개 메커니즘 proxy 배열 반환.
    - 강제 sieving:
        p_eff(Å)가 두 분자 지름 사이면:
          작은 분자 gas → Sieving만 활성(아주 큼), 나머지 0
          큰  분자 gas → Blocked=1, 나머지 0
        p_eff <= min_d → 모두 Blocked
    - 그 외: 일반 proxy 경쟁
    """
    M, d_ang = GASES[gas]
    d_other = GASES[other][1]
    d_small, d_large = sorted([d_ang, d_other])
    p_eff = d_nm*10.0 + FLEX_A

    relP = np.asarray(relP)
    zeros = np.zeros_like(relP, dtype=float)
    ones  = np.ones_like(relP, dtype=float)

    if p_eff <= d_small:
        return {"Blocked": ones, "Sieving": zeros, "Knudsen": zeros,
                "Surface": zeros, "Capillary": zeros, "Solution": zeros}

    if d_small < p_eff < d_large:
        if d_ang == d_small:
            siev = 1e3 * ones  # dominant 보장
            return {"Blocked": zeros, "Sieving": siev, "Knudsen": zeros,
                    "Surface": zeros, "Capillary": zeros, "Solution": zeros}
        else:
            return {"Blocked": ones, "Sieving": zeros, "Knudsen": zeros,
                    "Surface": zeros, "Capillary": zeros, "Solution": zeros}

    cap_scale = 0.6 if gas == "CO2" else 0.5
    siev = proxy_sieving(d_nm, d_ang, T) * np.ones_like(relP)
    knud = proxy_knudsen(d_nm, M) * np.ones_like(relP)
    surf = proxy_surface(d_nm, T, load_self)
    capp = proxy_capillary(d_nm, T, relP, cap_scale)
    solu = proxy_solution(T, load_self, M)

    return {"Blocked": zeros, "Sieving": siev, "Knudsen": knud,
            "Surface": surf, "Capillary": capp, "Solution": solu}

def pick_mechanism(prox_dict):
    arr = np.vstack([prox_dict[m] for m in MECHS])
    idx = np.argmax(arr, axis=0)
    return np.array(MECHS)[idx]

def permeance_from_proxies(prox_dict, load_self, load_other):
    arr = np.vstack([prox_dict[m] for m in MECHS])
    maxp = arr.max(axis=0)
    y = load_self / (load_self + load_other + 1e-30)
    return (maxp * y) / THICK_M

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Membrane mechanisms", layout="wide")
st.title("Membrane Transport Mechanisms — forced sieving + ± controls")

left, right = st.columns([6, 7], gap="large")  # 왼쪽 더 넓게

with left:
    st.subheader("Global")
    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("H2"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("CH4"))

    T    = control_with_buttons("Temperature (K)",       10.0, 600.0, 300.0, 1.0,  "T",    host=left)
    Pbar = control_with_buttons("Total pressure (bar)",   0.1,  10.0,   1.0, 0.1,  "P",    host=left)
    d_nm = control_with_buttons("Pore diameter (nm)",    0.01,  50.0,  0.34, 0.01, "d",    host=left)

    st.subheader("Double-site Langmuir — Gas 1")
    Q11 = control_with_buttons("Qst1 (kJ/mol) [Gas 1]", 0.0, 100.0, 12.0, 0.1, "Q11", host=left)
    Q12 = control_with_buttons("Qst2 (kJ/mol) [Gas 1]", 0.0, 100.0, 10.0, 0.1, "Q12", host=left)
    q11 = control_with_buttons("q1 (mmol/g) [Gas 1]",   0.0,  10.0,  0.20, 0.01, "q11", host=left)
    q12 = control_with_buttons("q2 (mmol/g) [Gas 1]",   0.0,  10.0,  0.10, 0.01, "q12", host=left)

    st.subheader("Double-site Langmuir — Gas 2")
    Q21 = control_with_buttons("Qst1 (kJ/mol) [Gas 2]", 0.0, 100.0, 12.0, 0.1, "Q21", host=left)
    Q22 = control_with_buttons("Qst2 (kJ/mol) [Gas 2]", 0.0, 100.0, 10.0, 0.1, "Q22", host=left)
    q21 = control_with_buttons("q1 (mmol/g) [Gas 2]",   0.0,  10.0,  0.20, 0.01, "q21", host=left)
    q22 = control_with_buttons("q2 (mmol/g) [Gas 2]",   0.0,  10.0,  0.10, 0.01, "q22", host=left)

# -------------------- Simulation --------------------
relP = np.linspace(0.01, 0.99, 400)

# DSL loading (mmol/g)
load1 = dsl_loading(T, Pbar, relP, Q11, Q12, q11, q12)
load2 = dsl_loading(T, Pbar, relP, Q21, Q22, q21, q22)

# Proxies + forced sieving rule
prox1 = proxies_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
prox2 = proxies_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)

# Dominant mechanism band (Gas1)
mechs_g1 = pick_mechanism(prox1)

# Permeance & Selectivity
perm1 = permeance_from_proxies(prox1, load1, load2)
perm2 = permeance_from_proxies(prox2, load2, load1)
sel   = np.where(perm2 > 0, perm1/perm2, 0.0)

# -------------------- Plots --------------------
with right:
    # Mechanism band
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mechs_g1])[None, :, :]
    figB, axB = plt.subplots(figsize=(9, 1.6))
    axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
    axB.set_yticks([]); axB.set_xlim(0, 1)
    axB.set_xlabel("Relative pressure (P/P0)")
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    st.pyplot(figB); plt.close(figB)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
    figL, axL = plt.subplots(figsize=(9, 1.2))
    axL.axis("off")
    figL.legend(handles=handles, loc="center", ncol=6, frameon=True)
    st.pyplot(figL); plt.close(figL)

    # Permeance
    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    ax1.plot(relP, perm1, label=f"Permeance {gas1}")
    ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
    ax1.set_ylabel("Permeance (arb. units)")
    ax1.set_xlabel("Relative pressure (P/P0)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    # Selectivity
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (-)")
    ax2.set_xlabel("Relative pressure (P/P0)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    # Summary
    mid = len(relP)//2
    st.write(f"Gas1 dominant near P/P0={relP[mid]:.2f}: {mechs_g1[mid]}")
