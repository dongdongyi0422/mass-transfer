# app.py — Forced-sieving + Permeance & Selectivity (Streamlit)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# -------------------- Constants & Gas DB --------------------
R = 8.314  # J/mol/K

# 튜닝 파라미터 (sieving이 잘 드러나도록)
FLEX_A = 0.2        # Å, 유효기공 보정 (p_eff = d_nm*10 + FLEX_A)
SIEV_A = 2.5        # sieving 가중치
CAP_A  = 0.45       # capillary 가중치
RELPA_CONST = -150  # capillary 임계 상대압력 계수(보수적으로; 쉬이 켜지지 않도록)

THICK_M = 100e-9    # m, 임의 두께(Permeance 산정 스케일)
E_D_SOL = 1.8e4     # J/mol, solution-diffusion 활성화
E_D_SRF = 9.0e3     # J/mol, surface-diffusion 활성화

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

# -------------------- Adsorption (Double-site Langmuir) --------------------
def dsl_loading(T, P_bar, relP, Q1_kJ, Q2_kJ, q1, q2, b0=1e-4):
    """
    q(relP) = q1*b1*P/(1+b1*P) + q2*b2*P/(1+b2*P)
    - Q in kJ/mol, q in mmol/g
    - P = relP * (P_bar*1e5) [Pa], 단순 스케일용
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
    # 크면 커짐, 가벼울수록 커짐
    return (d_nm/2.0) / np.sqrt(M)

def proxy_sieving(d_nm, d_ang, T):
    # 기공이 분자보다 살짝 클수록 커짐 + 온도 낮을수록 유리
    p_eff = d_nm*10.0 + FLEX_A  # nm->Å
    x = 1.0 - (d_ang/p_eff)**2
    return SIEV_A * np.clip(x, 0.0, None)**2 * np.exp(-9.0e3/(R*T))

def proxy_surface(d_nm, T, loading_mmolg):
    # 흡착량이 클수록, T 낮을수록 유리; 작은 기공에서 유리
    return np.exp(-E_D_SRF/(R*T)) * loading_mmolg / np.maximum(d_nm, 1e-9)

def proxy_capillary(d_nm, T, relP, cap_scale):
    # 임계 relP_th 이상에서만 활성; 작은 기공에서 임계가 높음
    r = max(d_nm/2.0, 1e-9)  # nm
    relP_th = np.exp(RELPA_CONST/(r*T))  # 보수적
    return CAP_A * cap_scale*np.sqrt(r) * np.clip(relP - relP_th, 0.0, None)

def proxy_solution(T, loading_mmolg, M):
    # 용해·확산: 흡착량 + 1/sqrt(M)
    return np.exp(-E_D_SOL/(R*T)) / np.sqrt(M) * np.maximum(loading_mmolg, 0.0)

# -------------------- Forced-sieving rule per gas --------------------
def proxies_for_gas(gas, other, T, P_bar, d_nm, relP, load_self, load_other):
    """
    각 relP에서 gas에 대한 6개 메커니즘 proxy 배열을 반환.
    - 강제 sieving 규칙:
        p_eff(Å)가 두 분자 지름 사이면:
          작은 분자 gas → Sieving만 활성(큼), 나머지 0
          큰  분자 gas → Blocked=1, 나머지 0
        p_eff <= min_d → 모두 Blocked
    - 그 외: 일반 proxy 계산
    """
    M, d_ang = GASES[gas]
    d_other = GASES[other][1]
    d_small, d_large = sorted([d_ang, d_other])
    p_eff = d_nm*10.0 + FLEX_A

    # 공통 배열 생성
    relP = np.asarray(relP)
    zeros = np.zeros_like(relP, dtype=float)
    ones  = np.ones_like(relP, dtype=float)

    # 완전 차단 영역
    if p_eff <= d_small:
        return {"Blocked": ones, "Sieving": zeros, "Knudsen": zeros,
                "Surface": zeros, "Capillary": zeros, "Solution": zeros}

    # 강제 sieving 영역
    if d_small < p_eff < d_large:
        if d_ang == d_small:
            # 현재 gas가 작은 분자 → sieving 통과
            siev = 1e3 * ones  # 매우 크게 하여 dominant 보장
            return {"Blocked": zeros, "Sieving": siev, "Knudsen": zeros,
                    "Surface": zeros, "Capillary": zeros, "Solution": zeros}
        else:
            # 현재 gas가 큰 분자 → 차단
            return {"Blocked": ones, "Sieving": zeros, "Knudsen": zeros,
                    "Surface": zeros, "Capillary": zeros, "Solution": zeros}

    # 일반 영역: proxy 경쟁
    cap_scale = 0.6 if gas == "CO2" else 0.5  # 간단 가중(원하면 테이블화 가능)
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
    y = load_self / (load_self + load_other + 1e-30)  # 조성 가중
    return (maxp * y) / THICK_M

# -------------------- UI --------------------
st.set_page_config(page_title="Membrane mechanisms", layout="wide")
st.title("Membrane Transport Mechanisms — forced sieving with permeance/selectivity")

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Global")
    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("H2"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("CH4"))
    T    = st.slider("Temperature (K)", 10.0, 600.0, 300.0, 1.0)
    Pbar = st.slider("Total pressure (bar)", 0.1, 10.0, 1.0, 0.1)
    d_nm = st.slider("Pore diameter (nm)", 0.01, 50.0, 0.34, 0.01)

    st.subheader("Double-site Langmuir — Gas 1")
    Q11 = st.slider("Qst1 (kJ/mol) [Gas 1]", 0.0, 100.0, 12.0, 0.1)
    Q12 = st.slider("Qst2 (kJ/mol) [Gas 1]", 0.0, 100.0, 10.0, 0.1)
    q11 = st.slider("q1 (mmol/g) [Gas 1]",   0.0, 10.0, 0.20, 0.01)
    q12 = st.slider("q2 (mmol/g) [Gas 1]",   0.0, 10.0, 0.10, 0.01)

    st.subheader("Double-site Langmuir — Gas 2")
    Q21 = st.slider("Qst1 (kJ/mol) [Gas 2]", 0.0, 100.0, 12.0, 0.1)
    Q22 = st.slider("Qst2 (kJ/mol) [Gas 2]", 0.0, 100.0, 10.0, 0.1)
    q21 = st.slider("q1 (mmol/g) [Gas 2]",   0.0, 10.0, 0.20, 0.01)
    q22 = st.slider("q2 (mmol/g) [Gas 2]",   0.0, 10.0, 0.10, 0.01)

# -------------------- Simulation --------------------
relP = np.linspace(0.01, 0.99, 400)

# DSL loading (mmol/g)
load1 = dsl_loading(T, Pbar, relP, Q11, Q12, q11, q12)
load2 = dsl_loading(T, Pbar, relP, Q21, Q22, q21, q22)

# Proxies + forced sieving rule
prox1 = proxies_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
prox2 = proxies_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)

# Dominant mechanism band (Gas1 기준)
mechs_g1 = pick_mechanism(prox1)

# Permeance & Selectivity
perm1 = permeance_from_proxies(prox1, load1, load2)
perm2 = permeance_from_proxies(prox2, load2, load1)
sel   = np.where(perm2 > 0, perm1/perm2, 0.0)

# -------------------- Plots --------------------
with right:
    # Mechanism band (Gas1 dominant)
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mechs_g1])[None, :, :]
    figB, axB = plt.subplots(figsize=(9, 1.6))
    axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
    axB.set_yticks([]); axB.set_xlim(0, 1)
    axB.set_xlabel("Relative pressure (P/P0)")
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    st.pyplot(figB); plt.close(figB)

    # Legend (separate)
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

    # Text summary at mid P/P0
    mid = len(relP)//2
    # 현재 조건에서 두 가스 각각의 지배 메커니즘
    g1_mech_now = pick_mechanism({k: v for k, v in prox1.items()})
    g2_mech_now = pick_mechanism({k: v for k, v in prox2.items()})
    st.write(f"Gas1 dominant near P/P0={relP[mid]:.2f}: {g1_mech_now[mid]}")
    st.write(f"Gas2 dominant near P/P0={relP[mid]:.2f}: {g2_mech_now[mid]}")
