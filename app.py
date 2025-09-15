# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import exp

# -------------------------------
# 기본 데이터베이스: 분자량(kg/kmol), 분자 직경(Å)
# -------------------------------
GASES = {
    "H2": (2.016, 2.89),
    "D2": (4.028, 2.89),
    "He": (4.003, 2.60),
    "N2": (28.0134, 3.64),
    "O2": (31.998, 3.46),
    "CO2": (44.01, 3.30),
    "CH4": (16.043, 3.80),
    "C2H6": (30.070, 4.44),
    "C3H6": (42.081, 4.68),
    "C3H8": (44.097, 4.65),
}

R = 8.314  # J/mol/K

# -------------------------------
# 메커니즘 proxy 함수들
# -------------------------------
def proxy_knudsen(M):
    return 1.0 / np.sqrt(M)

def proxy_surface(q, Qst, T):
    return q * np.exp(Qst*1000/(R*T))

def proxy_solution(solubility, diff):
    return solubility * diff

def proxy_capillary(d_nm, T, relP):
    r = max(d_nm/2.0, 1e-9)
    relP_th = np.exp(-120.0/(r*T))
    return np.clip(relP - relP_th, 0.0, None)

# -------------------------------
# Sieving 강제 규칙
# -------------------------------
def mechanism_selector(d_nm, gas1, gas2, T, relP, q1, Q1, q2, Q2):
    _, d1 = GASES[gas1]
    _, d2 = GASES[gas2]
    d_small, d_large = sorted([d1, d2])
    p_eff = d_nm * 10  # Å

    # ✅ sieving 강제 규칙
    if d_small < p_eff < d_large:
        return "Sieving"

    # 그렇지 않으면 proxy 비교
    score_knud = proxy_knudsen(GASES[gas1][0]) + proxy_knudsen(GASES[gas2][0])
    score_surf = proxy_surface(q1, Q1, T) + proxy_surface(q2, Q2, T)
    score_sol = proxy_solution(1.0, 1.0)   # 단순 placeholder
    score_cap = proxy_capillary(d_nm, T, relP)

    scores = {
        "Knudsen": score_knud,
        "Surface": score_surf,
        "Solution": score_sol,
        "Capillary": score_cap,
    }
    return max(scores, key=scores.get)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Membrane Mechanisms", layout="wide")
st.title("Membrane Transport Mechanism Simulator")

left, right = st.columns([2,3])

with left:
    st.header("Inputs")

    gas1 = st.selectbox("Gas 1", list(GASES.keys()), index=0)
    gas2 = st.selectbox("Gas 2", list(GASES.keys()), index=3)

    T = st.slider("Temperature (K)", 10.0, 600.0, 300.0, 10.0)
    P = st.slider("Total Pressure (bar)", 0.1, 10.0, 1.0, 0.1)
    d_nm = st.slider("Pore Diameter (nm)", 0.01, 50.0, 0.3, 0.01)

    st.markdown("**Double-site Langmuir Parameters (mmol/g, kJ/mol)**")
    q1 = st.slider("q1 (mmol/g)", 0.0, 10.0, 0.5, 0.1)
    Q1 = st.slider("Qst1 (kJ/mol)", 0.0, 100.0, 20.0, 1.0)
    q2 = st.slider("q2 (mmol/g)", 0.0, 10.0, 0.5, 0.1)
    Q2 = st.slider("Qst2 (kJ/mol)", 0.0, 100.0, 20.0, 1.0)

with right:
    st.header("Results")

    relP = np.linspace(0.01, 1.0, 100)
    mech_list = [mechanism_selector(d_nm, gas1, gas2, T, rp, q1, Q1, q2, Q2) for rp in relP]

    # 그래프 (메커니즘 밴드)
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(relP, [1]*len(relP), alpha=0)  # dummy

    colors = {"Knudsen":"orange", "Surface":"green", "Capillary":"purple",
              "Solution":"gray", "Sieving":"blue"}
    for mech in colors:
        mask = np.array(mech_list) == mech
        ax.fill_between(relP, 0, 1, where=mask, color=colors[mech], alpha=0.4, label=mech)

    ax.set_xlabel("Relative Pressure (P/P0)")
    ax.set_ylabel("Mechanism band")
    ax.set_ylim(0,1)
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.25))
    st.pyplot(fig)
    plt.close(fig)

    # 지배적 메커니즘 표시
    st.write(f"**Dominant mechanism at current conditions:** {mechanism_selector(d_nm, gas1, gas2, T, 0.5, q1, Q1, q2, Q2)}")
