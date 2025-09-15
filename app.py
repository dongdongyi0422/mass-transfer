# app.py — Membrane mechanism simulator (Streamlit)
# - 슬라이더와 숫자입력칸 완전 동기화 (session_state + on_change)
# - 각 P/P0 지점에서 6가지(proxy) 중 최대값으로 지배 메커니즘 결정
# - DSL 흡착 → Surface / Solution 반영
# - Blocked 구간은 밴드에 표시 + Permeance=0 + Selectivity 안전 처리
# - 밴드와 범례를 분리 렌더링해 겹침 완전 차단

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# ----------------------------- Page & theme -----------------------------
st.set_page_config(page_title="Membrane Mechanisms Simulator", layout="wide")
st.title("Membrane Transport Mechanisms – data-driven simulator")

# ----------------------------- Constants ------------------------------
R = 8.314  # J/mol/K
FLEX_A = 0.8          # Å, pore flexibility 보정
THICK_M = 100e-9      # 막 두께(임의 스케일) — permeance 산정 시 사용
E_D_SOL = 1.8e4       # 용해 확산 활성화에너지(기본값, J/mol)
E_D_SRF = 9.0e3       # 표면 확산 활성화에너지(기본값, J/mol)

# 대표 물성 (질량, 운동지름[Å], 모세관 응축 감도 스케일)
GASES = {
    "H2":  {"M": 2.016,  "d": 2.89, "cap": 0.10},
    "D2":  {"M": 4.028,  "d": 2.89, "cap": 0.10},
    "He":  {"M": 4.003,  "d": 2.60, "cap": 0.08},
    "N2":  {"M": 28.013, "d": 3.64, "cap": 0.20},
    "O2":  {"M": 31.998, "d": 3.46, "cap": 0.20},
    "CO2": {"M": 44.01,  "d": 3.30, "cap": 0.60},
    "CH4": {"M": 16.043, "d": 3.80, "cap": 0.50},
    "C2H6":{"M": 30.070, "d": 4.44, "cap": 0.70},
    "C3H6":{"M": 42.081, "d": 4.00, "cap": 0.80},
    "C3H8":{"M": 44.097, "d": 4.30, "cap": 0.85},
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

# ----------------------------- Synced controls ---------------------------
def slider_with_box(label, minv, maxv, default, step, key):
    """
    슬라이더 + 숫자입력칸 동기화:
    - session_state[f"{key}_val"]를 단일 소스로 사용
    - 두 위젯은 서로의 변경을 즉시 반영
    """
    ss = st.session_state
    vkey = f"{key}_val"
    skey = f"{key}_slider"
    bkey = f"{key}_box"

    if vkey not in ss:
        ss[vkey] = float(default)
    if skey not in ss:
        ss[skey] = float(default)
    if bkey not in ss:
        ss[bkey] = float(default)

    def _sync_from_slider():
        ss[vkey] = float(ss[skey])
        ss[bkey] = float(ss[skey])

    def _sync_from_box():
        ss[vkey] = float(ss[bkey])
        ss[skey] = float(ss[bkey])

    c1, c2 = st.columns([4, 1])
    c1.slider(
        label,
        min_value=float(minv), max_value=float(maxv),
        value=float(ss[vkey]),
        step=float(step),
        key=skey,
        on_change=_sync_from_slider,
    )
    c2.number_input(
        " ",
        min_value=float(minv), max_value=float(maxv),
        value=float(ss[vkey]),
        step=float(step),
        key=bkey,
        label_visibility="collapsed",
        on_change=_sync_from_box,
    )
    return float(ss[vkey])

# ----------------------------- Core models -----------------------------
def dsl_loading_series(T, P_bar, relP, Q1_kJ, Q2_kJ, q1_mmolg, q2_mmolg, b0=1e-4):
    """
    Double-site Langmuir: q = q1*b1P/(1+b1P) + q2*b2P/(1+b2P)
    - Q: kJ/mol (양수), q: mmol/g
    """
    P0 = P_bar * 1e5
    Q1 = max(Q1_kJ, 0.0) * 1e3
    Q2 = max(Q2_kJ, 0.0) * 1e3
    b1 = b0 * np.exp(Q1 / (R * T))
    b2 = b0 * np.exp(Q2 / (R * T))
    P = np.clip(relP, 1e-9, 1.0) * P0
    theta1 = (b1 * P) / (1.0 + b1 * P)
    theta2 = (b2 * P) / (1.0 + b2 * P)
    return q1_mmolg * theta1 + q2_mmolg * theta2  # mmol/g

def proxy_knudsen(pore_d_nm, M):
    rp = np.maximum(pore_d_nm / 2.0, 1e-9)
    return rp / np.sqrt(M)

def proxy_sieving(pore_d_nm, d_ang, T):
    p_eff = pore_d_nm * 10.0 + FLEX_A  # nm→Å + 유연성
    beta = 2.0
    Ea = 9.0e3
    x = 1.0 - (d_ang / p_eff) ** 2
    x = np.clip(x, 0.0, None)
    return (x ** beta) * np.exp(-Ea / (R * T))

def proxy_surface(pore_d_nm, T, loading_mmolg):
    Ds = np.exp(-E_D_SRF / (R * T))
    return Ds * loading_mmolg * np.maximum(1.0 / np.maximum(pore_d_nm, 1e-9), 1e-9)

def proxy_capillary(pore_d_nm, T, relP, cap_scale):
    r = np.maximum(pore_d_nm / 2.0, 1e-9)
    relP_th = np.exp(-120.0 / (r * T))  # Kelvin-like threshold (질감용)
    act = np.clip(relP - relP_th, 0.0, None)
    return cap_scale * np.sqrt(r) * act

def proxy_solution(T, loading_mmolg, M):
    D = np.exp(-E_D_SOL / (R * T)) / np.sqrt(M)
    return D * np.maximum(loading_mmolg, 0.0)

def proxies_all_for_gas(gas, other_gas, T, P_bar, pore_d_nm, relP, loading_self, loading_other):
    """각 relP에서 Gas의 6개 메커니즘 proxy + Blocked (전부 배열)"""
    M = GASES[gas]["M"]
    d = GASES[gas]["d"]
    cap_scale = GASES[gas]["cap"]

    siev = proxy_sieving(pore_d_nm, d, T) * np.ones_like(relP)
    knud = proxy_knudsen(pore_d_nm, M) * np.ones_like(relP)
    surf = proxy_surface(pore_d_nm, T, loading_self)            # array
    capp = proxy_capillary(pore_d_nm, T, relP, cap_scale)        # array
    solu = proxy_solution(T, loading_self, M)                    # array

    # Blocked: p_eff <= min(d_gas, d_other) → 1.0 배열, 아니면 0.0 배열
    d_min = min(GASES[gas]["d"], GASES[other_gas]["d"])
    p_eff = pore_d_nm * 10.0 + FLEX_A
    blocked_scalar = 1.0 if (p_eff <= d_min) else 0.0
    blocked = blocked_scalar * np.ones_like(relP, dtype=float)

    mask = (1.0 - blocked)
    siev *= mask
    knud *= mask
    surf *= mask
    capp *= mask
    solu *= mask

    return {
        "Blocked": blocked,
        "Sieving": siev,
        "Knudsen": knud,
        "Surface": surf,
        "Capillary": capp,
        "Solution": solu,
    }

def pick_mechanism_from_proxies(prox_dict):
    arr = np.vstack([prox_dict[m] for m in MECHS])
    idx = np.argmax(arr, axis=0)
    return np.array(MECHS)[idx]

def permeance_from_proxies(prox_dict, loading_self, loading_other):
    """
    합성 permeance:
    - (각 지점 최대 proxy) * 조성가중(DSL loading 비율) / 두께
    - Blocked면 mask에 의해 최대값도 0 → Permeance=0
    """
    arr = np.vstack([prox_dict[m] for m in MECHS])
    max_proxy = arr.max(axis=0)
    y = loading_self / (loading_self + loading_other + 1e-30)
    return (max_proxy * y) / THICK_M

# ----------------------------- UI -----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Global conditions")

    T = slider_with_box("Temperature, T (K)", 10.0, 600.0, 300.0, 1.0, key="T")
    Pbar = slider_with_box("Total pressure, P (bar)", 0.1, 10.0, 1.0, 0.1, key="P")
    d_nm = slider_with_box("Pore diameter (nm)", 0.01, 50.0, 0.34, 0.01, key="pore")

    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("C3H8"))

    st.subheader("Double-site Langmuir parameters — Gas 1")
    st.caption("Qst: 0–100 kJ/mol, q: 0–5 mmol/g  (슬라이더와 숫자칸 완전 동기화)")
    Q11 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 27.0, 0.1, key="g1_Q1")
    Q12 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 18.0, 0.1, key="g1_Q2")
    q11 = slider_with_box("q1 (mmol/g)", 0.0, 5.0, 0.70, 0.01, key="g1_q1")
    q12 = slider_with_box("q2 (mmol/g)", 0.0, 5.0, 0.30, 0.01, key="g1_q2")

    st.subheader("Double-site Langmuir parameters — Gas 2")
    Q21 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 26.5, 0.1, key="g2_Q1")
    Q22 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 17.0, 0.1, key="g2_Q2")
    q21 = slider_with_box("q1 (mmol/g)", 0.0, 5.0, 0.70, 0.01, key="g2_q1")
    q22 = slider_with_box("q2 (mmol/g)", 0.0, 5.0, 0.30, 0.01, key="g2_q2")

with right:
    st.subheader("Results")

    relP = np.linspace(0.01, 0.99, 500)

    # DSL loading (mmol/g)
    load1 = dsl_loading_series(T, Pbar, relP, Q11, Q12, q11, q12)
    load2 = dsl_loading_series(T, Pbar, relP, Q21, Q22, q21, q22)

    # proxies & mechanism (Gas1 기준으로 밴드)
    prox1 = proxies_all_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
    prox2 = proxies_all_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)
    mech_names = pick_mechanism_from_proxies(prox1)

    # -------------- Mechanism band (밴드/범례 분리 렌더링) --------------
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mech_names])[None, :, :]

    # (1) 밴드만
    figB, axB = plt.subplots(figsize=(8, 1.4))
    axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
    axB.set_yticks([])
    axB.set_xlim(0, 1)
    axB.set_xlabel(r"Relative pressure, $P/P_0$ (–)", labelpad=10)
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    st.pyplot(figB)
    plt.close(figB)

    # (2) 범례만
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
    leg_fig, leg_ax = plt.subplots(figsize=(8, 1.1))
    leg_ax.axis("off")
    leg = leg_fig.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=True,
        borderpad=0.8,
        handlelength=1.8,
        columnspacing=1.4,
    )
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_facecolor("white")
    plt.tight_layout()
    st.pyplot(leg_fig)
    plt.close(leg_fig)

    # -------------- Permeance & Selectivity --------------
    perm1 = permeance_from_proxies(prox1, load1, load2)
    perm2 = permeance_from_proxies(prox2, load2, load1)

    # Blocked 구간에서 선택도/그래프 NaN 방지
    denom = np.where(perm2 > 0, perm2, np.inf)
    sel = perm1 / denom
    sel = np.where(np.isfinite(sel), sel, 0.0)

    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(relP, perm1, label=f"Permeance {gas1}")
    ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
    ax1.set_ylabel(r"$\Pi$ (arb. units)")
    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (–)")
    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # -------------- Text summary --------------
    mid = len(relP) // 2
    summary = {m: prox1[m][mid] for m in MECHS}
    best_m = max(summary, key=summary.get)
    st.markdown(
        f"**Dominant mechanism at P/P0 ≈ {relP[mid]:.2f} (Gas1)**: `{best_m}`  \n"
        + " • ".join([f"{m}: {summary[m]:.3e}" for m in MECHS])
    )

st.caption("Note: qualitative proxies; permeance is scaled for interactive comparison (not absolute prediction).")
