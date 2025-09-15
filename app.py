# app.py — Membrane mechanism simulator (Streamlit)
# - 메커니즘 밴드는 '룰'이 아니라 각 메커니즘 proxy 값의 **실제 계산 결과로** 결정됩니다.
# - Permeance(G1/G2)와 Selectivity(G1/G2) vs 상대압(P/P0)
# - Double-site Langmuir(DSL) 흡착으로 loading 계산 → 표면/용해 확산에 반영

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import streamlit as st

# ----------------------------- App layout -----------------------------
st.set_page_config(page_title="Membrane Mechanisms Simulator", layout="wide")

st.title("Membrane Transport Mechanisms – data-driven simulator")

R = 8.314  # J/mol/K
FLEX_A = 0.8            # Å, pore flexibility 보정
THICK_M = 100e-9        # 막 두께(임의 스케일) — permeance 산정 시 사용
E_D_SOL = 1.8e4         # 용해-확산 확산활성화에너지(기본값, J/mol)
E_D_SRF = 9.0e3         # 표면확산 활성화에너지(기본값, J/mol)

# 기체 물성(대표 질량, 운동지름[Å], 캡릴러리 감도 스케일)
GASES = {
    "H2":  {"M": 2.016,  "d": 2.89, "cap": 0.10},
    "D2":  {"M": 4.028,  "d": 2.89, "cap": 0.10},
    "He":  {"M": 4.003,  "d": 2.60, "cap": 0.08},
    "N2":  {"M": 28.013,"d": 3.64, "cap": 0.20},
    "O2":  {"M": 31.998,"d": 3.46, "cap": 0.20},
    "CO2": {"M": 44.01, "d": 3.30, "cap": 0.60},
    "CH4": {"M": 16.043,"d": 3.80, "cap": 0.50},
    "C2H6":{"M": 30.070,"d": 4.44, "cap": 0.70},
    "C3H6":{"M": 42.081,"d": 4.00, "cap": 0.80},
    "C3H8":{"M": 44.097,"d": 4.30, "cap": 0.85},
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

# ----------------------------- Core models -----------------------------
def mean_free_path_nm(T, P_bar, d_ang):
    """평균자유행로(기체 혼합 대표지름) nm"""
    kB = 1.380649e-23
    P = P_bar * 1e5
    d = d_ang * 1e-10
    lam_m = kB*T / (np.sqrt(2)*np.pi*d*d*P)
    return lam_m*1e9

def dsl_loading_series(T, P_bar, relP, Q1_kJ, Q2_kJ, q1_mmolg, q2_mmolg, b0=1e-4):
    """
    Double-site Langmuir: q = q1*b1P/(1+b1P) + q2*b2P/(1+b2P)
    - Q: kJ/mol (positivized), q(mm0l/g)
    반환: relP와 동일 길이의 loading(mm0l/g)
    """
    P0 = P_bar * 1e5
    Q1 = max(Q1_kJ, 0.0)*1e3
    Q2 = max(Q2_kJ, 0.0)*1e3
    b1 = b0*np.exp(Q1/(R*T))
    b2 = b0*np.exp(Q2/(R*T))
    P = np.clip(relP, 1e-9, 1.0)*P0
    theta1 = (b1*P)/(1.0 + b1*P)
    theta2 = (b2*P)/(1.0 + b2*P)
    return q1_mmolg*theta1 + q2_mmolg*theta2  # mmol/g

# --- Intrinsic proxies (커지는 값이 지배 메커니즘) ---
def proxy_knudsen(pore_d_nm, M):
    rp = np.maximum(pore_d_nm/2.0, 1e-9)
    return rp/np.sqrt(M)

def proxy_sieving(pore_d_nm, d_ang, T):
    p_eff = pore_d_nm*10.0 + FLEX_A  # nm→Å + 유연성
    beta = 2.0
    Ea  = 9.0e3
    x = 1.0 - (d_ang/p_eff)**2
    x = np.clip(x, 0.0, None)
    return (x**beta) * np.exp(-Ea/(R*T))

def proxy_surface(pore_d_nm, T, loading_mmolg):
    # 표면확산: Ds(Arrhenius) * loading * (1/pore) — 스케일링
    Ds = np.exp(-E_D_SRF/(R*T))
    return Ds * loading_mmolg * np.maximum(1.0/np.maximum(pore_d_nm,1e-9), 1e-9)

def proxy_capillary(pore_d_nm, T, relP, cap_scale):
    # Kelvin-like 임계: relP_th = exp(-K/(r*T)) with K≈120 (임의 스케일)
    r = np.maximum(pore_d_nm/2.0, 1e-9)
    relP_th = np.exp(-120.0/(r*T))
    act = (relP - relP_th)  # >0 이면 응축 기여 시작
    act = np.clip(act, 0.0, None)
    return cap_scale*np.sqrt(r)*act

def proxy_solution(T, loading_mmolg, M):
    # P = D*S ~ exp(-E/RT)/sqrt(M) * loading
    D = np.exp(-E_D_SOL/(R*T))/np.sqrt(M)
    return D*np.maximum(loading_mmolg, 0.0)

def proxies_all_for_gas(gas, other_gas, T, P_bar, pore_d_nm, relP, loading_self, loading_other):
    """각 relP에서 Gas의 5개 메커니즘 proxy + Blocked"""
    M = GASES[gas]["M"]
    d = GASES[gas]["d"]
    cap_scale = GASES[gas]["cap"]

    # vector outputs
    siev = proxy_sieving(pore_d_nm, d, T)
    knud = proxy_knudsen(pore_d_nm, M)
    surf = proxy_surface(pore_d_nm, T, loading_self)
    capp = proxy_capillary(pore_d_nm, T, relP, cap_scale)
    solu = proxy_solution(T, loading_self, M)

    # Blocked: p_eff <= min(d1,d2) 이면 1, 아니면 0 (강제 차단)
    d_min = min(GASES[gas]["d"], GASES[other_gas]["d"])
    p_eff = pore_d_nm*10.0 + FLEX_A
    blocked = (p_eff <= d_min).astype(float)

    # Blocked면 나머지 proxy를 0으로 만들어 명확히 차단
    siev = siev * (1.0 - blocked)
    knud = knud * (1.0 - blocked)
    surf = surf * (1.0 - blocked)
    capp = capp * (1.0 - blocked)
    solu = solu * (1.0 - blocked)

    # dict of arrays
    return {
        "Blocked": blocked,
        "Sieving": siev,
        "Knudsen": knud,
        "Surface": surf,
        "Capillary": capp,
        "Solution": solu
    }

def pick_mechanism_from_proxies(prox_dict):
    """각 지점별 proxy가 최대인 메커니즘 이름 배열"""
    # stack [n_mech, n_points]
    arr = np.vstack([prox_dict[m] for m in MECHS])
    idx = np.argmax(arr, axis=0)
    return np.array(MECHS)[idx]

def permeance_from_proxies(prox_dict, loading_self, loading_other):
    """
    간단 합성: (가장 큰 proxy) * 조성가중 / 두께
    조성가중은 DSL loading 비율 사용
    """
    arr = np.vstack([prox_dict[m] for m in MECHS])
    max_proxy = arr.max(axis=0)
    y = loading_self/(loading_self + loading_other + 1e-30)
    return (max_proxy * y) / THICK_M  # (임의 스케일)

# ----------------------------- UI -----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Global conditions")
    T = st.slider("Temperature, T (K)", 10.0, 600.0, 300.0, 1.0)
    P_bar = st.slider("Total pressure, P (bar)", 0.1, 10.0, 1.0, 0.1)
    d_nm = st.slider("Pore diameter (nm)", 0.01, 50.0, 0.34, 0.01)

    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("C3H8"))

    st.subheader("Double-site Langmuir parameters (Gas 1)")
    st.caption("Qst: 0–100 kJ/mol, q: 0–5 mmol/g")
    Q11 = st.slider("Qst1 (kJ/mol) – Gas1", 0.0, 100.0, 27.0, 0.1)
    Q12 = st.slider("Qst2 (kJ/mol) – Gas1", 0.0, 100.0, 18.0, 0.1)
    q11 = st.slider("q1 (mmol/g) – Gas1", 0.0, 5.0, 0.70, 0.01)
    q12 = st.slider("q2 (mmol/g) – Gas1", 0.0, 5.0, 0.30, 0.01)

    st.subheader("Double-site Langmuir parameters (Gas 2)")
    Q21 = st.slider("Qst1 (kJ/mol) – Gas2", 0.0, 100.0, 26.5, 0.1)
    Q22 = st.slider("Qst2 (kJ/mol) – Gas2", 0.0, 100.0, 17.0, 0.1)
    q21 = st.slider("q1 (mmol/g) – Gas2", 0.0, 5.0, 0.70, 0.01)
    q22 = st.slider("q2 (mmol/g) – Gas2", 0.0, 5.0, 0.30, 0.01)

with right:
    st.subheader("Results")

    relP = np.linspace(0.01, 0.99, 500)

    # DSL loading (mmol/g)
    load1 = dsl_loading_series(T, P_bar, relP, Q11, Q12, q11, q12)
    load2 = dsl_loading_series(T, P_bar, relP, Q21, Q22, q21, q22)

    # proxies and best mechanism (Gas1 기준으로 밴드 그리기)
    prox1 = proxies_all_for_gas(gas1, gas2, T, P_bar, d_nm, relP, load1, load2)
    prox2 = proxies_all_for_gas(gas2, gas1, T, P_bar, d_nm, relP, load2, load1)
    mech_names = pick_mechanism_from_proxies(prox1)

    # ---- Mechanism band (vector color) ----
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mech_names])[None, :, :]
    figB, axB = plt.subplots(figsize=(8, 1.0))
    axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
    axB.set_yticks([])
    axB.set_xlim(0, 1)
    axB.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    handles = [plt.Rectangle((0,0),1,1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
    leg = axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.9),
                     ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")
    st.pyplot(figB); plt.close(figB)

    # ---- Permeance/Selectivity ----
    perm1 = permeance_from_proxies(prox1, load1, load2)
    perm2 = permeance_from_proxies(prox2, load2, load1)
    sel   = np.divide(perm1, perm2, out=np.zeros_like(perm1), where=(perm2>0))

    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(relP, perm1, label=f"Permeance {gas1}")
    ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
    ax1.set_ylabel(r"$\Pi$ (arb. units)")
    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (–)")
    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    # ---- Instant summary at mid-pressure ----
    mid = len(relP)//2
    summary = {m: prox1[m][mid] for m in MECHS}
    best_m = max(summary, key=summary.get)
    st.markdown(
        f"**Dominant mechanism at P/P0 ≈ {relP[mid]:.2f} (Gas1)**: `{best_m}`  \n"
        + " • ".join([f"{m}: {summary[m]:.3e}" for m in MECHS])
    )

st.caption("Note: This simulator is qualitative. Proxies and permeance are scaled for interactive comparison, not absolute prediction.")
