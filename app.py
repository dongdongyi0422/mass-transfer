# ===============================================================
# Unified Transport Simulator
# Gas membrane + Ion membrane + Drug transport
# UI 유지 / 오류 제거 / log_slider 안정화
# ===============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

# --------------------------------------------------
# Constants
# --------------------------------------------------
R  = 8.314462618
kB = 1.380649e-23
h  = 6.62607015e-34
NA = 6.02214076e23
F  = 96485.33212

GPU_UNIT = 3.35e-10
PI_TINY  = 1e-14


# --------------------------------------------------
# UI helpers
# --------------------------------------------------
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3, help=None):
    if key not in st.session_state:
        st.session_state[key] = float(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    fmt = f"%.{decimals}f"
    lab = f"{label} [{unit}]" if unit else label

    st.slider(
        lab, float(vmin), float(vmax),
        float(st.session_state[key]),
        float(vstep),
        key=f"{key}__s", format=fmt, help=help,
        on_change=_mark_s
    )

    st.number_input(
        "", float(vmin), float(vmax),
        float(st.session_state[key]), float(vstep),
        key=f"{key}__n", format=fmt,
        on_change=_mark_n
    )

    if st.session_state[f"{key}__who"] == "n":
        new = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = float(st.session_state[f"{key}__s"])
    else:
        new = st.session_state[key]

    st.session_state[key] = float(np.clip(new, vmin, vmax))
    return st.session_state[key]


def nudged_int(label, vmin, vmax, vstep, vinit, key):
    if key not in st.session_state:
        st.session_state[key] = int(vinit)
    if f"{key}__who" not in st.session_state:
        st.session_state[f"{key}__who"] = ""

    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"

    st.slider(
        label, int(vmin), int(vmax),
        int(st.session_state[key]),
        int(vstep),
        key=f"{key}__s",
        on_change=_mark_s
    )

    st.number_input(
        "", int(vmin), int(vmax),
        int(st.session_state[key]), int(vstep),
        key=f"{key}__n", on_change=_mark_n
    )

    if st.session_state[f"{key}__who"] == "n":
        new = int(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = int(st.session_state[f"{key}__s"])
    else:
        new = st.session_state[key]

    st.session_state[key] = int(np.clip(new, vmin, vmax))
    return st.session_state[key]


# --------------------------------------------------------------
# FIXED log_slider (오류 없이 작동)
# --------------------------------------------------------------
def log_slider(label, exp_min, exp_max, exp_step, exp_init, key, unit=""):

    # SAFE INIT
    if key not in st.session_state:
        st.session_state[key] = float(exp_init)
    if f"{key}_who" not in st.session_state:
        st.session_state[f"{key}_who"] = ""

    def _mark_s(): st.session_state[f"{key}_who"] = "s"
    def _mark_n(): st.session_state[f"{key}_who"] = "n"

    lab = f"{label} ({unit})" if unit else label

    st.slider(
        lab,
        exp_min, exp_max,
        float(st.session_state[key]),
        step=exp_step,
        key=f"{key}_s", format="%.2f",
        on_change=_mark_s
    )

    st.number_input(
        "exp (10^x)",
        exp_min, exp_max,
        float(st.session_state[key]),
        step=exp_step,
        key=f"{key}_n", format="%.2f",
        on_change=_mark_n
    )

    who = st.session_state.get(f"{key}_who", "")
    if who == "s":
        new = float(st.session_state[f"{key}_s"])
    elif who == "n":
        new = float(st.session_state[f"{key}_n"])
    else:
        new = float(st.session_state[key])

    st.session_state[key] = new
    return 10 ** new


# --------------------------------------------------------------
# GAS MEMBRANE (이상적 모델)
# --------------------------------------------------------------
GAS_PARAMS = {
    "H2":  {"M":2.016e-3,  "d":2.89, "Ea_s":8e3},
    "D2":  {"M":4.028e-3,  "d":2.89, "Ea_s":8.5e3},
    "He":  {"M":4.003e-3,  "d":2.60, "Ea_s":6e3},
    "N2":  {"M":28.013e-3, "d":3.64, "Ea_s":9e3},
    "O2":  {"M":31.998e-3, "d":3.46, "Ea_s":9e3},
    "CO2": {"M":44.01e-3,  "d":3.30, "Ea_s":9.5e3},
    "CH4": {"M":16.043e-3, "d":3.80, "Ea_s":9.5e3},
    "C3H6":{"M":42.081e-3, "d":4.00, "Ea_s":1.05e4},
    "C3H8":{"M":44.097e-3, "d":4.30, "Ea_s":1.05e4},
}


def de_broglie(T, M):
    m = M / NA
    return h / np.sqrt(2 * np.pi * m * kB * T)


def effective_diameter(gas, T, alpha):
    dA = GAS_PARAMS[gas]["d"]
    lam = de_broglie(T, GAS_PARAMS[gas]["M"]) * 1e10
    return max(0.5 * dA, dA - alpha * lam)


def alpha_auto(T, d_nm):
    a = 0.05 * np.sqrt(300 / max(T, 1e-6))
    return float(np.clip(a, 0, 0.6))


# Ideal model constants
DAMP_FACTOR = 1e-3
SURFACE_SCALE = 2e6
SOLUTION_SCALE = 2e6
CAPILLARY_D_MIN = 1.5
CAPILLARY_RP_MIN = 0.4


def Pi_sieving(d_nm, gas, T, L_m, d_eff):
    pA = d_nm * 10
    x = max(1 - (d_eff/pA)**2, 0)
    f = x**2
    return max(1e-6 * f * np.exp(-6000 / (R*T)), PI_TINY)


def Pi_knudsen(d_nm, T, M, L_m):
    r = d_nm * 1e-9 / 2
    Dk = (2/3)*r*np.sqrt((8*R*T)/(np.pi*M))
    Pi = Dk/(L_m*R*T)
    if d_nm <= 0.5:
        Pi *= DAMP_FACTOR
    return max(Pi, PI_TINY)


def Pi_surface(d_nm, gas, T, L_m, dqdp):
    Ds = 1e-10 * np.exp(-GAS_PARAMS[gas]["Ea_s"]/(R*T))
    return max((Ds/L_m) * (dqdp*SURFACE_SCALE), PI_TINY)


def Pi_solution(gas, T, L_m, dqdp):
    Dsol = 3e-9 * np.exp(-1.8e4/(R*T))
    return max((Dsol/L_m)*(dqdp*SOLUTION_SCALE), PI_TINY)


def Pi_capillary(d_nm, rp, L_m):
    if d_nm < CAPILLARY_D_MIN or rp < CAPILLARY_RP_MIN:
        return 0.0
    r = d_nm*1e-9/2
    return max(1e-6*np.sqrt(r)/L_m, 0.0)


def DSL(gas, T, P_bar, rp, q1, q2, b1, b2):
    P = rp * P_bar * 1e5
    th1 = (b1 * P)/(1+b1*P)
    th2 = (b2 * P)/(1+b2*P)
    q = (q1*1e-3*1e3)*th1 + (q2*1e-3*1e3)*th2
    dqdp = (q1*1e-3*1e3)*(b1/(1+b1*P)**2) + (q2*1e-3*1e3)*(b2/(1+b2*P)**2)
    return q/1e3, dqdp


def run_gas():
    st.header("Gas Membrane")

    T    = nudged_slider("T",10,600,1,300,"Tg")
    Pbar = nudged_slider("P(bar)",0.1,10,0.1,1,"Pg")
    d_nm = nudged_slider("Pore size(nm)",0.01,10,0.01,0.36,"dg")
    L_nm = nudged_slider("Thickness(nm)",5,20000,1,100,"Lg")

    gases=list(GAS_PARAMS.keys())
    g1=st.sidebar.selectbox("Gas1",gases,index=gases.index("CO2"))
    g2=st.sidebar.selectbox("Gas2",gases,index=gases.index("CH4"))

    alpha=alpha_auto(T,d_nm)
    d_eff=effective_diameter(g1,T,alpha)

    q11=nudged_slider("q1 Gas1",0,100,0.01,0.7,"q11")
    q12=nudged_slider("q2 Gas1",0,100,0.01,0.3,"q12")
    b11=nudged_slider("b1 Gas1",1e-10,1e-3,1e-8,1e-5,"b11",decimals=8)
    b12=nudged_slider("b2 Gas1",1e-10,1e-3,1e-8,5e-6,"b12",decimals=8)

    q21=nudged_slider("q1 Gas2",0,100,0.01,0.7,"q21")
    q22=nudged_slider("q2 Gas2",0,100,0.01,0.3,"q22")
    b21=nudged_slider("b1 Gas2",1e-10,1e-3,1e-8,1e-5,"b21",decimals=8)
    b22=nudged_slider("b2 Gas2",1e-10,1e-3,1e-8,5e-6,"b22",decimals=8)

    rp=np.linspace(0.01,0.99,400)
    L_m=L_nm*1e-9

    q1_vec=np.zeros_like(rp)
    dq1_vec=np.zeros_like(rp)
    q2_vec=np.zeros_like(rp)
    dq2_vec=np.zeros_like(rp)

    for i,r in enumerate(rp):
        q1_vec[i],dq1_vec[i]=DSL(g1,T,Pbar,r,q11,q12,b11,b12)
        q2_vec[i],dq2_vec[i]=DSL(g2,T,Pbar,r,q21,q22,b21,b22)

    Pi1=np.zeros_like(rp)
    Pi2=np.zeros_like(rp)

    for i,r in enumerate(rp):
        Pi1_intr=[
            Pi_sieving(d_nm,g1,T,L_m,d_eff),
            Pi_knudsen(d_nm,T,GAS_PARAMS[g1]["M"],L_m),
            Pi_surface(d_nm,g1,T,L_m,dq1_vec[i]),
            Pi_solution(g1,T,L_m,dq1_vec[i]),
            Pi_capillary(d_nm,r,L_m)
        ]
        Pi2_intr=[
            Pi_sieving(d_nm,g2,T,L_m,effective_diameter(g2,T,alpha)),
            Pi_knudsen(d_nm,T,GAS_PARAMS[g2]["M"],L_m),
            Pi_surface(d_nm,g2,T,L_m,dq2_vec[i]),
            Pi_solution(g2,T,L_m,dq2_vec[i]),
            Pi_capillary(d_nm,r,L_m)
        ]
        Pi1[i]=sum(Pi1_intr)
        Pi2[i]=sum(Pi2_intr)

    Pi1_gpu=Pi1/GPU_UNIT
    Pi2_gpu=Pi2/GPU_UNIT
    Sel=np.divide(Pi1,Pi2,out=np.zeros_like(Pi1),where=Pi2>0)

    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(rp,Pi1_gpu,label=g1)
    ax.plot(rp,Pi2_gpu,"--",label=g2)
    ax.set_ylabel("Permeance (GPU)")
    ax.set_xlabel("P/P0")
    ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close()

    fig2,ax2=plt.subplots(figsize=(8,3))
    ax2.plot(rp,Sel,label=f"{g1}/{g2}")
    ax2.set_ylabel("Selectivity")
    ax2.set_xlabel("P/P0")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close()

    # ===============================================================
# ION MEMBRANE
# ===============================================================

ION_DB = {
    "Na+":      {"z": +1, "D": 1.33e-9},
    "K+":       {"z": +1, "D": 1.96e-9},
    "Li+":      {"z": +1, "D": 1.03e-9},
    "Ca2+":     {"z": +2, "D": 0.79e-9},
    "Mg2+":     {"z": +2, "D": 0.706e-9},
    "Cl-":      {"z": -1, "D": 2.03e-9},
    "NO3-":     {"z": -1, "D": 1.90e-9},
    "SO4^2-":   {"z": -2, "D": 1.065e-9},
    "HCO3-":    {"z": -1, "D": 1.18e-9},
    "Acetate-": {"z": -1, "D": 1.09e-9},
}

def eta_water(T_K: float) -> float:
    """물 점도 근사 (Pa·s). VTF식."""
    A, B, C = 2.414e-5, 247.8, 140.0
    T_C = T_K - 273.15
    return A * (10.0 ** (B / (T_C - C)))

def D_temp(D_ref: float, T_ref: float, T: float) -> float:
    """Stokes–Einstein 기반 D(T) 스케일링."""
    return D_ref * (T / T_ref) * (eta_water(T_ref) / eta_water(T))

def donnan_potential(c_bulk, z_map, K_map, Xf, T):
    """
    다성분 Donnan 전위:
        sum z_i*C_m,i + Xf = 0
        C_m,i = K_i C_b,i exp(-z_i F psi / RT)
    """
    psi = 0.0
    for _ in range(80):
        t  = -F * psi / (R * T)
        g  = Xf
        dg = 0.0
        for sp, cb in c_bulk.items():
            z  = z_map[sp]
            Ki = K_map.get(sp, 1.0)
            cm = Ki * cb * np.exp(-z * t)
            g  += z * cm
            dg += (F / (R * T)) * (z**2) * cm
        step = -g / (dg + 1e-30)
        psi += step
        if abs(step) < 1e-12:
            break
    return psi

def NP_flux_constfield(D_eff, z, Cin_m, Cout_m, dphi_mem, L, T, v_s=0.0):
    """
    1D steady Nernst–Planck (상수 계수):
    J = -D dC/dx - (z D F / RT) C_avg dphi/dx + v_s C_avg
    """
    L = max(float(L), 1e-12)
    Cavg = 0.5 * (Cin_m + Cout_m)
    dC   = float(Cout_m - Cin_m)
    term_d = -D_eff * (dC / L)
    term_e = -(z * D_eff * F / (R*T)) * Cavg * (dphi_mem / L)
    term_c = v_s * Cavg
    return term_d + term_e + term_c

def run_ion():
    st.header("Ion-Exchange / NF Membrane")

    # ---- Sidebar: 막 / 운전 조건 ----
    with st.sidebar:
        st.subheader("Membrane & Operation")
        T   = nudged_slider("T (K)", 273.15, 360.0, 0.5, 298.15, key="Ti", unit="K")
        Lnm = nudged_slider("Active layer thickness", 10.0, 5000.0, 1.0, 200.0, key="Lnm_i", unit="nm")
        eps = nudged_slider("Porosity ε", 0.05, 0.8, 0.01, 0.30, key="eps_i", unit="–")
        tau = nudged_slider("Tortuosity τ", 1.0, 5.0, 0.1, 2.0, key="tau_i", unit="–")
        Cf  = nudged_slider("Fixed charge C_f", -3000.0, 3000.0, 10.0, -500.0, key="Cf_i", unit="mol/m³")
        dV  = nudged_slider("Applied potential ΔV", -0.3, 0.3, 0.005, 0.05, key="dV_i", unit="V")
        v_s = nudged_slider("Solvent velocity v_s", -1e-4, 1e-4, 1e-6, 0.0, key="v_s_i", unit="m/s")

        st.subheader("Species set")
        all_cations = [k for k in ION_DB if ION_DB[k]["z"] > 0]
        all_anions  = [k for k in ION_DB if ION_DB[k]["z"] < 0]
        sel_cat = st.multiselect("Cations", all_cations, default=["Na+","K+"], key="sel_cat_i")
        sel_an  = st.multiselect("Anions",  all_anions,  default=["Cl-","NO3-"], key="sel_an_i")

        st.subheader("Bulk concentrations (mol/m³)")
        c_feed = {}; c_perm = {}
        for sp in sel_cat + sel_an:
            c_feed[sp] = nudged_slider(f"{sp} feed", 0.0, 2000.0, 1.0, 100.0, key=f"cf_{sp}", unit="mol/m³")
            c_perm[sp] = nudged_slider(f"{sp} perm", 0.0, 2000.0, 1.0, 10.0,  key=f"cp_{sp}", unit="mol/m³")

        st.subheader("Partition K & Diffusivity D")
        autoKD = st.checkbox("Auto D(T) via water viscosity (K=1)", value=True, key="autoKD_i")

    species = sel_cat + sel_an
    if not species:
        st.warning("최소 한 개 이상의 이온을 선택해 주세요.")
        return

    # ---- Partition & Diffusivity ----
    if autoKD:
        K_map = {sp: 1.0 for sp in species}
        D_map = {sp: D_temp(ION_DB[sp]["D"], 298.15, T) for sp in species}
    else:
        K_map = {}
        D_map = {}
        with st.sidebar:
            for sp in species:
                K_map[sp] = nudged_slider(f"K {sp}", 0.01, 10.0, 0.01, 1.0, key=f"K_{sp}")
                D_map[sp] = log_slider(f"D {sp}", -11.0, -8.0, 0.1, np.log10(ION_DB[sp]["D"]), key=f"D_{sp}", unit="m²/s")

    # ---- Core 계산 ----
    L = Lnm * 1e-9
    z_map = {sp: ION_DB[sp]["z"] for sp in species}
    D_eff = {sp: D_map[sp] * eps / max(tau, 1e-9) for sp in species}

    psi_f = donnan_potential(c_feed, z_map, K_map, Cf, T)
    psi_p = donnan_potential(c_perm, z_map, K_map, Cf, T)

    Cm_f = {sp: K_map.get(sp,1.0)*c_feed[sp]*np.exp(-(z_map[sp]*F*psi_f)/(R*T)) for sp in species}
    Cm_p = {sp: K_map.get(sp,1.0)*c_perm[sp]*np.exp(-(z_map[sp]*F*psi_p)/(R*T)) for sp in species}

    dphi_mem = float(dV) - float(psi_p - psi_f)

    J = {}
    for sp in species:
        J[sp] = NP_flux_constfield(D_eff[sp], z_map[sp],
                                   Cm_f[sp], Cm_p[sp],
                                   dphi_mem, L, T, v_s=v_s)
    i_net = F * sum(z_map[sp]*J[sp] for sp in species)

    # ---- 출력 ----
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Potentials")
        st.metric("ψ_feed (V)", f"{psi_f:.3e}")
        st.metric("ψ_perm (V)", f"{psi_p:.3e}")
        st.metric("Δφ_mem (V)", f"{dphi_mem:.3e}")
        st.subheader("Net current")
        st.metric("i_net (A/m²)", f"{i_net:.3e}")

    with col2:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(range(len(species)), [J[sp] for sp in species])
        ax.set_xticks(range(len(species)))
        ax.set_xticklabels(species)
        ax.set_ylabel("Flux J (mol m⁻² s⁻¹)")
        ax.grid(True, axis="y")
        st.pyplot(fig); plt.close(fig)


# ===============================================================
# DRUG IN VESSEL (1D ADR, transient)
# ===============================================================
DRUG_DB = {
    "Caffeine":      {"Db": 6.5e-10, "Pv": 2.0e-6, "kel": 1.0e-4},
    "Acetaminophen": {"Db": 6.0e-10, "Pv": 1.5e-6, "kel": 8.0e-5},
    "Ibuprofen":     {"Db": 5.0e-10, "Pv": 8.0e-7, "kel": 1.0e-4},
    "Insulin":       {"Db": 1.8e-10, "Pv": 5.0e-8, "kel": 3.0e-5},
}

def run_drug():
    st.header("Drug Transport in a Vessel (1D ADR)")

    with st.sidebar:
        st.subheader("Geometry & Flow")
        Rv_um = nudged_slider("Vessel radius", 2.0, 100.0, 1.0, 4.0, key="Rv_um_v", unit="μm")
        L_mm  = nudged_slider("Segment length", 1.0, 200.0, 1.0, 20.0, key="Lv_mm_v", unit="mm")
        U     = nudged_slider("Mean velocity U", 0.1e-3, 20e-3, 0.1e-3, 1.0e-3, key="U_v", unit="m/s")

        st.subheader("Drug & Kinetics")
        drug = st.selectbox("Drug", list(DRUG_DB.keys()), index=0, key="drug_v")

        st.subheader("Thermophysical (for Db auto)")
        T_body = nudged_slider("Temperature", 293.15, 312.15, 0.5, 310.15, key="Tdrug_v", unit="K")
        auto_Db = st.checkbox("Auto Db from T & viscosity", value=True, key="autoDb_v")

        if auto_Db:
            Db = D_temp(DRUG_DB[drug]["Db"], 298.15, T_body)
            st.caption(f"D_b auto: {Db:.3e} m²/s  (ref {DRUG_DB[drug]['Db']:.3e} @25°C)")
        else:
            Db = log_slider("D_b", -12.0, -8.0, 0.1, np.log10(DRUG_DB[drug]["Db"]), key="Db_v", unit="m²/s")

        Pv  = log_slider("P_v",   -9.0, -5.0, 0.1, np.log10(DRUG_DB[drug]["Pv"]), key="Pv_v",  unit="m/s")
        kel = log_slider("k_elim",-6.0, -2.0, 0.1, np.log10(DRUG_DB[drug]["kel"]), key="kel_v", unit="s⁻¹")

        st.subheader("Inlet profile")
        C0      = nudged_slider("Reference conc. C₀", 0.0, 5.0, 0.01, 1.0, key="C0_v", unit="mol/m³")
        profile = st.selectbox("Profile", ["Bolus (Gaussian)", "Constant infusion"], index=0, key="pulse_v")
        t_end   = nudged_slider("Sim time", 0.1, 600.0, 0.1, 60.0, key="tend_v", unit="s")
        dt      = nudged_slider("Δt", 1e-3, 0.5, 1e-3, 0.01, key="dt_v", unit="s")
        Nx      = nudged_int("Grid Nx", 50, 600, 10, 200, key="Nx_v")

    # ---- Grid & numbers ----
    Rv = Rv_um * 1e-6
    L  = L_mm  * 1e-3
    x  = np.linspace(0, L, Nx)
    if len(x) < 2:
        st.error("Nx too small."); return
    dx = x[1] - x[0]

    t  = np.arange(0.0, t_end+dt, dt)
    if len(t) < 2:
        st.error("Simulation time too short."); return

    Pe_r = U*Rv/Db
    Deff = Db*(1.0 + (Pe_r**2)/192.0)
    k_leak = 2.0*Pv/max(Rv, 1e-12)

    if profile == "Bolus (Gaussian)":
        t0   = 0.2*t_end
        sig  = 0.05*t_end if t_end>0 else 1.0
        Cin_t = C0*np.exp(-((t-t0)**2)/(2*sig**2))
    else:
        Cin_t = C0*np.ones_like(t)

    C = np.zeros((len(t), Nx), dtype=float)
    C[0,:] = 0.0

    lam_a = U*dt/dx
    lam_d = Deff*dt/(dx*dx)
    if lam_a > 1.0:
        st.warning(f"Advection CFL>1 (UΔt/Δx={lam_a:.2f}).")
    if lam_d > 0.5:
        st.warning(f"Diffusion number>0.5 (D_effΔt/Δx²={lam_d:.2f}).")

    for n in range(1, len(t)):
        Cn  = C[n-1,:].copy()
        Cnp = Cn.copy()
        Cnp[0] = Cin_t[n]

        adv   = -lam_a*(Cn[1:] - Cn[:-1])
        dif   = lam_d*(Cn[2:] - 2*Cn[1:-1] + Cn[:-2])
        react = -dt*(k_leak + kel)*Cn[1:-1]

        Cnp[1:-1] = Cn[1:-1] + adv[:-1] + dif + react
        Cnp[-1]   = Cnp[-2]
        C[n,:]    = np.maximum(Cnp, 0.0)

    Cout_t = C[:, -1]

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Key numbers")
        st.metric("Pe_r = U·R/D_b", f"{Pe_r:.2f}")
        st.metric("D_eff",          f"{Deff:.3e} m²/s")
        st.metric("k_leak = 2P_v/R",f"{k_leak:.3e} s⁻¹")

    with col2:
        fig, ax = plt.subplots(figsize=(8,3))
        im = ax.imshow(C.T, aspect='auto', origin='lower',
                       extent=(t[0], t[-1], x[0]*1e3, x[-1]*1e3))
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (mm)")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("C (mol/m³)")
        st.pyplot(fig); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(t, Cout_t, label="Outlet")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("C_out (mol/m³)")
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2); plt.close(fig2)


# ===============================================================
# MAIN APP SHELL
# ===============================================================
st.title("Unified Transport Simulators")

mode_main = st.sidebar.radio(
    "Select simulation",
    ["Gas membrane", "Ion membrane", "Drug in vessel"],
    index=0
)

if mode_main == "Gas membrane":
    run_gas()
elif mode_main == "Ion membrane":
    run_ion()
else:
    run_drug()

