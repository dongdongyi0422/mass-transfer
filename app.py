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

# ===== Utilities for TIME (transient LDF) =====
def pressure_schedule_series(t, P0_bar, ramp, tau):
    """Return P_bar(t) in bar for a given schedule."""
    if isinstance(ramp, str) and ramp.lower().startswith("step"):
        return np.full_like(t, float(P0_bar), dtype=float)
    # exponential ramp: P(t) = P0*(1 - exp(-t/tau))
    tau = max(float(tau), 1e-9)
    return float(P0_bar) * (1.0 - np.exp(-t/tau))

# --- TIME 모드에서 (ldf_evolve_q 앞쪽) 콜백을 이렇게 고치세요 ---

def qeq_slope_cb_g1(Pbar_scalar: float):
    # Pbar_scalar [bar] -> relP in [1e-6, 0.9999]
    relP_scalar = float(np.clip(Pbar_scalar / float(Pbar), 1e-6, 0.9999))
    qv, dv = dsl_loading_and_slope_b(
        gas1, T, Pbar,
        np.array([relP_scalar], dtype=float),
        q11, q12, b11, b12
    )
    return float(qv[0]), float(dv[0])

def qeq_slope_cb_g2(Pbar_scalar: float):
    relP_scalar = float(np.clip(Pbar_scalar / float(Pbar), 1e-6, 0.9999))
    qv, dv = dsl_loading_and_slope_b(
        gas2, T, Pbar,
        np.array([relP_scalar], dtype=float),
        q21, q22, b21, b22
    )
    return float(qv[0]), float(dv[0])


def ldf_evolve_q(t, P_bar_t, q_eq_fn, kLDF, q0=0.0):
    """
    LDF integrator: dq/dt = k_LDF * (q*(P) - q)
    q_eq_fn(P_bar) -> (q_eq[mmol/g], dqdp_eq[mol/kg/Pa])
    """
    q_dyn = np.zeros_like(t, dtype=float)
    dqdp_series = np.zeros_like(t, dtype=float)
    q = float(q0)
    for i in range(len(t)):
        Pbar_i = float(P_bar_t[i])
        q_eq_i, dqdp_i = q_eq_fn(Pbar_i)
        dqdp_series[i] = dqdp_i
        if i > 0:
            dt = float(t[i] - t[i-1])
            q += dt * float(kLDF) * (q_eq_i - q)
        q_dyn[i] = q
    return q_dyn, dqdp_series
# ===== end utilities =====

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

def pressure_schedule_series(t, P0_bar, ramp, tau):
        """t[s]에서의 절대압력 [bar] 시계열 생성"""
        if ramp.startswith("Step"):
            P_bar_t = np.full_like(t, float(P0_bar))
        else:  # Exp ramp
            P_bar_t = P0_bar*(1.0 - np.exp(-t/float(tau)))
        return P_bar_t

def ldf_evolve_q(t, P_bar_t, q_eq_fn, kLDF, q0=0.0):
        """
        LDF: dq/dt = k*(q*(P) - q)
        q_eq_fn(P_bar)->(q_eq, dqdp_eq) 를 제공해야 함. (단위: q[mmol/g], dqdp[mol/kg/Pa])
        반환: q_dyn[mmol/g], dqdp_eq_series[mol/kg/Pa]  (같은 길이)
        """
        q_dyn = np.zeros_like(t, float)
        dqdp_series = np.zeros_like(t, float)
        q = float(q0)
        for i in range(len(t)):
            Pbar_i = float(P_bar_t[i])
            # q_eq (mmol/g), dqdp_eq (mol/kg/Pa) at current pressure
            q_eq_i, dqdp_i = q_eq_fn(Pbar_i)
            dqdp_series[i] = dqdp_i
            # Explicit Euler (안정성을 위해 dt*kLDF <= ~0.2 권장)
            if i == 0:
                q = q  # initial
            else:
                dt = float(t[i]-t[i-1])
                q += dt * kLDF * (q_eq_i - q)
            q_dyn[i] = q
        return q_dyn, dqdp_series

# -------- Heuristic mechanism weights (0~1), sum to 1 --------
def mechanism_weights(gas, other, T, P_bar, pore_d_nm, rp, dqdp_mkpa):
    """
    입력
      gas, other : 가스명 (PARAMS에 존재)
      T [K], P_bar [bar], pore_d_nm [nm], rp (=P/P0), dqdp_mkpa [mol/kg/Pa]
    출력
      w : dict  (keys: "Blocked","Sieving","Knudsen","Surface","Capillary","Solution")
          각 가중치의 합은 1 (노멀라이즈)
    """
    # 초기값
    w = {k: 0.0 for k in ["Blocked","Sieving","Knudsen","Surface","Capillary","Solution"]}

    d1 = PARAMS[gas]["d"]
    d2 = PARAMS[other]["d"]
    dmin = min(d1, d2)           # [Å]
    pA   = pore_d_nm * 10.0      # [Å]
    lam  = mean_free_path_nm(T, P_bar, 0.5*(d1+d2))  # [nm]

    # 1) 완전 차단 근처
    if pA <= dmin - SIEVE_BAND_A:
        w["Blocked"] = 1.0
        return w

    # 2) 솔루션-확산(매우 작은 기공)
    if pore_d_nm <= SOL_TH_NM:
        w["Solution"] = 1.0
        return w

    # 보조 시그모이드(부드러운 경계)
    def sig(x, s=1.0):
        return 1.0/(1.0 + np.exp(-x/s))

    # 3) Sieving: 분자 직경 근방일수록 ↑
    #    pA-dmin 가 0 부근이면 가중치가 크고, 멀어질수록 감소
    w_sieve = np.exp(-((pA - dmin)/max(SIEVE_BAND_A,1e-6))**2)

    # 4) Knudsen: λ 가 기공보다 훨씬 크면 ↑  (pore_d_nm << λ)
    r = pore_d_nm / max(lam, 1e-9)
    w_kn = 1.0 / (1.0 + (r/0.5)**2)   # r→0 이면 ~1, r 커지면 감소

    # 5) Capillary: 큰 기공(≥~2 nm) + 높은 rp 에서 ↑
    w_cap = sig((pore_d_nm - 2.0)/0.2) * sig((rp - 0.5)/0.05)

    # 6) Surface: 흡착기울기(dq/dp)가 클수록 ↑ (단위 의존 → 완만한 스케일)
    alpha = 5e5   # 필요시 조정
    w_surf = 1.0 - np.exp(-alpha*max(float(dqdp_mkpa),0.0))

    # 7) Solution(경계가 아닌 영역에서는 낮게만 기여)
    w_sol = sig((SOL_TH_NM - pore_d_nm)/0.02)

    # 8) Blocked(경계 완충): dmin 아래 조금만 벗어나도 약간 남겨둠
    if pA < dmin:
        w_blk = np.exp(-((dmin - pA)/max(SIEVE_BAND_A,1e-6))**2)
    else:
        w_blk = 0.0

    # 합치고 정규화
    w["Blocked"]  = float(w_blk)
    w["Sieving"]  = float(w_sieve)
    w["Knudsen"]  = float(w_kn)
    w["Surface"]  = float(w_surf)
    w["Capillary"]= float(w_cap)
    w["Solution"] = float(w_sol)

    s = sum(w.values())
    if s <= 1e-12:
        w["Surface"] = 1.0
        return w
    for k in w:
        w[k] /= s
    return w

def permeance_series_SI(pore_d_nm, gas, other, T, P_bar, relP, L_nm,
                        q_mmolg, dqdp_molkgPa, q_other_mmolg):
    """
    혼합 공존 모델:
      1) 각 메커니즘 intrinsic Pi0 계산
      2) 조건 기반 가중치 w_i 산출 (합=1)
      3) Pi0_mix = sum_i w_i * Pi0_i
      4) 경쟁흡착 점유율 theta 곱
    """
    L_m = max(L_nm, 1e-3) * 1e-9
    M = PARAMS[gas]["M"]
    Pi = np.zeros_like(relP, float)

    for i, rp in enumerate(relP):
        # (1) 메커니즘별 intrinsic
        Pi_intr = {
            "Blocked":   PI_TINY,
            "Sieving":   pintr_sieving_SI(pore_d_nm, gas, T, L_m),
            "Knudsen":   pintr_knudsen_SI(pore_d_nm, T, M, L_m),
            "Surface":   pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa[i]),
            "Capillary": pintr_capillary_SI(pore_d_nm, rp, L_m),
            "Solution":  pintr_solution_SI(gas, T, L_m, dqdp_molkgPa[i]),
        }
        # 가스별 보정 스케일 적용
        for k in Pi_intr:
            Pi_intr[k] *= GAS_SCALE.get(gas, 1.0)

        # (2) 가중치
        w = mechanism_weights(gas, other, T, P_bar, pore_d_nm, rp,
                              dqdp_molkgPa[i])

        # (3) 직렬-병렬 혼합
        # 3-1) pore-like 그룹(공극 통과 경로)과 diffusion-like 그룹(흡착/확산 경로)으로 분리
        Pi_pore = (
            w["Sieving"]  * Pi_intr["Sieving"]  +
            w["Knudsen"]  * Pi_intr["Knudsen"]  +
            w["Capillary"]* Pi_intr["Capillary"]
        )

        Pi_diff = (
            w["Surface"]  * Pi_intr["Surface"]  +
            w["Solution"] * Pi_intr["Solution"]
        )

        # 3-2) 직렬 저항 결합 (harmonic mean 형태)
        #     1/Π_mix = 1/Π_pore + 1/Π_diff
        eps = 1e-30  # 0 나눗셈 방지용 작은 값
        if Pi_pore <= eps and Pi_diff <= eps:
            Pi0_mix = PI_TINY
        elif Pi_pore <= eps:
            Pi0_mix = Pi_diff
        elif Pi_diff <= eps:
            Pi0_mix = Pi_pore
        else:
           Pi0_mix = 1.0 / ( (1.0/(Pi_pore + eps)) + (1.0/(Pi_diff + eps)) )


        # (4) 경쟁흡착 점유율
        qi = q_mmolg[i]; qj = q_other_mmolg[i]
        theta = (qi/(qi+qj)) if (qi+qj) > 0 else 0.0

        Pi[i] = Pi0_mix * theta

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

    # === Axis / transient mode toggle ===
    mode = st.radio(
    "X-axis / Simulation mode",
    ["Relative pressure (P/P0)", "Time (transient LDF)"],
    index=0,
    )

    if mode == "Time (transient LDF)":
        st.subheader("Transient (LDF) settings")
        t_end = nudged_slider("Total time", 0.1, 3600.0, 0.1, 120.0, key="t_end", unit="s")
        dt    = nudged_slider("Time step",  1e-3, 10.0, 1e-3, 0.1,   key="dt",    unit="s")
        kLDF  = nudged_slider("k_LDF",      1e-4, 10.0, 1e-4, 0.05,  key="kLDF",  unit="s⁻¹")
        P0bar = nudged_slider("Feed P₀",    0.1,  10.0, 0.1,  Pbar,  key="P0bar", unit="bar")

        ramp = st.selectbox("Pressure schedule P(t)",
                            ["Step (P=P₀)", "Exp ramp: P₀(1-exp(-t/τ))"],
                            index=1)
        tau  = nudged_slider("τ (only for exp ramp)", 1e-3, 1000.0, 1e-3, 5.0,
                             key="tau", unit="s")

# ---------------------------- Compute ----------------------------
GPU = 3.35e-10
time_mode = (mode == "Time (transient LDF)")

if time_mode:
    # ===== Time (transient LDF) branch =====
    t = np.arange(0.0, t_end + dt, dt)
    P_bar_t = pressure_schedule_series(t, P0bar, ramp, tau)

    # 메커니즘 분류/모세관 분기 등에 쓰는 상대압력(rp)
    relP = relP_plot = np.clip(P_bar_t/float(P0bar), 1e-6, 0.9999)

    # q*(P) 콜백 (각 가스)
    def qeq_slope_cb_g1(Pbar_scalar: float):
        rp = float(np.clip(Pbar_scalar/float(Pbar), 1e-6, 0.9999))
        qv, dv = dsl_loading_and_slope_b(
            gas1, T, Pbar, np.array([rp]), q11, q12, b11, b12
        )
        return float(qv[0]), float(dv[0])

    def qeq_slope_cb_g2(Pbar_scalar: float):
        rp = float(np.clip(Pbar_scalar/float(Pbar), 1e-6, 0.9999))
        qv, dv = dsl_loading_and_slope_b(
            gas2, T, Pbar, np.array([rp]), q21, q22, b21, b22
        )
        return float(qv[0]), float(dv[0])

    # LDF 적분 → q_dyn(t), dqdp_eq(t)
    q1_dyn, dqdp1 = ldf_evolve_q(t, P_bar_t, qeq_slope_cb_g1, kLDF, q0=0.0)
    q2_dyn, dqdp2 = ldf_evolve_q(t, P_bar_t, qeq_slope_cb_g2, kLDF, q0=0.0)

    # Permeance(t)
    Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, Pbar, relP, L_nm, q1_dyn, dqdp1, q2_dyn)
    Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, Pbar, relP, L_nm, q2_dyn, dqdp2, q1_dyn)

    x_axis  = t
    x_label = "Time (s)"

else:
    # ===== Relative pressure (P/P0) branch =====
    relP = relP_plot = np.linspace(0.01, 0.99, 500)

    # DSL (b 직접 입력 사용)
    q1_mmolg, dqdp1 = dsl_loading_and_slope_b(gas1, T, Pbar, relP, q11, q12, b11, b12)
    q2_mmolg, dqdp2 = dsl_loading_and_slope_b(gas2, T, Pbar, relP, q21, q22, b21, b22)

    # (옵션) 자동 보정
    ensure_gas_scale_once(gas1, gas2, q11, q12, b11, b12)
    ensure_gas_scale_once(gas2, gas1, q21, q22, b21, b22)

    # Permeance(P/P0)
    Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, Pbar, relP, L_nm, q1_mmolg, dqdp1, q2_mmolg)
    Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, Pbar, relP, L_nm, q2_mmolg, dqdp2, q1_mmolg)

    x_axis  = relP
    x_label = r"Relative pressure, $P/P_0$ (–)"

# 공통: 선택도 & GPU 변환
Sel      = np.divide(Pi1, Pi2, out=np.zeros_like(Pi1), where=(Pi2 > 0))
Pi1_gpu  = Pi1 / GPU
Pi2_gpu  = Pi2 / GPU

# ---------------------------- Layout: Plots & Info ----------------------------
colA, colB = st.columns([1,2])

with colB:
    st.subheader("Mechanism map (along relative pressure)")
    rgba, mech_names = mechanism_band_rgba(gas1, gas2, T, Pbar, d_nm, relP_plot)
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

    # Permeance (GPU) plot
    ax1.plot(x_axis, Pi1_gpu, label=f"{gas1}")
    ax1.plot(x_axis, Pi2_gpu, '--', label=f"{gas2}")
    ax1.set_ylabel(r"$\Pi$  (GPU)")
    ax1.set_xlabel(x_label)

    ax1.set_ylabel(r"$\Pi$  (GPU)")

    # y축에서 1e6 같은 오프셋 표기 제거
    from matplotlib.ticker import ScalarFormatter
    ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.get_yaxis().get_offset_text().set_visible(False)

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

    ax2.set_xlabel(x_label)
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
