"""Data-center onsite power plant digital twin for the DCPower-ICS dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import random


NOM_V = 480.0
NOM_F = 60.0
BATT_CAP_KWH = 720.0
BATT_MAX_KW = 650.0
GEN_RATED_KW = 900.0
GEN_RAMP_KW_S = 55.0


@dataclass
class HiddenHealth:
    sensor_bias: Dict[str, float] = field(default_factory=dict)
    sensor_scale: Dict[str, float] = field(default_factory=dict)
    freeze_points: Dict[str, float] = field(default_factory=dict)
    stale_age_s: float = 0.0
    gen_efficiency: Dict[str, float] = field(default_factory=lambda: {"gen1": 1.0, "gen2": 1.0})
    gen_start_delay_s: Dict[str, float] = field(default_factory=lambda: {"gen1": 0.0, "gen2": 0.0})
    battery_capacity_factor: float = 1.0
    battery_resistance_factor: float = 1.0
    cooling_efficiency: float = 1.0
    ats_delay_factor: float = 1.0
    breaker_delay_s: float = 0.0
    comm_latency_s: float = 0.0
    maintenance_bypass: bool = False
    ups_efficiency: float = 0.965
    line_loss_factor: float = 0.018


@dataclass
class PowerPlantState:
    t: float = 0.0
    controller_mode: str = "Grid-connected normal"
    focus: str = "Maintenance Focus"
    active_scenario: Optional[str] = None
    scenario_age: float = 0.0
    rng: random.Random = field(default_factory=lambda: random.Random(7))
    health: HiddenHealth = field(default_factory=HiddenHealth)

    utility_available: bool = True
    utility_degraded: bool = False
    pcc_breaker_cmd: int = 1
    pcc_breaker: int = 1
    ats_state: str = "UTILITY"
    ats_timer: float = 0.0

    gen1_cmd: int = 0
    gen2_cmd: int = 0
    gen1_running: int = 0
    gen2_running: int = 0
    gen1_ready: int = 1
    gen2_ready: int = 1
    gen1_fault: int = 0
    gen2_fault: int = 0
    gen1_breaker: int = 0
    gen2_breaker: int = 0
    gen1_p: float = 0.0
    gen2_p: float = 0.0
    gen1_runtime_s: float = 0.0
    gen2_runtime_s: float = 0.0

    ups_mode: str = "ONLINE"
    ups_bypass: int = 0
    bess_state: str = "IDLE"
    bess_cmd_kw: float = 0.0
    bess_p: float = 0.0
    bess_soc: float = 82.0

    load_shed_stage: int = 0
    it_load_kw: float = 620.0
    cooling_kw: float = 210.0
    noncritical_kw: float = 110.0
    ambient_c: float = 26.0

    bus_v: float = NOM_V
    bus_f: float = NOM_F
    pcc_p: float = 940.0
    pcc_q: float = 190.0
    pcc_i: float = 1130.0
    ups_in_p: float = 640.0
    ups_out_p: float = 620.0
    battery_current_a: float = 0.0

    last_obs: Dict[str, float] = field(default_factory=dict)
    event_note: str = ""


def _noise(s: PowerPlantState, sigma: float) -> float:
    return s.rng.gauss(0.0, sigma)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def set_scenario(s: PowerPlantState, code: Optional[str]) -> None:
    s.active_scenario = code
    s.scenario_age = 0.0
    s.health = HiddenHealth()
    s.event_note = ""
    if not code:
        s.utility_available = True
        s.utility_degraded = False
        s.controller_mode = "Grid-connected normal"
        return
    if code in {"GRID_DISTURBANCE", "GEN_START_DELAY", "ATS_SLOW_TRANSFER", "LOAD_SHED_MISCONFIG", "GEN_EFF_LOSS", "BATTERY_DEGRADATION", "STEALTH_GEN_BIAS", "SWAPPED_SENSOR_MAPPING"}:
        s.utility_degraded = True
    if code == "UNSCHEDULED_BLACK_START":
        s.utility_available = False
        s.utility_degraded = True
        s.pcc_breaker = 0
        s.pcc_breaker_cmd = 0
        s.controller_mode = "Unscheduled black start / restart sequence"


def _apply_hidden_health(s: PowerPlantState) -> None:
    code = s.active_scenario
    age = s.scenario_age
    if not code:
        return
    h = s.health
    if code == "GEN_EFF_LOSS":
        h.gen_efficiency["gen1"] = 1.0 - min(0.34, age / 130.0)
    elif code == "GEN_START_DELAY":
        h.gen_start_delay_s["gen1"] = 18.0
    elif code in {"BREAKER_POSITION_MISMATCH", "BREAKER_STATUS_SPOOF"}:
        h.breaker_delay_s = 99.0
    elif code == "ATS_SLOW_TRANSFER":
        h.ats_delay_factor = 3.5
    elif code == "BATTERY_DEGRADATION":
        h.battery_capacity_factor = 0.34
        h.battery_resistance_factor = 1.85
    elif code == "SOC_CAL_DRIFT":
        h.sensor_bias["BESS_SOC"] = -min(10.0, age * 0.055)
    elif code == "COOLING_EFF_LOSS":
        h.cooling_efficiency = 1.28
    elif code == "PCC_METER_BIAS":
        h.sensor_scale["PCC_P"] = 0.91
        h.sensor_scale["PCC_I"] = 0.94
    elif code == "BUS_VOLTAGE_FREEZE":
        h.freeze_points.setdefault("bus_voltage", s.bus_v)
    elif code == "UPS_BYPASS_STUCK":
        h.maintenance_bypass = True
    elif code == "WRONG_ATS_TIMING":
        h.ats_delay_factor = 0.25
    elif code == "WRONG_BATT_SETPOINT":
        h.sensor_bias["BESS_cmd_bias"] = -260.0
    elif code == "LOAD_SHED_MISCONFIG":
        h.sensor_bias["shed_disabled"] = 1.0
    elif code == "SWAPPED_SENSOR_MAPPING":
        h.sensor_bias["swap_gen"] = 1.0
    elif code == "BYPASS_LEFT_ENABLED":
        h.maintenance_bypass = True
    elif code == "POWER_METER_SPOOF":
        h.sensor_scale["PCC_P"] = 1.18
    elif code == "COORDINATED_MASKING":
        h.sensor_scale["PCC_P"] = 1.10
        h.sensor_bias["BESS_SOC"] = 8.0
    elif code == "STEALTH_GEN_BIAS":
        h.sensor_scale["GEN1_P"] = 0.93
    elif code == "FALSE_HEALTHY_SUBSYSTEM":
        h.sensor_bias["false_oem_healthy"] = 1.0


def _controller(s: PowerPlantState, dt: float) -> None:
    load = s.it_load_kw + s.cooling_kw + s.noncritical_kw
    emergency = (not s.utility_available) or s.utility_degraded

    if not emergency and s.active_scenario != "UNSCHEDULED_BLACK_START":
        s.controller_mode = "Battery assist / peak shaving" if load > 950 and s.bess_soc > 45 else "Grid-connected normal"
        s.pcc_breaker_cmd = 1
        s.gen1_cmd = 0
        s.gen2_cmd = 0
        s.load_shed_stage = 0
        s.ats_state = "UTILITY"
    else:
        if s.active_scenario == "UNSCHEDULED_BLACK_START" and s.scenario_age < 22:
            s.controller_mode = "Unscheduled black start / restart sequence"
            s.gen1_cmd = 1 if s.scenario_age > 6 else 0
            s.gen2_cmd = 1 if s.scenario_age > 14 else 0
            s.load_shed_stage = 2
        else:
            s.controller_mode = "Grid disturbance / utility degraded"
            s.gen1_cmd = 1
            s.gen2_cmd = 1 if load > 780 else 0
            s.load_shed_stage = 1 if load > 850 else 0
        transfer_delay = 8.0 * s.health.ats_delay_factor
        s.ats_timer += dt
        if s.ats_timer > transfer_delay and (s.gen1_running or s.gen2_running):
            s.controller_mode = "Islanded operation"
            s.ats_state = "GENERATOR"
            s.pcc_breaker_cmd = 0

    if s.active_scenario == "WRONG_ATS_TIMING":
        s.controller_mode = "Transfer to backup generation"
    if s.active_scenario == "BYPASS_LEFT_ENABLED":
        s.controller_mode = "Maintenance / bypass / test mode"

    target = 0.0
    if s.controller_mode == "Battery assist / peak shaving":
        target = min(BATT_MAX_KW, max(0.0, load - 900.0))
    elif s.controller_mode in {"Grid disturbance / utility degraded", "Transfer to backup generation"}:
        target = min(360.0, max(80.0, load * 0.22))
    elif s.controller_mode == "Unscheduled black start / restart sequence":
        target = min(420.0, load * 0.35)
    elif s.bess_soc < 74 and s.utility_available and not s.utility_degraded:
        target = -130.0
    target += s.health.sensor_bias.get("BESS_cmd_bias", 0.0)
    s.bess_cmd_kw = _clamp(target, -BATT_MAX_KW, BATT_MAX_KW)


def _gen_step(cmd: int, running: int, ready: int, fault: int, p: float, runtime: float,
              target: float, delay: float, age: float, eff: float, dt: float) -> tuple[int, int, float, float]:
    if fault:
        return 0, 0, 0.0, 0.0
    if cmd and ready and age >= delay:
        running = 1
        runtime += dt
        derated_target = target * eff
        p += _clamp(derated_target - p, -GEN_RAMP_KW_S * dt, GEN_RAMP_KW_S * dt)
    else:
        p = max(0.0, p - 2.0 * GEN_RAMP_KW_S * dt)
        if p < 5:
            running = 0
            runtime = 0.0
    breaker = 1 if running and p > 35 else 0
    return running, breaker, _clamp(p, 0.0, GEN_RATED_KW), runtime


def step(s: PowerPlantState, dt: float = 1.0) -> dict:
    s.t += dt
    if s.active_scenario:
        s.scenario_age += dt
    else:
        s.scenario_age = 0.0
        s.ats_timer = 0.0

    _apply_hidden_health(s)

    day = 0.5 + 0.5 * math.sin(2 * math.pi * (s.t % 600.0) / 600.0)
    s.it_load_kw = _clamp(590.0 + 155.0 * day + _noise(s, 14.0), 470.0, 820.0)
    s.ambient_c = 25.0 + 4.0 * math.sin(2 * math.pi * (s.t % 1200.0) / 1200.0)
    cooling_target = (0.27 + 0.006 * max(0.0, s.ambient_c - 24.0)) * s.it_load_kw * s.health.cooling_efficiency
    s.cooling_kw += (cooling_target - s.cooling_kw) * min(1.0, dt / 25.0)
    s.cooling_kw = _clamp(s.cooling_kw + _noise(s, 2.4), 120.0, 390.0)
    base_noncrit = 120.0 + 10.0 * math.sin(2 * math.pi * s.t / 300.0)
    if s.load_shed_stage >= 2:
        shed_factor = 0.18
    elif s.load_shed_stage >= 1:
        shed_factor = 0.48
    else:
        shed_factor = 1.0
    if s.health.sensor_bias.get("shed_disabled"):
        shed_factor = 1.0
    s.noncritical_kw = _clamp(base_noncrit * shed_factor + _noise(s, 2.0), 15.0, 145.0)

    _controller(s, dt)

    load = s.it_load_kw + s.cooling_kw + s.noncritical_kw
    target_per_gen = min(GEN_RATED_KW * 0.82, max(120.0, load / max(1, s.gen1_cmd + s.gen2_cmd)))
    s.gen1_running, s.gen1_breaker, s.gen1_p, s.gen1_runtime_s = _gen_step(
        s.gen1_cmd, s.gen1_running, s.gen1_ready, s.gen1_fault, s.gen1_p, s.gen1_runtime_s,
        target_per_gen, s.health.gen_start_delay_s["gen1"], s.scenario_age, s.health.gen_efficiency["gen1"], dt)
    s.gen2_running, s.gen2_breaker, s.gen2_p, s.gen2_runtime_s = _gen_step(
        s.gen2_cmd, s.gen2_running, s.gen2_ready, s.gen2_fault, s.gen2_p, s.gen2_runtime_s,
        target_per_gen, s.health.gen_start_delay_s["gen2"], s.scenario_age, s.health.gen_efficiency["gen2"], dt)

    if s.active_scenario == "GEN_START_DELAY" and s.scenario_age < 16:
        s.gen1_p = 0.0
        s.gen1_breaker = 0
        s.gen1_running = 0

    if s.health.maintenance_bypass:
        s.ups_mode = "BYPASS"
        s.ups_bypass = 1
    elif s.ats_state == "GENERATOR" and (s.gen1_p + s.gen2_p) < 0.75 * load:
        s.ups_mode = "BATTERY"
        s.ups_bypass = 0
    else:
        s.ups_mode = "ONLINE"
        s.ups_bypass = 0

    s.bess_p = _clamp(s.bess_cmd_kw + _noise(s, 7.0), -BATT_MAX_KW, BATT_MAX_KW)
    if s.ups_mode == "BATTERY":
        s.bess_p = max(s.bess_p, min(BATT_MAX_KW, 0.55 * load))
    usable = BATT_CAP_KWH * s.health.battery_capacity_factor
    eff = 0.94 / s.health.battery_resistance_factor
    if s.bess_p >= 0:
        soc_delta = -(s.bess_p * dt / 3600.0) / max(usable, 1.0) * 100.0 / max(eff, 0.5)
        s.bess_state = "DISCHARGE" if s.bess_p > 15 else "IDLE"
    else:
        soc_delta = -s.bess_p * dt / 3600.0 / max(usable, 1.0) * 100.0 * eff
        s.bess_state = "CHARGE"
    s.bess_soc = _clamp(s.bess_soc + soc_delta, 4.0, 100.0)

    gen_total = s.gen1_p + s.gen2_p
    if s.pcc_breaker_cmd and not s.health.breaker_delay_s:
        s.pcc_breaker = 1
    elif not s.pcc_breaker_cmd:
        s.pcc_breaker = 0
    if s.active_scenario in {"BREAKER_POSITION_MISMATCH", "BREAKER_STATUS_SPOOF"}:
        s.pcc_breaker = 0

    if s.pcc_breaker and s.ats_state == "UTILITY" and s.utility_available:
        s.pcc_p = max(0.0, load - gen_total - s.bess_p)
    else:
        s.pcc_p = 0.0
    losses = s.health.line_loss_factor * load + (0.000018 * load * load)
    imbalance = s.pcc_p + gen_total + s.bess_p - load - losses
    source_stiffness = 1.0 if s.pcc_p > 50 else 0.45 + min(0.5, gen_total / 1200.0)
    v_sag = (load / 1000.0) * (2.8 / max(source_stiffness, 0.2))
    f_droop = -imbalance * 0.0015 if s.pcc_p < 50 else -imbalance * 0.0003
    if s.utility_degraded and s.ats_state == "UTILITY":
        s.bus_v = 462.0 + _noise(s, 1.8)
        s.bus_f = 59.72 + _noise(s, 0.025)
        s.controller_mode = "Grid disturbance / utility degraded"
    else:
        s.bus_v = _clamp(NOM_V - v_sag + _noise(s, 0.7), 430.0, 505.0)
        s.bus_f = _clamp(NOM_F + f_droop + _noise(s, 0.012), 58.4, 60.4)

    s.ups_out_p = s.it_load_kw
    if s.ups_mode == "BYPASS":
        s.ups_in_p = s.ups_out_p + _noise(s, 4.0)
    elif s.ups_mode == "BATTERY":
        s.ups_in_p = max(0.0, s.ups_out_p - min(s.bess_p, s.ups_out_p))
    else:
        s.ups_in_p = s.ups_out_p / s.health.ups_efficiency
    s.pcc_q = 0.24 * max(s.pcc_p, 0.0) + 0.10 * gen_total + _noise(s, 6.0)
    s.pcc_i = max(0.0, (math.sqrt(s.pcc_p ** 2 + s.pcc_q ** 2) * 1000.0) / (math.sqrt(3) * max(s.bus_v, 1.0)))
    s.battery_current_a = abs(s.bess_p) * 1000.0 / 720.0 + _noise(s, 1.5)

    return observe(s)


def observe(s: PowerPlantState) -> dict:
    obs = {
        "t": round(s.t, 1),
        "controller_mode": s.controller_mode,
        "focus": s.focus,
        "active_scenario": s.active_scenario,
        "scenario_age": round(s.scenario_age, 1),
        "utility_available": int(s.utility_available),
        "utility_degraded": int(s.utility_degraded),
        "PCC_breaker": float(s.pcc_breaker),
        "PCC_breaker_cmd": float(s.pcc_breaker_cmd),
        "ATS_state": s.ats_state,
        "GEN1_cmd": float(s.gen1_cmd),
        "GEN2_cmd": float(s.gen2_cmd),
        "GEN1_running": float(s.gen1_running),
        "GEN2_running": float(s.gen2_running),
        "GEN1_ready": float(s.gen1_ready),
        "GEN2_ready": float(s.gen2_ready),
        "GEN1_fault": float(s.gen1_fault),
        "GEN2_fault": float(s.gen2_fault),
        "GEN1_breaker": float(s.gen1_breaker),
        "GEN2_breaker": float(s.gen2_breaker),
        "GEN1_P": s.gen1_p,
        "GEN2_P": s.gen2_p,
        "GEN1_Q": 0.18 * s.gen1_p,
        "GEN2_Q": 0.18 * s.gen2_p,
        "GEN1_f": s.bus_f + _noise(s, 0.006),
        "GEN2_f": s.bus_f + _noise(s, 0.006),
        "GEN1_V": s.bus_v + _noise(s, 0.5),
        "GEN2_V": s.bus_v + _noise(s, 0.5),
        "UPS_mode": s.ups_mode,
        "UPS_bypass": float(s.ups_bypass),
        "UPS_in_P": s.ups_in_p,
        "UPS_out_P": s.ups_out_p,
        "BESS_state": s.bess_state,
        "BESS_cmd": s.bess_cmd_kw,
        "BESS_P": s.bess_p,
        "BESS_SOC": s.bess_soc,
        "BESS_current": s.battery_current_a,
        "P_IT": s.it_load_kw,
        "P_cooling": s.cooling_kw,
        "P_noncritical": s.noncritical_kw,
        "load_shed_stage": float(s.load_shed_stage),
        "bus_voltage": s.bus_v,
        "bus_frequency": s.bus_f,
        "PCC_P": s.pcc_p,
        "PCC_Q": s.pcc_q,
        "PCC_I": s.pcc_i,
        "ambient_c": s.ambient_c,
        "maintenance_bypass": float(s.health.maintenance_bypass),
    }
    for point, value in list(obs.items()):
        if isinstance(value, (int, float)):
            v = float(value)
            if point in s.health.freeze_points:
                v = s.health.freeze_points[point]
            v = v * s.health.sensor_scale.get(point, 1.0) + s.health.sensor_bias.get(point, 0.0)
            if point in {"PCC_P", "GEN1_P", "GEN2_P", "BESS_P", "P_IT", "P_cooling", "P_noncritical"}:
                v += _noise(s, 1.8)
            elif point in {"bus_voltage", "GEN1_V", "GEN2_V"}:
                v += _noise(s, 0.5)
            elif point in {"bus_frequency", "GEN1_f", "GEN2_f"}:
                v += _noise(s, 0.006)
            obs[point] = round(v, 3)
    if s.health.sensor_bias.get("swap_gen"):
        obs["GEN1_P"], obs["GEN2_P"] = obs["GEN2_P"], obs["GEN1_P"]
    if s.active_scenario in {"BREAKER_POSITION_MISMATCH", "BREAKER_STATUS_SPOOF"}:
        obs["PCC_breaker"] = 1.0
        obs["PCC_I"] = max(0.0, min(obs["PCC_I"], 2.0))
    if s.active_scenario == "UPS_BYPASS_STUCK":
        obs["UPS_mode"] = "ONLINE"
        obs["UPS_bypass"] = 0.0
    if s.active_scenario == "FALSE_HEALTHY_SUBSYSTEM":
        obs["OEM_device_health"] = "healthy"
    s.last_obs = obs
    return obs
