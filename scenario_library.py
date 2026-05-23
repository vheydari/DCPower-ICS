"""Scenario catalogue for the DCPower-ICS dataset generator and demo.

This public catalogue intentionally contains only dataset metadata needed to
reproduce the Scientific Data paper: scenario code, title, group, category,
affected assets, and a short scenario summary. It does not implement or expose
any detector-specific explanation, evidence, or diagnostic framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Scenario:
    code: str
    title: str
    group: str
    category: str
    affected_assets: List[str]
    summary: str


SCENARIOS: Dict[str, Scenario] = {
    'GEN_EFF_LOSS': Scenario(
        code='GEN_EFF_LOSS',
        title='Generator efficiency loss',
        group='maintenance',
        category='degradation',
        affected_assets=['generator_1'],
        summary='Generator 1 requires more mechanical effort for the same electrical output.',
    ),
    'GEN_START_DELAY': Scenario(
        code='GEN_START_DELAY',
        title='Generator start delay',
        group='maintenance',
        category='switching/control',
        affected_assets=['generator_1'],
        summary='Generator start command is present but voltage and power rise late.',
    ),
    'BREAKER_POSITION_MISMATCH': Scenario(
        code='BREAKER_POSITION_MISMATCH',
        title='Breaker position mismatch',
        group='maintenance',
        category='instrumentation',
        affected_assets=['pcc_breaker'],
        summary='Breaker status reports closed while feeder current indicates an open path.',
    ),
    'ATS_SLOW_TRANSFER': Scenario(
        code='ATS_SLOW_TRANSFER',
        title='ATS slow transfer',
        group='maintenance',
        category='switching/control',
        affected_assets=['ats'],
        summary='ATS transfer sequence takes longer than the policy envelope.',
    ),
    'BATTERY_DEGRADATION': Scenario(
        code='BATTERY_DEGRADATION',
        title='Battery capacity degradation',
        group='maintenance',
        category='degradation',
        affected_assets=['battery', 'ups'],
        summary='Battery SOC falls faster than expected for the measured discharge power.',
    ),
    'SOC_CAL_DRIFT': Scenario(
        code='SOC_CAL_DRIFT',
        title='Battery SOC calibration drift',
        group='maintenance',
        category='instrumentation',
        affected_assets=['battery', 'ups'],
        summary='SOC trend is inconsistent with battery charge/discharge telemetry.',
    ),
    'COOLING_EFF_LOSS': Scenario(
        code='COOLING_EFF_LOSS',
        title='Cooling efficiency loss',
        group='maintenance',
        category='degradation',
        affected_assets=['cooling_load'],
        summary='Cooling power is high relative to IT load and ambient proxy.',
    ),
    'PCC_METER_BIAS': Scenario(
        code='PCC_METER_BIAS',
        title='PCC meter bias',
        group='maintenance',
        category='instrumentation',
        affected_assets=['pcc_meter'],
        summary='PCC power disagrees with bus voltage and current cross-check.',
    ),
    'BUS_VOLTAGE_FREEZE': Scenario(
        code='BUS_VOLTAGE_FREEZE',
        title='Bus voltage sensor freeze',
        group='maintenance',
        category='instrumentation',
        affected_assets=['main_bus'],
        summary='Bus voltage remains unnaturally flat through source and load changes.',
    ),
    'UPS_BYPASS_STUCK': Scenario(
        code='UPS_BYPASS_STUCK',
        title='UPS bypass stuck',
        group='maintenance',
        category='switching/control',
        affected_assets=['ups'],
        summary='UPS reports online intent while measurements match bypass behavior.',
    ),
    'WRONG_ATS_TIMING': Scenario(
        code='WRONG_ATS_TIMING',
        title='Wrong ATS timing parameter',
        group='configuration',
        category='commissioning/configuration',
        affected_assets=['ats', 'site_controller'],
        summary='ATS sequence violates configured operating policy after commissioning.',
    ),
    'WRONG_BATT_SETPOINT': Scenario(
        code='WRONG_BATT_SETPOINT',
        title='Wrong battery dispatch setpoint',
        group='configuration',
        category='commissioning/configuration',
        affected_assets=['battery', 'site_controller'],
        summary='Battery dispatch occurs outside the safe policy envelope.',
    ),
    'LOAD_SHED_MISCONFIG': Scenario(
        code='LOAD_SHED_MISCONFIG',
        title='Load-shed stage misconfiguration',
        group='configuration',
        category='commissioning/configuration',
        affected_assets=['noncritical_load', 'site_controller'],
        summary='Non-critical load remains energized during an emergency mode that requires shedding.',
    ),
    'SWAPPED_SENSOR_MAPPING': Scenario(
        code='SWAPPED_SENSOR_MAPPING',
        title='Swapped sensor mapping',
        group='configuration',
        category='commissioning/configuration',
        affected_assets=['generator_1', 'generator_2'],
        summary='Generator telemetry follows the wrong generator channel after a configuration change.',
    ),
    'BYPASS_LEFT_ENABLED': Scenario(
        code='BYPASS_LEFT_ENABLED',
        title='Maintenance bypass left enabled',
        group='configuration',
        category='commissioning/configuration',
        affected_assets=['ups', 'switchgear'],
        summary='Maintenance bypass remains enabled after return-to-service.',
    ),
    'BREAKER_STATUS_SPOOF': Scenario(
        code='BREAKER_STATUS_SPOOF',
        title='Breaker status spoof',
        group='cyber',
        category='cyber-like inconsistency',
        affected_assets=['pcc_breaker'],
        summary='Breaker status and current measurements cannot both be true.',
    ),
    'POWER_METER_SPOOF': Scenario(
        code='POWER_METER_SPOOF',
        title='Power meter spoof',
        group='cyber',
        category='cyber-like inconsistency',
        affected_assets=['pcc_meter'],
        summary='PCC power is manipulated relative to generation, battery, and load totals.',
    ),
    'COORDINATED_MASKING': Scenario(
        code='COORDINATED_MASKING',
        title='Coordinated masking',
        group='cyber',
        category='cyber-like inconsistency',
        affected_assets=['pcc_meter', 'battery'],
        summary='Multiple measurements remain individually plausible but fail cross-system consistency.',
    ),
    'STEALTH_GEN_BIAS': Scenario(
        code='STEALTH_GEN_BIAS',
        title='Stealthy generator bias',
        group='cyber',
        category='cyber-like inconsistency',
        affected_assets=['generator_1'],
        summary='Generator power bias stays small per sample but accumulates as a persistent mismatch.',
    ),
    'FALSE_HEALTHY_SUBSYSTEM': Scenario(
        code='FALSE_HEALTHY_SUBSYSTEM',
        title='False healthy signal from subsystem',
        group='cyber',
        category='cyber-like inconsistency',
        affected_assets=['ups', 'battery'],
        summary='OEM/device health reports healthy while plant-level consistency checks show inconsistent behavior.',
    ),
    'GRID_DISTURBANCE': Scenario(
        code='GRID_DISTURBANCE',
        title='Utility degraded / transfer drill',
        group='operations',
        category='operations disturbance',
        affected_assets=['utility', 'ats', 'generators'],
        summary='Utility voltage/frequency degrades and controller begins backup transfer.',
    ),
    'UNSCHEDULED_BLACK_START': Scenario(
        code='UNSCHEDULED_BLACK_START',
        title='Unscheduled black start / restart sequence',
        group='operations',
        category='operations disturbance',
        affected_assets=['site_controller', 'generators', 'ups'],
        summary='Site restarts from no utility source using battery ride-through and generator sequencing.',
    ),
}


def scenario_groups() -> Dict[str, List[dict]]:
    """Return scenarios grouped for the demo UI."""
    groups: Dict[str, List[dict]] = {
        "maintenance": [],
        "configuration": [],
        "operations": [],
        "cyber": [],
    }
    for s in SCENARIOS.values():
        groups.setdefault(s.group, []).append({
            "code": s.code,
            "title": s.title,
            "category": s.category,
            "affected_assets": s.affected_assets,
            "summary": s.summary,
        })
    return groups
