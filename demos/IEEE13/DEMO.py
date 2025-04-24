"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

from pathlib import Path

import sys

cases = {}
cases["1"] = "PV_BATT"
cases["2"] = "PV_BATT_TMM"
cases["3"] = "PV_BATT_ADMS_POWER"
cases["4"] = "PV_BATT_ADMS_VOLTAGE"
cases["5"] = "PV_BATT_ADMS_POWER_TMM"

path_to_here = Path(__file__).resolve().parent
path_to_demos = path_to_here / ".."

# Add Demos to Python Path
sys.path.append(str(path_to_demos))

from FULL_DEMO import run

baselines = {}
baselines["Baseline"] = {
    "cancelled_DERs": ["BAT1", "BAT2", "BAT3", "BAT4", "BAT5", "PV1", "PV2", "PV3"],
    "ignored_DERs": [],
}
baselines["Baseline PV"] = {
    "cancelled_DERs": ["BAT1", "BAT2", "BAT3", "BAT4", "BAT5"],
    "ignored_DERs": ["PV1", "PV2", "PV3"],
}

run(cases, baselines, path_to_here)
