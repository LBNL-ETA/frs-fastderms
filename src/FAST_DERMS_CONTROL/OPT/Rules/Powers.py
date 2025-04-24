"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ...Modeling.networkCase import NetworkCase

import pyomo.environ as pe

import logging

logger = logging.getLogger(("FAST DERMS / OPT / Rules / Powers"))


# RULE to initialize load
def Load_init_rule(
    model, busID, phi_nr, frame_idx, scenario_nr, fcast_type: str = "PL"
):
    """
    RULE : Initialize Load
    """
    frame = pe.value(model.networkCase).horizon[frame_idx - 1]
    list_loadIDs = frame.get_ressources_bus(busID, "load")

    L_value = 0

    for loadID in list_loadIDs:
        load = frame.get_loads([loadID])[0]
        if load.get_active_status():
            one_value = load.get_samples(
                forecast_type=fcast_type, scenario_nr=scenario_nr - 1
            )[phi_nr]
            L_value += one_value
            logger.debug(
                f"{loadID}: {fcast_type}: {one_value:.3f} on {busID}, ph {phi_nr}, f {frame_idx}, s {scenario_nr}"
            )

    return L_value


def PL_init_rule(model, busID, phi_nr, frame_idx, scenario_nr):
    return Load_init_rule(model, busID, phi_nr, frame_idx, scenario_nr, "PL")


def QL_init_rule(model, busID, phi_nr, frame_idx, scenario_nr):
    return Load_init_rule(model, busID, phi_nr, frame_idx, scenario_nr, "QL")
