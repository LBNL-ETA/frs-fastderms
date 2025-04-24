"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import pyomo.environ as pe
import numpy as np
import logging
import math

from ...Modeling.network import Network

logger = logging.getLogger("FAST DERMS / OPT / Constraints / DERs")


### DER Constraints
def DER_constraints(model: pe.Model, **kw_args) -> pe.Model:
    """Adds DER specific constraints and variables for each DER type present in network."""

    # Get options out of kw_args
    unity_powerfactor = kw_args.get("unity_powerfactor", False)

    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network

    # General Mapping
    def activeDERmapping_rule(
        model, busID, phi_nr, frame_idx, schedule_case, scenario_nr
    ):

        Net = networkCase.horizon[frame_idx - 1]

        if schedule_case == "sched":
            cons = model.PLDER[busID, phi_nr, frame_idx, scenario_nr] == sum(
                Net.CD[derType][busID][derID][phi_nr]
                * model.PDER[derType, derID, frame_idx, scenario_nr]
                for derType in Net.get_DER_types(True)
                for derID in Net.CD[derType][busID].keys()
            )
        elif schedule_case == "up":
            cons = model.RLup[busID, phi_nr, frame_idx, scenario_nr] == sum(
                Net.CD[derType][busID][derID][phi_nr]
                * model.rDER_up[derType, derID, frame_idx, scenario_nr]
                for derType in Net.get_DER_types(True)
                for derID in Net.CD[derType][busID].keys()
            )
        elif schedule_case == "dn":
            cons = model.RLdn[busID, phi_nr, frame_idx, scenario_nr] == sum(
                Net.CD[derType][busID][derID][phi_nr]
                * model.rDER_dn[derType, derID, frame_idx, scenario_nr]
                for derType in Net.get_DER_types(True)
                for derID in Net.CD[derType][busID].keys()
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.activeDERmapping = pe.Constraint(model.Pkt.keys(), rule=activeDERmapping_rule)

    def reactiveDERmapping_rule(model, busID, phi_nr, frame_idx, scenario_nr):
        Net = networkCase.horizon[frame_idx - 1]

        cons = model.QLDER[busID, phi_nr, frame_idx, scenario_nr] == sum(
            Net.CD[derType][busID][derID][phi_nr]
            * model.QDER[derType, derID, frame_idx, scenario_nr]
            for derType in Net.get_DER_types(True)
            for derID in Net.CD[derType][busID].keys()
        )
        return cons

    model.reactiveDERmapping = pe.Constraint(
        model.Qkt.keys(), rule=reactiveDERmapping_rule
    )

    # Check Composite Resources and enforce no reserves
    for resource in Net.get_composite_resources():
        for derID in resource.get_DER_ids():
            derType = Net.get_DERs(mRIDs=derID)[0].get_type()
            model.rDER_up[derType, derID, ...].fix(0)
            model.rDER_dn[derType, derID, ...].fix(0)
            logger.debug(f"Fixed DER {derType} {derID} reserves to zero")

    # Fix reactive power to zero if no q control / unity_powerfactor is enabled
    if unity_powerfactor:
        logger.info("Unity Power Factor requested")

        for der in Net.get_DERs():
            model.QDER[der.get_type(), der.getID(), ...].fix(0)
            logger.debug(
                f"Fixed DER {der.get_type()} {der.getID()} reactive power to zero"
            )

    # Fix DER Schedule Setpoint and Reserves to zero if DER is inactive
    for der in Net.get_DERs():

        def func_list(Network: Network, mRID):
            der = Network.get_DERs(mRIDs=mRID)[0]
            return der.get_active_status()

        status_horizon = networkCase.get_horizon(func_list, der.getID())

        for frame_idx, status in zip(model.T, status_horizon):
            # Fix DER Schedule Setpoint and Reserves to zero if DER is inactive
            if not status:
                model.PDER[der.get_type(), der.getID(), frame_idx, ...].fix(0)
                model.QDER[der.get_type(), der.getID(), frame_idx, ...].fix(0)
                model.rDER_up[der.get_type(), der.getID(), frame_idx, ...].fix(0)
                model.rDER_dn[der.get_type(), der.getID(), frame_idx, ...].fix(0)
                logger.info(
                    f"Frame {frame_idx}: Fixed DER {der.get_type()} {der.getID()} to zero"
                )

    # DER Specific Constraints
    model.Battery_list = pe.Set(
        initialize=[
            der.getID() for der in Net.get_DERs() if der.get_type() == "Battery"
        ],
        doc="Set of batteries",
    )
    model.PV_list = pe.Set(
        initialize=[der.getID() for der in Net.get_DERs() if der.get_type() == "PV"],
        doc="Set of PVs",
    )
    model.EV_list = pe.Set(
        initialize=[
            der.getID() for der in Net.get_DERs() if der.get_type() == "SmartChargeEV"
        ],
        doc="Set of SmartChargeEV",
    )
    model.FlexibleLoad_list = pe.Set(
        initialize=[
            der.getID() for der in Net.get_DERs() if der.get_type() == "FlexibleLoad"
        ],
        doc="Set of Flexible Loads",
    )
    model.VPP_list = pe.Set(
        initialize=[der.getID() for der in Net.get_DERs() if der.get_type() == "VPP"],
        doc="Set of VPP",
    )
    model.TR_list = pe.Set(
        initialize=[
            der.getID()
            for der in Net.get_DERs()
            if der.get_type() == "TransactiveResource"
        ],
        doc="Set of TransactiveResource",
    )

    for derType in model.DERtype:
        if derType == "Battery":
            model = Battery_Model(model)
        elif derType == "PV":
            model = PV_Model(model)
        elif derType == "SmartChargeEV":
            model = SmartChargeEV_Model(model)
        elif derType == "FlexibleLoad":
            model = FlexibleLoad_Model(model)
        elif derType == "VPP":
            model = VPP_Model(model)
        elif derType == "TransactiveResource":
            model = TransactiveResource_Model(model)
        else:
            logger.error(f"DER type {derType} is not yet supported !!!")
    return model


def Battery_Model(model: pe.Model) -> pe.Model:
    groupID = "Battery"

    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network
    # need dealtaT in hours
    deltT = networkCase.timestep_period / 60

    # model.Battery_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of batteries')

    # CREATE BATTERY SPECIFIC VARIABLES
    # Discharging power of battery i at time t (non-negative)
    model.Pd_BAT = pe.Var(
        model.Battery_list,
        model.T,
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Discharging power of battery i at time t",
    )
    # Charging power of battery i at time t (non-positive)
    model.Pc_BAT = pe.Var(
        model.Battery_list,
        model.T,
        model.Scenarios,
        within=pe.NonPositiveReals,
        doc="Charging power of battery i at time t",
    )
    # Energy stored in battery i at time t
    model.E_BAT = pe.Var(
        model.Battery_list,
        model.T,
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Energy stored in battery i at time t",
    )

    # CREATE BATTERY SPECIFIC EQUALITY CONSTRAINTS
    def BAT_Power_rule(model, derID, frame_idx, scenario_nr):
        # Total battery power = sum of charge and discharge power
        cons = (
            model.PDER[groupID, derID, frame_idx, scenario_nr]
            == model.Pd_BAT[derID, frame_idx, scenario_nr]
            + model.Pc_BAT[derID, frame_idx, scenario_nr]
        )
        return cons

    model.BAT_power = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_Power_rule
    )

    def BAT_Energy_rule(model, derID, frame_idx, scenario_nr):
        battery = networkCase.Network.get_DERs(mRIDs=derID)[0]
        # Calculate battery energy levels
        if frame_idx == 1:
            cons = (
                model.E_BAT[derID, frame_idx, scenario_nr]
                == battery.E_0
                - (
                    battery.eff_c * model.Pc_BAT[derID, frame_idx, scenario_nr]
                    + 1 / battery.eff_d * model.Pd_BAT[derID, frame_idx, scenario_nr]
                )
                * deltT
            )
        else:
            cons = (
                model.E_BAT[derID, frame_idx, scenario_nr]
                == model.E_BAT[derID, frame_idx - 1, scenario_nr]
                - (
                    battery.eff_c * model.Pc_BAT[derID, frame_idx, scenario_nr]
                    + 1 / battery.eff_d * model.Pd_BAT[derID, frame_idx, scenario_nr]
                )
                * deltT
            )
        return cons

    model.BAT_energy = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_Energy_rule
    )

    def BAT_FinalEnergy_rule(model, derID, scenario_nr):
        battery = networkCase.Network.get_DERs(mRIDs=derID)[0]
        # Final energy level must be equal to initial energy level
        cons = model.E_BAT[derID, model.T.at(-1), scenario_nr] == battery.E_F
        return cons

    model.BAT_finalEnergy = pe.Constraint(
        model.Battery_list, model.Scenarios, rule=BAT_FinalEnergy_rule
    )

    # CREATE BATTERY SPECIFIC INEQUALITY CONSTRAINTS
    def BAT_QLim_rule(model, derID, frame_idx, scenario_nr):
        # Battery Reactive Limits
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if battery.get_active_status() and battery.Q_min != None:
            cons = (
                battery.Q_min,
                model.QDER[groupID, derID, frame_idx, scenario_nr],
                battery.Q_max,
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.BAT_QLim = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_QLim_rule
    )

    # Thermal Line Limits
    def BAT_ThermalLim_rule(
        model, derID, frame_idx, schedule_case, scenario_nr, polytope_nr
    ):
        # Battery Thermal Limits
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]

        # Net = pe.value(model.networkCase).horizon[frame_idx-1]
        # line = Net.get_lines([lineID])[0]

        if battery.get_active_status() and battery.S_max != None:
            # dn = line.to_bus
            # check whether NPolytope >=3
            if 3 in model.NPolytope:
                alpha = 2 * math.pi / len(model.NPolytope)
                m = (
                    math.sin(polytope_nr * alpha) - math.sin((polytope_nr - 1) * alpha)
                ) / (
                    math.cos(polytope_nr * alpha) - math.cos((polytope_nr - 1) * alpha)
                )
                if polytope_nr <= len(model.NPolytope) / 2:
                    cons = (
                        battery.S_max * math.sin(polytope_nr * alpha)
                        - m * battery.S_max * math.cos(polytope_nr * alpha)
                        >= model.QDER[groupID, derID, frame_idx, scenario_nr]
                        - m * model.PDER[groupID, derID, frame_idx, scenario_nr]
                    )
                else:
                    cons = (
                        battery.S_max * math.sin(polytope_nr * alpha)
                        - m * battery.S_max * math.cos(polytope_nr * alpha)
                        <= model.QDER[groupID, derID, frame_idx, scenario_nr]
                        - m * model.PDER[groupID, derID, frame_idx, scenario_nr]
                    )

            elif [1] == model.NPolytope:
                print(
                    "Quadratic Constraint for "
                    + battery.mRID
                    + " - Scenario "
                    + scenario_nr
                )
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    + model.QDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    <= battery.S_max**2
                )

            else:
                print(
                    "Error Generating Maximum Apparent Power Flow Approximation. n_thermal not large enough. Using quadratic constraint instead. "
                )
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    + model.QDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    <= battery.S_max**2
                )

        else:
            cons = pe.Constraint.Skip
        return cons

    model.BAT_ThermalLim = pe.Constraint(
        model.Battery_list,
        model.T,
        model.cases,
        model.Scenarios,
        model.NPolytope,
        rule=BAT_ThermalLim_rule,
    )

    def BAT_ChargeLim_rule(model, derID, frame_idx, scenario_nr):
        # Battery Charge Limits
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if battery.get_active_status():
            cons = model.Pc_BAT[derID, frame_idx, scenario_nr] >= battery.get_samples(
                forecast_type="Pmin", scenario_nr=scenario_nr - 1
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.BAT_ChargeLim = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_ChargeLim_rule
    )

    def BAT_DischargeLim_rule(model, derID, frame_idx, scenario_nr):
        # Battery Discharge Limits
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if battery.get_active_status():
            cons = model.Pd_BAT[derID, frame_idx, scenario_nr] <= battery.get_samples(
                forecast_type="Pmax", scenario_nr=scenario_nr - 1
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.BAT_DischargeLim = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_DischargeLim_rule
    )

    def BAT_PowerLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        # Battery power limits
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if battery.get_active_status():
            if reserve == "up":
                P_max_fcast = battery.get_samples(
                    forecast_type="Pmax", scenario_nr=scenario_nr - 1
                )
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    + model.rDER_up[groupID, derID, frame_idx, scenario_nr]
                    <= P_max_fcast
                )
            elif reserve == "dn":
                P_min_fcast = battery.get_samples(
                    forecast_type="Pmin", scenario_nr=scenario_nr - 1
                )
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    - model.rDER_dn[groupID, derID, frame_idx, scenario_nr]
                    >= P_min_fcast
                )
            else:
                cons = pe.Constraint.Skip
        else:
            if reserve == "up":
                cons = model.Pd_BAT[derID, frame_idx, scenario_nr] == 0
            elif reserve == "dn":
                cons = model.Pc_BAT[derID, frame_idx, scenario_nr] == 0
            else:
                cons = pe.Constraint.Skip

        return cons

    model.BAT_PowerLim = pe.Constraint(
        model.Battery_list,
        model.T,
        ["up", "dn"],
        model.Scenarios,
        rule=BAT_PowerLim_rule,
    )

    def BAT_EnergyLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Battery energy limits
        if reserve == "up":
            # supply more
            cons = model.E_BAT[derID, frame_idx, scenario_nr] - model.rDER_up[
                groupID, derID, frame_idx, scenario_nr
            ] * deltT >= battery.get_samples(
                forecast_type="Emin", scenario_nr=scenario_nr - 1
            )
        elif reserve == "dn":
            # dn, store more
            cons = model.E_BAT[derID, frame_idx, scenario_nr] + model.rDER_dn[
                groupID, derID, frame_idx, scenario_nr
            ] * deltT <= battery.get_samples(
                forecast_type="Emax", scenario_nr=scenario_nr - 1
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.BAT_EnergyLim = pe.Constraint(
        model.Battery_list,
        model.T,
        ["up", "dn"],
        model.Scenarios,
        rule=BAT_EnergyLim_rule,
    )

    def BAT_ChargeDischarge_rule(model, derID, frame_idx, scenario_nr):
        battery = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Constraint that discourages simultaneous charging and discharging
        P_min_batt = battery.get_samples(
            forecast_type="Pmin", scenario_nr=scenario_nr - 1
        )
        P_max_batt = battery.get_samples(
            forecast_type="Pmax", scenario_nr=scenario_nr - 1
        )
        cons = model.Pc_BAT[derID, frame_idx, scenario_nr] >= P_min_batt + model.Pd_BAT[
            derID, frame_idx, scenario_nr
        ] * (P_min_batt / P_max_batt)
        return cons

    model.BAT_ChargeDischarge = pe.Constraint(
        model.Battery_list, model.T, model.Scenarios, rule=BAT_ChargeDischarge_rule
    )

    return model


def FlexibleLoad_Model(model: pe.Model) -> pe.Model:
    groupID = "FlexibleLoad"

    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network
    # need dealtaT in hours
    deltT = networkCase.timestep_period / 60

    # model.FlexibleLoad_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of Flexible Loads')

    # CREATE FLEXIBLE LOAD SPECIFIC VARIABLES
    # Energy stored in flexible load
    model.E_FL = pe.Var(
        model.FlexibleLoad_list, model.T, model.Scenarios, within=pe.NonNegativeReals
    )

    # CREATE FLEXIBLE LOAD SPECIFIC EQUALITY CONSTRAINTS
    def FL_Energy_rule(model, derID, frame_idx, scenario_nr):
        flexload = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Calculate flexible load energy levels
        # Check if DER is active
        if flexload.get_active_status():
            if frame_idx == 1:
                cons = (
                    model.E_FL[derID, frame_idx, scenario_nr]
                    == flexload.get_samples(
                        forecast_type="alpha", scenario_nr=scenario_nr - 1
                    )
                    * flexload.E_0
                    - model.PDER[groupID, derID, frame_idx, scenario_nr] * deltT
                )
            else:
                cons = (
                    model.E_FL[derID, frame_idx, scenario_nr]
                    == flexload.get_samples(
                        forecast_type="alpha", scenario_nr=scenario_nr - 1
                    )
                    * model.E_FL[derID, frame_idx - 1, scenario_nr]
                    - model.PDER[groupID, derID, frame_idx, scenario_nr] * deltT
                )
        else:
            # If DER not active, set energy levels equal to previous time period
            if frame_idx == 1:
                cons = model.E_FL[derID, frame_idx, scenario_nr] == flexload.E_0
            else:
                cons = (
                    model.E_FL[derID, frame_idx, scenario_nr]
                    == model.E_FL[derID, frame_idx - 1, scenario_nr]
                )
        return cons

    model.FL_Energy = pe.Constraint(
        model.FlexibleLoad_list, model.T, model.Scenarios, rule=FL_Energy_rule
    )

    # CREATE FLEXIBLE LOAD SPECIFIC INEQUALITY CONSTRAINTS
    def FL_PowerLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        # Flexible load power limits
        flexload = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if flexload.get_active_status():
            if reserve == "up":
                cons = (
                    flexload.get_samples(
                        forecast_type="Pmin", scenario_nr=scenario_nr - 1
                    ),
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    + model.rDER_up[groupID, derID, frame_idx, scenario_nr],
                    flexload.get_samples(
                        forecast_type="Pmax", scenario_nr=scenario_nr - 1
                    ),
                )
            elif reserve == "dn":
                cons = (
                    flexload.get_samples(
                        forecast_type="Pmin", scenario_nr=scenario_nr - 1
                    ),
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    - model.rDER_dn[groupID, derID, frame_idx, scenario_nr],
                    flexload.get_samples(
                        forecast_type="Pmax", scenario_nr=scenario_nr - 1
                    ),
                )
            else:
                cons = pe.Constraint.Skip
        else:
            cons = pe.Constraint.Skip
        return cons

    model.FL_PowerLim = pe.Constraint(
        model.FlexibleLoad_list,
        model.T,
        ["up", "dn"],
        model.Scenarios,
        rule=FL_PowerLim_rule,
    )

    def FL_EnergyLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        flexload = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Flexible load energy limits
        if reserve == "up":
            cons = (
                flexload.get_samples(forecast_type="Emin", scenario_nr=scenario_nr - 1),
                model.E_FL[derID, frame_idx, scenario_nr]
                - model.rDER_up[groupID, derID, frame_idx, scenario_nr] * deltT,
                flexload.get_samples(forecast_type="Emax", scenario_nr=scenario_nr - 1),
            )
        elif reserve == "dn":
            cons = (
                flexload.get_samples(forecast_type="Emin", scenario_nr=scenario_nr - 1),
                model.E_FL[derID, frame_idx, scenario_nr]
                + model.rDER_dn[groupID, derID, frame_idx, scenario_nr] * deltT,
                flexload.get_samples(forecast_type="Emax", scenario_nr=scenario_nr - 1),
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.FL_EnergyLim = pe.Constraint(
        model.FlexibleLoad_list,
        model.T,
        ["up", "dn"],
        model.Scenarios,
        rule=FL_EnergyLim_rule,
    )

    return model


def PV_Model(model: pe.Model) -> pe.Model:
    groupID = "PV"

    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network

    # model.PV_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of PVs')

    # CREATE PV SPECIFIC VARIABLES: None

    # CREATE PV SPECIFIC EQUALITY CONSTRAINTS: None

    # CREATE PV SPECIFIC INEQUALITY CONSTRAINTS
    def PV_PowerLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        # PV real power limits (P_min, P_sun)
        pv = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if pv.get_active_status():

            if reserve == "up":
                cons = (
                    pv.get_samples(forecast_type="Pmin", scenario_nr=scenario_nr - 1),
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    + model.rDER_up[groupID, derID, frame_idx, scenario_nr],
                    pv.get_samples(forecast_type="Pmax", scenario_nr=scenario_nr - 1),
                )

            elif reserve == "dn":
                cons = (
                    pv.get_samples(forecast_type="Pmin", scenario_nr=scenario_nr - 1),
                    model.PDER[groupID, derID, frame_idx, scenario_nr]
                    - model.rDER_dn[groupID, derID, frame_idx, scenario_nr],
                    pv.get_samples(forecast_type="Pmax", scenario_nr=scenario_nr - 1),
                )
            else:
                cons = pe.Constraint.Skip
        else:
            cons = pe.Constraint.Skip
        return cons

    model.PV_PowerLim = pe.Constraint(
        model.PV_list, model.T, ["up", "dn"], model.Scenarios, rule=PV_PowerLim_rule
    )

    def PV_QLim_rule(model, derID, frame_idx, scenario_nr):
        # PV Thermal Line Limits (Maximum Apparent Power Constraint if Utility Scale or Reactive Power Limits if Aggregation)
        pv = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if pv.get_active_status() and pv.Q_min != None:
            cons = (
                pv.get_samples(forecast_type="Qmin", scenario_nr=scenario_nr - 1),
                model.QDER[groupID, derID, frame_idx, scenario_nr],
                pv.get_samples(forecast_type="Qmax", scenario_nr=scenario_nr - 1),
            )  # This is NOT a thermal limit.
        else:
            cons = pe.Constraint.Skip
        return cons

    model.PV_QLim = pe.Constraint(
        model.PV_list, model.T, model.Scenarios, rule=PV_QLim_rule
    )

    def PV_ThermalLim_rule(
        model, derID, frame_idx, schedule_case, scenario_nr, polytope_nr
    ):
        # PV Thermal Limits
        PV = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]

        if PV.get_active_status() and PV.S_max != None:
            # check whether NPolytope >=3
            if 3 in model.NPolytope:
                alpha = 2 * math.pi / len(model.NPolytope)
                m = (
                    math.sin(polytope_nr * alpha) - math.sin((polytope_nr - 1) * alpha)
                ) / (
                    math.cos(polytope_nr * alpha) - math.cos((polytope_nr - 1) * alpha)
                )
                if polytope_nr <= len(model.NPolytope) / 2:
                    cons = (
                        PV.S_max * math.sin(polytope_nr * alpha)
                        - m * PV.S_max * math.cos(polytope_nr * alpha)
                        >= model.QDER[groupID, derID, frame_idx, scenario_nr]
                        - m * model.PDER[groupID, derID, frame_idx, scenario_nr]
                    )
                else:
                    cons = (
                        PV.S_max * math.sin(polytope_nr * alpha)
                        - m * PV.S_max * math.cos(polytope_nr * alpha)
                        <= model.QDER[groupID, derID, frame_idx, scenario_nr]
                        - m * model.PDER[groupID, derID, frame_idx, scenario_nr]
                    )
            elif [1] == model.NPolytope:
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    + model.QDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    <= PV.S_max**2
                )
                print(
                    "Quadratic Constraint for " + PV.mRID + " - Scenario " + scenario_nr
                )

            else:
                print(
                    "Error Generating Maximum Apparent Power Flow Approximation. n_thermal not large enough. Using quadratic constraint instead. "
                )
                cons = (
                    model.PDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    + model.QDER[groupID, derID, frame_idx, scenario_nr] ** 2
                    <= PV.S_max**2
                )

        else:
            cons = pe.Constraint.Skip
        return cons

    model.PV_ThermalLim = pe.Constraint(
        model.PV_list,
        model.T,
        model.cases,
        model.Scenarios,
        model.NPolytope,
        rule=PV_ThermalLim_rule,
    )
    # ThermalLineLim_rule(model, lineID, phi_nr, frame_idx, schedule_case, scenario_nr, polytope_nr)
    # model.Lines, model.Phases, model.T, model.cases, model.Scenarios, model.NPolytope
    return model


def SmartChargeEV_Model(model: pe.Model) -> pe.Model:
    groupID = "SmartChargeEV"

    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network
    # need dealtaT in hours
    deltT = networkCase.timestep_period / 60

    # model.EV_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of SmartChargeEV')

    # CREATE EV SPECIFIC VARIABLES
    # Energy stored in SmartChargeEV resource
    model.E_EV = pe.Var(
        model.EV_list, model.T, model.Scenarios, within=pe.NonNegativeReals
    )

    # CREATE EV SPECIFIC EQUALITY CONSTRAINTS
    def EV_Energy_rule(model, derID, frame_idx, scenario_nr):
        # Calculate battery energy levels
        ev = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        if frame_idx == 1:
            cons = model.E_EV[derID, frame_idx, scenario_nr] == ev.E_0 - (
                ev.eff_c * model.PDER[groupID, derID, frame_idx, scenario_nr] * deltT
                + ev.get_samples(forecast_type="Del_Eev", scenario_nr=scenario_nr - 1)
            )
        else:
            cons = model.E_EV[derID, frame_idx, scenario_nr] == model.E_EV[
                derID, frame_idx - 1, scenario_nr
            ] - (
                ev.eff_c * model.PDER[groupID, derID, frame_idx, scenario_nr] * deltT
                + ev.get_samples(forecast_type="Del_Eev", scenario_nr=scenario_nr - 1)
            )
        return cons

    model.EV_Energy = pe.Constraint(
        model.EV_list, model.T, model.Scenarios, rule=EV_Energy_rule
    )

    # CREATE EV SPECIFIC INEQUALITY CONSTRAINTS
    def EV_PowerLimUp_rule(model, derID, frame_idx, scenario_nr):
        # EV Real Power Limits (P_min, 0)
        ev = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if ev.get_active_status():
            cons = (
                model.PDER[groupID, derID, frame_idx, scenario_nr]
                + model.rDER_up[groupID, derID, frame_idx, scenario_nr]
                <= 0
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.EV_PowerLimUp = pe.Constraint(
        model.EV_list, model.T, model.Scenarios, rule=EV_PowerLimUp_rule
    )

    def EV_PowerLimDn_rule(model, derID, frame_idx, scenario_nr):
        # EV Real Power Limits (P_min, 0)
        ev = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if ev.get_active_status():
            cons = (
                model.PDER[groupID, derID, frame_idx, scenario_nr]
                - model.rDER_dn[groupID, derID, frame_idx, scenario_nr]
                >= ev.get_samples(forecast_type="EVperc", scenario_nr=scenario_nr - 1)
                * ev.P_min
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.EV_PowerLimDn = pe.Constraint(
        model.EV_list, model.T, model.Scenarios, rule=EV_PowerLimDn_rule
    )

    # Is it OK to rename this EV_Reactive_Power_Limit_rule
    def EV_ThermalLim_rule(model, derID, frame_idx, scenario_nr):
        # EV Thermal Line Limits (Maximum Apparent Power Constraint if Utility Scale or Reactive Power Limits if Aggregation)
        ev = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if ev.get_active_status():
            EVperc = ev.get_samples(forecast_type="EVperc", scenario_nr=scenario_nr - 1)
            cons = (
                EVperc * ev.Q_min,
                model.QDER[groupID, derID, frame_idx, scenario_nr],
                EVperc * ev.Q_max,
            )

        else:
            cons = pe.Constraint.Skip
        return cons

    model.EV_ThermalLim = pe.Constraint(
        model.EV_list, model.T, model.Scenarios, rule=EV_ThermalLim_rule
    )

    def EV_EnergyLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        # EV Energy Limits (E_min, E_max)
        ev = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        EVperc = ev.get_samples(forecast_type="EVperc", scenario_nr=scenario_nr - 1)
        if reserve == "up":
            # supply more
            cons = model.E_EV[derID, frame_idx, scenario_nr] - model.rDER_up[
                groupID, derID, frame_idx, scenario_nr
            ] * deltT >= EVperc * ev.get_samples(
                forecast_type="Emin", scenario_nr=scenario_nr - 1
            )
        elif reserve == "dn":
            # dn, store more
            cons = (
                model.E_EV[derID, frame_idx, scenario_nr]
                + model.rDER_dn[groupID, derID, frame_idx, scenario_nr] * deltT
                <= EVperc * ev.E_max
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.EV_EnergyLim = pe.Constraint(
        model.EV_list, model.T, ["up", "dn"], model.Scenarios, rule=EV_EnergyLim_rule
    )

    return model


def VPP_Model(model: pe.Model) -> pe.Model:
    groupID = "VPP"
    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network

    # model.VPP_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of VPP')

    # CREATE VPP-SPECIFIC VARIABLES: None

    # CREATE VPP-SPECIFIC EQUALITY CONSTRAINTS: None

    # CREATE VPP-SPECIFIC INEQUALITY CONSTRAINTS
    def VPP_PowerLim_rule(model, derID, frame_idx, reserve, scenario_nr):
        # VPP Real Power Limits
        vpp = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if vpp.get_active_status():
            if reserve == "up":
                cons = model.PDER[
                    groupID, derID, frame_idx, scenario_nr
                ] + model.rDER_up[
                    groupID, derID, frame_idx, scenario_nr
                ] <= vpp.get_samples(
                    forecast_type="Pmax", scenario_nr=scenario_nr - 1
                )
            elif reserve == "dn":
                cons = model.PDER[
                    groupID, derID, frame_idx, scenario_nr
                ] - model.rDER_dn[
                    groupID, derID, frame_idx, scenario_nr
                ] >= vpp.get_samples(
                    forecast_type="Pmin", scenario_nr=scenario_nr - 1
                )
            else:
                cons = pe.Constraint.Skip
        else:
            cons = pe.Constraint.Skip
        return cons

    model.VPP_PowerLim = pe.Constraint(
        model.VPP_list, model.T, ["up", "dn"], model.Scenarios, rule=VPP_PowerLim_rule
    )

    def VPP_ReactivePowerLim_rule(model, derID, frame_idx, scenario_nr):
        # VPP Reactive Power Limits
        vpp = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check if DER is active
        if vpp.get_active_status():
            cons = (
                vpp.get_samples(forecast_type="Qmin", scenario_nr=scenario_nr - 1),
                model.QDER[groupID, derID, frame_idx, scenario_nr],
                vpp.get_samples(forecast_type="Qmax", scenario_nr=scenario_nr - 1),
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.VPP_ReactivePowerLim = pe.Constraint(
        model.VPP_list, model.T, model.Scenarios, rule=VPP_ReactivePowerLim_rule
    )

    return model


def TransactiveResource_Model(model):
    """
    Creates TransactiveResource constraints for all TransactiveResources within the network.

    Parameters
    ----------
    model : pyomo model to add TransactiveResource constraints to


    Returns
    -------
    model with added TransactiveResource variables and constraints
    """

    groupID = "TransactiveResource"
    networkCase = pe.value(model.networkCase)
    Net = networkCase.Network

    # model.TR_list = pe.Set(initialize = [der.getID() for der in Net.get_DERs() if der.get_type() == groupID], doc = 'Set of TransactiveResource')

    # CREATE TR SPECIFIC VARIABLES: already created in main code since it is referenced in objective function
    # if there are no transactive resources, then the variable PTR_shed will be empty

    # CREATE TR SPECIFIC EQUALITY CONSTRAINTS
    # Define power
    def TR_Power_rule(model, derID, frame_idx, scenario_nr):
        # Total TR power = sum of deviations from Pmin
        tr = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        cons = model.PDER[groupID, derID, frame_idx, scenario_nr] == tr.get_samples(
            forecast_type="TRpower", scenario_nr=scenario_nr - 1
        )[0] + sum(
            model.PTR_shed[derID, pwl, frame_idx, scenario_nr]
            for pwl in range(
                len(
                    tr.get_samples(forecast_type="TRprice", scenario_nr=scenario_nr - 1)
                )
                - 1
            )
        )
        return cons

    model.TR_power = pe.Constraint(
        model.TR_list, model.T, model.Scenarios, rule=TR_Power_rule
    )

    ## Fix reserves to zero
    def TR_zeroreservesUP_rule(model, derID, frame_idx, scenario_nr):
        cons = model.rDER_up[groupID, derID, frame_idx, scenario_nr] == 0
        return cons

    model.TR_zeroreservesUP = pe.Constraint(
        model.TR_list, model.T, model.Scenarios, rule=TR_zeroreservesUP_rule
    )

    def TR_zeroreservesDN_rule(model, derID, frame_idx, scenario_nr):
        cons = model.rDER_dn[groupID, derID, frame_idx, scenario_nr] == 0
        return cons

    model.TR_zeroreservesDN = pe.Constraint(
        model.TR_list, model.T, model.Scenarios, rule=TR_zeroreservesDN_rule
    )

    # CREATE TR SPECIFIC INEQUALITY CONSTRAINTS

    ## Bounds on the delta P components
    def TR_PowerShedLim_rule(model, derID, pwl, frame_idx, scenario_nr):
        # TransactiveResource power limits on the delta P components
        tr = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        pwls = tr.get_samples(forecast_type="TRpower", scenario_nr=scenario_nr - 1)
        cons = (
            0,
            model.PTR_shed[derID, pwl, frame_idx, scenario_nr],
            pwls[pwl + 1] - pwls[pwl],
        )
        return cons

    model.TR_PowerShedLim = pe.Constraint(
        model.TRshed, model.T, model.Scenarios, rule=TR_PowerShedLim_rule
    )

    ###### This should be dropped, and reactive power should be a fixed PF. ######
    # def TR_ThermalLim_rule(model, derID, frame_idx, scenario_nr):
    #     # TransactiveResource Reactive power limits
    #     tr = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs = derID)[0]
    #     if tr.get_active_status():
    #         cons = (tr.Q_min, model.QDER[groupID, derID, frame_idx, scenario_nr], tr.Q_max)
    #     else:
    #         cons = pe.Constraint.Skip
    #     return cons

    # model.TR_ThermalLim = pe.Constraint(model.TR_list, model.T, model.Scenarios, rule=TR_ThermalLim_rule)

    def TR_QLim_rule(model, derID, frame_idx, scenario_nr):
        tr = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        if tr.get_active_status():
            pf = tr.PF
            cons = model.QDER[groupID, derID, frame_idx, scenario_nr] == model.PDER[
                groupID, derID, frame_idx, scenario_nr
            ] * np.tan(np.arccos(pf))
        else:
            cons = pe.Constraint.Skip
        return cons

    model.TR_QLim = pe.Constraint(
        model.TR_list, model.T, model.Scenarios, rule=TR_QLim_rule
    )

    def TR_PowerLim_rule(model, derID, frame_idx, scenario_nr):
        # TransactiveResource power limits
        tr = networkCase.horizon[frame_idx - 1].get_DERs(mRIDs=derID)[0]
        # Check to see what values are acceptable based on the ISO cleared price
        price = networkCase.read_horizon_samples(
            "price", None, "pi", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
        )  # Get the ISO prices from the network case
        TRPower = tr.get_samples(forecast_type="TRpower", scenario_nr=scenario_nr - 1)
        TRPrices = tr.get_samples(forecast_type="TRprice", scenario_nr=scenario_nr - 1)
        Pmin = TRPower[0]  # This is the load at the lowest price
        Pmax = TRPower[len(TRPower) - 1]  # This is the load at the highest price.
        logger.debug(
            f"Building TMM PowerLimit: ISO Price: {price} ; Bid Prices: {TRPrices} ; Bid Powers: {TRPower}"
        )
        priceBoolean = np.array(TRPrices) >= price
        try:
            Minimum_ISO_power = min(
                [num for num, include in zip(TRPower, priceBoolean) if include]
            )  # This returns the lowest Power value of the TMM price bid that are acceptable given the ISO price.
            logger.debug(f"It worked, and the Minimum Power is {Minimum_ISO_power}")
        except:  # this could happen if the ISO price is greater than all prices in the bid.
            Minimum_ISO_power = Pmax
            logger.debug(
                f"Failed to identify a minimum power, so defaulting to Pmax: {Minimum_ISO_power} , meaning no flexibility"
            )
        if tr.get_active_status():
            cons = (
                max(Pmin, Minimum_ISO_power),
                model.PDER[groupID, derID, frame_idx, scenario_nr],
                Pmax,
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.TR_PowerLim = pe.Constraint(
        model.TR_list, model.T, model.Scenarios, rule=TR_PowerLim_rule
    )

    return model


def TR_ComputePrices(model: pe.Model, **kw_args):
    """
    Computes the prices for each TransactiveResource in the network.

    Parameters
    ----------
    model : pyomo model to add TransactiveResource constraints to

    Returns
    -------
    """
    case = pe.value(model.networkCase)
    selected_scenario_nr = kw_args.get("scenario", 1)
    try:
        selected_scenario = model.Scenarios.at(selected_scenario_nr)
    except:
        # Default to scenario 1 (expected value / single scenario)
        selected_scenario = model.Scenarios.at(1)

    TR_pi = {}

    for derID in model.TR_list:
        try:
            TR_setpowers = [
                pe.value(model.PDER["TransactiveResource", derID, t, selected_scenario])
                for t in model.T
            ]
            bid_samples = case.read_horizon_samples(
                "DER", derID, "TRprice", scenario_nr=selected_scenario_nr - 1
            )
            power_samples = case.read_horizon_samples(
                "DER", derID, "TRpower", scenario_nr=selected_scenario_nr - 1
            )

            # Determine the closest, larger price-quantity pair
            TR_pi[derID] = [0 for t in model.T]

            for TRprices, TRpowers, TR_setpower, frame_idx in zip(
                bid_samples, power_samples, TR_setpowers, model.T
            ):
                try:
                    TR_pi[derID][frame_idx - 1] = TRprices[
                        TRpowers >= TR_setpower
                    ].min()
                except:
                    TR_pi[derID][frame_idx - 1] = TRprices[-1]
                    logger.error(
                        f"No price found for TR {derID} at time {frame_idx} and power {TR_setpower}. Setting price to Max"
                    )
                    logger.debug(f"TRprices: {TRprices}")
                    logger.debug(f"TRpowers: {TRpowers}")
        except Exception as e:
            logger.error(f"Error while computing price for TR {derID}: {e}")

    return TR_pi


def DER_MPC_constraints(
    model: pe.Model,
    E_BAT_final_flag: bool = False,
    E_EV_final_flag: bool = False,
    E_FL_final_flag: bool = False,
) -> pe.Model:

    # Add slack variables
    model.eta_energy = pe.Var(
        model.Battery_list | model.FlexibleLoad_list | model.EV_list,
        within=pe.NonNegativeReals,
        doc="Final energy level slack variables",
    )

    networkCase = pe.value(model.networkCase)
    if E_BAT_final_flag:
        # Hard constraints on final battery energy
        for derID in model.Battery_list:
            model.eta_energy[derID] = 0
    else:
        # Modify final energy constraint to desired one
        model.BAT_finalEnergy.deactivate()

        def BAT_FinalEnergyUB_rule(model, derID, scenario_nr):
            battery = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_BAT[derID, model.T.at(-1), scenario_nr] - battery.E_F
                <= model.eta_energy[derID]
            )
            return cons

        model.BAT_finalEnergyUB = pe.Constraint(
            model.Battery_list, model.Scenarios, rule=BAT_FinalEnergyUB_rule
        )

        def BAT_FinalEnergyLB_rule(model, derID, scenario_nr):
            battery = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_BAT[derID, model.T.at(-1), scenario_nr] - battery.E_F
                >= -model.eta_energy[derID]
            )
            return cons

        model.BAT_finalEnergyLB = pe.Constraint(
            model.Battery_list, model.Scenarios, rule=BAT_FinalEnergyLB_rule
        )

    # Modify final energy constraint to desired one
    if E_FL_final_flag:

        def FL_FinalEnergy_rule(model, derID, scenario_nr):
            flexibleLoad = networkCase.Network.get_DERs(mRIDs=derID)[0]
            cons = model.E_FL[derID, model.T.at(-1), scenario_nr] == flexibleLoad.E_F
            return cons

        # Final energy level must be equal to final (reference) deterministic energy level
        model.FL_finalEnergy = pe.Constraint(
            model.FlexibleLoad_list, model.Scenarios, rule=FL_FinalEnergy_rule
        )

        for derID in model.FlexibleLoad_list:
            model.eta_energy[derID] = 0

    else:

        def FL_FinalEnergyUB_rule(model, derID, scenario_nr):
            flexibleLoad = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_FL[derID, model.T.at(-1), scenario_nr] - flexibleLoad.E_F
                <= model.eta_energy[derID]
            )
            return cons

        model.FL_finalEnergyUB = pe.Constraint(
            model.FlexibleLoad_list, model.Scenarios, rule=FL_FinalEnergyUB_rule
        )

        def FL_FinalEnergyLB_rule(model, derID, scenario_nr):
            flexibleLoad = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_FL[derID, model.T.at(-1), scenario_nr] - flexibleLoad.E_F
                >= -model.eta_energy[derID]
            )
            return cons

        model.FL_finalEnergyLB = pe.Constraint(
            model.FlexibleLoad_list, model.Scenarios, rule=FL_FinalEnergyLB_rule
        )

    # Modify final energy constraint to desired one
    if E_EV_final_flag:

        def EV_FinalEnergy_rule(model, derID, scenario_nr):
            ev = networkCase.Network.get_DERs(mRIDs=derID)[0]
            cons = model.E_EV[derID, model.T.at(-1), scenario_nr] == ev.E_F
            return cons

        # Final energy level must be equal to final (reference) deterministic energy level
        model.EV_finalEnergy = pe.Constraint(
            model.EV_list, model.Scenarios, rule=EV_FinalEnergy_rule
        )

        for derID in model.EV_list:
            model.eta_energy[derID] = 0

    else:

        def EV_FinalEnergyUB_rule(model, derID, scenario_nr):
            ev = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_EV[derID, model.T.at(-1), scenario_nr] - ev.E_F
                <= model.eta_energy[derID]
            )
            return cons

        model.EV_finalEnergyUB = pe.Constraint(
            model.EV_list, model.Scenarios, rule=EV_FinalEnergyUB_rule
        )

        def EV_FinalEnergyLB_rule(model, derID, scenario_nr):
            ev = networkCase.Network.get_DERs(mRIDs=derID)[0]
            # Final energy level constraint with slack variable
            cons = (
                model.E_EV[derID, model.T.at(-1), scenario_nr] - ev.E_F
                >= -model.eta_energy[derID]
            )
            return cons

        model.EV_finalEnergyLB = pe.Constraint(
            model.EV_list, model.Scenarios, rule=EV_FinalEnergyLB_rule
        )

    return model
