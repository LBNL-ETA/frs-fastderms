"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import pyomo.environ as pe
from ...Modeling.network import Network
from ...Modeling.networkCase import NetworkCase

# from math import sqrt, pow


def power_cost(model: pe.Model, networkCase: NetworkCase, frame_idx, scenario_nr):

    price = networkCase.read_horizon_samples(
        "price", None, "pi", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
    )
    substation_ID = networkCase.Network.get_substation_ID()

    cost = price * sum(
        model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
        for phi in model.Phases
    )

    return cost


def reverse_pf_cost(model: pe.Model, networkCase: NetworkCase, frame_idx, scenario_nr):

    substation_ID = networkCase.Network.get_substation_ID()
    price = networkCase.read_horizon_samples(
        "price", None, "pi", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
    )
    cost = price * sum(
        model.Psubnegative[substation_ID, phi, frame_idx, "sched", scenario_nr]
        for phi in model.Phases
    )

    return cost


def transactive_cost(model: pe.Model, networkCase: NetworkCase, frame_idx, scenario_nr):
    # min_TR_load_cost = sum((-networkCase.read_horizon_samples('DER', der.getID(), 'TRprice', frame_idx = frame_idx-1, scenario_nr = scenario_nr-1)[0] * networkCase.read_horizon_samples('DER', der.getID(), 'TRpower', frame_idx = frame_idx-1, scenario_nr = scenario_nr-1)[0]) for der in networkCase.Network.get_DERs() if der.get_type() == 'TransactiveResource') #this value doesn't include a variable, so we can ignore it.

    load_shed_cost = sum(
        networkCase.read_horizon_samples(
            "DER",
            derID,
            "TRprice",
            frame_idx=frame_idx - 1,
            scenario_nr=scenario_nr - 1,
        )[pwl + 1]
        * model.PTR_shed[derID, pwl, frame_idx, scenario_nr]
        for (derID, pwl) in model.TRshed
    )

    cost = load_shed_cost  # + min_TR_load_cost

    return cost


def loss_cost(model: pe.Model, networkCase: NetworkCase, frame_idx, scenario_nr):

    price = networkCase.read_horizon_samples(
        "price", None, "pi", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
    )

    cost = price * sum(
        line.Raa
        * (
            model.Pkt[line.to_bus, 0, frame_idx, "sched", scenario_nr] ** 2
            + model.Qkt[line.to_bus, 0, frame_idx, scenario_nr] ** 2
        )
        + line.Rbb
        * (
            model.Pkt[line.to_bus, 1, frame_idx, "sched", scenario_nr] ** 2
            + model.Qkt[line.to_bus, 1, frame_idx, scenario_nr] ** 2
        )
        + line.Rcc
        * (
            model.Pkt[line.to_bus, 2, frame_idx, "sched", scenario_nr] ** 2
            + model.Qkt[line.to_bus, 2, frame_idx, scenario_nr] ** 2
        )
        for line in networkCase.horizon[frame_idx - 1].get_lines()
        if line.get_active_status()
    )

    return cost


def reserve_cost(model: pe.Model, networkCase: NetworkCase, frame_idx, scenario_nr):
    price_up = networkCase.read_horizon_samples(
        "price", None, "pi_rup", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
    )
    price_dn = networkCase.read_horizon_samples(
        "price", None, "pi_rdn", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
    )

    cost = -(price_up * model.Rup[frame_idx] + price_dn * model.Rdn[frame_idx])

    return cost


# deviations from day-ahead optimal substation power setpoint
def substation_deviation_cost(
    model: pe.Model, frame_idx, scenario_nr, schedule_cases, price_override=None
):
    networkCase = pe.value(model.networkCase)
    if price_override is None:
        price = networkCase.read_horizon_samples(
            "price", None, "pi", frame_idx=frame_idx - 1, scenario_nr=scenario_nr - 1
        )
    else:
        price = price_override
    cost = price * sum(
        sum(
            model.eta[phi_nr, frame_idx, schedule_case, scenario_nr]
            for phi_nr in model.Phases
        )
        for schedule_case in schedule_cases
    )
    return cost


def reserves_deviation_cost(
    model: pe.Model, frame_idx, scenario_nr, price_override=None
):
    networkCase = pe.value(model.networkCase)
    if price_override is None:
        price_up = networkCase.read_horizon_samples(
            "price",
            None,
            "pi_rup",
            frame_idx=frame_idx - 1,
            scenario_nr=scenario_nr - 1,
        )
        price_dn = networkCase.read_horizon_samples(
            "price",
            None,
            "pi_rdn",
            frame_idx=frame_idx - 1,
            scenario_nr=scenario_nr - 1,
        )
    else:
        price_up = price_override
        price_dn = price_override
    cost = (
        price_up * model.eta_up_reserves[frame_idx]
        + price_dn * model.eta_dn_reserves[frame_idx]
    )
    return cost


# deviations final battery storage
def battery_deviation_cost(model: pe.Model, derID):
    cost = model.eta_energy[derID]
    return cost


# discourage battery charging / discharging within each time interval
def battery_charge_discharge_cost(model: pe.Model, derID, frame_idx, scenario_nr):
    # the price should be balanced in such a way as it would reflect the
    cost = (
        model.Pd_BAT[derID, frame_idx, scenario_nr] ** 2
        + model.Pc_BAT[derID, frame_idx, scenario_nr] ** 2
    )
    return cost


def PV_curtailment_cost(model: pe.Model, derID, frame_idx, scenario_nr):
    # Current formulation is simply a very mild incentive to increase the power out of PV.  It may be more realistic to think that the opportunity cost for curtailment is what would be paid to the PV (assuming it is a merchant generator). This would mean the price would be the real-ISO price times the curtailed power.
    cost = model.PDER["PV", derID, frame_idx, scenario_nr]
    return cost


### Objective construction below


def load_serving_cost(model: pe.Model, **kw_args):

    # Get options out of kw_args
    power_weight = kw_args.get("power_weight", 1)
    transactive_weight = kw_args.get("transactive_weight", 1)
    reserve_weight = kw_args.get("reserve_weight", 1)
    loss_weight = kw_args.get("loss_weight", 0)
    reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
    substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
    substation_deviation_price_override = kw_args.get(
        "substation_deviation_price_override", None
    )

    networkCase = pe.value(model.networkCase)

    # Linear Cost Function (no losses)
    base_cost = sum(
        sum(
            power_weight * power_cost(model, networkCase, frame_idx, scenario_nr)
            + transactive_weight
            * transactive_cost(model, networkCase, frame_idx, scenario_nr)
            + reserve_weight * reserve_cost(model, networkCase, frame_idx, scenario_nr)
            + substation_deviation_weight
            * substation_deviation_cost(
                model,
                frame_idx,
                scenario_nr,
                ["sched", "up", "dn"],
                substation_deviation_price_override,
            )
            for scenario_nr in model.Scenarios
        )
        for frame_idx in model.T
    )

    optional_cost = 0
    # Penalty for reverse power flow
    if not (reverse_pf_weight is None or reverse_pf_weight == 0):
        optional_cost += reverse_pf_weight * sum(
            sum(
                reverse_pf_cost(model, networkCase, frame_idx, scenario_nr)
                for scenario_nr in model.Scenarios
            )
            for frame_idx in model.T
        )

    # Losses (Quadratic !)
    if loss_weight != 0:
        optional_cost += loss_weight * sum(
            sum(
                loss_cost(model, networkCase, frame_idx, scenario_nr)
                for scenario_nr in model.Scenarios
            )
            for frame_idx in model.T
        )

    # Normalize the cost by number of scenarios
    total_cost = (1 / len(model.Scenarios)) * (base_cost + optional_cost)

    return total_cost


def load_serving_cost_DET(model: pe.Model, **kw_args):

    # Get options out of kw_args
    power_weight = kw_args.get("power_weight", 1)
    transactive_weight = kw_args.get("transactive_weight", 1)
    loss_weight = kw_args.get("loss_weight", 1)
    reserve_weight = kw_args.get("reserve_weight", 1)
    reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
    substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
    substation_deviation_price_override = kw_args.get(
        "substation_deviation_price_override", None
    )
    reserves_deviation_weight = kw_args.get("reserves_deviation_weight", 1)
    reserves_deviation_price_override = kw_args.get(
        "reserves_deviation_price_override", None
    )

    networkCase = pe.value(model.networkCase)

    # Linear Cost Function (no losses)
    base_cost = sum(
        (1 / len(model.Scenarios))
        * sum(
            power_weight * power_cost(model, networkCase, frame_idx, scenario_nr)
            + transactive_weight
            * transactive_cost(model, networkCase, frame_idx, scenario_nr)
            + reserve_weight * reserve_cost(model, networkCase, frame_idx, scenario_nr)
            + substation_deviation_weight
            * substation_deviation_cost(
                model,
                frame_idx,
                scenario_nr,
                ["sched"],
                substation_deviation_price_override,
            )
            + reserves_deviation_weight
            * reserves_deviation_cost(
                model, frame_idx, scenario_nr, reserves_deviation_price_override
            )
            for scenario_nr in model.Scenarios
        )
        for frame_idx in model.T
    )
    optional_cost = 0
    # Penalty for reverse power flow
    if not (reverse_pf_weight is None or reverse_pf_weight == 0):
        optional_cost += reverse_pf_weight * sum(
            sum(
                reverse_pf_cost(model, networkCase, frame_idx, scenario_nr)
                for scenario_nr in model.Scenarios
            )
            for frame_idx in model.T
        )

    # Losses (Quadratic !)
    if loss_weight != 0:
        optional_cost += loss_weight * sum(
            sum(
                loss_cost(model, networkCase, frame_idx, scenario_nr)
                for scenario_nr in model.Scenarios
            )
            for frame_idx in model.T
        )

    # Normalize the cost by number of scenarios
    total_cost = (1 / len(model.Scenarios)) * (base_cost + optional_cost)

    return total_cost


def discrepancy_cost(model: pe.Model, **kw_args):

    # Get options out of kw_args
    substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
    substation_deviation_price_override = kw_args.get(
        "substation_deviation_price_override", None
    )
    reserves_deviation_weight = kw_args.get("reserves_deviation_weight", 1)
    reserves_deviation_price_override = kw_args.get(
        "reserves_deviation_price_override", None
    )
    pv_curtailment_weight = kw_args.get("pv_curtailment_weight", 1)
    battery_charge_discharge_weight = kw_args.get("battery_charge_discharge_weight", 1)
    battery_deviation_weight = kw_args.get("battery_deviation_weight", 1000)

    base_cost = sum(
        sum(
            substation_deviation_weight
            * substation_deviation_cost(
                model,
                frame_idx,
                scenario_nr,
                ["sched"],
                substation_deviation_price_override,
            )
            + reserves_deviation_weight
            * reserves_deviation_cost(
                model, frame_idx, scenario_nr, reserves_deviation_price_override
            )
            + pv_curtailment_weight
            * sum(
                PV_curtailment_cost(model, derID, frame_idx, scenario_nr)
                for derID in model.PV_list
            )
            + battery_charge_discharge_weight
            * sum(
                battery_charge_discharge_cost(model, derID, frame_idx, scenario_nr)
                for derID in model.Battery_list
            )
            for frame_idx in model.T
        )
        for scenario_nr in model.Scenarios
    ) + battery_deviation_weight * sum(
        battery_deviation_cost(model, derID)
        for derID in model.Battery_list | model.FlexibleLoad_list | model.EV_list
    )

    # Normalize the cost by number of scenarios
    total_cost = (1 / len(model.Scenarios)) * base_cost

    return total_cost


def MPC_cost(model: pe.Model, **kw_args):

    # Get options out of kw_args
    power_weight = kw_args.get("power_weight", 1)
    transactive_weight = kw_args.get("transactive_weight", 1)
    reserve_weight = kw_args.get("reserve_weight", 1)
    substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
    substation_deviation_price_override = kw_args.get(
        "substation_deviation_price_override", None
    )
    reserves_deviation_weight = kw_args.get("reserves_deviation_weight", 1)
    reserves_deviation_price_override = kw_args.get(
        "reserves_deviation_price_override", None
    )
    pv_curtailment_weight = kw_args.get("pv_curtailment_weight", 1)
    battery_charge_discharge_weight = kw_args.get("battery_charge_discharge_weight", 1)
    battery_deviation_weight = kw_args.get("battery_deviation_weight", 1000)

    networkCase = pe.value(model.networkCase)

    total_cost = sum(
        (1 / len(model.Scenarios))
        * sum(
            power_weight * power_cost(model, networkCase, frame_idx, scenario_nr)
            + transactive_weight
            * transactive_cost(model, networkCase, frame_idx, scenario_nr)
            + reserve_weight * reserve_cost(model, networkCase, frame_idx, scenario_nr)
            + substation_deviation_weight
            * substation_deviation_cost(
                model,
                frame_idx,
                scenario_nr,
                ["sched"],
                substation_deviation_price_override,
            )
            + reserves_deviation_weight
            * reserves_deviation_cost(
                model, frame_idx, scenario_nr, reserves_deviation_price_override
            )
            + pv_curtailment_weight
            * sum(
                PV_curtailment_cost(model, derID, frame_idx, scenario_nr)
                for derID in model.PV_list
            )
            + battery_charge_discharge_weight
            * sum(
                battery_charge_discharge_cost(model, derID, frame_idx, scenario_nr)
                for derID in model.Battery_list
            )
            for scenario_nr in model.Scenarios
        )
        for frame_idx in model.T
    ) + battery_deviation_weight * sum(
        battery_deviation_cost(model, derID)
        for derID in model.Battery_list | model.FlexibleLoad_list | model.EV_list
    )

    return total_cost
