"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import pyomo.environ as pe

from ...Modeling.networkCase import NetworkCase


def hard_no_reverse_pf(model: pe.ConcreteModel):
    """
    Constraints to prevent reverse power flow at the substation.
    """

    def reverse_SubstationPower_rule(
        model, frame_idx, phi_nr, schedule_case, scenario_nr
    ):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr] >= 0
        )
        return cons

    model.reverse_pf_hard_rule = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=reverse_SubstationPower_rule,
    )
    return model


def soft_no_reverse_pf(model: pe.ConcreteModel, case: NetworkCase):
    """
    Constraints to prevent reverse power flow at the substation.
    """

    # Create tuple of active bus, phase, and time period for entire scheduling horizon
    active_flow = [
        (busID, phi_nr, frame_idx)
        for frame_idx in model.T
        for busID in case.horizon[frame_idx - 1].get_active_nodes()
        for phi_nr in [0, 1, 2]
    ]

    # Additional variables for reverse power flow constraint
    model.Psubpostive = pe.Var(
        active_flow, model.cases, model.Scenarios, within=pe.NonNegativeReals
    )
    model.Psubnegative = pe.Var(
        active_flow, model.cases, model.Scenarios, within=pe.NonNegativeReals
    )
    model.Psubpostive_flag_nonzero = pe.Var(
        active_flow, model.cases, model.Scenarios, within=pe.Binary
    )
    model.Psubnegative_flag_nonzero = pe.Var(
        active_flow, model.cases, model.Scenarios, within=pe.Binary
    )

    # Soft constraint on reverse powerflow at the substation

    def reverse_sum_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr]
            == model.Psubpostive[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            - model.Psubnegative[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
        )
        return cons

    model.reverse_pf_sum_hard_rule = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=reverse_sum_rule,
    )

    def reverse_flagpos_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Psubpostive[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            <= model.Psubpostive_flag_nonzero[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            * 10
        )
        return cons

    model.reverse_pf_flagpos_rule = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=reverse_flagpos_rule,
    )

    def reverse_flagneg_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Psubnegative[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            <= model.Psubnegative_flag_nonzero[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            * 10
        )
        return cons

    model.reverse_pf_flagneg_rule = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=reverse_flagneg_rule,
    )

    def reverse_flagzero_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Psubpostive_flag_nonzero[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            + model.Psubnegative_flag_nonzero[
                substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr
            ]
            <= 1
        )
        return cons

    model.reverse_pf_flagzero_rule = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=reverse_flagzero_rule,
    )

    return model
