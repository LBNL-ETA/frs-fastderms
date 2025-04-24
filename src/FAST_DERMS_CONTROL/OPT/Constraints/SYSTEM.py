"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import pyomo.environ as pe
import math
import logging

logger = logging.getLogger("FAST DERMS / OPT / Constraints / SYSTEM")


def Lin3DistFlow(model: pe.ConcreteModel):
    """
    Linearized AC Power Flow Constraints.
    State Variables: Real & Reactive Power into nodes, Nodal Voltage Magnitude Squared
    Returns model with the added Lin3DistFlow constraints
    """

    Net = pe.value(model.networkCase).Network

    # Setting the voltage magnitude squared at the substation head at all time periods
    model.Ykt[Net.get_substation_ID(), ...].fix(Net.get_substation_voltage() ** 2)

    # Voltage Magnitude Squared: Sched, Up, Dn (Only consider active lines)
    def voltage_rule(model, lineID, phi_nr, frame_idx, schedule_case, scenario_nr):
        Net = pe.value(model.networkCase).horizon[frame_idx - 1]
        line = Net.get_lines([lineID])[0]
        # Only consider active lines
        if line.get_active_status():
            voltage_drop = sum(
                line.MP[phi_nr][i]
                * model.Pkt[line.to_bus, i, frame_idx, schedule_case, scenario_nr]
                + line.MQ[phi_nr][i] * model.Qkt[line.to_bus, i, frame_idx, scenario_nr]
                for i in model.Phases
            )

            cons = (
                model.Ykt[line.from_bus, phi_nr, frame_idx, schedule_case, scenario_nr]
                == model.Ykt[line.to_bus, phi_nr, frame_idx, schedule_case, scenario_nr]
                - voltage_drop
            )
        else:
            cons = pe.Constraint.Skip
        return cons

    model.voltage = pe.Constraint(
        model.Lines,
        model.Phases,
        model.T,
        model.cases,
        model.Scenarios,
        rule=voltage_rule,
    )

    # Active Power Flow: Sched, Up, Dn
    def active_power_rule(model, busID, phi_nr, frame_idx, schedule_case, scenario_nr):

        Net = pe.value(model.networkCase).horizon[frame_idx - 1]
        active_nodes = Net.get_active_nodes()

        if schedule_case == "sched":
            cons = model.Pkt[
                busID, phi_nr, frame_idx, schedule_case, scenario_nr
            ] == model.PL[busID, phi_nr, frame_idx, scenario_nr] - model.PLDER[
                busID, phi_nr, frame_idx, scenario_nr
            ] + sum(
                Net.CG[active_nodes == busID, active_nodes == to_bus, phi_nr]
                * model.Pkt[to_bus, phi_nr, frame_idx, schedule_case, scenario_nr]
                for to_bus in active_nodes
            )

        elif schedule_case == "up":
            cons = model.Pkt[
                busID, phi_nr, frame_idx, schedule_case, scenario_nr
            ] == model.PL[busID, phi_nr, frame_idx, scenario_nr] - model.PLDER[
                busID, phi_nr, frame_idx, scenario_nr
            ] - model.RLup[
                busID, phi_nr, frame_idx, scenario_nr
            ] + sum(
                Net.CG[active_nodes == busID, active_nodes == to_bus, phi_nr]
                * model.Pkt[to_bus, phi_nr, frame_idx, schedule_case, scenario_nr]
                for to_bus in active_nodes
            )

        elif schedule_case == "dn":
            cons = model.Pkt[
                busID, phi_nr, frame_idx, schedule_case, scenario_nr
            ] == model.PL[busID, phi_nr, frame_idx, scenario_nr] - model.PLDER[
                busID, phi_nr, frame_idx, scenario_nr
            ] + model.RLdn[
                busID, phi_nr, frame_idx, scenario_nr
            ] + sum(
                Net.CG[active_nodes == busID, active_nodes == to_bus, phi_nr]
                * model.Pkt[to_bus, phi_nr, frame_idx, schedule_case, scenario_nr]
                for to_bus in active_nodes
            )
        else:
            cons = pe.Constraint.Skip

        return cons

    model.activepower = pe.Constraint(model.Pkt.keys(), rule=active_power_rule)

    # Reactive Power Flow: Sched case = Up case = Dn case
    def reactive_power_rule(model, busID, phi_nr, frame_idx, scenario_nr):
        Net = pe.value(model.networkCase).horizon[frame_idx - 1]
        active_nodes = Net.get_active_nodes()

        cons = model.Qkt[busID, phi_nr, frame_idx, scenario_nr] == model.QL[
            busID, phi_nr, frame_idx, scenario_nr
        ] - model.QLDER[busID, phi_nr, frame_idx, scenario_nr] + sum(
            Net.CG[active_nodes == busID, active_nodes == to_bus, phi_nr]
            * model.Qkt[to_bus, phi_nr, frame_idx, scenario_nr]
            for to_bus in active_nodes
        )

        return cons

    model.reactivepower = pe.Constraint(model.Qkt.keys(), rule=reactive_power_rule)

    return model


def system_level(model: pe.ConcreteModel):
    """
    Additional Power System Constraints.
    """

    # Voltage Magnitude Squared Limits
    def VoltageLim_rule(model, busID, phi_nr, frame_idx, schedule_case, scenario_nr):
        cons = (
            model.Vmin[busID] ** 2,
            model.Ykt[busID, phi_nr, frame_idx, schedule_case, scenario_nr],
            model.Vmax[busID] ** 2,
        )
        return cons

    model.VoltageLim = pe.Constraint(model.Ykt.keys(), rule=VoltageLim_rule)

    # Thermal Line Limits
    def ThermalLineLim_rule(
        model, lineID, phi_nr, frame_idx, schedule_case, scenario_nr, polytope_nr
    ):

        Net = pe.value(model.networkCase).horizon[frame_idx - 1]
        line = Net.get_lines([lineID])[0]

        if line.get_active_status():
            dn = line.to_bus
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
                        line.Smax * math.sin(polytope_nr * alpha)
                        - m * line.Smax * math.cos(polytope_nr * alpha)
                        >= model.Qkt[dn, phi_nr, frame_idx, scenario_nr]
                        - m
                        * model.Pkt[dn, phi_nr, frame_idx, schedule_case, scenario_nr]
                    )
                else:
                    cons = (
                        line.Smax * math.sin(polytope_nr * alpha)
                        - m * line.Smax * math.cos(polytope_nr * alpha)
                        <= model.Qkt[dn, phi_nr, frame_idx, scenario_nr]
                        - m
                        * model.Pkt[dn, phi_nr, frame_idx, schedule_case, scenario_nr]
                    )
            elif [1] == model.NPolytope:
                cons = (
                    model.Pkt[dn, phi_nr, frame_idx, schedule_case, scenario_nr] ** 2
                    + model.Qkt[dn, phi_nr, frame_idx, scenario_nr] ** 2
                    <= line.Smax**2
                )

            else:
                logger.error(
                    "Error Generating Maximum Apparent Power Flow Approximation. n_thermal not large enough. Using quadratic constraint instead. "
                )
                cons = (
                    model.Pkt[dn, phi_nr, frame_idx, schedule_case, scenario_nr] ** 2
                    + model.Qkt[dn, phi_nr, frame_idx, scenario_nr] ** 2
                    <= line.Smax**2
                )

        else:
            cons = pe.Constraint.Skip
        return cons

    model.ThermalLineLim = pe.Constraint(
        model.Lines,
        model.Phases,
        model.T,
        model.cases,
        model.Scenarios,
        model.NPolytope,
        rule=ThermalLineLim_rule,
    )

    return model


def specific_DA(model: pe.Model) -> pe.Model:

    model.eta = pe.Var(
        model.Phases,
        model.T,
        ["sched", "up", "dn"],
        model.Scenarios,
        within=pe.NonNegativeReals,
    )

    # Soft constraint of substation power
    # Enforce that the substation power is constant over all scenarios
    def softSubstationPowerUB_rule(
        model, frame_idx, phi_nr, schedule_case, scenario_nr
    ):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        if scenario_nr == 1:
            model.eta[phi_nr, frame_idx, schedule_case, scenario_nr].fix(0)
            cons = pe.Constraint.Skip
        else:
            cons = (
                model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr]
                - model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, 1]
                <= model.eta[phi_nr, frame_idx, schedule_case, scenario_nr]
            )
        return cons

    model.softSubstationPowerUB = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=softSubstationPowerUB_rule,
    )

    def softSubstationPowerLB_rule(
        model, frame_idx, phi_nr, schedule_case, scenario_nr
    ):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        if scenario_nr == 1:
            model.eta[phi_nr, frame_idx, schedule_case, scenario_nr].fix(0)
            cons = pe.Constraint.Skip
        else:
            cons = (
                model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr]
                - model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, 1]
                >= -model.eta[phi_nr, frame_idx, schedule_case, scenario_nr]
            )
        return cons

    model.softSubstationPowerLB = pe.Constraint(
        model.T,
        model.Phases,
        ["sched", "up", "dn"],
        model.Scenarios,
        rule=softSubstationPowerLB_rule,
    )

    # Define Rup in terms of reserve powerflow at substation
    # This constraint is used to compute the up reserve
    def defineRup_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        return model.Rup[frame_idx] == sum(
            -model.Pkt[substation_ID, phi, frame_idx, "up", scenario_nr]
            + model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
            for phi in model.Phases
        )

    model.defineRup = pe.Constraint(model.T, model.Scenarios, rule=defineRup_rule)

    # Define Rdn in terms of reserve powerflow at substation
    def defineRdn_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        return model.Rdn[frame_idx] == sum(
            -model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
            + model.Pkt[substation_ID, phi, frame_idx, "dn", scenario_nr]
            for phi in model.Phases
        )

    model.defineRdn = pe.Constraint(model.T, model.Scenarios, rule=defineRdn_rule)

    ## OLD CODE
    def softSubstationPower_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        if scenario_nr == 1:
            model.eta[phi_nr, frame_idx, schedule_case, scenario_nr].fix(0)
            cons = pe.Constraint.Skip
        else:
            cons = (
                -model.eta[phi_nr, frame_idx, schedule_case, scenario_nr],
                model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr]
                - model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, 1],
                model.eta[phi_nr, frame_idx, schedule_case, scenario_nr],
            )
        return cons

    # model.softSubstationPower = pe.Constraint(model.T, model.Phases, ['sched', 'up', 'dn'], model.Scenarios, rule=softSubstationPower_rule)

    # Hard Substation Power Constraint was replaced by a soft constraint
    def hardSubstationPower_rule(model, frame_idx, phi_nr, schedule_case, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        if scenario_nr == 1:
            model.eta[phi_nr, frame_idx, schedule_case, scenario_nr].fix(0)
            cons = pe.Constraint.Skip
        else:
            cons = (
                model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, scenario_nr]
                == model.Pkt[substation_ID, phi_nr, frame_idx, schedule_case, 1]
            )
        return cons

    # model.hardSubstationPower = pe.Constraint(model.T, model.Phases, ['sched', 'up', 'dn'], model.Scenarios, rule=hardSubstationPower_rule)

    return model


def specific_DA_DET(model: pe.Model) -> pe.Model:

    # non-negative slack variable that penalizes deviations from day-ahead optimal substation power setpoint
    model.eta = pe.Var(
        model.Phases,
        model.T,
        ["sched"],
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft power setpoint constraint",
    )

    # SOFT CONSTRAINT OF SUBSTATION POWER
    def softSubstationPowerUB_rule(model, frame_idx, phase_nr, scenario_nr):
        case = pe.value(model.networkCase)
        substation_ID = case.Network.get_substation_ID()
        Net = case.horizon[frame_idx - 1]
        cons = (
            sum(
                model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            - Net.get_substation_power(None)
            <= model.eta[phase_nr, frame_idx, "sched", scenario_nr]
        )
        return cons

    model.softSubstationPowerUB = pe.Constraint(
        model.T, model.Phases, model.Scenarios, rule=softSubstationPowerUB_rule
    )

    def softSubstationPowerLB_rule(model, frame_idx, phase_nr, scenario_nr):
        case = pe.value(model.networkCase)
        substation_ID = case.Network.get_substation_ID()
        Net = case.horizon[frame_idx - 1]
        cons = (
            sum(
                model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            - Net.get_substation_power(None)
            >= -model.eta[phase_nr, frame_idx, "sched", scenario_nr]
        )
        return cons

    model.softSubstationPowerLB = pe.Constraint(
        model.T, model.Phases, model.Scenarios, rule=softSubstationPowerLB_rule
    )

    # non-negative slack variable that penalizes deviations from day-ahead reserve solutions
    model.eta_up_reserves = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft up reserves constraint",
    )

    # Define Rup in terms of reserve powerflow at substation
    def defineRupUB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rup[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "up", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            <= model.eta_up_reserves[frame_idx]
        )
        return cons

    model.defineRupUB = pe.Constraint(model.T, model.Scenarios, rule=defineRupUB_rule)

    def defineRupLB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rup[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "up", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            >= -model.eta_up_reserves[frame_idx]
        )
        return cons

    model.defineRupLB = pe.Constraint(model.T, model.Scenarios, rule=defineRupLB_rule)

    # non-negative slack variable that penalizes deviations from day-ahead reserve solutions
    model.eta_dn_reserves = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft dn reserves constraint",
    )

    # Define Rdn in terms of reserve powerflow at substation
    def defineRdnUB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rdn[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "dn", scenario_nr]
                for phi in model.Phases
            )
            <= model.eta_dn_reserves[frame_idx]
        )
        return cons

    model.defineRdnUB = pe.Constraint(model.T, model.Scenarios, rule=defineRdnUB_rule)

    def defineRdnLB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rdn[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "dn", scenario_nr]
                for phi in model.Phases
            )
            >= -model.eta_dn_reserves[frame_idx]
        )
        return cons

    model.defineRdnLB = pe.Constraint(model.T, model.Scenarios, rule=defineRdnLB_rule)

    return model


def specific_MPC(model: pe.Model) -> pe.Model:
    # non-negative slack variable that penalizes deviations from day-ahead optimal substation power setpoint
    model.eta = pe.Var(
        model.Phases,
        model.T,
        ["sched"],
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft power setpoint constraint",
    )

    # SOFT CONSTRAINT OF SUBSTATION POWER
    def softSubstationPowerUB_rule(model, frame_idx, phase_nr, scenario_nr):
        case = pe.value(model.networkCase)
        substation_ID = case.Network.get_substation_ID()
        Net = case.horizon[frame_idx - 1]
        cons = (
            sum(
                model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            - Net.get_substation_power(None)
            <= model.eta[phase_nr, frame_idx, "sched", scenario_nr]
        )
        return cons

    model.softSubstationPowerUB = pe.Constraint(
        model.T, model.Phases, model.Scenarios, rule=softSubstationPowerUB_rule
    )

    def softSubstationPowerLB_rule(model, frame_idx, phase_nr, scenario_nr):
        case = pe.value(model.networkCase)
        substation_ID = case.Network.get_substation_ID()
        Net = case.horizon[frame_idx - 1]
        cons = (
            sum(
                model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            - Net.get_substation_power(None)
            >= -model.eta[phase_nr, frame_idx, "sched", scenario_nr]
        )
        return cons

    model.softSubstationPowerLB = pe.Constraint(
        model.T, model.Phases, model.Scenarios, rule=softSubstationPowerLB_rule
    )

    # Distribution up reserves calculated in preprocessing
    model.Rup_dist = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Desired (distribution, preprocessing) up reserves",
    )
    # non-negative slack variable that penalizes deviations from day-ahead reserve solutions
    model.eta_up_reserves = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft up reserves constraint",
    )

    # Define Rup in terms of reserve powerflow at substation
    def defineRupUB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rup[frame_idx]
            + model.Rup_dist[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "up", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            <= model.eta_up_reserves[frame_idx]
        )
        return cons

    model.defineRupUB = pe.Constraint(model.T, model.Scenarios, rule=defineRupUB_rule)

    def defineRupLB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rup[frame_idx]
            + model.Rup_dist[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "up", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                for phi in model.Phases
            )
            >= -model.eta_up_reserves[frame_idx]
        )
        return cons

    model.defineRupLB = pe.Constraint(model.T, model.Scenarios, rule=defineRupLB_rule)

    # Distribution dn reserves calculated in preprocessing
    model.Rdn_dist = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Desired (distribution, preprocessing) dn reserves",
    )
    # non-negative slack variable that penalizes deviations from day-ahead reserve solutions
    model.eta_dn_reserves = pe.Var(
        model.T,
        within=pe.NonNegativeReals,
        doc="Slack variable for soft dn reserves constraint",
    )

    # Define Rdn in terms of reserve powerflow at substation
    def defineRdnUB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rdn[frame_idx]
            + model.Rdn_dist[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "dn", scenario_nr]
                for phi in model.Phases
            )
            <= model.eta_dn_reserves[frame_idx]
        )
        return cons

    model.defineRdnUB = pe.Constraint(model.T, model.Scenarios, rule=defineRdnUB_rule)

    def defineRdnLB_rule(model, frame_idx, scenario_nr):
        substation_ID = pe.value(model.networkCase).Network.get_substation_ID()
        cons = (
            model.Rdn[frame_idx]
            + model.Rdn_dist[frame_idx]
            - sum(
                -model.Pkt[substation_ID, phi, frame_idx, "sched", scenario_nr]
                + model.Pkt[substation_ID, phi, frame_idx, "dn", scenario_nr]
                for phi in model.Phases
            )
            >= -model.eta_dn_reserves[frame_idx]
        )
        return cons

    model.defineRdnLB = pe.Constraint(model.T, model.Scenarios, rule=defineRdnLB_rule)

    return model
