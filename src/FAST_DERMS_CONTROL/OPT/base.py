"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import pyomo.environ as pe
import numpy as np

import logging

from ..Modeling.networkCase import NetworkCase
from .Rules.Powers import PL_init_rule, QL_init_rule


def initPowerSystem(
    model: pe.Model, case: NetworkCase = None, n_thermal: int = None
) -> pe.Model:

    Net = case.Network
    settings = case.get_case_settings()

    # Create Sets
    # Set of time periods within scheduling horizon
    model.T = pe.RangeSet(settings["n_timestep"], doc="Set of Time Periods")
    # Number of scenarios included in DA problem
    model.Scenarios = pe.RangeSet(settings["n_scenario"], doc="Set of Scenarios")
    # Set of scheduled, up reserve, and down reserve cases
    model.cases = pe.Set(initialize=["sched", "up", "dn"], doc="Set of Cases")
    # Set of nodes, includes substation/substation head
    model.Nodes = pe.Set(
        initialize=Net.get_nodes(),
        ordered=pe.Set.SortedOrder,
        doc="Set of Nodes by mRID",
    )
    # Set of phases (a,b,c)
    model.Phases = pe.Set(
        initialize=[0, 1, 2], ordered=pe.Set.SortedOrder, doc="Set of Phases"
    )
    # Index of Lines by mRID
    model.Lines = pe.Set(
        initialize=[line.getID() for line in Net.get_lines()],
        doc="Set of Lines by mRID",
    )
    # Set of DER types present in network
    model.DERtype = pe.Set(
        initialize=Net.get_DER_types(), doc="Set of Present DER classes"
    )

    # Number of linear constraint segments generated in inner approx if N>=3 else quadratic constraint (N=1)
    model.NPolytope = pe.RangeSet(
        1 if n_thermal == None else n_thermal, doc="Set of Line Segments in Lin Approx"
    )

    # Transactive Resources Schedule
    # excluding the min power-price pair. Indexed by DER ID and pair number after Pmin-Qmin .e.g. ('TR_Agg1',0), ('TR_Agg1', 1)
    model.TRshed = pe.Set(
        initialize=[
            (der.getID(), s)
            for der in Net.get_DERs()
            if der.get_type() == "TransactiveResource"
            for s in range(
                case.horizon[0].get_DERs(mRIDs=der.getID())[0].get_number_pairs() - 1
            )
        ],
        doc="Set of transactive resource piecewise components",
    )

    # Create Parameters
    # Store a copy of NetworkCase
    model.networkCase = pe.Param(
        initialize=case,
        within=pe.Any,
        doc="copy of the current NetworkCase",
        mutable=True,
    )
    # Minimum voltage magnitude [pu]
    model.Vmin = pe.Param(
        model.Nodes, initialize=Net.Vmin, doc="Min Voltage Magnitude [pu]"
    )
    # Maximum voltage magnitude [pu]
    model.Vmax = pe.Param(
        model.Nodes, initialize=Net.Vmax, doc="Max Voltage Magnitude [pu]"
    )

    # Scenarios of real power load at bus k, phase phi, time t [pu]
    model.PL = pe.Param(
        model.Nodes,
        model.Phases,
        model.T,
        model.Scenarios,
        initialize=PL_init_rule,
        doc="Active Power Demand Scenarios [pu]",
    )

    # Scenarios of reactive power load at bus k, phase phi, time t [pu]
    model.QL = pe.Param(
        model.Nodes,
        model.Phases,
        model.T,
        model.Scenarios,
        initialize=QL_init_rule,
        doc="Reactive Power Demand Scenarios [pu]",
    )

    # Create tuple of active bus, phase, and time period for entire scheduling horizon
    active_flow = [
        (busID, phi_nr, frame_idx)
        for frame_idx in model.T
        for busID in case.horizon[frame_idx - 1].get_active_nodes()
        for phi_nr in [0, 1, 2]
    ]
    # List of tuples of (DER type, DER id, time period, scenario)
    active_DERs = [
        (der.get_type(), der.getID(), frame_idx, scenario_nr)
        for frame_idx in model.T
        for der in case.horizon[frame_idx - 1].get_DERs()
        for scenario_nr in model.Scenarios
    ]

    # Create Power Flow State Variables (only considering active lines/nodes)
    # Real Power Flowing into node i, phase phi, time t [pu]
    model.Pkt = pe.Var(
        active_flow, model.cases, model.Scenarios, doc="Real Power Flow [pu]"
    )
    # Reactive Power Flowing into node i (Sched=Up=Dn) [pu]
    model.Qkt = pe.Var(active_flow, model.Scenarios, doc="Reactive Power Flow [pu]")
    # Voltage magnitude squared at node i [pu]^2
    model.Ykt = pe.Var(
        active_flow,
        model.cases,
        model.Scenarios,
        doc="Voltage Magnitude Squared [pu]^2",
    )

    # Create Scheduled Decision Variables -- varies with every scenario (only considering active lines/nodes)
    # Scheduled real power set point of DERs at bus k, phase phi, time t. Generator sign convention.  [pu]
    model.PLDER = pe.Var(
        active_flow, model.Scenarios, doc="Total Network DER Real Power Setpoints [pu]"
    )
    # Scheduled reactive power set point of DERs at bus k, phase phi, time t  [pu]
    model.QLDER = pe.Var(
        active_flow,
        model.Scenarios,
        doc="Total Network DER Reactive Power Setpoints [pu]",
    )
    # Scheduled real power set point of DERs, by component type and id, at time t and scenario s [pu]
    model.PDER = pe.Var(active_DERs, doc="Real Power Setpoints of DERs [pu]")
    # Scheduled reactive power set point of DERs, by component type and id, at time t and scenario s [pu]
    model.QDER = pe.Var(active_DERs, doc="Reactive Power Setpoints of DERs [pu]")
    # delta P, piecewise deviations in power (load shed) from min power
    model.PTR_shed = pe.Var(
        model.TRshed,
        model.T,
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="pwl load shed components of transactive resources",
    )

    # Create Reserve Decision Variables
    # Total up reserves capacity offer for the network at substation head, time t (sum of 3-phases) [pu]
    model.Rup = pe.Var(
        model.T, within=pe.NonNegativeReals, doc="Substation Up Reserves [pu]"
    )
    # Total dn reserves capacity offer for the network at substation head, time t (sum of 3-phases)[pu]
    model.Rdn = pe.Var(
        model.T, within=pe.NonNegativeReals, doc="Substation Dn Reserves [pu]"
    )
    # Up real power reserve capacity of DERs at bus k, phase phi, time t  [pu]
    model.RLup = pe.Var(
        active_flow,
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Network Up Reserves [pu]",
    )
    # Down real power reserve capacity of DERs at bus k, phase phi, time t  [pu]
    model.RLdn = pe.Var(
        active_flow,
        model.Scenarios,
        within=pe.NonNegativeReals,
        doc="Network Dn Reserves [pu]",
    )
    # Up real power reserve capacity of DERs, by component type and id, at time t [pu]
    model.rDER_up = pe.Var(
        active_DERs, within=pe.NonNegativeReals, doc="Up Reserves of DERs [pu]"
    )
    # Down real power reserve capacity of DERs, by component type and id, at time t [pu]
    model.rDER_dn = pe.Var(
        active_DERs, within=pe.NonNegativeReals, doc="Dn Reserves of DERs [pu]"
    )

    return model
