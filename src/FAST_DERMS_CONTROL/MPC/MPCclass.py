"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..Modeling.networkCase import NetworkCase
from ..OPT.helpers import OptEngine
from ..OPT.base import initPowerSystem
from ..OPT.Rules.Objective import discrepancy_cost, MPC_cost
from ..OPT.Constraints.SYSTEM import Lin3DistFlow, system_level, specific_MPC
from ..OPT.Constraints.DERs import (
    DER_constraints,
    DER_MPC_constraints,
    TR_ComputePrices,
)

from typing import Dict, List
from tabulate import tabulate
from pathlib import Path
from itertools import chain

import pyomo.environ as pe
import math


class ModelPredict(OptEngine):
    def __init__(self, IOmodule, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        # Store MPC settings
        self.IO = IOmodule

        # Initialize Attributes
        self.E_init = None
        self.MPC_model = None

        self.logger.info(f"MPC object initialized")

    def init_MPC_model(self, case: NetworkCase = None, **kw_args):
        """
        Create model for linearized, unbalanced power flow, system constraints, and DER constraints.
        Solve for up/down reserve capacity at the substation head given N scenarios.
        """
        # Get options out of kw_args
        n_thermal = kw_args.get("n_thermal", 20)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        # Objective weights
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
        E_BAT_final_flag = kw_args.get("E_BAT_final_flag", True)
        E_EV_final_flag = kw_args.get("E_EV_final_flag", True)
        E_FL_final_flag = kw_args.get("E_FL_final_flag", True)

        # Set up Model
        self.logger.info("MPC Model Building")
        # Initialize Model
        model = pe.ConcreteModel(name="MPC")

        # Sets, Params and Variables
        model = initPowerSystem(model, case, n_thermal)
        self.logger.info("MPC Model initPowerSystem: Done")
        # Include Linearized Unbalanced 3-Phase Power Flow Constraints: Sched, Up, Dn
        model = Lin3DistFlow(model)
        self.logger.info("MPC Model Lin3DistFlow: Done")

        # Include System-Level Constraints: Sched, Up, Dn
        model = system_level(model)
        self.logger.info("MPC Model system_level: Done")

        # Specific to MPC model
        model = specific_MPC(model)
        self.logger.info("MPC Model specific_MPC: Done")

        # Include DER specific constraints
        model = DER_constraints(model, unity_powerfactor=unity_powerfactor)
        self.logger.info("MPC Model DER_constraints: Done")

        model = DER_MPC_constraints(
            model, E_BAT_final_flag, E_EV_final_flag, E_FL_final_flag
        )
        self.logger.info("MPC Model DER_MPC_constraints: Done")

        # Add Objective Function
        # model.obj = pe.Objective(rule=discrepancy_cost, sense=pe.minimize)
        # Using partial function to pass options objective
        def MPC_cost_partial(model):
            return MPC_cost(
                model,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                reserve_weight=reserve_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
                reserves_deviation_weight=reserves_deviation_weight,
                reserves_deviation_price_override=reserves_deviation_price_override,
            )

        model.obj = pe.Objective(rule=MPC_cost_partial, sense=pe.minimize)

        self.MPC_model = model

        self.logger.info("MPC Model Created")

    def step_MPC(self, case: NetworkCase = None, solver: str = "glpk", **kw_args):
        # Initialize Model with Current Case settings
        self.init_MPC_model(case, **kw_args)
        instance = self.MPC_model
        # Fix Reserve Setpoints based on DA results
        for frame_idx in instance.T:
            Net = case.horizon[frame_idx - 1]
            instance.Rup[frame_idx].fix(Net.get_system_reserves("up"))
            instance.Rdn[frame_idx].fix(Net.get_system_reserves("dn"))
            instance.Rup_dist[frame_idx].fix(Net.get_system_reserves("up_dist"))
            instance.Rdn_dist[frame_idx].fix(Net.get_system_reserves("dn_dist"))

        self.opt = pe.SolverFactory(solver, tee=True)

        try:
            self.results = self.opt.solve(instance)
            try:
                termination_condition = getattr(
                    self.results.solver, "termination_condition", None
                )
                self.logger.warning(
                    f"The solver returned a status of: {termination_condition}"
                )
                solver_time = getattr(self.results.solver, "time", None)
                if solver_time is None:
                    solver_time = getattr(self.results.solver, "wallclock_time", None)
                if solver_time is None:
                    solver_time = getattr(self.results.solver, "cpu_time", "N/A")
                self.logger.info(f"Solver Time: {solver_time:6.3f} seconds")
            except:
                pass
        except:
            try:
                self.logger.debug(self.results.solver)
                self.logger.debug(self.results.write())
            except:
                pass
            raise

    def get_MPC_solver_status(self):
        return self.results.solver

    def set_E_init(self, E_init: Dict):
        """
        Store the E_init for the next MPC step
        """

        self.E_init = E_init

    def get_E_init(self):
        """
        Return the E_init for the next MPC step
        """

        return self.E_init

    def export_E_init(self, n_skip: int = 1):
        """
        Export initial E0 values for each DER for the next iteration
        """

        model = self.MPC_model
        scenario = model.Scenarios.at(1)
        t = model.T.at(n_skip)

        try:
            E_BAT = {
                der_ID: pe.value(model.E_BAT[der_ID, t, scenario])
                for der_ID in model.Battery_list
            }
        except:
            E_BAT = {}
        try:
            E_FL = {
                der_ID: pe.value(model.E_FL[der_ID, t, scenario])
                for der_ID in model.FlexibleLoad_list
            }
        except:
            E_FL = {}
        try:
            E_EV = {
                der_ID: pe.value(model.E_EV[der_ID, t, scenario])
                for der_ID in model.EV_list
            }
        except:
            E_EV = {}

        self.E_init = {**E_BAT, **E_FL, **E_EV}

    def export_results(self, topic: str):
        """
        Export MPC results for RT controller
        """

        model = self.MPC_model
        case = pe.value(model.networkCase)
        Net = case.Network

        no_scenario = model.Scenarios.at(1)

        # P0 schedule
        substation_ID = Net.get_substation_ID()
        P0_set = {
            substation_ID: {
                "p_set": [
                    sum(
                        pe.value(
                            model.Pkt[
                                substation_ID, phi, frame_idx, "sched", no_scenario
                            ]
                        )
                        for phi in model.Phases
                    )
                    for frame_idx in model.T
                ]
            }
        }

        # DER Setpoints
        DER_data = {
            der.getID(): {
                "p_set": [
                    pe.value(
                        model.PDER[der.get_type(), der.getID(), frame_idx, no_scenario]
                    )
                    for frame_idx in model.T
                ],
                "q_set": [
                    pe.value(
                        model.QDER[der.get_type(), der.getID(), frame_idx, no_scenario]
                    )
                    for frame_idx in model.T
                ],
                "r_up": [
                    pe.value(
                        model.rDER_up[
                            der.get_type(), der.getID(), frame_idx, no_scenario
                        ]
                    )
                    for frame_idx in model.T
                ],
                "r_dn": [
                    pe.value(
                        model.rDER_dn[
                            der.get_type(), der.getID(), frame_idx, no_scenario
                        ]
                    )
                    for frame_idx in model.T
                ],
            }
            for der in Net.get_DERs()
        }

        # Post processing for special resources
        # Composite Resources
        CR_data = {}
        for resource in Net.get_composite_resources():
            total_setpoint = [0 for frame_idx in model.T]
            # DERs
            for derID in resource.get_DER_ids():
                # Removing DER from MPC data
                der = DER_data.pop(derID)
                total_setpoint += der["p_set"]
            # Loads
            load_ids = resource.get_load_ids()
            for loadID in load_ids:
                load_samples = case.read_horizon_samples(
                    "load", loadID, fcast_type="PL", scenario_nr=no_scenario
                )
                total_setpoint += load_samples
            CR_data[resource.getID()] = {"p_set": total_setpoint}

        # Publish results
        case_settings = case.get_case_settings()
        t0 = case_settings["t0"]
        n_timestep = case_settings["n_timestep"]
        timestep_period = case_settings["timestep_period"]
        # Publish MPC results for DER controller
        MPC_data = {**P0_set, **DER_data}
        self.IO.publish_MPC_result(MPC_data, t0, n_timestep, timestep_period, topic)
        # Publish MPC results for MPC dispatch controller
        MPC_dispatch_data = {**CR_data}
        self.IO.publish_MPC_result(
            MPC_dispatch_data,
            t0,
            n_timestep,
            timestep_period,
            topic,
            datatype="MPC_dispatch",
        )

        TR_pi = TR_ComputePrices(model)

        fcasts = {}
        for derID in TR_pi.keys():
            fcasts[derID] = TR_pi[derID]
        self.IO.publish_fcasts_gapps(
            "pi_local", fcasts, t0, n_timestep, timestep_period, topic=topic
        )

        All_data = {"P0_set": P0_set, "DER_data": DER_data, "TR_pi": TR_pi}
        return All_data

    def printMPC(self):
        model = self.MPC_model
        case = pe.value(model.networkCase)

        print("")
        print(f"MPC Solution for {case.getSettings()['t0']}")
        print("")

        try:
            print("")
            print("Slack Variable for Substation power constraints")
            print("")
            model.eta.pprint()

            print("")
            print("Slack Variable for Up and Down Reserves Constraints")
            print("")
            model.eta_up_reserves.pprint()
            model.eta_dn_reserves.pprint()
        except Exception as e:
            pass

        h_voltages = [
            "Time Period",
            "Node",
            "Range Ph. A (pu)",
            "Range Ph. B (pu)",
            "Range Ph. C (pu)",
        ]
        d_voltages = []

        h_reserves = ["Time Period", "Up Reserves(pu)", "Dn Reserves(pu)"]
        d_reserves = []
        for frame_idx in model.T:
            Net = case.horizon[frame_idx - 1]
            d_voltages.append([f"t ={frame_idx:2d}:"])

            for node in Net.get_nodes():
                y = ["---"] * 3
                for phase in [0, 1, 2]:
                    voltages = pe.value(model.Ykt[node, phase, frame_idx, :, :])
                    if len(voltages):
                        y[phase] = (
                            f"{math.sqrt(min(voltages)):6.4f}-{math.sqrt(max(voltages)):6.4f}"
                        )
                d_voltages.append(["", node, y[0], y[1], y[2]])

            d_reserves.append(
                [
                    frame_idx,
                    pe.value(model.Rup[frame_idx]),
                    pe.value(model.Rdn[frame_idx]),
                ]
            )
        print("")
        print("Scenario-Based Day Ahead Problem Results")
        print("")
        print(tabulate(d_voltages, headers=h_voltages))

        print("")
        print("Total Reserve Capacity of the DERs")
        print("")
        print(tabulate(d_reserves, headers=h_reserves))

        h_der_reserves = [
            "Type",
            "DER ID",
            "Time Period",
            "Schedule (pu)",
            "Up Reserves (pu)",
            "Down Reserves (pu)",
        ]
        d_der_reserves = []

        for der in case.Network.get_DERs():
            derType = der.get_type()
            derID = der.getID()
            for frame_idx in model.T:
                sched = pe.value(model.PDER[derType, derID, frame_idx, :])
                rup = pe.value(model.rDER_up[derType, derID, frame_idx, :])
                rdn = pe.value(model.rDER_dn[derType, derID, frame_idx, :])
                if frame_idx == 1:
                    el1 = derType
                    el2 = derID
                else:
                    el1 = ""
                    el2 = ""
                d_der_reserves.append(
                    [
                        el1,
                        el2,
                        frame_idx,
                        f"({min(sched):6.4f},{max(sched):6.4f})",
                        f"({min(rup):6.4f},{max(rup):6.4f})",
                        f"({min(rdn):6.4f},{max(rdn):6.4f})",
                    ]
                )

        print("")
        print("Schedule and Reserves Ranges of DERs over Scenarios")
        print("")
        print(tabulate(d_der_reserves, headers=h_der_reserves))
