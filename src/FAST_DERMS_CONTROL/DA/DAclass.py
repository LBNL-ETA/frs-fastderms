"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..IOs.IOclass import IOmodule
from ..Modeling.networkCase import NetworkCase
from ..OPT.helpers import OptEngine
from ..OPT.base import initPowerSystem
from ..OPT.Rules.Objective import load_serving_cost, load_serving_cost_DET
from ..OPT.Constraints.SYSTEM import (
    Lin3DistFlow,
    system_level,
    specific_DA,
    specific_DA_DET,
)
from ..OPT.Constraints.REVERSE_PF import hard_no_reverse_pf, soft_no_reverse_pf
from ..OPT.Constraints.DERs import DER_constraints, TR_ComputePrices

from typing import Dict, List
from tabulate import tabulate
from io import StringIO

import datetime as dt
import pyomo.environ as pe
from pyomo.util.infeasible import log_infeasible_constraints


class DayAhead(OptEngine):
    def __init__(self, IOmodule: IOmodule, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        self.IO = IOmodule

        # Initialize Attributes
        self.DA_model = None
        self.DET_model = None
        self.results = None
        self.results2 = None

        self.logger.info("DayAhead class initialized")

    def build_DA_scenarios(self, case: NetworkCase = None, **kw_args):
        """
        Create model for linearized, unbalanced power flow, system constraints, and DER constraints.
        Solve for up/down reserve capacity at the substation head given N scenarios.
        """
        # Get options out of kw_args
        n_thermal = kw_args.get("n_thermal", 20)
        remove_reserves = kw_args.get("remove_reserves", False)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        ## Objective weights
        power_weight = kw_args.get("power_weight", 1)
        transactive_weight = kw_args.get("transactive_weight", 1)
        reserve_weight = kw_args.get("reserve_weight", 1)
        loss_weight = kw_args.get("loss_weight", 0)
        reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
        substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
        substation_deviation_price_override = kw_args.get(
            "substation_deviation_price_override", None
        )

        # Set up Model
        self.logger.info("DA Scenarios Model Building")
        # Initialize Model
        model = pe.ConcreteModel(name="DA - Scenario Based")
        # Sets, Params and Variables
        model = initPowerSystem(model, case, n_thermal)
        self.logger.info("DA Scenarios Model initPowerSystem: Done")
        # Include Linearized Unbalanced 3-Phase Power Flow Constraints: Sched, Up, Dn
        model = Lin3DistFlow(model)
        self.logger.info("DA Scenarios Model Lin3DistFlow: Done")
        # Include System-Level Constraints: Sched, Up, Dn
        model = system_level(model)
        self.logger.info("DA Scenarios Model system_level: Done")
        # DA specifics
        model = specific_DA(model)
        self.logger.info("DA Scenarios Model specific_DA: Done")

        if reverse_pf_weight is not None:
            # Include reverse power flow constraints
            if reverse_pf_weight == 0:
                model = hard_no_reverse_pf(model)
                self.logger.info("DA Scenarios Model hard_no_reverse_pf: Done")
            else:
                model = soft_no_reverse_pf(model, case)
                self.logger.info("DA Scenarios Model soft_no_reverse_pf: Done")

        # Include DER specific constraints
        model = DER_constraints(model, unity_powerfactor=unity_powerfactor)
        self.logger.info("DA Scenarios Model DER_constraints: Done")

        # Fix system reserves to zero
        if remove_reserves:
            for frame_idx in model.T:
                model.Rup[frame_idx].fix(0)
                model.Rdn[frame_idx].fix(0)

        # Add Objective Function
        # Using partial function to pass options to load_serving_cost
        def load_serving_cost_partial(model):
            return load_serving_cost(
                model,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                reserve_weight=reserve_weight,
                loss_weight=loss_weight,
                reverse_pf_weight=reverse_pf_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
            )

        model.obj = pe.Objective(rule=load_serving_cost_partial, sense=pe.minimize)

        self.DA_model = model
        self.logger.info("DA Scenarios Model Created")

    def solve_DA_scenarios(self, solver: str = "ipopt"):
        self.opt = pe.SolverFactory(solver, tee=True)
        try:
            self.logger.info("DA Scenarios Started Solving")
            self.results = self.opt.solve(self.DA_model)
            self.logger.info("DA Scenarios Solved")
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

    def get_DA_model(self):
        return self.DA_model

    def get_DA_solver_status(self):
        return self.results.solver

    def print_DA_scenarios(self):
        model = self.DA_model
        case = pe.value(model.networkCase)

        if (
            self.results.solver.termination_condition
            == pe.TerminationCondition.infeasible
        ):
            log_infeasible_constraints(model)
            self.logger.error("Infeasible Model")

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
        print_report = "\n"
        print_report += "####################################################\n"
        print_report += "Schedule and Reserves Ranges of DERs over Scenarios\n"
        print_report += "####################################################\n"
        print_report += tabulate(d_der_reserves, headers=h_der_reserves)
        self.logger.warning(print_report)
        try:
            print_report = "\n"
            print_report += "######################\n"
            print_report += "Slack Variable for Substation power across scenarios\n"
            print_report += "######################\n"
            _out = StringIO()
            self.DA_model.eta.pprint(ostream=_out)
            print_report += _out.getvalue()
            print_report += "\n"

        except:
            pass
        finally:
            self.logger.debug(print_report)

    def export_DA_scenarios_res(self, model=None):
        if model is None:
            model = self.DA_model
        case = pe.value(model.networkCase)
        # P0 same over all scenarios
        substation_ID = case.Network.get_substation_ID()
        P0_set = [
            [
                pe.value(model.Pkt[substation_ID, phi, frame_idx, "sched", 1])
                for phi in model.Phases
            ]
            for frame_idx in model.T
        ]
        Rup_set = [pe.value(model.Rup[t]) for t in model.T]
        Rdn_set = [pe.value(model.Rdn[t]) for t in model.T]

        exportDict = {"P0_set": P0_set, "Rup_set": Rup_set, "Rdn_set": Rdn_set}

        # Publish Data
        self.IO.publish_DA_result(exportDict)
        return exportDict

    def export_DA_self_sched_bid(self, model=None, feeder_ID: str = "FRS"):
        if model is None:
            model = self.DA_model
        case = pe.value(model.networkCase)
        S_base = case.get_network().Sbase
        n_timestep = case.get_case_settings(setting="n_timestep")
        t0 = case.get_case_settings(setting="t0")

        # n_timestep = len(model.T)
        # P0 same over all scenarios
        substation_ID = case.Network.get_substation_ID()
        P0_set = [
            [
                pe.value(model.Pkt[substation_ID, phi, frame_idx, "sched", 1])
                for phi in model.Phases
            ]
            for frame_idx in model.T
        ]
        Rup_set = [pe.value(model.Rup[t]) for t in model.T]
        Rdn_set = [pe.value(model.Rdn[t]) for t in model.T]

        # Format the bids appropriately
        # Energy Bid in MW
        EnBid = {
            (
                t0.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=i)
            ).timestamp(): {"s1": {"price": 0, "power": sum(P0_set[i]) * S_base / 1e6}}
            for i in range(n_timestep)
        }
        # Reserves Bid in MW
        ResBid = {
            (
                t0.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=i)
            ).timestamp(): {
                "Rup": {"price": 0, "power": Rup_set[i] * S_base / 1e6},
                "Rdn": {"price": 0, "power": Rdn_set[i] * S_base / 1e6},
            }
            for i in range(n_timestep)
        }

        exportDict = {"ID": feeder_ID, "EnBid": EnBid, "ResBid": ResBid}
        # Publish Data
        self.IO.publish_results(exportDict, filename="_dump_self_sched_bid.pkl")

        return exportDict

    def build_DA_deterministic(self, case: NetworkCase = None, **kw_args):
        """
        Create 2nd run model (deterministic) for linearized, unbalanced power flow, system constraints, and DER constraints.
        Solve for DER schedule and reserve setpoint given substation head setpoints under perfect forecast.
        Return deterministic model.
        """

        # Get options out of kw_args
        n_thermal = kw_args.get("n_thermal", 20)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        ## Objective weights
        power_weight = kw_args.get("power_weight", 1)
        transactive_weight = kw_args.get("transactive_weight", 1)
        reserve_weight = kw_args.get("reserve_weight", 1)
        loss_weight = kw_args.get("loss_weight", 0)
        reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
        substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
        substation_deviation_price_override = kw_args.get(
            "substation_deviation_price_override", None
        )
        reserves_deviation_weight = kw_args.get("reserves_deviation_weight", 1)
        reserves_deviation_price_override = kw_args.get(
            "reserves_deviation_price_override", None
        )

        # Set up Model
        self.logger.info("DET Deterministic Model Building")
        # Initialize Model
        model = pe.ConcreteModel(name="DA - Deterministic")

        # Sets, Params and Variables
        model = initPowerSystem(model, case, n_thermal)
        self.logger.info("DET Deterministic Model initPowerSystem: Done")

        # Fix Substation Real Power and Reserve Setpoints based on First Run solution
        for frame_idx in model.T:
            Net = case.horizon[frame_idx - 1]
            model.Rup[frame_idx].fix(Net.get_system_reserves("up"))
            model.Rdn[frame_idx].fix(Net.get_system_reserves("dn"))

        # Include Linearized Unbalanced 3-Phase Power Flow Constraints: Sched, Up, Dn
        model = Lin3DistFlow(model)
        self.logger.info("DET Deterministic Model Lin3DistFlow: Done")

        # Include System-Level Constraints: Sched, Up, Dn
        model = system_level(model)
        self.logger.info("DET Deterministic Model system_level: Done")

        # DA reserves
        model = specific_DA_DET(model)
        self.logger.info("DET Deterministic Model specific_DA_DET: Done")

        # Include DER specific constraints
        model = DER_constraints(model, unity_powerfactor=unity_powerfactor)
        self.logger.info("DET Deterministic Model DER_constraints: Done")

        if reverse_pf_weight is not None:
            # Include reverse power flow constraints
            if reverse_pf_weight == 0:
                model = hard_no_reverse_pf(model)
                self.logger.info("DET Deterministic Model hard_no_reverse_pf: Done")
            else:
                model = soft_no_reverse_pf(model, case)
                self.logger.info("DET Deterministic Model soft_no_reverse_pf: Done")

        # Add Objective Function
        # Using partial function to pass options to load_serving_cost
        def load_serving_cost_DET_partial(model):
            return load_serving_cost_DET(
                model,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                loss_weight=loss_weight,
                reserve_weight=reserve_weight,
                reverse_pf_weight=reverse_pf_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
                reserves_deviation_weight=reserves_deviation_weight,
                reserves_deviation_price_override=reserves_deviation_price_override,
            )

        model.obj = pe.Objective(rule=load_serving_cost_DET_partial, sense=pe.minimize)

        self.DET_model = model
        self.logger.info("DET Deterministic Model Created")

    def solve_DA_deterministic(self, solver: str = "ipopt"):
        self.opt2 = pe.SolverFactory(solver, tee=True)
        self.opt2.options["linear_solver"] = "mumps"
        try:
            self.logger.info("DET model Started Solving")
            self.results2 = self.opt2.solve(self.DET_model)
            self.logger.info("Deterministic Perfect Forecast Problem")
            try:
                termination_condition = getattr(
                    self.results2.solver, "termination_condition", None
                )
                self.logger.warning(
                    f"The solver returned a status of: {termination_condition}"
                )
                solver_time = getattr(self.results2.solver, "time", None)
                if solver_time is None:
                    solver_time = getattr(self.results2.solver, "wallclock_time", None)
                if solver_time is None:
                    solver_time = getattr(self.results2.solver, "cpu_time", "N/A")
                self.logger.info(f"Solver Time: {solver_time:6.3f} seconds")
            except:
                pass
        except:
            try:
                self.logger.debug(self.results2.solver)
                self.logger.debug(self.results2.write())
            except:
                pass
            raise

    def get_DA_DET_model(self, use_backup: bool = False):

        if use_backup:
            # using DA solution (in particular the expected scenario)
            model = self.DA_model
            self.logger.warning(
                "Using DA solution as backup for deterministic solution"
            )
        else:
            model = self.DET_model
        return model

    def get_DET_solver_status(self):
        return self.results2.solver

    def print_DA_deterministic(self, model=None):
        if model is None:
            model = self.get_DA_DET_model()

        case = pe.value(model.networkCase)

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
        print_report = "####################################################\n"
        print_report += "Schedule and Reserves Ranges of DERs\n"
        print_report += "####################################################\n"
        print_report += tabulate(d_der_reserves, headers=h_der_reserves)

        try:
            print_report += "\n######################\n"
            print_report += 'Slack Variable for Substation power constraints"\n'
            print_report += "######################\n"
            _out = StringIO()
            self.DET_model.eta.pprint(ostream=_out)
            print_report += _out.getvalue()
            print_report += "\n"

            print_report += "######################\n"
            print_report += 'Slack Variable for Up and Down Reserves Constraints"\n'
            print_report += "######################\n"
            _out = StringIO()
            self.DET_model.eta_up_reserves.pprint(ostream=_out)
            self.DET_model.eta_dn_reserves.pprint(ostream=_out)
            print_report += _out.getvalue()
            print_report += "\n"
        finally:
            self.logger.debug(print_report)

    def export_DA_deterministic_res(self, model=None):
        if model is None:
            model = self.get_DA_DET_model()

        case = pe.value(model.networkCase)
        det_scenario = model.Scenarios.at(1)
        Net = case.Network

        # Added E_0 at index 0 in recorded Data to facilitate the processing in MPC
        try:
            E_BAT = {
                der_ID: [Net.get_DERs(mRIDs=der_ID)[0].E_0]
                + [pe.value(model.E_BAT[der_ID, t, det_scenario]) for t in model.T]
                for der_ID in model.Battery_list
            }
        except:
            E_BAT = {}
        try:
            E_FL = {
                der_ID: [Net.get_DERs(mRIDs=der_ID)[0].E_0]
                + [pe.value(model.E_FL[der_ID, t, det_scenario]) for t in model.T]
                for der_ID in model.FlexibleLoad_list
            }
        except:
            E_FL = {}
        try:
            E_EV = {
                der_ID: [Net.get_DERs(mRIDs=der_ID)[0].E_0]
                + [pe.value(model.E_EV[der_ID, t, det_scenario]) for t in model.T]
                for der_ID in model.EV_list
            }
        except:
            E_EV = {}

        TR_pi = TR_ComputePrices(model)
        self.logger.debug(f"TR_pi: {TR_pi}")

        substation_ID = case.Network.get_substation_ID()
        P0_set = [
            [
                pe.value(
                    model.Pkt[substation_ID, phi, frame_idx, "sched", det_scenario]
                )
                for phi in model.Phases
            ]
            for frame_idx in model.T
        ]
        Rup_set = [pe.value(model.Rup[t]) for t in model.T]
        Rdn_set = [pe.value(model.Rdn[t]) for t in model.T]

        settings = case.get_case_settings()

        exportDict = {
            "P0_set": P0_set,
            "Rup_set": Rup_set,
            "Rdn_set": Rdn_set,
            "E_All": {**E_BAT, **E_FL, **E_EV},
            "E_FL": E_FL,
            "E_EV": E_EV,
            "TR_pi": TR_pi,
            "t0": settings["t0"],
            "timestep_period": settings["timestep_period"],
        }

        # Publish Data
        self.IO.publish_DET_result(exportDict)

        return exportDict
