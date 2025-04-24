"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import FAST_DERMS_CONTROL.IOs.IOclass as IO
import FAST_DERMS_CONTROL.Modeling.modelHandler as mod_hd
import FAST_DERMS_CONTROL.Modeling.Scenarios as sce
import FAST_DERMS_CONTROL.DA.DAclass as DA
import FAST_DERMS_CONTROL.MPC.MPCclass as MPCc

from FAST_DERMS_CONTROL.common import (
    fastderms_app,
    init_logging,
    Pyomo_Exception,
    Implementation_Exception,
)
from FAST_DERMS_CONTROL.IOs.IObackup import IObackup

from gridappsd import GridAPPSD, utils
from pathlib import Path

import datetime as dt

import argparse
import importlib
import json
import logging
import time
import pytz
import os
import sys

default_FRS_mrid = "FRS"

# Network Data
default_file_static_data = "IO13_gonogo_static_data.json"
default_file_fcast_data = "IO13_gonogo_fcast_data.pkl"

logger_name = "__Main__Orchestrator"

# The reference start time for the period of simulation, it is assumed to be in Pacific time
next_day = dt.datetime(2022, 4, 1, 0, 0, 0)
default_sim_start = dt.datetime(2022, 4, 1, 14, 0)

# simulation duration (in seconds)
default_sim_length = -1

# MPC specific
# Offset to run the MPC in seconds before(-) /after(+) the timesteps
MPC_offset = -5 * 60
# MPC Additional Reserve requirements
default_opt_R_dist = {}
default_opt_R_dist["beta_load"] = 0.9
default_opt_R_dist["beta_DER"] = 0.9
default_opt_R_dist["R_sigma_up"] = 1.5
default_opt_R_dist["R_sigma_dn"] = 1.5

# ADMS specific
default_adms_mrid = "ADMS"
default_adms_strptime = "%a, %b %d %H:%M"

_main_logger = logging.getLogger(logger_name)


class Orchestrator(fastderms_app):
    """
    Main orchestrator class for the FAST-DERMS control system.

    This class coordinates the Day-Ahead (DA) and Model Predictive Control (MPC) operations for distributed energy resources. It handles initialization, model setup, DA scheduling, deterministic runs, and real-time MPC control.

    The orchestrator responds to various commands received through GridAPPS-D messaging system:
    - init: Initialize and setup the network model
    - init_red: Initialize and reduce the network model
    - DA: Run Day-Ahead scheduling
    - DET: Run deterministic analysis
    - export_models: Export DA and DET models
    - export_DA: Export DA process data
    - DEMO: Run complete demonstration sequence
    - hotstart: Import previously saved DA process
    - MPC: Initialize and run MPC

    Inherits from:
        fastderms_app: Base application class providing common functionality

    Attributes:
        IOs (IOmodule): Input/Output module for data handling
        handler (modelHandler): Network model handler
        DA (DayAhead): Day-ahead scheduling module
        builder (ScenarioBuilder): Scenario generation module
    """

    def __init__(self, model_id, IOmodule: IO.IOmodule, **kw_args):
        """
        Initialize the Orchestrator class.

        This constructor sets up the main components and communication channels for the FAST-DERMS control system. It initializes the IO module, model handler, day-ahead scheduler, and scenario builder, as well as configures various messaging topics for GridAPPS-D communication.

        Args:
            model_id (str): Identifier for the power system model
            IOmodule (IO.IOmodule): Input/Output module instance for data handling
            \**kw_args: Keyword arguments including:
                - name (str): Logger name (defaults to __name__)
                - mrid (str): MRID identifier (defaults to default_FRS_mrid)
                - rng_seed (int): Random number generator seed (defaults to 412474)
                - Sbase (float): Base power in VA (defaults to 100e6)
                - adms_topic (str): ADMS communication topic
                - automation_topic (str): Automation service topic

        Attributes:
            IOs (IOmodule): Input/Output module instance
            handler (modelHandler): Network model handler
            DA (DayAhead): Day-ahead scheduling module
            builder (ScenarioBuilder): Scenario generation module
            _simout_topic (str): Simulation output topic
            _orchestrator_topic (str): Orchestrator input topic
            _adms_topic (str): ADMS communication topic
            _publish_DA_topic (str): Day-ahead results publication topic
            _publish_DET_topic (str): Deterministic results publication topic
            _publish_MPC_topic (str): MPC results publication topic
            _automation_topic (str): Automation service topic
            _MPC_ON (bool): Flag indicating if MPC is running
            _MPC_run_count (int): Counter for MPC iterations
            _error_code (bool): Error status flag

        Note:
            The constructor sets up all necessary communication channels and initializes the core components needed for the FAST-DERMS control system operation.
        """
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        kw_args.update({"mrid": kw_args.get("mrid", default_FRS_mrid)})
        # Update kw_args with default values
        super().__init__(**kw_args)

        rng_seed = kw_args.get("rng_seed", 412474)
        self.logger.info(f"Random seed is: {rng_seed}")
        Sbase = kw_args.get("Sbase", 100e6)
        self.logger.info(f"Sbase is: {Sbase} VA")

        self.IOs = IOmodule
        self.handler = mod_hd.modelHandler(model_id, IOmodule, Sbase, tz=self.local_tz)
        self.DA = DA.DayAhead(IOmodule=self.IOs, tz=self.local_tz)
        self.builder = sce.ScenarioBuilder(seed=rng_seed, tz=self.local_tz)

        self._simout_topic = self.IOs.simulation_topic("output")

        self._orchestrator_topic = self.IOs.service_topic(self.mrid, "input")

        self._adms_topic = kw_args.get(
            "adms_topic", self.IOs.application_topic(default_adms_mrid, "output")
        )

        self._publish_DA_topic = self.IOs.application_topic(
            self.mrid, "output", sub_id="DA"
        )
        self._publish_DET_topic = self.IOs.application_topic(
            self.mrid, "output", sub_id="DET"
        )
        self._publish_MPC_topic = self.IOs.application_topic(
            self.mrid, "output", sub_id="MPC"
        )
        self._automation_topic = kw_args.get(
            "automation_topic", self.IOs.service_topic("automation", "input")
        )

        self.logger.warning(
            f"[Internal] subscribing to: \n{self._simout_topic} \n{self._orchestrator_topic} \n{self._adms_topic}"
        )
        self.logger.warning(
            f"publishing to: \n{self._publish_DA_topic} \n{self._publish_DET_topic} \n{self._publish_MPC_topic}"
        )

        # Initialize Attributes
        self._MPC_ON = False
        self._MPC_run_count = 0
        self.MPC_options = {}
        self._error_code = False

        self.logger.warning("Orchestrator Initialized")
        self.IOs.send_gapps_message(self._automation_topic, {"command": "stop_task"})

    def running(self):
        """
        Check if the orchestrator is running without errors.

        Returns:
            bool: True if the orchestrator is running without errors, False otherwise. The status is determined by checking if there's no error code set.
        """
        # Check if any error code
        running = not bool(self._error_code)
        return running

    def error(self):
        """
        Get the current error code of the orchestrator.

        Returns:
            int or bool: The error code value. Possible values:
                - 0 or False: No error
                - 1 or True: General error
                - 2: Controlled termination
        """
        return self._error_code

    def on_message(self, headers, message):
        """
        Handle incoming messages from various topics in the GridAPPS-D platform.

        This method processes messages from three main topics:
        1. Orchestrator topic: Handles control commands for model setup, DA scheduling, and MPC operations
        2. Simulation output topic: Processes simulation timestamps and triggers MPC runs
        3. ADMS topic: Processes network management system constraints and configurations

        The method supports the following commands on the orchestrator topic:
        - init: Initialize and setup the network model
        - init_red: Initialize and reduce the network model
        - DA: Run Day-Ahead scheduling
        - DET: Run deterministic analysis
        - export_models: Export DA and DET models
        - export_DA: Export DA process data
        - DEMO: Run complete demonstration sequence
        - hotstart: Import previously saved DA process
        - MPC: Initialize and run MPC

        Parameters
        ----------
        headers : dict
            Message headers containing metadata such as:
            - destination: Topic where the message was published
            - timestamp: Message timestamp in milliseconds
        message : dict
            Message payload containing command and parameter data. Structure varies by message type:
            - For orchestrator messages: Contains 'command' and command-specific parameters
            - For simulation output: Contains simulation timestamp
            - For ADMS messages: Contains network constraints and configurations

        Raises
        ------
        Exception
            Any unhandled exception during message processing will set the error code and re-raise the exception

        Notes
        -----
        - MPC operations are triggered based on simulation timestamps and configured periods
        - ADMS messages are only processed when MPC is running
        - Error handling includes logging of line numbers and error details
        """
        self.logger.debug(f"Received message on topic: {headers['destination']}")
        message_timestamp = int(headers["timestamp"]) / 1000
        self.logger.debug(f"Message timestamp: {message_timestamp}")

        try:
            if self._orchestrator_topic in headers["destination"]:
                # Remote Control message
                command = message["command"]
                if command == "init":
                    # Setup the model
                    self.run_and_ack(self.setup_Model, **message)

                elif command == "init_red":
                    # Setup the model
                    self.setup_Model(**message)
                    self.run_and_ack(self.reduce_Model, **message)

                elif command == "DA":
                    # DA run case
                    self.run_and_ack(self.run_DA, **message)

                elif command == "DET":
                    # DA DET run case
                    self.run_and_ack(self.run_DET, **message)

                elif command == "export_models":
                    # Export models
                    self.run_and_ack(self.export_DayAhead_Models, **message)

                elif command == "export_DA":
                    # Export process
                    self.run_and_ack(self.export_DayAhead_Process, **message)

                elif command == "DEMO":
                    # Setup the model
                    self.setup_Model(**message)
                    # DA run case
                    self.run_DA(**message)
                    # DA DET run case
                    self.run_DET(**message)
                    # Export
                    self.export_DayAhead_Process(**message)

                elif command == "hotstart":
                    # Import process
                    self.run_and_ack(self.import_DayAhead_Process, **message)

                elif command == "MPC":
                    self.iteration_offset = message.get(
                        "iteration_offset", self.iteration_offset
                    )
                    self.logger.info(
                        f"Iteration offset is updated to: {self.iteration_offset} s."
                    )
                    # Activate MPC
                    tmstp_start = message.get(
                        "tmstp_start",
                        self.local_tz.localize(default_sim_start).timestamp(),
                    )
                    self.set_next_iteration(tmstp_start, force=True)

                    n_skip = message.get("n_skip", 1)
                    timestep_period = message.get("timestep_period", 15)
                    time_multiplier = message.get("time_multiplier", 1)
                    if time_multiplier != 1:
                        self.logger.warning(
                            f"Time multiplier is set to {time_multiplier}"
                        )
                    # MPC period
                    self.timestep = int(n_skip * timestep_period * 60 / time_multiplier)

                    # Update message
                    message.update(
                        {
                            "n_skip": n_skip,
                            "timestep_period": timestep_period,
                        }
                    )
                    self.run_and_ack(self.init_MPC, t0=self.next_iteration, **message)
                else:
                    self.logger.error(f"Unknown command: {command}")

            elif self._simout_topic in headers["destination"]:
                # Simulation Output Received
                simulation_timestamp = message["message"]["timestamp"]
                # Case to run the MPC Problem every MPC_period
                if self._MPC_ON:
                    if self.next_iteration is None:
                        self.logger.debug(
                            "Next iteration is None, executing at first simulation output"
                        )
                        self.set_next_iteration(simulation_timestamp, force=True)

                    if simulation_timestamp >= self.next_offset_timestamp:
                        self._MPC_run_count += 1
                        self.logger.info(
                            f"Time: {self.timestamp_to_datetime(simulation_timestamp)}"
                        )
                        self.logger.info(
                            f"MPC run: {self._MPC_run_count} for t0: {self.next_iteration}"
                        )

                        # Step MPC
                        self.run_MPC(t0=self.next_iteration, **self.MPC_options)

                        # Set iterations
                        self.set_next_iteration(simulation_timestamp)
                    else:
                        self.logger.info(
                            f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}, waiting until {self.timestamp_to_datetime(self.next_offset_timestamp)}"
                        )
                else:
                    self.logger.debug(
                        f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}, MPC is not running"
                    )

            elif self._adms_topic in headers["destination"]:
                # ADMS message Received
                if self._MPC_ON:
                    # Only process ADMS message if MPC is running
                    self.process_ADMS_message(message)
                    # Set next iteration to previous to trigger immediate MPC run
                    self.set_next_iteration(self.previous_iteration, force=True)
                    # Give Gridapps a few seconds to ingest the fcasts
                    self.logger.debug("Waiting 4s...")
                    time.sleep(4)
                else:
                    self.logger.critical(
                        f"ADMS message received while MPC is not running"
                    )

            else:
                self.logger.warning(f"Unknown topic: {headers['destination']}")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(
                f"Error on line {exc_tb.tb_lineno}: {repr(e)},\n{exc_type}, {exc_obj}"
            )
            self._error_code = True
            raise

    def run_and_ack(self, func, **kw_args):
        func(**kw_args)
        self.IOs.send_gapps_message(self._automation_topic, {"command": "stop_task"})

    def setup_Model(self, **kw_args):
        """
        Set up and initialize the power system model with specified parameters.

        This method performs the following operations:
        1. Initializes the network model
        2. Updates voltage limits
        3. Converts network values to per-unit system

        Args:
            \**kw_args: Keyword arguments including:
                - Vmin (float): Minimum voltage limit in per unit (default: 0.9)
                - Vmax (float): Maximum voltage limit in per unit (default: 1.1)
                - force_static (bool): Force using static data (passed to initialize_model)
                - Any other arguments accepted by handler.initialize_model()

        Note:
            The method uses the model handler (self.handler) to perform the actual
            model setup operations. The handler must be properly initialized before
            calling this method.
        """
        # Get options out of kw_args
        Vmin = kw_args.get("Vmin", 0.9)
        Vmax = kw_args.get("Vmax", 1.1)
        force_static = kw_args.get("force_static", False)

        # Loading Model
        self.handler.initialize_model(force_static=force_static)
        self.handler.update_model_voltage_limits(Vmin, Vmax)
        self.handler.per_unitize_network()
        self.logger.warning("Model initialized")

    def reduce_Model(self, **kw_args):
        """
        Reduce the complexity of the power system model by applying specified reduction rules.

        This method applies model reduction techniques to simplify the network model while preserving essential characteristics. It dynamically loads and applies reduction rules from the model_reduction module.

        Args:
            \**kw_args: Keyword arguments including:
                - rules (list[str]): List of reduction rule names to apply. Default is ['remove_leaves']. Each rule should correspond to a class name in the model_reduction module.

        Note:
            - The method will skip any rules that cannot be found in the model_reduction module
            - Rules are applied in the order they are specified in the rules list
            - The model handler must be properly initialized before calling this method

        Raises:
            Any exceptions from the model reduction process are logged but not raised
        """
        rules = kw_args.get("rules", ["remove_leaves"])
        rule_classes = []

        mod_red = importlib.import_module("FAST_DERMS_CONTROL.Modeling.model_reduction")

        for rule in rules:
            try:
                rule_classes.append(getattr(mod_red, rule))
            except:
                self.logger.error(f"Rule {rule} not found")
                continue
        self.handler.set_reduce_model_rules(*rule_classes)
        self.handler.reduce_model()
        self.logger.warning("Model reduction performed")

    def run_DA(self, **kw_args):
        """
        Run the Day-Ahead (DA) scheduling optimization process.

        This method performs the following operations:
        1. Updates network model and initializes case with specified parameters
        2. Updates forecasts and generates scenarios
        3. Builds and solves the DA optimization problem
        4. Exports results and self-schedule bids

        Args:
            \**kw_args: Keyword arguments including:
                - n_timestep (int): Number of timesteps in the scheduling horizon (default: 24)
                - timestep_period (int): Duration of each timestep in minutes (default: 60)
                - n_scenario (int): Number of scenarios to generate (default: 10)
                - t0 (float): Start time in epoch seconds (default: current time)
                - remove_reserves (bool): Flag to exclude reserve requirements (default: False)
                - n_init (int): Number of initial scenarios for selection (default: 1000)
                - n_thermal (int): Number of thermal constraints to consider (default: 20)

        Note:
            The method temporarily deactivates thermal line limits during optimization. Results are exported if logging level is INFO or lower.

        Raises:
            Pyomo_Exception: If the scenario-based optimization problem fails to solve
            Exception: For other unexpected errors during execution

        """
        # Get options out of kw_args
        ## DA options
        n_timestep = kw_args.get("n_timestep", 24)
        timestep_period = kw_args.get("timestep_period", 60)
        n_scenario = kw_args.get("n_scenario", 10)
        t0 = self.timestamp_to_datetime(
            float(kw_args.get("t0", self.local_tz.localize(next_day).timestamp()))
        )
        ## Scenario selection options
        n_init = kw_args.get("n_init", 1000)
        add_prices = kw_args.get("add_prices", False)
        TS_metrics = kw_args.get("TS_metrics", False)
        use_exp_value = kw_args.get("use_exp_value", True)
        max_loop_nr = int(kw_args.get("max_loop_nr", 10))
        sigma_multiplier = kw_args.get("sigma_multiplier", 1)
        offset_scenario = int(kw_args.get("offset_scenario", -1))  # No longer used ?
        ## DA model options
        n_thermal = kw_args.get("n_thermal", 20)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        remove_reserves = kw_args.get("remove_reserves", False)
        ### Objective weights
        power_weight = kw_args.get("power_weight", 1)
        transactive_weight = kw_args.get("transactive_weight", 1)
        reserve_weight = kw_args.get("reserve_weight", 1)
        loss_weight = kw_args.get("loss_weight", 0)
        reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
        substation_deviation_weight = kw_args.get("substation_deviation_weight", 1)
        substation_deviation_price_override = kw_args.get(
            "substation_deviation_price_override", None
        )
        ## Solver options
        solver = kw_args.get("solver", "ipopt")

        solver_status = None
        try:
            start_time = time.time()
            self.logger.warning("DA run started")

            self.handler.update_network()
            self.handler.init_case(
                n_timestep=n_timestep,
                timestep_period=timestep_period,
                n_scenario=n_scenario,
                t0=t0,
            )
            self.handler.update_forecasts()

            case_scenarios = self.builder.build_select_scenarios(
                self.handler.export_case(),
                n_timestep=n_timestep,
                n_scenario=n_scenario,
                n_init=n_init,
                add_prices=add_prices,
                TS_metrics=TS_metrics,
                use_exp_value=use_exp_value,
                max_loop_nr=max_loop_nr,
                multiplier=sigma_multiplier,
                offset_scenario=offset_scenario,
            )
            self.handler.import_case(case_scenarios)

            self.DA.build_DA_scenarios(
                self.handler.export_case(),
                n_thermal=n_thermal,
                remove_reserves=remove_reserves,
                unity_powerfactor=unity_powerfactor,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                reserve_weight=reserve_weight,
                loss_weight=loss_weight,
                reverse_pf_weight=reverse_pf_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
            )

            # Deactivate thermal line limits
            self.DA.DA_model.ThermalLineLim.deactivate()

            self.DA.solve_DA_scenarios(solver=solver)
            self.DA.print_DA_scenarios()

            solver_status = self.DA.get_DA_solver_status()
            if not (
                solver_status.status == "ok"
                and solver_status.termination_condition == "optimal"
            ):
                raise Pyomo_Exception("Scenario based problem failed to solve")

            self.DA.export_DA_scenarios_res()
            self.DA.export_DA_self_sched_bid(feeder_ID=self.handler.mRID)
            self.logger.warning("DA solved!")
            self.logger.warning(f"Total DA time: {time.time() - start_time:0.3f} s")
        except Pyomo_Exception as e:
            self.logger.error(e)
            if solver_status is not None:
                self.logger.error(solver_status)
        except Exception as e:
            self.logger.error(e)
            raise

        finally:
            if self.logger.isEnabledFor(logging.INFO):
                DA_res = self.DA.export_full_pyomo_data(
                    self.DA.DA_model,
                    power_weight=power_weight,
                    transactive_weight=transactive_weight,
                    reserve_weight=reserve_weight,
                    loss_weight=loss_weight,
                    reverse_pf_weight=reverse_pf_weight,
                    substation_deviation_price_override=substation_deviation_price_override,
                    substation_deviation_weight=substation_deviation_weight,
                )
                self.IOs.publish_results(DA_res, filename="DA_data.pkl", archive=True)

    def run_DET(self, **kw_args):
        """
        Run the Deterministic (DET) optimization process for Day-Ahead scheduling.

        This method performs a deterministic optimization run using expected values instead of scenarios. The process includes:
            1. Updating deterministic forecast data
            2. Building and solving the deterministic optimization model
            3. Exporting results and handling any solver failures

        Args:
            \**kw_args: Keyword arguments including:
                - n_thermal (int): Number of thermal constraints to consider (default: 20)

        Raises:
            Pyomo_Exception: If the deterministic optimization problem fails to solve
            Exception: For other unexpected errors during execution

        Note:
            - If the solver fails, the method will fall back to using backup data
            - Results are exported if logging level is INFO or lower
            - Solver status and execution time are logged at WARNING level

        """
        # Get options out of kw_args
        ## DET model options
        n_thermal = kw_args.get("n_thermal", 20)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        ### Objective weights
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
        ## Solver options
        solver = kw_args.get("solver", "ipopt")

        solver_status = None
        use_backup = False
        try:
            start_time = time.time()
            self.logger.warning("DET run started")
            self.handler.updateDeterministicData()
            self.DA.build_DA_deterministic(
                self.handler.export_case(),
                n_thermal=n_thermal,
                unity_powerfactor=unity_powerfactor,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                reserve_weight=reserve_weight,
                loss_weight=loss_weight,
                reverse_pf_weight=reverse_pf_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
                reserves_deviation_weight=reserves_deviation_weight,
                reserves_deviation_price_override=reserves_deviation_price_override,
            )
            self.DA.solve_DA_deterministic(solver=solver)
            self.DA.print_DA_deterministic()

            solver_status = self.DA.get_DET_solver_status()
            if not (
                solver_status.status == "ok"
                and solver_status.termination_condition == "optimal"
            ):
                raise Pyomo_Exception("Deterministic problem failed to solve")
            self.logger.warning("DET solved!")
            self.logger.warning(f"Total DET time: {time.time() - start_time:0.3f} s")
        except Pyomo_Exception as e:
            use_backup = True
            self.logger.error(e)
            if solver_status is not None:
                self.logger.error(solver_status)
        except Exception as e:
            use_backup = True
            self.logger.error(repr(e))
            raise

        finally:
            model = self.DA.get_DA_DET_model(use_backup=use_backup)
            self.DA.export_DA_deterministic_res(model)

            if self.logger.isEnabledFor(logging.INFO):
                DET_res = self.DA.export_full_pyomo_data(
                    model,
                    power_weight=power_weight,
                    transactive_weight=transactive_weight,
                    reserve_weight=reserve_weight,
                    loss_weight=loss_weight,
                    reverse_pf_weight=reverse_pf_weight,
                    substation_deviation_price_override=substation_deviation_price_override,
                    reserves_deviation_price_override=reserves_deviation_price_override,
                    substation_deviation_weight=substation_deviation_weight,
                    reserves_deviation_weight=reserves_deviation_weight,
                )
                self.IOs.publish_results(DET_res, filename="DET_data.pkl", archive=True)

    def export_DayAhead_Models(self, **kw_args):
        """
        Export Day-Ahead optimization models and their associated statistics.

        This method exports both the scenario-based (DA) and deterministic (DET) optimization models along with their solver statistics. The exported data includes the full Pyomo model data and solver performance metrics.

        Args:
            \**kw_args: Keyword arguments including:
                - filename (str): Name of the export file (default: 'DayAhead_Models.pkl')
                - quit_after (bool): Flag to terminate orchestrator after export (default: True)

        The exported data dictionary contains:
            - DA_model: Full Pyomo data for the scenario-based model (if available)
            - DA_stats: Solver statistics for the DA model including:
                * solver_status: Termination condition
                * solver_time: Solution time
            - DET_model: Full Pyomo data for the deterministic model (if available)
            - DET_stats: Solver statistics for the DET model including:
                * solver_status: Termination condition
                * solver_time: Solution time

        Note:
            - The method will only export models and statistics that are available
            - Results are archived with a timestamp based on the case settings
            - If quit_after is True, sets error_code to 2 for controlled termination
        """
        # Get options out of kw_args
        filename = kw_args.get("filename", "DayAhead_Models.pkl")
        quit_after = kw_args.get("quit_after", True)

        export_data = {}
        if self.DA.DA_model is not None:
            export_data.update(
                {"DA_model": self.DA.export_full_pyomo_data(self.DA.DA_model)}
            )
        if self.DA.results is not None:
            stats = {}
            try:
                stats["solver_status"] = self.DA.results.solver.termination_condition
                stats["solver_time"] = self.DA.results.solver.time
            except:
                pass
            export_data.update({"DA_stats": stats})
        if self.DA.DET_model is not None:
            export_data.update(
                {"DET_model": self.DA.export_full_pyomo_data(self.DA.DET_model)}
            )
        if self.DA.results2 is not None:
            stats = {}
            try:
                stats["solver_status"] = self.DA.results2.solver.termination_condition
                stats["solver_time"] = self.DA.results2.solver.time
            except:
                pass
            export_data.update({"DET_stats": stats})

        self.IOs.publish_results(
            export_data,
            filename=filename,
            archive=True,
            apdx=f'{self.handler.case.get_case_settings(setting = "t0").strftime("%Y%m%d")}',
        )

        if quit_after:
            # Terminate the orchestrator
            self._error_code = 2

    def export_DayAhead_Process(self, **kw_args):
        """
        Export the complete Day-Ahead process data including handler, builder, and IO results.

        This method exports the current state of the Day-Ahead process, including:
        - Network model handler state
        - Scenario builder configuration
        - IO module results

        The exported data is saved to a pickle file with an optional custom filename and automatically archived with a timestamp.

        Args:
            \**kw_args: Keyword arguments including:
                - filename (str): Name of the export file (default: 'DayAhead_Process.pkl')
                - quit_after (bool): Flag to terminate orchestrator after export (default: True)

        Note:
            - The export includes the complete state of the handler, builder, and IO results
            - The archive filename includes the case timestamp in YYYYMMDD format
            - If quit_after is True, sets error_code to 2 for controlled termination
        """
        # Get options out of kw_args
        filename = kw_args.get("filename", "DayAhead_Process.pkl")
        quit_after = kw_args.get("quit_after", True)

        export_data = {}
        export_data.update({"handler": self.handler.export_self()})
        export_data.update({"builder": self.builder})
        export_data.update({"IOresults": self.IOs.results})

        self.IOs.publish_results(
            export_data,
            filename=filename,
            archive=True,
            apdx=f'{self.handler.case.get_case_settings(setting = "t0").strftime("%Y%m%d")}',
        )

        if quit_after:
            # Terminate the orchestrator
            self._error_code = 2

    def import_DayAhead_Process(self, **kw_args):
        """
        Import previously saved Day-Ahead process data from a file.

        This method loads and restores the state of the Day-Ahead process components, including the IO module results, model handler, and scenario builder from a previously exported file.

        Args:
            filename (str): Path to the file containing the saved Day-Ahead process data
            \**kw_args: Additional keyword arguments (currently unused but maintained for future extensibility)

        Note:
            - The method attempts to restore three main components:
                1. IO module results (self.IOs.results)
                2. Model handler (self.handler)
                3. Scenario builder (self.builder)
            - For the model handler import, the mRID and Sbase values from the saved data are ignored in favor of the current instance\'s values
            - If any component is not found in the imported data, that component remains unchanged
        """
        # Get options out of kw_args
        path_to_file = kw_args.get("path_to_file", "DayAhead_Process.pkl")

        import_data = self.IOs.load_results(path_to_file)

        IOresults = import_data.get("IOresults", None)
        if IOresults is not None:
            self.IOs.results = IOresults

        handler = import_data.get("handler", None)
        if handler is not None:
            place_holder = 0
            # Note that unlike for IOs, the mRID and Sbase provided will be ignored
            self.handler = mod_hd.modelHandler(
                place_holder, self.IOs, place_holder, new_self=handler
            )

        builder = import_data.get("builder", None)
        if builder is not None:
            self.builder = sce.ScenarioBuilder(new_self=builder)

    def init_MPC(self, **kw_args):
        """
        Initialize the Model Predictive Control (MPC) module with specified parameters.

        This method sets up the MPC module by:
        1. Configuring time-related parameters (skip intervals, timesteps, periods)
        2. Setting up the initial time and case settings
        3. Initializing the MPC case in the model handler
        4. Creating a new ModelPredict instance with specified parameters

        Args:
            \**kw_args: Keyword arguments including:
                - n_skip (int): Number of timesteps to skip between MPC runs (default: 1)
                - n_timestep (int): Number of timesteps in MPC horizon (default: 24)
                - timestep_period (int): Duration of each timestep in minutes (default: 15)
                - t0 (datetime): Initial time for MPC (default: current case t0)
                - n_thermal (int): Number of thermal constraints to consider (default: 20)
                - E_BAT_final_flag (bool): Flag for battery final state constraint (default: False)
                - E_EV_final_flag (bool): Flag for EV final state constraint (default: False)
                - E_FL_final_flag (bool): Flag for flexible load final state constraint (default: False)

        Note:
            - The MPC period is calculated as n_skip * timestep_period
            - The MPC horizon is calculated as n_timestep * timestep_period
            - All time-related parameters are logged at debug or info level
            - The method creates a new MPCc.ModelPredict instance stored in self.MPC
        """
        # Get options out of kw_args
        n_skip = kw_args.get("n_skip", 1)
        n_timestep = kw_args.get("n_timestep", 24)
        timestep_period = kw_args.get("timestep_period", 15)

        # t0 is removed from kw_args to be passed to MPC_options
        t0 = self.handler.case.get_case_settings(setting="t0")
        t0 = kw_args.pop("t0", t0)

        # Store all options in self.MPC_options
        kw_args.update(
            {
                "n_skip": n_skip,
                "n_timestep": n_timestep,
                "timestep_period": timestep_period,
                "n_scenario": 1,
            }
        )
        self.MPC_options = kw_args

        self.logger.debug(f"MPC n_skip is: {n_skip}")
        self.logger.debug(f"MPC n_timestep is: {n_timestep}")
        self.logger.debug(f"MPC timestep_period is: {timestep_period}")
        self.logger.info(f"MPC t0 is: {t0}")
        self.logger.info(
            f"MPC Period is {n_skip*timestep_period}, with Horizon: {n_timestep*timestep_period} min"
        )

        self.handler.init_MPC_case(
            t0=t0, n_timestep=n_timestep, timestep_period=timestep_period
        )

        self.MPC = MPCc.ModelPredict(IOmodule=self.IOs, tz=self.local_tz)
        self._MPC_ON = True

    def run_MPC(self, t0, **kw_args):
        """
        Execute a Model Predictive Control (MPC) optimization step.

        This method performs a complete MPC optimization cycle including:
        1. Updating the case time and initial energy states
        2. Applying reserve requirements and updating the MPC case
        3. Building scenarios using expected values
        4. Solving the MPC optimization problem
        5. Exporting and publishing results

        Args:
            \**kw_args: Keyword arguments including:
                - t0 (datetime): Start time for this MPC step (default: previous t0 + skip period)
                - opt_R_dist (dict): Reserve distribution parameters (default: default_opt_R_dist)
                    - beta_load (float): Load reserve factor
                    - beta_DER (float): DER reserve factor
                    - R_sigma_up (float): Upward reserve sigma
                    - R_sigma_dn (float): Downward reserve sigma
                - sigma_multiplier (float): Multiplier for scenario standard deviation (default: 1)

        Raises:
            Pyomo_Exception: If the MPC optimization problem fails to solve
            Exception: For other unexpected errors during execution

        Note:
            - Results are published to the MPC topic if successful
            - Detailed results are archived if logging level is INFO or lower
            - Initial energy states are exported regardless of success/failure
            - Execution time is logged at WARNING level
        """
        # Get options out of kw_args
        ## MPC options
        n_skip = kw_args.get("n_skip", 1)
        n_timestep = kw_args.get("n_timestep", 24)
        timestep_period = kw_args.get("timestep_period", 15)
        n_scenario = kw_args.get("n_scenario", 1)
        ### Uncertainty Propagation
        opt_R_dist = kw_args.get("opt_R_dist", default_opt_R_dist)
        sigma_multiplier = kw_args.get("sigma_multiplier", 1)
        ## MPC Model options
        n_thermal = kw_args.get("n_thermal", 20)
        unity_powerfactor = kw_args.get("unity_powerfactor", False)
        ## Objective weights
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
        battery_charge_discharge_weight = kw_args.get(
            "battery_charge_discharge_weight", 1
        )
        battery_deviation_weight = kw_args.get("battery_deviation_weight", 1000)
        E_BAT_final_flag = kw_args.get("E_BAT_final_flag", False)
        E_EV_final_flag = kw_args.get("E_EV_final_flag", False)
        E_FL_final_flag = kw_args.get("E_FL_final_flag", False)
        # Solver options
        solver = kw_args.get("solver", "ipopt")

        solver_status = None
        try:
            start_time = time.time()
            if t0 is None:
                old_t0 = self.handler.case.get_case_settings(setting="t0")
                t0 = old_t0 + dt.timedelta(minutes=n_skip * timestep_period)
            self.logger.warning(f"MPC run started for t0: {t0}")

            E_init_step = self.MPC.get_E_init()
            # MPC Additional Reserve requirements:
            self.handler.update_MPC_case(
                t0=t0, E_init=E_init_step, opt_R_dist=opt_R_dist
            )
            # Note: Since we use the same code for the MPC and the DA, we need to move the all the forecast data we just updated to the "samples" field in the case object by using the build_scenarios method
            case_with_samples = self.builder.build_scenarios(
                self.handler.export_MPC_case(),
                n_timestep=n_timestep,
                n_scenario=n_scenario,
                use_exp_values=True,
                multiplier=sigma_multiplier,
            )

            self.handler.import_MPC_case(case_with_samples)
            self.MPC.step_MPC(
                self.handler.export_MPC_case(),
                solver=solver,
                n_thermal=n_thermal,
                unity_powerfactor=unity_powerfactor,
                power_weight=power_weight,
                transactive_weight=transactive_weight,
                reserve_weight=reserve_weight,
                substation_deviation_weight=substation_deviation_weight,
                substation_deviation_price_override=substation_deviation_price_override,
                reserves_deviation_weight=reserves_deviation_weight,
                reserves_deviation_price_override=reserves_deviation_price_override,
                pv_curtailment_weight=pv_curtailment_weight,
                battery_charge_discharge_weight=battery_charge_discharge_weight,
                battery_deviation_weight=battery_deviation_weight,
                E_BAT_final_flag=E_BAT_final_flag,
                E_EV_final_flag=E_EV_final_flag,
                E_FL_final_flag=E_FL_final_flag,
            )

            solver_status = self.MPC.get_MPC_solver_status()
            if not (
                solver_status.status == "ok"
                and solver_status.termination_condition == "optimal"
            ):
                raise Pyomo_Exception("MPC problem failed to solve")

            MPC_results = self.MPC.export_results(self._publish_MPC_topic)
            if self.logger.isEnabledFor(logging.INFO):
                MPC_res = self.MPC.export_full_pyomo_data(
                    self.MPC.MPC_model,
                    power_weight=power_weight,
                    transactive_weight=transactive_weight,
                    reserve_weight=reserve_weight,
                    substation_deviation_price_override=substation_deviation_price_override,
                    reserves_deviation_price_override=reserves_deviation_price_override,
                    substation_deviation_weight=substation_deviation_weight,
                    reserves_deviation_weight=reserves_deviation_weight,
                    pv_curtailment_weight=pv_curtailment_weight,
                    battery_charge_discharge_weight=battery_charge_discharge_weight,
                    battery_deviation_weight=battery_deviation_weight,
                )
                self.IOs.publish_results(
                    MPC_res,
                    filename="MPC_data.pkl",
                    archive=True,
                    apdx=f'{t0.strftime("%Y%m%d_%H%M")}',
                )

        except Pyomo_Exception as e:
            self.logger.error(e)
            if solver_status is not None:
                self.logger.error(solver_status)

        except Exception as e:
            self.logger.error(e)

        finally:
            self.MPC.export_E_init(n_skip=n_skip)
            self.logger.warning("MPC solved!")
            self.logger.warning(f"Total MPC time: {time.time() - start_time:0.3f} s")

    def process_ADMS_message(self, message):
        """
        Process messages received from the Advanced Distribution Management System (ADMS).

        This method handles two types of ADMS messages:
        1. NMS constraints: Network Management System constraints for power limits
        2. NMS switch: Network switch configuration updates

        For NMS constraints, the method:
        - Validates the feeder ID matches the current handler
        - Converts timestamps to MPC timesteps
        - Standardizes power units to Watts
        - Creates and publishes forecasts with the new constraints

        Args:
            message (dict): Message payload containing:
                - datatype (str): Type of ADMS message ('NMS constraints' or 'NMS switch')
                - message (list): For NMS constraints, list of constraint dictionaries with:
                    * Substation (str): Substation name
                    * Circuit (str): Circuit identifier
                    * Feeder ID (str): Feeder identifier
                    * Upper Limit (str/int): Power upper limit
                    * Lower Limit (str/int): Power lower limit
                    * Units (str): Power units ('W', 'kW', or 'MW')
                    * Start Time (str): Constraint start time
                    * End Time (str): Constraint end time

        Note:
            - Times are rounded to nearest MPC timestep (floor for start, ceil for end)
            - All power values are converted to Watts internally
            - Forecasts are published to MPC topic for the constraint duration
        """
        try:
            datatype = message["datatype"]
            if datatype == "NMS power constraints" or datatype == "NMS constraints":
                self.logger.info(f"NMS power constraints received")
            elif datatype == "NMS voltage constraints":
                self.logger.info(f"NMS voltage constraints received")
            elif datatype == "NMS switch":
                self.logger.info(f"NMS switch received")
            else:
                raise Implementation_Exception(f"Unknown ADMS message type: {datatype}")

            message = message["message"]
            self.logger.debug(f"Received {len(message)} constraints")
            for constraint in message:
                feeder_name = f'{constraint["Substation"]}_{constraint["Circuit"]}'
                feeder_ID = constraint["Feeder ID"]
                start_time = constraint["Start Time"]
                end_time = constraint["End Time"]

                # Check this is for our Feeder
                if feeder_ID == self.handler.mRID:
                    # Process Start and End times for the constraint
                    MPC_timestep = dt.timedelta(
                        minutes=self.MPC_options.get("timestep_period", 15)
                    )
                    # Rounding the data to the nearest MPC timestep
                    # Startime with floor
                    try:
                        start_time = dt.datetime.strptime(
                            start_time, default_adms_strptime
                        )
                        start_time = self.time_round(
                            start_time, MPC_timestep, method="floor"
                        )
                    except ValueError:
                        self.logger.error(f"Invalid start time: {start_time}")
                        start_time = self.timestamp_to_datetime(
                            self.next_offset_timestamp
                        )

                    # Endtime with ceil
                    try:
                        end_time = dt.datetime.strptime(end_time, default_adms_strptime)
                        end_time = self.time_round(
                            end_time, MPC_timestep, method="ceil"
                        )
                    except ValueError:
                        self.logger.error(f"Invalid end time: {end_time}")
                        end_time = start_time + dt.timedelta(hours=24)

                    constraint_length = (end_time - start_time).total_seconds()

                    # Forcing received info to be mapped on simulation day
                    start_time = self.next_iteration.replace(
                        hour=start_time.hour,
                        minute=start_time.minute,
                        second=start_time.second,
                    )
                    end_time = start_time + dt.timedelta(seconds=constraint_length)

                    # How many steps
                    n_steps = int(constraint_length / MPC_timestep.total_seconds())

                    # Process Constraint data specific to this feeder
                    if (
                        datatype == "NMS power constraints"
                        or datatype == "NMS constraints"
                    ):
                        upper_limit = int(constraint["Upper Limit"])
                        lower_limit = int(constraint["Lower Limit"])
                        units = constraint["Units"]

                        # Fixing unit to W
                        if units == "kW":
                            upper_limit = upper_limit * 1000
                            lower_limit = lower_limit * 1000
                        elif units == "MW":
                            upper_limit = upper_limit * 1000000
                            lower_limit = lower_limit * 1000000
                        elif units == "W":
                            pass
                        else:
                            self.logger.error(f"Unknown units: {units}")
                        self.logger.info(
                            f"{start_time} - {end_time}: Setting {feeder_ID} limits to {lower_limit/1000} kW and {upper_limit/1000} kW"
                        )

                        # Package as fcast
                        fcast = {}
                        fcast["ADMS_P0_up_limit"] = [upper_limit] * n_steps
                        fcast["ADMS_P0_dn_limit"] = [lower_limit] * n_steps
                    elif datatype == "NMS voltage constraints":
                        value = float(constraint["Value"])
                        units = constraint["Units"]

                        # For now we only support p.u. for substation voltage constraint.
                        if units == "p.u." or units == "pu":
                            field_name = "ADMS_V_set_pu"
                            pass
                        elif units == "V":
                            field_name = "ADMS_V_set"
                            pass
                        elif units == "kV":
                            field_name = "ADMS_V_set"
                            value = value * 1000
                            units = "V"
                        else:
                            self.logger.error(f"Unknown units: {units}")
                        self.logger.info(
                            f"{start_time} - {end_time}: Setting {feeder_ID} Substation Voltage {field_name} to {value} {units}"
                        )

                        # Package as fcast
                        fcast = {}
                        fcast[field_name] = [value] * n_steps

                    elif datatype == "NMS switch":
                        raise Implementation_Exception(
                            f"NMS switch constraint processing"
                        )

                    # Publish fcast
                    self.IOs.publish_fcasts_gapps(
                        feeder_ID,
                        fcast,
                        start_time,
                        n_steps,
                        MPC_timestep.total_seconds() / 60,
                        topic=self._publish_MPC_topic,
                    )

        except Implementation_Exception as e:
            self.logger.warning(e)

        except Exception as e:
            self.logger.error("Error processing ADMS message: {e}")
            raise


if __name__ == "__main__":
    """
    Main execution block for the FAST-DERMS Orchestrator.

    This script initializes and runs the FAST-DERMS Orchestrator, which coordinates distributed energy resources in a power system. It handles:
    1. Command line argument parsing
    2. Logging setup
    3. GridAPPS-D connection
    4. IO module initialization
    5. Orchestrator instantiation and subscription setup
    6. Main execution loop

    Command Line Arguments:
        simulation_id: Identifier for responses on the message bus
        request: Path to the simulation request file
        config: Application configuration file path

    Configuration Parameters:
        log_level: Logging level (default: logging.INFO)
        path_to_export: Path for log exports (default: './logs')
        tz: Timezone ('PST' or 'EST')
        use_GAPPS: Whether to use GridAPPS-D connection (default: True)
        file_all_data: Path to combined data file (optional)
        file_static_data: Path to static data file (if file_all_data not used)
        file_fcast_data: Path to forecast data file (if file_all_data not used)
        path_to_repo: Path to repository (default: '../../')
        path_to_archive: Path for archiving (default: './archive')
        sim_time: Simulation duration in seconds (-1 for indefinite)

    The script runs until either:
        - The simulation time is reached
        - The orchestrator terminates normally (error code 2)
        - An error occurs in the orchestrator
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "simulation_id", help="Simulation id to use for responses on the message bus."
    )
    parser.add_argument("request", help="Path to the simulation request file")
    parser.add_argument("config", help="App Config")

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("'", ""))
    app_config = json.loads(opts.config.replace("'", ""))

    # Logging Facility
    log_level = app_config.get("log_level", logging.INFO)
    path_to_export = app_config.get("path_to_export", "./logs")
    path_to_export = Path(path_to_export).resolve()

    init_logging(
        app_name="Orchestrator", log_level=log_level, path_to_logs=path_to_export
    )

    # Restrict logging on somecomponents
    logging.getLogger("pyomo.core").setLevel(logging.WARNING)
    logging.getLogger("stomp.py").setLevel(logging.ERROR)

    _main_logger.warning(
        "Orchestrator starting!!!-------------------------------------------------------"
    )

    sim_id = opts.simulation_id
    _main_logger.debug(f"Info received from remote: Simulation ID {sim_id}")

    model_mrid = sim_request["power_system_config"]["Line_name"]
    _main_logger.debug(f"Info received from remote: Model ID {model_mrid}")

    tz = app_config.get("tz", "PST")
    if tz == "PST":
        tz = "US/Pacific"
    elif tz == "EST":
        tz = "US/Eastern"

    tz = pytz.timezone(tz)
    _main_logger.debug(f"Info received from remote: timezone {tz}")

    use_gapps = app_config.get("use_GAPPS", True)
    if use_gapps:
        gapps = GridAPPSD(simulation_id=sim_id)
        if gapps.connected:
            _main_logger.debug(f"GridAPPSD connected to simulation {sim_id}")
        else:
            _main_logger.error("GridAPPSD not Connected")
    else:
        gapps = None
        _main_logger.warning("No GridAPPSD connection")

    file_all_data = app_config.get("file_all_data", None)
    kw_args = {}
    if file_all_data is None:
        # old way with 2 files
        file_static_data = app_config.get("static_data", default_file_static_data)
        _main_logger.debug(
            f"Info received from remote: Static data file {file_static_data}"
        )
        kw_args.update({"file_static_data": file_static_data})

        file_fcast_data = app_config.get("fcast_data", default_file_fcast_data)
        _main_logger.debug(
            f"Info received from remote: Forecast data file {file_fcast_data}"
        )
        kw_args.update({"file_fcast_data": file_fcast_data})
    else:
        # new way with 1 file
        _main_logger.debug(f"Info received from remote: All data file {file_all_data}")
        kw_args.update({"file_all_data": file_all_data})

    path_to_repo = app_config.get("path_to_repo", "../../")
    path_to_repo = Path(path_to_repo).resolve()
    path_to_archive = app_config.get("path_to_archive", "./archive")
    path_to_archive = Path(path_to_archive).resolve()

    # Instantiate IO module
    IO = IObackup(
        simulation_id=sim_id,
        model_id=model_mrid,
        path_to_repo=path_to_repo,
        path_to_export=path_to_export,
        path_to_archive=path_to_archive,
        tz=tz,
        **kw_args,
    )

    # Instantiate the Orchestrator
    orchestrator = Orchestrator(
        model_id=model_mrid, IOmodule=IO, name=logger_name, local_tz=tz, **app_config
    )

    # Subscriptions
    # Orchestrator input for commands
    FRS_mrid = app_config.get("mrid", default_FRS_mrid)
    input_topic = IO.service_topic(FRS_mrid, "input")
    gapps.subscribe(input_topic, orchestrator)
    # Simout
    simout_topic = IO.simulation_topic("output")
    gapps.subscribe(simout_topic, orchestrator)
    # ADMS Output
    adms_topic = app_config.get("adms_topic", IO.application_topic("ADMS", "output"))
    gapps.subscribe(adms_topic, orchestrator)

    _main_logger.debug(
        f"[Main] Orchestrator subscribed to: \n{simout_topic} \n{input_topic} \n{adms_topic}"
    )

    sim_time = app_config.get("sim_time", default_sim_length)
    if sim_time == -1:
        _main_logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _main_logger.info(f"Info received from remote: sim_time {sim_time} s.")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:
        if not orchestrator.running():
            if orchestrator.error() == 2:
                _main_logger.warning("Orchestrator Terminated")
                orchestrator.IOs.send_gapps_message(
                    orchestrator._automation_topic, {"command": "stop_task"}
                )
            else:
                _main_logger.error("Orchestrator failed")
                orchestrator.IOs.send_gapps_message(
                    orchestrator._automation_topic, {"command": "error"}
                )
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)
    if elapsed_time >= sim_time and sim_time != -1:
        _main_logger.warning("Orchestrator reached sim_time")
        orchestrator.IOs.send_gapps_message(
            orchestrator._automation_topic, {"command": "stop_task"}
        )
    _main_logger.warning(
        "Orchestrator finished!!!\n -------------------------------------------------------------------------------------"
    )
