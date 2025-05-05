"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import FAST_DERMS_CONTROL.IOs.IOclass as IO

from FAST_DERMS_CONTROL.common import fastderms_app, init_logging
from FAST_DERMS_CONTROL.IOs.IObackup import IObackup

from gridappsd import GridAPPSD
from simple_pid import PID
from pathlib import Path

import pandas as pd
import numpy as np

import argparse
import logging
import sys
import time
import json
import os

# App Name
app_name = "MPC_Dispatch_Control"
__version__ = "0.9"

# Topics
default_FRS_topic = "FRS"

## Default values
# The reference start time for the period of simulation, it is assumed to be in Pacific time.
default_tz = "US/Pacific"

# MPC_Dispatch_Control Settings
# Controller Timestep
default_timestep = 0
# Offset to run the MPC Dispatch controller in seconds ahead of the timesteps
default_iteration_offset = 0
# simulation duration (in seconds)
default_sim_length = -1
# number of decimal in setpoints in kW
default_setpoint_precision = 3
# default Sbase in VA
default_Sbase = 100e6

# Logger
logger_name = f"__Main__{app_name}"
_main_logger = logging.getLogger(logger_name)


class MPC_Dispatch_Control(fastderms_app):

    def __init__(self, IO: IO.IOmodule, **kw_args):
        """
        Create a new MPC_Dispatch_Control object
        """
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", logger_name)})
        kw_args.update({"mrid": kw_args.get("mrid", app_name)})
        # Update kw_args with controller specific default values
        kw_args.update(
            {"message_period": kw_args.get("message_period", default_timestep)}
        )
        kw_args.update(
            {
                "iteration_offset": kw_args.get(
                    "iteration_offset", default_iteration_offset
                )
            }
        )
        super().__init__(**kw_args)

        self.IOs = IO

        self._simout_topic = self.IOs.simulation_topic("output")
        FRS_topic = kw_args.get("FRS_topic", default_FRS_topic)
        self._mpc_topic = self.IOs.application_topic(FRS_topic, "output", sub_id="MPC")
        self.logger.warning(
            f"[internal] subscribing to:\n {self._simout_topic}\n {self._mpc_topic}"
        )

        self._publish_to_topic = self.IOs.application_topic(
            FRS_topic, "output", sub_id=self.mrid
        )
        self.logger.warning(f"publishing to:\n {self._publish_to_topic}")

        self._automation_topic = kw_args.get(
            "automation_topic", self.IOs.service_topic("automation", "input")
        )
        # Controller Type
        # current implementation dispatches MPC results as they arrive
        # future plans for an advanced implementation where the MPC results are dispatched on MPC timestamps
        self._advanced = kw_args.get("advanced", False)

        # List of DER that this controller will manage
        self.DERs = kw_args.get("DERs", [])
        if self.DERs:
            self.logger.warning(
                f"Controlling the following DERs with MPC Dispatch: {self.DERs}"
            )

        ## MPC specific
        # MPC timestep in minutes
        self._MPC_timestamp = time.time()
        self._MPC_timestep = int(kw_args.get("MPC_timestep", 5))
        # number of MPC intervals
        self._MPC_n_interval = int(
            kw_args.get(
                "MPC_n_interval",
                60 * kw_args.get("MPC_horizon", 4) / self._MPC_timestep,
            )
        )
        # Sbase in kVA (for MPC)
        self._MPC_Sbase = kw_args.get("Sbase", default_Sbase) / 1000
        self.logger.debug(f"MPC Sbase is: {self._MPC_Sbase} kVA")
        # initializing the MPC next timestamp as the start of simulation.
        self._MPC_next_timestamp = np.nan

        # initialize all the variables:
        self._message_count = 0
        self._setpoints = {}
        self._MPC_data = None

        self.logger.info(f"{self.mrid} Initialized")
        self.IOs.send_gapps_message(self._automation_topic, {"command": "stop_task"})

    def on_message(self, headers, message):
        """Handle incoming messages on the simulation_output_topic for the simulation_id
        Parameters
        ----------
        headers: dict
            A dictionary of headers that could be used to determine topic of origin and
            other attributes.
        message: object
            A data structure following the protocol defined in the message structure
            of ``GridAPPSD``. Most message payloads will be serialized dictionaries, but that is
            not a requirement.
        """

        message_timestamp = int(headers["timestamp"]) / 1000
        try:
            # MPC message
            if self._mpc_topic in headers["destination"]:
                self.logger.info(f"Processing message from MPC")
                self.logger.debug(message.keys())
                datatype = message.get("datatype", "")
                # An output message was produced by the MPC and should be processed
                if datatype == "MPC_dispatch":
                    self._MPC_timestamp = message["timestamp"]
                    self.logger.warning(
                        f"MPC Data received for t0 = {self.timestamp_to_datetime(self._MPC_timestamp)}"
                    )
                    self.process_new_MPC_result(message)

                    if not self._advanced:
                        self._message_count += 1
                        self.logger.info(
                            f"Control iteration {int(self._message_count)}"
                        )
                        self.iterate_basic_control(self._MPC_timestamp)
                else:
                    self.logger.info(f"MPC message type {datatype} not processed")

            # SIMOUT message
            elif self._simout_topic in headers["destination"]:
                # Simulation Output Received
                simulation_timestamp = message["message"]["timestamp"]
                self.logger.info(
                    f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}"
                )
                # The simout message processing will only be considered for the adavanced implementation
                if self._advanced:
                    # If next iteration is None, set it to the first simulation message
                    if self.next_iteration is None:
                        self.logger.warning(
                            "Next iteration is None, starting execution at first simulation message"
                        )
                        self.set_next_iteration(simulation_timestamp, force=True)

                    # If the simulation timestamp is greater than or equal to the next offset timestamp
                    if simulation_timestamp >= self.next_offset_timestamp:

                        # This control iteration is for:
                        next_iteration_timestamp = self.next_iteration.timestamp()

                        self._message_count += 1
                        # Every message_period messages we are going to iterate the advanced control
                        self.logger.info(
                            f"Control iteration {int(self._message_count)}"
                        )
                        self.iterate_advanced_control(next_iteration_timestamp)

                        # Iteration is now complete, update the previous iteration
                        self.set_next_iteration(simulation_timestamp)

                    else:
                        self.logger.debug(
                            f"Waiting until next Iteration at {self.timestamp_to_datetime(self.next_offset_timestamp)}"
                        )
            else:
                self.logger.info(
                    f"Unknown message received on topic: {headers['destination']}\n{message}"
                )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(
                f"Error on line {exc_tb.tb_lineno}: {e},\n{exc_type}, {exc_obj}"
            )
            self._error_code = True
            raise

    def process_new_MPC_result(self, message):

        _MPC_DER_power = {}
        _MPC_DER_power_q = {}
        timestamps = []
        for data_dict in message["message"]:
            # Process the message
            derID = data_dict["equipment_mrid"]
            timestamp = data_dict["setpoint_timestamp"]
            timestamps.append(timestamp)

            # DER data
            if derID not in _MPC_DER_power.keys():
                # initializing empty lists
                _MPC_DER_power[derID] = {}
                _MPC_DER_power_q[derID] = {}

            _MPC_DER_power[derID][timestamp] = data_dict["p_set"]
            _MPC_DER_power_q[derID][timestamp] = data_dict["q_set"]

        # Get the active DER from the MPC results
        self.active_DERs = [
            der_ID for der_ID in _MPC_DER_power.keys() if der_ID in self.DERs
        ]

        timestamps = list(np.unique(timestamps))

        # Create the dataframe
        iterables = [self.active_DERs, timestamps]
        new_MPC_data = pd.DataFrame(
            columns=["P_set", "Q_set"],
            index=pd.MultiIndex.from_product(iterables, names=["mRID", "timestamp"]),
        )

        # DERs
        for derID in self.active_DERs:
            new_MPC_data.loc[derID, "P_set"] = [
                _MPC_DER_power[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]
            new_MPC_data.loc[derID, "Q_set"] = [
                _MPC_DER_power_q[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]

            self.logger.debug(f"MPC data for {derID}:\n{new_MPC_data.loc[derID]}")

        # Update stored MPC data to include the new data (update existing timestamps, add new timestamps).
        if self._MPC_data is None:
            self._MPC_data = new_MPC_data
            # Update the next timestamp to not NaN
            self._MPC_next_timestamp = self._MPC_timestamp
        else:
            self._MPC_data = new_MPC_data.combine_first(self._MPC_data)

    def iterate_basic_control(self, timestamp):
        # Determine which MPC interval is active
        MPC_data, current_timestamp = self.locate_MPC_data(timestamp)

        # New Setpoints
        new_setpoints = {
            der_ID: MPC_data.loc[der_ID, "P_set"] for der_ID in self.active_DERs
        }
        # Formatting the setpoints: removing NaN values and rounding to the default setpoint precision
        new_setpoints = {
            der_ID: np.round(setpoint, default_setpoint_precision)
            for der_ID, setpoint in new_setpoints.items()
            if not np.isnan(setpoint)
        }
        self.logger.info(f"New Setpoints:\n{new_setpoints}")

        # Dispatch the setpoints
        if new_setpoints:
            self.dispatch_der_mpc_control(new_setpoints)

        # Update the setpoints
        self._setpoints.update(new_setpoints)
        self.logger.debug(f"Updated Setpoints:\n{self._setpoints}")

    def iterate_advanced_control(self, timestamp):
        pass

    def dispatch_der_mpc_control(self, der_control_data):
        """ """
        # Dispatch the der_control_data
        self.logger.info("Dispatching DER MPC control")
        self.logger.debug(der_control_data)

        timestamp = time.time()

        # Create the payload
        def payload(derID, setpoint):
            return {
                "setpoint_timestamp": timestamp,
                "equipment_mrid": derID,
                "p_set": setpoint,
            }

        # Create the message with each DER setpoint
        msg = {}
        msg["message"] = [
            payload(der_ID, setpoint) for der_ID, setpoint in der_control_data.items()
        ]

        # Publish the message
        self.IOs.send_gapps_message(self._publish_to_topic, msg)

    def get_current_MPC_timestamp(self, timestamp):
        # Determine which MPC interval is active
        MPC_timestamps = self._MPC_data.index.get_level_values("timestamp").unique()
        try:
            current_timestamp = MPC_timestamps[MPC_timestamps <= timestamp][-1]
        except IndexError:
            current_timestamp = MPC_timestamps[0]

        return current_timestamp

    def get_next_MPC_timestamp(self, timestamp):
        # Determine which MPC interval is active
        MPC_timestamps = self._MPC_data.index.get_level_values("timestamp").unique()
        try:
            next_timestamp = MPC_timestamps[MPC_timestamps > timestamp][0]
        except IndexError:
            next_timestamp = MPC_timestamps[0]

        return next_timestamp

    def locate_MPC_data(self, timestamp):
        if self._MPC_data is None:
            self.logger.error("No MPC data available")
            return None, None

        # Determine which MPC interval is active
        current_timestamp = self.get_current_MPC_timestamp(timestamp)
        current_data = self._MPC_data.xs(current_timestamp, level="timestamp")
        return current_data, current_timestamp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "simulation_id", help="Simulation id to use for responses on the message bus."
    )
    parser.add_argument("request", help="Path to the simulation request file")
    parser.add_argument("config", help="App Config")

    # Authenticate with GridAPPS-D Platform
    os.environ["GRIDAPPSD_APPLICATION_ID"] = app_name
    os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
    os.environ["GRIDAPPSD_USER"] = "app_user"
    os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("'", ""))
    app_config = json.loads(opts.config.replace("'", ""))

    # Logging Facility
    log_level = app_config.get("log_level", logging.INFO)
    path_to_export = app_config.get("path_to_export", "./logs")
    path_to_export = Path(path_to_export).resolve()

    init_logging(app_name=app_name, log_level=log_level, path_to_logs=path_to_export)

    _main_logger.warning(
        f"{app_name} starting!!!-------------------------------------------------------"
    )

    simulation_id = opts.simulation_id
    _main_logger.debug(f"Info received from remote: Simulation ID {simulation_id}")

    model_id = sim_request["power_system_config"]["Line_name"]
    _main_logger.debug(f"Info received from remote: Model ID {model_id}")

    # GridAPPSD
    _gapps = GridAPPSD(simulation_id=simulation_id)
    if _gapps.connected:
        _main_logger.debug(f"GridAPPSD connected to simulation {simulation_id}")
    else:
        _main_logger.error("GridAPPSD not Connected")
        raise Exception("GridAPPSD not Connected")

    path_to_repo = app_config.get("path_to_repo", "../../../")
    path_to_repo = Path(path_to_repo).resolve()

    # Instantiate IO module
    IO = IObackup(
        simulation_id=simulation_id,
        model_id=model_id,
        path_to_repo=path_to_repo,
        path_to_export=path_to_export,
    )

    mpc_dispatch_controller = MPC_Dispatch_Control(IO, **app_config)

    # All ther subscriptions:
    # SIM Output
    simout_topic = IO.simulation_topic("output")
    _gapps.subscribe(simout_topic, mpc_dispatch_controller)
    # MPC OUT
    FRS_topic = app_config.get("FRS_topic", default_FRS_topic)
    mpc_topic = IO.application_topic(FRS_topic, "output", sub_id="MPC")
    _gapps.subscribe(mpc_topic, mpc_dispatch_controller)
    _main_logger.debug(f"[external] subscribing to:\n {simout_topic}\n {mpc_topic}")

    sim_time = app_config.get("sim_time", default_sim_length)
    if sim_time == -1:
        _main_logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _main_logger.info(f"Info received from remote: sim_time {sim_time} s.")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not mpc_dispatch_controller.running():
            if mpc_dispatch_controller.error() == 2:
                _main_logger.warning("MPC Dispatch Controller Terminated")
            else:
                _main_logger.error("MPC Dispatch Controller Failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)

    _main_logger.warning(
        f"{app_name} finished!!!-------------------------------------------------------"
    )
