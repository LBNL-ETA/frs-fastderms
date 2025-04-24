"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

import FAST_DERMS_CONTROL.IOs.IOclass as IO

from FAST_DERMS_CONTROL.common import fastderms_app, init_logging
from FAST_DERMS_CONTROL.IOs.IObackup import IObackup

from gridappsd import GridAPPSD
from simple_pid import PID
from pathlib import Path

import datetime as dt
import pandas as pd
import numpy as np

import argparse
import logging
import sys
import time
import pytz
import json
import os
import operator

# App Name
app_name = "RT_Control"
__version__ = "0.9"

# Topics
default_FRS_topic = "FRS"
default_ISO_topic = "ISO"

## Default values
# The reference start time for the period of simulation, it is assumed to be in Pacific time.
default_tz = "US/Pacific"
#
# Offset to run the RT controller in seconds ahead of the timesteps
default_RT_offset = 0

# RT Control Settings
# timestep [s.]
default_timestep = 60
# simulation duration (in seconds)
default_sim_length = -1
# number of decimal in setpoints in kW
default_setpoint_precision = 3
# default Sbase in VA
default_Sbase = 100e6

# model ID
default_model_id = "_C07972A7-600D-4AA5-B254-4CAA4263560E"
# Feeder Head ID
default_substation_ID = "node_650"

# IDs of the measurements associated with substationhead
default_substation_measurement_IDs = [
    "_359535a6-cdb5-4e13-9ee7-5558f932006f",
    "_c339f153-0f4c-4f88-9e13-9a6a2c0ec280",
    "_faaf4974-a562-48fc-9039-a6ef7f327188",
]

# PV IDs and meaqsurements
default_PV_measurement_IDs = {
    "PV1": [
        "_2a290d9d-3e75-4ffd-86d7-b69f6d0ef95f",
        "_5c262f2b-89da-4495-a6db-b633ed1cba02",
    ],
    "PV2": ["_86927daa-a0c9-42ff-bf7e-164042ed2989"],
    "PV3": ["_2cff4ce1-53dc-4777-a668-110cc61287c2"],
}

# Logger
logger_name = f"__Main__{app_name}"
_main_logger = logging.getLogger(logger_name)


class DER_RT_Control(fastderms_app):

    def __init__(self, IO: IO.IOmodule, **kw_args):
        """
        Create a new DER_RT_Control object
        """
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", logger_name)})
        # Update kw_args with RT controller specific default values
        kw_args.update({"timestep": kw_args.get("timestep", default_timestep)})
        kw_args.update(
            {"iteration_offset": kw_args.get("iteration_offset", default_RT_offset)}
        )
        super().__init__(**kw_args)

        self.IOs = IO

        self._simout_topic = self.IOs.simulation_topic("output")
        FRS_topic = kw_args.get("FRS_topic", default_FRS_topic)
        self._mpc_topic = self.IOs.application_topic(FRS_topic, "output", sub_id="MPC")
        ISO_topic = kw_args.get("ISO_topic", default_ISO_topic)
        self._iso_topic = self.IOs.application_topic(ISO_topic, "output")
        self.logger.warning(
            f"[internal] subscribing to:\n {self._simout_topic}\n {self._mpc_topic}\n {self._iso_topic}"
        )

        self._publish_to_topic = self.IOs.application_topic(
            FRS_topic, "output", sub_id=app_name
        )
        self.logger.warning(f"publishing to:\n {self._publish_to_topic}")

        self._automation_topic = kw_args.get(
            "automation_topic", self.IOs.service_topic("automation", "input")
        )

        self.model_mrid = kw_args.get("model_id", default_model_id)
        self.logger.debug(f"Model mrid is: {self.model_mrid}")

        self.substation_ID = kw_args.get("substation_ID", default_substation_ID)
        self.logger.debug(f"Substation ID is: {self.substation_ID}")

        self.substation_measurement_IDs = kw_args.get(
            "substation_measurement_IDs", default_substation_measurement_IDs
        )
        self.logger.debug(
            f"Substation measurement IDs are: {self.substation_measurement_IDs}"
        )

        self.PV_measurement_IDs = kw_args.get(
            "PV_measurement_IDs", default_PV_measurement_IDs
        )
        self.logger.debug(f"PV measurement IDs are: {self.PV_measurement_IDs}")

        # List of DER that will be ignored, i.e. not controlled
        self.ignored_DERs = kw_args.get("ignored_DERs", [])
        if self.ignored_DERs:
            self.logger.warning(f"Ignoring DERs (No Control): {self.ignored_DERs}")

        # List of DER that will be cancelled, i.e. output zeroed out
        self.cancelled_DERs = kw_args.get("cancelled_DERs", [])
        if self.cancelled_DERs:
            self.logger.warning(
                f"Cancelling DERs (Output Zeroed Out): {self.cancelled_DERs}"
            )

        ## PID Controller
        Kp = kw_args.get("Kp", 1.0)
        Ki = kw_args.get("Ki", 0.0)
        Kd = kw_args.get("Kd", 0.0)

        self._pid_control = PID(Kp, Ki, Kd, setpoint=0.0)

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
        # Flag for error correction
        self._MPC_error_rollover = kw_args.get("MPC_error_rollover", True)

        ## ISO specific
        self._ISO_ramp = kw_args.get("ISO_ramp", False)
        self._ISO_ramp_n_steps = kw_args.get("ISO_ramp_n_steps", 4)

        # initialize all the variables:
        self.status = True
        self._message_count = 0
        self._setpoints = {}
        self._MPC_data = None
        self._dispatch = None
        self.subs_power = None
        # self.MPC_DispatchError = 0.0 #I may want to carry the difference between the PID dispatch and the MPC dispatch right before the MPC change to hot start the PID with the new values.

        self.logger.info(f"{app_name} Initialized")
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

        # - DER Setpoints and Reserve Allocations from MPC.

        self.logger.debug(f"Received message on topic: {headers['destination']}")

        message_timestamp = int(headers["timestamp"]) / 1000
        try:
            # MPC message
            if self._mpc_topic in headers["destination"]:
                self.logger.info(f"Processing message from MPC")
                self.logger.debug(message.keys())
                datatype = message.get("datatype", "")
                # An output message was produced by the MPC and should be processed
                if datatype == "MPC_result":
                    self._MPC_timestamp = message["timestamp"]
                    self.logger.warning(
                        f"MPC Data received at t0 = {self.timestamp_to_datetime(self._MPC_timestamp)}"
                    )
                    self.process_new_MPC_result(message)
                elif datatype == "forecasts":
                    # Look for ADMS data in forecasts
                    try:
                        # Substation Constraint uses feeder ID as identifier
                        resource_ID = message.get("equipment_mrid", "")
                        if resource_ID == self.model_mrid:
                            data_dict = message["message"][0]
                            if "ADMS_P0_up_limit" in data_dict.keys():
                                self.logger.info(
                                    f"Received New ADMS Substation Contraints"
                                )
                                self.process_new_ADMS_constraints(message)
                    except Exception as e:
                        self.logger.error(f"Error parsing fcast received: {repr(e)}")
                else:
                    self.logger.info(f"MPC message type {datatype} not processed")

            # ISO message (dispatch)
            elif self._iso_topic in headers["destination"]:
                self.logger.info(f"Processing message from ISO")
                self.logger.debug(message.keys())
                # What kind of ISO message is it?
                datatype = message.get("datatype", "dispatch")
                if datatype == "dispatch":
                    # An output message was produced by the ISO and should be processed
                    for data_dict in message["message"]:
                        if data_dict["model_mrid"] == self.model_mrid:
                            dispatch_timestamp = data_dict["dispatch_timestamp"]
                            # Dispatch in kW (MW to kW conversion))
                            dispatch = data_dict["dispatch"] * 1000

                            self.logger.warning(
                                f"ISO Dispatch received at t0 = {self.timestamp_to_datetime(dispatch_timestamp)}"
                            )
                            self.logger.debug(f"Dispatch for {self.model_mrid}")
                            self.logger.info(f"Subs_iso_p0 = {dispatch:.3f} kW")
                            self.logger.info(
                                f"Price {data_dict['cleared_price']} USD/MWh"
                            )

                            # process new dispatch
                            self.process_new_dispatch(dispatch, dispatch_timestamp)
                else:
                    self.logger.info(f"ISO message type {datatype} not processed")

            # SIMOUT message
            elif self._simout_topic in headers["destination"]:
                # Case to run the RT control
                # starting after t_start and iterating every dispatch_timestep
                simulation_timestamp = message["message"]["timestamp"]

                if self.next_iteration is None:
                    self.logger.debug(
                        "Next iteration is None, starting execution at first simulation message"
                    )
                    self.set_next_iteration(simulation_timestamp, force=True)

                if simulation_timestamp >= self.next_offset_timestamp:

                    # The next control iteration is for:
                    next_iteration_timestamp = self.next_iteration.timestamp()

                    # Check if new MPC data is available for the next iteration
                    if next_iteration_timestamp >= self._MPC_next_timestamp:
                        try:
                            self.logger.info(
                                f"Found new MPC data for t0 = {self.timestamp_to_datetime(self._MPC_next_timestamp)}"
                            )
                            self.process_new_MPC_data(next_iteration_timestamp)
                        except:  # I think this is running before the first MPC data arrives and is processed, so
                            self.logger.info(
                                f"process_new_MPC_Data failed, the dataframe is currently:\n {self._MPC_data}"
                            )
                    else:
                        self.logger.info(
                            f"No new MPC data available for t0: {self.next_iteration}, using previous MPC data"
                        )
                        self.logger.debug(
                            f"Next MPC data available at t0: {self.timestamp_to_datetime(self._MPC_next_timestamp)}"
                        )

                    self._message_count += 1
                    # Every message_period messages we are going to iterate the RT control
                    self.logger.info(f"Control iteration {int(self._message_count)}")

                    measurements = message["message"]["measurements"]
                    # Get current substation power
                    for meas_ID in self.substation_measurement_IDs:
                        try:
                            self.logger.debug(
                                f"measurements {meas_ID}: {measurements[meas_ID]}"
                            )
                        except Exception as e:
                            self.logger.error(e)
                    try:
                        self.subs_power = sum(
                            np.real(
                                measurements[meas_ID]["magnitude"]
                                * np.exp(
                                    1j * np.deg2rad(measurements[meas_ID]["angle"])
                                )
                            )
                            for meas_ID in measurements
                            if meas_ID in self.substation_measurement_IDs
                        )
                        # Computte substation power in kW
                        self.subs_power = self.subs_power / 1000
                        self.logger.info(
                            f"Substation power received at t0 = {self.timestamp_to_datetime(simulation_timestamp)}"
                        )
                        self.logger.info(f"Subs_p0 = {self.subs_power:.3f} kW")
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error(f"Substation power not received")
                        self.subs_power = None

                    for PV in self.PV_measurement_IDs.keys():
                        try:
                            pv_power = sum(
                                np.real(
                                    measurements[meas_ID]["magnitude"]
                                    * np.exp(
                                        1j * np.deg2rad(measurements[meas_ID]["angle"])
                                    )
                                )
                                for meas_ID in measurements
                                if meas_ID in self.PV_measurement_IDs[PV]
                            )
                            self.logger.debug(f"{PV} power {pv_power/1000:.3f} kW")
                        except Exception as e:
                            self.logger.error(e)
                            self.logger.error(f"{PV} power not received")

                    if self._dispatch is None:
                        self.logger.warning(
                            "Dispatch hasn't been received, skipping RT control"
                        )
                    else:
                        # Compute the error
                        dispatch, _ = self.locate_dispatch_data(
                            next_iteration_timestamp
                        )
                        self.logger.debug(
                            f"Dispatch is {dispatch} and Substation Power is {self.subs_power}"
                        )
                        dispatch_err = dispatch - self.subs_power
                        # Note:
                        # Dispatch Error is in load convention (if I need to increase load, dispatch error is positive)
                        # temporary for debug
                        # dispatch_err = 200
                        self.process_dispatch_error(
                            dispatch_err, self.next_iteration.timestamp()
                        )

                    # Iteration is now complete, update the previous iteration
                    self.set_next_iteration(simulation_timestamp)

                else:
                    self.logger.info(
                        f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}, waiting until {self.timestamp_to_datetime(self.next_offset_timestamp)}"
                    )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(
                f"Error on line {exc_tb.tb_lineno}: {e},\n{exc_type}, {exc_obj}"
            )
            self.status = False
            raise

    def get_current_dispatch_timestamp(self, timestamp):
        # Determine which Dispatch is active
        _timestamps = self._dispatch.index.unique()
        try:
            current_timestamp = _timestamps[_timestamps <= timestamp][-1]
        except IndexError:
            current_timestamp = _timestamps[0]

        return current_timestamp

    def locate_dispatch_data(self, timestamp, **kw_args):
        # Determine which dispatch is active
        current_timestamp = self.get_current_dispatch_timestamp(timestamp)
        current_data = self._dispatch.xs(current_timestamp)

        ignore_limits = kw_args.get("ignore_limits", False)
        if ignore_limits:
            current_dispatch = current_data["dispatch"]
        else:
            try:
                up_limit = current_data.get("up_limit", np.nan)
                dn_limit = current_data.get("dn_limit", np.nan)
                current_dispatch = min(
                    max(current_data["dispatch"], dn_limit), up_limit
                )
            except Exception:
                # if there is no up / dn limits
                current_dispatch = current_data["dispatch"]

        return current_dispatch, current_timestamp

    def process_new_dispatch(self, dispatch, dispatch_timestamp):
        if self._ISO_ramp:
            if self._dispatch is None:
                new_dispatch = pd.DataFrame(
                    {"dispatch": [dispatch]}, index=[dispatch_timestamp]
                )
            else:
                previous_dispatch, _ = self.locate_dispatch_data(
                    self.previous_iteration.timestamp()
                )
                # Create a ramp between the previous dispatch and the new dispatch
                step_dispatch = np.linspace(
                    previous_dispatch, dispatch, self._ISO_ramp_n_steps + 1
                )[1:]
                start_index = int(np.floor(self._ISO_ramp_n_steps / 2) - 1)
                step_timestamps = [
                    dispatch_timestamp + i * self.timestep
                    for i in range(-start_index, self._ISO_ramp_n_steps - start_index)
                ]
                new_dispatch = pd.DataFrame(
                    {"dispatch": step_dispatch}, index=step_timestamps
                )
                self.logger.info(
                    f"Ramping dispatch from {previous_dispatch} to {dispatch} over {self._ISO_ramp_n_steps} steps"
                )
                self.logger.debug(f"New dispatch:\n{new_dispatch}")
        else:
            # Update the dispatch
            new_dispatch = pd.DataFrame(
                {"dispatch": [dispatch]}, index=[dispatch_timestamp]
            )
        # Update stored Dispatch data to include the new data.
        if self._dispatch is None:
            self._dispatch = new_dispatch
        else:
            self._dispatch = new_dispatch.combine_first(self._dispatch).ffill()

    def process_new_ADMS_constraints(self, message):
        # Parse the fcast
        fcast_messages = message["message"]
        # Sort incomming message
        fcast_messages.sort(key=operator.itemgetter("data_timestamp"))
        # Get the timestamps and the up and down limits
        timestamps = [data_dict["data_timestamp"] for data_dict in fcast_messages]
        extra_timestamp = timestamps[-1] + (timestamps[-1] - timestamps[-2])
        timestamps = timestamps + [extra_timestamp]
        # Adding a trailing 99999999999 to signify the end of the constraint
        # Converting incoming W into kW
        up_limit = [
            data_dict["ADMS_P0_up_limit"] / 1000 for data_dict in fcast_messages
        ] + [999999999999]
        dn_limit = [
            data_dict["ADMS_P0_dn_limit"] / 1000 for data_dict in fcast_messages
        ] + [-999999999999]

        # Create the DataFrame
        new_constraints = pd.DataFrame(
            {"up_limit": up_limit, "dn_limit": dn_limit}, index=timestamps
        )
        self.logger.debug(f"Constraint Received: \n{new_constraints}")

        if self._dispatch is None:
            self.logger.error(
                f"Dispatch still hasn't been received, skipping ADMS constraints"
            )
        else:
            self._dispatch = new_constraints.combine_first(self._dispatch).ffill()
            self.logger.debug(f"Updated Dispatch Data: \n{self._dispatch}")

    def process_new_MPC_result(self, message):
        _MPC_SUBS_power = {}
        _MPC_DER_power = {}
        _MPC_DER_power_q = {}
        _MPC_DER_rup = {}
        _MPC_DER_rdn = {}
        timestamps = []
        for data_dict in message["message"]:
            # Process the message
            resourceID = data_dict["equipment_mrid"]
            timestamp = data_dict["setpoint_timestamp"]
            timestamps.append(timestamp)

            if resourceID == self.substation_ID:
                # process PL0
                _MPC_SUBS_power[timestamp] = data_dict["p_set"]

            else:
                # DER data
                derID = data_dict["equipment_mrid"]
                if derID not in _MPC_DER_power.keys():
                    # initializing empty lists
                    _MPC_DER_power[derID] = {}
                    _MPC_DER_power_q[derID] = {}
                    _MPC_DER_rup[derID] = {}
                    _MPC_DER_rdn[derID] = {}

                _MPC_DER_power[derID][timestamp] = data_dict["p_set"]
                _MPC_DER_power_q[derID][timestamp] = data_dict["q_set"]
                _MPC_DER_rup[derID][timestamp] = data_dict["r_up"]
                _MPC_DER_rdn[derID][timestamp] = data_dict["r_dn"]

        # Get the active DER from the MPC results
        self.active_DERs = [
            der_ID
            for der_ID in _MPC_DER_power.keys()
            if der_ID not in self.ignored_DERs + self.cancelled_DERs
        ]

        timestamps = list(np.unique(timestamps))
        all_ids = [self.substation_ID] + self.active_DERs

        # Create the dataframe
        iterables = [all_ids, timestamps]
        new_MPC_data = pd.DataFrame(
            columns=["P_set", "Q_set", "R_up", "R_dn"],
            index=pd.MultiIndex.from_product(iterables, names=["mRID", "timestamp"]),
        )

        # Substation
        new_MPC_data.loc[self.substation_ID, "P_set"] = [
            _MPC_SUBS_power[ts] * self._MPC_Sbase for ts in timestamps
        ]
        self.logger.info(f"Substation Schedule\n{new_MPC_data.loc[self.substation_ID]}")

        # DERs
        for derID in self.active_DERs:
            new_MPC_data.loc[derID, "P_set"] = [
                _MPC_DER_power[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]
            new_MPC_data.loc[derID, "Q_set"] = [
                _MPC_DER_power_q[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]
            new_MPC_data.loc[derID, "R_up"] = [
                _MPC_DER_rup[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]
            new_MPC_data.loc[derID, "R_dn"] = [
                _MPC_DER_rdn[derID][ts] * self._MPC_Sbase for ts in timestamps
            ]

            self.logger.debug(f"MPC data for {derID}:\n{new_MPC_data.loc[derID]}")

        # Update stored MPC data to include the new data (update existing timestamps, add new timestamps).
        if self._MPC_data is None:
            self._MPC_data = new_MPC_data
            # Update the next timestamp to not NaN
            self._MPC_next_timestamp = self._MPC_timestamp
        else:
            self._MPC_data = new_MPC_data.combine_first(self._MPC_data)

        ## Set setpoints as MPC dresults as a starting point
        # self._setpoints.update({der_ID: np.round(self._MPC_data.loc[der_ID, timestamps[0]]['P_set'], default_setpoint_precision) for der_ID in self.active_DERs})

        # self.logger.warning(f"setpoints: {self._setpoints}")

        # # Compute initial error
        # if self.dispatch is None:
        #     # In case we haven't received dispatch yet
        #     init_dispatch_err = 0
        # else:
        #     init_dispatch_err = self.dispatch - self.subs_power
        #     #init_dispatch_err = self.dispatch - self._MPC_data.loc[self.substation_ID, timestamps[0]]['P_set']

        # self.process_dispatch_error(init_dispatch_err, self._MPC_timestamp)

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
        # Determine which MPC interval is active
        current_timestamp = self.get_current_MPC_timestamp(timestamp)
        current_data = self._MPC_data.xs(current_timestamp, level="timestamp")

        return current_data, current_timestamp

    # def process_control_error(self, control_error, MPC_data, reserves_type = 'up'):

    #     if reserves_type == 'up':
    #         # Use up reserves to meet the dispatch
    #         total_reserves = sum(MPC_data['R_up'][der_ID] for der_ID in self.active_DERs)
    #         error_correction = {der_ID: control_error * MPC_data['R_up'][der_ID] / total_reserves for der_ID in self.active_DERs}

    #     elif reserves_type == 'down':
    #         # Use down reserves to meet the dispatch
    #         total_reserves = sum(MPC_data['R_dn'][der_ID] for der_ID in self.active_DERs)
    #         error_correction = {der_ID: control_error * MPC_data['R_dn'][der_ID] / total_reserves for der_ID in self.active_DERs}

    #     else:
    #         self.logger.error(f"Reserves type {reserves_type} not recognized")
    #         self.logger.info("Spliting control error equally between resources")
    #         total_reserves = len(self.active_DERs)
    #         error_correction = {der_ID: control_error / total_reserves for der_ID in self.active_DERs}

    #     return error_correction

    def recursive_redispatch(
        self,
        error_correction,
        total_error_correction,
        reserve_type,
        MPC_data,
        DER_min,
        DER_max,
        DERs_unconstrained,
    ):
        # This function is designed to recursively dispatch the control signal to the unconstrained resources.
        self.logger.debug(
            f"Total error: {total_error_correction}, DERs unconstrained: {DERs_unconstrained}, reserve type: {reserve_type}, error correction: {error_correction}"
        )

        if np.isnan(total_error_correction):
            self.logger.error("The control signal is NaN")
            return error_correction

        if len(DERs_unconstrained) == 0:
            self.logger.info(
                "All DER are constrained and the controller is unable to send the whole control setpoint to the resources"
            )
            return error_correction

        if reserve_type == "UP":
            # The initial check and augmentation if the resources are constrained:
            # create a logical array of the resources that are constrained
            DERs_constrained = [
                der_ID
                for der_ID in DERs_unconstrained
                if (self._setpoints.get(der_ID, 0) + error_correction[der_ID])
                >= DER_max[der_ID]
            ]
            DERs_unconstrained = [
                der_ID
                for der_ID in DERs_unconstrained
                if der_ID not in DERs_constrained
            ]

            # update the error_correction (output of processing the control error)
            error_correction.update(
                {
                    der_ID: DER_max[der_ID] - self._setpoints.get(der_ID, 0)
                    for der_ID in DERs_constrained
                }
            )

            # calculate the control error that has been unassigned because of constrained resources
            remaining_control_signal = total_error_correction - sum(
                error_correction[der_ID] for der_ID in self.active_DERs
            )

            # Tentative redispatch
            # Allocate the remaining control signal to the unconstrained DER
            available_reserves = sum(
                MPC_data["R_up"][der_ID] for der_ID in DERs_unconstrained
            )
            if available_reserves == 0:
                weights = {
                    der_ID: 1 / len(DERs_unconstrained) for der_ID in DERs_unconstrained
                }
            else:
                weights = {
                    der_ID: MPC_data["R_up"][der_ID] / available_reserves
                    for der_ID in DERs_unconstrained
                }

        elif reserve_type == "DOWN":
            # The initial check and augmentation if the resources are constrained:
            # create a logical array of the resources that are constrained
            DERs_constrained = [
                der_ID
                for der_ID in DERs_unconstrained
                if (self._setpoints.get(der_ID, 0) + error_correction[der_ID])
                <= DER_min[der_ID]
            ]
            DERs_unconstrained = [
                der_ID
                for der_ID in DERs_unconstrained
                if der_ID not in DERs_constrained
            ]

            # update the error_correction (output of processing the control error)
            error_correction.update(
                {
                    der_ID: DER_min[der_ID] - self._setpoints.get(der_ID, 0)
                    for der_ID in DERs_constrained
                }
            )

            # calculate the control error that has been unassigned because of constrained resources
            remaining_control_signal = total_error_correction - sum(
                error_correction[der_ID] for der_ID in self.active_DERs
            )

            # Tentative redispatch
            # Allocate the remaining control signal to the unconstrained DER
            available_reserves = sum(
                MPC_data["R_dn"][der_ID] for der_ID in DERs_unconstrained
            )
            if available_reserves == 0:
                weights = {
                    der_ID: 1 / len(DERs_unconstrained) for der_ID in DERs_unconstrained
                }
            else:
                weights = {
                    der_ID: MPC_data["R_dn"][der_ID] / available_reserves
                    for der_ID in DERs_unconstrained
                }

        else:
            self.logger.info(
                "We somehow do not know what reserve region we are in...uh oh"
            )
            # The initial check and augmentation if the resources are constrained:
            # create a logical array of the resources that are constrained
            DERs_constrained_dn = [
                der_ID
                for der_ID in DERs_unconstrained
                if (self._setpoints.get(der_ID, 0) + error_correction[der_ID])
                <= DER_min[der_ID]
            ]
            DERs_constrained_up = [
                der_ID
                for der_ID in DERs_unconstrained
                if (self._setpoints.get(der_ID, 0) + error_correction[der_ID])
                >= DER_max[der_ID]
            ]
            DERs_unconstrained = [
                der_ID
                for der_ID in DERs_unconstrained
                if der_ID not in DERs_constrained_up + DERs_constrained_dn
            ]

            # update the error_correction (output of processing the control error)
            error_correction.update(
                {
                    der_ID: DER_min[der_ID] - self._setpoints.get(der_ID, 0)
                    for der_ID in DERs_constrained_dn
                }
            )
            error_correction.update(
                {
                    der_ID: DER_max[der_ID] - self._setpoints.get(der_ID, 0)
                    for der_ID in DERs_constrained_up
                }
            )

            # calculate the control error that has been unassigned because of constrained resources
            remaining_control_signal = total_error_correction - sum(
                error_correction[der_ID] for der_ID in self.active_DERs
            )

            # Tentative redispatch
            # Allocate equally to the unconstrained DER
            weights = {
                der_ID: 1 / len(DERs_unconstrained) for der_ID in DERs_unconstrained
            }

        error_correction.update(
            {
                der_ID: error_correction[der_ID]
                + remaining_control_signal * weights[der_ID]
                for der_ID in DERs_unconstrained
            }
        )

        if abs(remaining_control_signal) <= 0.01:
            self.logger.info("All control signal has been dispatched")
            return error_correction

        return self.recursive_redispatch(
            error_correction,
            total_error_correction,
            reserve_type,
            MPC_data,
            DER_min,
            DER_max,
            DERs_unconstrained,
        )

    def respecting_MPC_DER_limits(self, error_correction, MPC_data, reserve_type):
        control_signal = sum(error_correction[der_ID] for der_ID in self.active_DERs)

        DER_max = {
            der_ID: (MPC_data["P_set"][der_ID] + MPC_data["R_up"][der_ID])
            for der_ID in self.active_DERs
        }
        DER_min = {
            der_ID: (MPC_data["P_set"][der_ID] - MPC_data["R_dn"][der_ID])
            for der_ID in self.active_DERs
        }

        updated_error_correction = self.recursive_redispatch(
            error_correction,
            control_signal,
            reserve_type,
            MPC_data,
            DER_min,
            DER_max,
            self.active_DERs,
        )

        return updated_error_correction

    def process_control_error(self, control_error, MPC_data):
        # input data:
        # we assume that the control error is the output of the PID, and that it is in load convention (dispatch - measurement).
        # MPC data is a dictionary with the MPC Setpoints (P, Q, R_up, and R_down) for each DER and the intended substation power relevant to the current timestep.

        total_MPC_setpoints = sum(
            MPC_data["P_set"][der_ID] for der_ID in self.active_DERs
        )  # the total power applied to the MPC
        self.logger.debug(f"Total MPC Setpoints: {total_MPC_setpoints}")
        total_current_setpoints = sum(
            self._setpoints.get(der_ID, 0) for der_ID in self.active_DERs
        )  # The total of the current setpoints being sent to the DER
        self.logger.debug(f"Total Current Setpoints: {total_current_setpoints}")
        total_reserves_up = sum(MPC_data["R_up"][der_ID] for der_ID in self.active_DERs)
        self.logger.debug(f"Total Up Reserves: {total_reserves_up}")
        total_reserves_dn = sum(MPC_data["R_dn"][der_ID] for der_ID in self.active_DERs)
        self.logger.debug(f"Total Down Reserves: {total_reserves_dn}")
        # what the anticipated setpoints will be after the control error is applied:
        ending_setpoints = total_current_setpoints + control_error

        # we determine how to apply the control error to reserves based on the collective location of the setpoints relative to the MPC at both before and after the control error is added in.

        if (
            total_current_setpoints >= total_MPC_setpoints
            and ending_setpoints >= total_MPC_setpoints
        ):  # reserves_type == 'up':
            self.logger.info(
                "P Control: Providing UP reserves, setpoints will result in increased GENERATION relative to the MPC Setpoints"
            )
            # Use up reserves only to meet the dispatch
            reserve_type = "UP"
            if total_reserves_up == 0:
                self.logger.warning(
                    "No Up Reserves are available system-wide, splitting the control error equally between resources."
                )
                error_correction = {
                    der_ID: control_error / len(self.active_DERs)
                    for der_ID in self.active_DERs
                }
            else:
                error_correction = {
                    der_ID: control_error * MPC_data["R_up"][der_ID] / total_reserves_up
                    for der_ID in self.active_DERs
                }

        elif (
            total_current_setpoints < total_MPC_setpoints
            and ending_setpoints <= total_MPC_setpoints
        ):  # reserves_type == 'down':
            # Use down reserves to meet the dispatch
            self.logger.info(
                "P Control: Providing DOWN Reserves, setpoints will result in increasing LOAD relative to the MPC Setpoints"
            )
            reserve_type = "DOWN"
            if total_reserves_dn == 0:
                self.logger.warning(
                    "No Down Reserves are available system-wide, splitting the control error equally between resources."
                )
                error_correction = {
                    der_ID: control_error / len(self.active_DERs)
                    for der_ID in self.active_DERs
                }
            else:
                error_correction = {
                    der_ID: control_error * MPC_data["R_dn"][der_ID] / total_reserves_dn
                    for der_ID in self.active_DERs
                }

        elif (
            total_current_setpoints >= total_MPC_setpoints
            and ending_setpoints <= total_MPC_setpoints
        ):
            # We start with Up reserves but finish with down reserves
            self.logger.info(
                "P Control: Starting with UP reserves but finishing in the DOWN reserves"
            )
            MPC_error = (
                total_MPC_setpoints - total_current_setpoints
            )  # Our overall control error is currently positive, implying more load was need based on the previous MPC point.
            error_remainder = {
                der_ID: (control_error - MPC_error)
                * MPC_data["R_dn"][der_ID]
                / total_reserves_dn
                for der_ID in self.active_DERs
            }
            desired_stpts = {
                der_ID: MPC_data["P_set"][der_ID] + error_remainder[der_ID]
                for der_ID in self.active_DERs
            }
            reserve_type = "DOWN"
            error_correction = {
                der_ID: desired_stpts[der_ID] - self._setpoints.get(der_ID, 0)
                for der_ID in self.active_DERs
            }

        elif (
            total_current_setpoints < total_MPC_setpoints
            and ending_setpoints >= total_MPC_setpoints
        ):
            # we start with dn reserves but finish with up reserves.
            self.logger.info(
                "P Control: Starting with DOWN reserves but finishing in the UP reserves"
            )
            MPC_error = (
                total_MPC_setpoints - total_current_setpoints
            )  # Our overall control error is positive, more load.
            error_remainder = {
                der_ID: (control_error - MPC_error)
                * MPC_data["R_up"][der_ID]
                / total_reserves_dn
                for der_ID in self.active_DERs
            }
            desired_stpts = {
                der_ID: MPC_data["P_set"][der_ID] + error_remainder[der_ID]
                for der_ID in self.active_DERs
            }
            reserve_type = "UP"
            error_correction = {
                der_ID: desired_stpts[der_ID] - self._setpoints.get(der_ID, 0)
                for der_ID in self.active_DERs
            }

        else:
            self.logger.error(f"Reserve choice undecided")
            self.logger.info("Spliting control error equally between resources")
            total_reserves = len(self.active_DERs)
            reserve_type = None
            error_correction = {
                der_ID: control_error / total_reserves for der_ID in self.active_DERs
            }

        self.logger.debug(f"RAW Control Error Correction: {error_correction}")

        # Correct the Error Correction such that no resource has a setpoint greater than their Max or Minimum assigned MPC power
        error_correction = self.respecting_MPC_DER_limits(
            error_correction, MPC_data, reserve_type
        )

        return error_correction

    def process_new_MPC_data(self, next_iteration_timestamp):
        # Reset the PID
        self._pid_control.reset()

        # When a new MPC setpoint is available, update the starting setpoint to the new MPC data plus a correction as the total difference between the previous MPC data and the current setpoint
        previous_MPC_data, _ = self.locate_MPC_data(self.previous_iteration.timestamp())
        self.logger.debug(
            f"Previous MPC data for time {self.previous_iteration.timestamp()}:\n{previous_MPC_data}"
        )

        MPC_data, _ = self.locate_MPC_data(next_iteration_timestamp)
        self.logger.debug(
            f"Current MPC data for time {next_iteration_timestamp}:\n{MPC_data}"
        )

        self.logger.info(
            f"Processing new MPC data point for time: {next_iteration_timestamp}"
        )

        # Calculate a control error that is the sum of the difference between the previous setpoints sent to the resources
        # and the MPC Data used during that period. This difference will be in generator convention (if the system had increased generation, it will be positive.).
        self.logger.debug(
            f"What do the setpoints currently look like:\n{self._setpoints}"
        )

        try:
            if self._MPC_error_rollover:
                Previous_Forecasted_Load = previous_MPC_data.loc[
                    self.substation_ID, "P_set"
                ] + sum(
                    previous_MPC_data.loc[der_ID, "P_set"]
                    for der_ID in self.active_DERs
                )
                Current_Forecasted_Load = MPC_data.loc[
                    self.substation_ID, "P_set"
                ] + sum(MPC_data.loc[der_ID, "P_set"] for der_ID in self.active_DERs)
                Delta_Forecast = Current_Forecasted_Load - Previous_Forecasted_Load
                Delta_Sub = (
                    MPC_data.loc[self.substation_ID, "P_set"]
                    - previous_MPC_data.loc[self.substation_ID, "P_set"]
                )
                control_error = sum(
                    self._setpoints[der_ID] - previous_MPC_data.loc[der_ID, "P_set"]
                    for der_ID in self.active_DERs
                )  # The old error correction
                # control_error = sum(self._setpoints[der_ID] - previous_MPC_data.loc[der_ID, 'P_set'] for der_ID in self.active_DERs)-Delta_Sub#-Delta_Forecast
            else:
                control_error = 0
                self.logger.debug(
                    "MPC error rollover is disabled, setting control error to 0"
                )
        except:
            control_error = 0
            self.logger.debug("Calculating Control Error failure, setting equal to 0")
        self.logger.info(
            f"Rolling over error from previous MPC setpoints equivalent to:\n{control_error}"
        )

        # Updating setpoint to new MPC data
        new_setpoints = {
            der_ID: MPC_data.loc[der_ID, "P_set"] for der_ID in self.active_DERs
        }
        self._setpoints.update(
            {
                der_ID: np.round(setpoint, default_setpoint_precision)
                for der_ID, setpoint in new_setpoints.items()
            }
        )

        # Updating the error correction with the new MPC data
        error_correction = self.process_control_error(control_error, MPC_data)
        self.logger.debug(f"Setpoints are corrected by: {error_correction}")

        new_setpoints = {
            der_ID: MPC_data.loc[der_ID, "P_set"] + error_correction.get(der_ID, 0)
            for der_ID in self.active_DERs
        }

        # Update the setpoints
        self._setpoints.update(
            {
                der_ID: np.round(setpoint, default_setpoint_precision)
                for der_ID, setpoint in new_setpoints.items()
            }
        )
        self.logger.debug(f"What do updated setpoints look like:\n{self._setpoints}")
        # Update the next timestamp to NaN to avoid reprocessing the same MPC data
        # Change NaN to next MPC timestep to update setpoints for every time we pass a new MPC timestep (not only new MPC message)
        self._MPC_next_timestamp = self.get_next_MPC_timestamp(next_iteration_timestamp)

    def process_dispatch_error(self, dispatch_err, message_timestamp):

        if self._MPC_data is None:
            self.logger.warning("MPC data hasn't been received, skipping RT control")
        else:
            self.logger.info(f"Current Dispatch Error: {dispatch_err}")
            # PID control
            control_Out = self._pid_control(dispatch_err)
            # Note: The Control Out is in generator convention.  If the desired output is an increase in generation, the value of the control out is positive.
            self.logger.debug(f"PID_out = {control_Out:.3f}")

            # Determine which MPC interval is active
            current_data, current_timestamp = self.locate_MPC_data(message_timestamp)

            # self.logger.critical(f"MPC timestamps: {current_timestamps}")
            self.logger.info(
                f"Processing control output using MPC Data for t0: {current_timestamp} // {self.timestamp_to_datetime(current_timestamp)}"
            )

            self.logger.debug(f"MPC Data used:\n{current_data}")

            new_setpoints = {}

            error_correction = self.process_control_error(control_Out, current_data)

            # if control_Out > 0: #control error based on dispatch error (dispatch (load convention) - measurement (load convention))
            #     self.logger.info("P Control: Using Down Reserves, need to increase effective LOAD")
            #     # Use down reserves (add load) to meet the dispatch
            #     error_correction = self.process_control_error(control_Out, current_data, reserves_type = 'down')

            # elif control_Out < 0:
            #     self.logger.info("P Control: Using Up Reserves, need to increase GENERATION")
            #     # Use up reserves (add gen) to meet the dispatch
            #     error_correction = self.process_control_error(control_Out, current_data, reserves_type = 'up')

            new_setpoints = {
                der_ID: self._setpoints.get(der_ID, 0) + error_correction.get(der_ID, 0)
                for der_ID in self.active_DERs
            }
            # Formatting the setpoints: removing NaN values and rounding to the default setpoint precision
            new_setpoints = {
                der_ID: np.round(setpoint, default_setpoint_precision)
                for der_ID, setpoint in new_setpoints.items()
                if not np.isnan(setpoint)
            }

            if new_setpoints:
                self.logger.debug(f"Old P setpoints: {self._setpoints}")
                self.logger.info(f"New P setpoints: {new_setpoints}")
                # New setpoints available to send to the DERs
                self.dispatch_der_rt_control(new_setpoints)
                # Update the setpoints for the next iteration
                self._setpoints.update(new_setpoints)

            # Q Control
            new_setpoints = {
                der_ID: np.round(
                    current_data["Q_set"][der_ID], default_setpoint_precision
                )
                for der_ID in self.active_DERs
            }
            self.logger.info("Q Control: Forwarding Q setpoints from MPC")
            self.logger.info(f"Setpoints: {new_setpoints}")

        if self.cancelled_DERs:
            # Zeros out the setpoints for the cancelled DERs
            new_setpoints = {der_ID: 0.0 for der_ID in self.cancelled_DERs}
            self.logger.info(f"New P setpoints: {new_setpoints}")
            # Dispatch the 0 setpoints to the DERs
            self.dispatch_der_rt_control(new_setpoints)
            # Q Control
            new_setpoints = {der_ID: 0.0 for der_ID in self.cancelled_DERs}
            self.logger.info(f"Q Setpoints: {new_setpoints}")

    def dispatch_der_rt_control(self, der_rt_control_data):
        """
        Dispatch the der_rt_control data to the simulation engine
        :param gridappsd_obj: gridappsd object
        :param der_rt_control_data: der_rt_control data
        :return:
        """
        # dispatch the der_rt_control data to the simulation engine
        self.logger.info("Dispatching der_rt_control data to the simulation engine")
        self.logger.debug(der_rt_control_data)

        # Publish the der_rt_control data to the simulation engine
        timestamp = time.time()

        def payload(derID, setpoint):
            return {
                "setpoint_timestamp": timestamp,
                "equipment_mrid": derID,
                "p_set": setpoint,
            }

        msg = {}
        msg["message"] = [
            payload(der_ID, setpoint)
            for der_ID, setpoint in der_rt_control_data.items()
        ]

        self.IOs.send_gapps_message(self._publish_to_topic, msg)


def _main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "simulation_id", help="Simulation id to use for responses on the message bus."
    )
    parser.add_argument("request", help="Path to the simulation request file")
    parser.add_argument("config", help="App Config")

    # Authenticate with GridAPPS-D Platform
    os.environ["GRIDAPPSD_APPLICATION_ID"] = logger_name
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

    init_logging(
        app_name="RT_control", log_level=log_level, path_to_logs=path_to_export
    )

    _main_logger.warning(
        "RT Control starting!!!-------------------------------------------------------"
    )

    sim_id = opts.simulation_id
    _main_logger.debug(f"Info received from remote: Simulation ID {sim_id}")

    time_multiplier = app_config.get("time_multiplier", 1)
    if time_multiplier != 1:
        _main_logger.warning(f"Time multiplier is set to {time_multiplier}")
        message_period = app_config.get("message_period", default_timestep)
        message_period = int(message_period / time_multiplier)
        _main_logger.warning(f"New message period is {message_period} s.")
        app_config.update({"message_period": message_period})

    model_id = sim_request["power_system_config"]["Line_name"]
    _main_logger.debug(f"Info received from remote: Model ID {model_id}")

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

    # Instantiate IO module
    IO = IObackup(
        simulation_id=sim_id,
        model_id=model_id,
        path_to_repo=path_to_repo,
        path_to_export=path_to_export,
        **kw_args,
    )

    rt_controller = DER_RT_Control(IO, model_id=model_id, **app_config)

    # All ther subscriptions:
    # SIM Output
    simout_topic = IO.simulation_topic("output")
    gapps.subscribe(simout_topic, rt_controller)
    # MPC OUT
    FRS_topic = app_config.get("FRS_topic", default_FRS_topic)
    mpc_topic = IO.application_topic(FRS_topic, "output", sub_id="MPC")
    gapps.subscribe(mpc_topic, rt_controller)
    # ISO OUT
    ISO_topic = app_config.get("ISO_topic", default_ISO_topic)
    iso_topic = IO.application_topic(ISO_topic, "output")
    gapps.subscribe(iso_topic, rt_controller)

    sim_time = app_config.get("sim_time", default_sim_length)
    if sim_time == -1:
        _main_logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _main_logger.info(f"Info received from remote: sim_time {sim_time} s.")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not rt_controller.status:
            _main_logger.error("RT Controller failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)

    _main_logger.warning(
        "RT Control finished!!!-------------------------------------------------------"
    )


if __name__ == "__main__":
    _main()
