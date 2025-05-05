"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

from pathlib import Path
from gridappsd import GridAPPSD, DifferenceBuilder, utils
from gridappsd.topics import (
    simulation_input_topic,
    application_output_topic,
    service_input_topic,
)

import argparse
import json
import logging
import time
import os

import datetime as dt

# App Name
app_name = "Dispatcher"
__version__ = "0.9"

# Topics
default_FRS_topic = "FRS_RT_Control"

# Default Values
# simulation duration (in hours)
default_sim_length = -1

default_BATT_list = {}
default_PV_list = {
    "PV1": [
        "_F34E9382-78F6-42A7-87D4-E39182A5C740",
        "_755E3B33-8001-45A9-9681-5C7B46396327",
    ],
    "PV2": "_8F3C3887-6D69-48E2-9BD9-EFB9E151DEBB",
    "PV3": "_C771288A-046A-411B-BDE9-2B5FCA365736",
}

# Logger
logger_name = f"__Main__{app_name}"
_logger = logging.getLogger(logger_name)


class Dispatcher(object):

    def __init__(self, simulation_id, **kw_args):
        """
        Create a new Dispatcher object
        """
        self.mrid = kw_args.get("mrid", app_name)
        # Re-authenticate with GridAPPS-D (the java security token doesn't get inherited well)
        os.environ["GRIDAPPSD_APPLICATION_ID"] = self.mrid
        os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

        try:
            # Connect to GridAPPS-D Platform
            _gapps = GridAPPSD(simulation_id=simulation_id)
            if not _gapps.connected:
                raise ConnectionError("Failed to connect to GridAPPS-D")
        except Exception as e:
            _logger.error(e)
            _gapps = None

        self._gapps = _gapps
        self._simulation_id = simulation_id

        FRS_topic = kw_args.get("FRS_topic", default_FRS_topic)
        self.frs_control_topic = application_output_topic(FRS_topic, None)
        _logger.warning(f"subscribing to:\n {self.frs_control_topic}")
        self._publish_to_topic = simulation_input_topic(simulation_id)
        _logger.warning(f"publishing to:\n {self._publish_to_topic}")
        self._automation_topic = kw_args.get(
            "automation_topic", service_input_topic("automation", simulation_id)
        )

        self.PV_list = kw_args.get("PV_list", default_PV_list)
        self.BATT_list = kw_args.get("BATT_list", default_BATT_list)

        # initialize all the variables:
        self._error_code = False
        self._message_count = 0
        _logger.info("Dispatcher Initialized")
        self.send_gapps_message(self._automation_topic, {"command": "stop_task"})

    def running(self):
        # Check if any error code
        running = not bool(self._error_code)
        return running

    def error(self):
        return self._error_code

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
        _logger.debug(f"Received message on topic: {headers['destination']}")

        message_timestamp = int(headers["timestamp"]) / 1000
        try:
            # RT controller message
            if self.frs_control_topic in headers["destination"]:
                self._message_count += 1
                _logger.info(
                    f"Processing message nr {int(self._message_count)} from RT controller"
                )
                _logger.debug(message.keys())

                setpoints = {}
                for data_dict in message["message"]:
                    # Process the message
                    resourceID = data_dict["equipment_mrid"]
                    setpoint = data_dict["p_set"]
                    setpoints[resourceID] = setpoint

                self.dispatch_der(setpoints)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            _logger.error(
                f"Error on line {exc_tb.tb_lineno}: {repr(e)},\n{exc_type}, {exc_obj}"
            )
            self._error_code = True
            raise

    def dispatch_der(self, der_rt_control_data):
        """
        Dispatch the der_rt_control data to the simulation engine
        :param gridappsd_obj: gridappsd object
        :param der_rt_control_data: der_rt_control data
        :return:
        """
        _logger.debug(der_rt_control_data)
        # dispatch the der_rt_control data to the simulation engine
        FRS_diffs = DifferenceBuilder(self._simulation_id)
        for resourceID, setpoint in der_rt_control_data.items():
            # converting setpoint to kW to match the units of the simulation engine
            setpoint = int(setpoint * 1000)
            if resourceID in self.PV_list.keys():
                PV_ids = self.PV_list[resourceID]
                if type(PV_ids) is list:
                    PV_count = len(PV_ids)
                    for PV_id in PV_ids:
                        FRS_diffs.add_difference(
                            PV_id,
                            "PowerElectronicsConnection.p",
                            int(setpoint / PV_count),
                            0.0,
                        )
                else:
                    FRS_diffs.add_difference(
                        self.PV_list[resourceID],
                        "PowerElectronicsConnection.p",
                        int(setpoint),
                        0.0,
                    )
                # FRS_diffs.add_difference(resourceID, "PowerElectronicsConnection.q", setpoint, 0)
            elif resourceID in self.BATT_list:
                FRS_diffs.add_difference(
                    self.BATT_list[resourceID],
                    "PowerElectronicsConnection.p",
                    int(setpoint),
                    0.0,
                )
                # FRS_diffs.add_difference(resourceID, "PowerElectronicsConnection.q", setpoint, 0)
        _logger.info("Dispatching der_rt_control data to the simulation engine")
        message = json.dumps(FRS_diffs.get_message())
        _logger.debug(f"Publishing message to topic: {self._publish_to_topic}")
        _logger.debug(message)
        self._gapps.send(self._publish_to_topic, message)

    def send_gapps_message(self, topic, message):
        out_message = {}
        try:
            for key in message.keys():
                out_message[key] = message[key]
        except:
            _logger.info("Message is not a dictionary")
            out_message["message"] = message
        finally:
            try:
                if self._gapps is None:
                    raise Exception("No GAPPS: Cannot send message")
                self._gapps.send(topic, json.dumps(out_message))
                _logger.info(f"Sent message to topic {topic} at {dt.datetime.now()}")
                _logger.debug(f"Message Content: {out_message}")
                return True
            except Exception as e:
                _logger.error(
                    f"Failed to send message to topic {topic} at {dt.datetime.now()}"
                )
                _logger.error(e)
                return False


########################### Main Program
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
    path_to_logs = app_config.get("path_to_export", "./logs/")
    log_filename = Path(path_to_logs) / "_log_common_demo.log"
    logging.basicConfig(
        filename=log_filename,
        format="%(levelname)-8s %(asctime)s | %(name)-12s: %(message)s",
        level=log_level,
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console._name = "console"
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    root_logger = logging.getLogger()

    if not any(handler._name in ["console"] for handler in root_logger.handlers):
        root_logger.addHandler(console)

    _logger.setLevel(log_level)

    ############## Individual Logger
    log_filename = Path(path_to_logs) / "_log_Dispatcher.log"
    debug_file = logging.FileHandler(log_filename)
    debug_file.setLevel(log_level)
    debug_file._name = "individual_log"
    debug_format = logging.Formatter(
        "%(levelname)-8s %(asctime)s | %(name)-12s: %(message)s"
    )
    debug_file.setFormatter(debug_format)
    root_logger = logging.getLogger()

    if not any(handler._name in ["individual_log"] for handler in root_logger.handlers):
        root_logger.addHandler(debug_file)
    ############### END OF INDIVIDUAL LOG FILE

    _logger.warning(
        "Dispatcher starting!!!-------------------------------------------------------"
    )

    simulation_id = opts.simulation_id

    _gapps = GridAPPSD(simulation_id=simulation_id)
    if _gapps.connected:
        _logger.debug(f"GridAPPSD connected to simulation {simulation_id}")
    else:
        _logger.error("GridAPPSD not Connected")

    dispatcher = Dispatcher(simulation_id, **app_config)

    # Subscribing to FRS Control
    FRS_topic = app_config.get("FRS_topic", default_FRS_topic)
    control_topic = application_output_topic(FRS_topic, None)
    _gapps.subscribe(control_topic, dispatcher)
    _logger.warning(f"subscribing to (main):\n {control_topic}")

    sim_time = app_config.get("sim_time", default_sim_length)
    if sim_time == -1:
        _logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _logger.info(f"Info received from remote: sim_time {sim_time} seconds")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not dispatcher.running():
            if dispatcher.error() == 2:
                _logger.warning("Dispatcher Terminated")
            else:
                _logger.error("Dispatcher Failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)
    _logger.warning(
        "Dispatcher finished!!!-------------------------------------------------------"
    )


if __name__ == "__main__":
    _main()
