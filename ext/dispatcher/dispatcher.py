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
default_FRS_topic = "FRS"

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
_main_logger = logging.getLogger(logger_name)


class Dispatcher(object):

    def __init__(self, gapps, sim_id, **kw_args):
        """
        Create a new Dispatcher object
        """
        self.logger = logging.getLogger(logger_name)

        self._message_count = 0
        self.gapps = gapps
        self.sim_id = sim_id

        FRS_topic = kw_args.get("FRS_topic", default_FRS_topic)
        self.frs_control_topic = application_output_topic(
            f"{FRS_topic}_RT_Control", None
        )
        self.logger.warning(f"subscribing to:\n {self.frs_control_topic}")
        self._publish_to_topic = simulation_input_topic(sim_id)
        self.logger.warning(f"publishing to:\n {self._publish_to_topic}")
        self._automation_topic = kw_args.get(
            "automation_topic", service_input_topic("automation", sim_id)
        )

        self.PV_list = kw_args.get("PV_list", default_PV_list)
        self.BATT_list = kw_args.get("BATT_list", default_BATT_list)

        self.status = True
        self.logger.info("Dispatcher Initialized")
        self.send_gapps_message(self._automation_topic, {"command": "stop_task"})

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
        self.logger.debug(f"Received message on topic: {headers['destination']}")

        message_timestamp = int(headers["timestamp"]) / 1000
        try:
            # RT controller message
            if self.frs_control_topic in headers["destination"]:
                self.logger.info(f"Processing message from RT controller")
                self.logger.debug(message.keys())

                setpoints = {}
                for data_dict in message["message"]:
                    # Process the message
                    resourceID = data_dict["equipment_mrid"]
                    setpoint = data_dict["p_set"]
                    setpoints[resourceID] = setpoint

                self.dispatch_der(setpoints)

        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.status = False
            raise

    def dispatch_der(self, der_rt_control_data):
        """
        Dispatch the der_rt_control data to the simulation engine
        :param gridappsd_obj: gridappsd object
        :param der_rt_control_data: der_rt_control data
        :return:
        """
        self.logger.debug(der_rt_control_data)
        # dispatch the der_rt_control data to the simulation engine
        FRS_diffs = DifferenceBuilder(self.sim_id)
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
        self.logger.info("Dispatching der_rt_control data to the simulation engine")
        message = json.dumps(FRS_diffs.get_message())
        self.logger.debug(f"Publishing message to topic: {self._publish_to_topic}")
        self.logger.debug(message)
        self.gapps.send(self._publish_to_topic, message)

    def send_gapps_message(self, topic, message):
        out_message = {}
        try:
            for key in message.keys():
                out_message[key] = message[key]
        except:
            self.logger.info("Message is not a dictionary")
            out_message["message"] = message
        finally:
            try:
                if self.gapps is None:
                    raise Exception("No GAPPS: Cannot send message")
                self.gapps.send(topic, json.dumps(out_message))
                self.logger.info(
                    f"Sent message to topic {topic} at {dt.datetime.now()}"
                )
                self.logger.debug(f"Message Content: {out_message}")
                return True
            except Exception as e:
                self.logger.error(
                    f"Failed to send message to topic {topic} at {dt.datetime.now()}"
                )
                self.logger.error(e)
                return False


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

    _main_logger.setLevel(log_level)
    _main_logger.warning(
        "Dispatcher starting!!!-------------------------------------------------------"
    )

    simulation_id = opts.simulation_id
    model_id = sim_request["power_system_config"]["Line_name"]
    _main_logger.debug(f"Model mrid is: {model_id}")

    gapps = GridAPPSD(simulation_id=simulation_id)
    if gapps.connected:
        _main_logger.debug(f"GridAPPSD connected to simulation {simulation_id}")
    else:
        _main_logger.error("GridAPPSD not Connected")
        gapps = None

    dispatcher = Dispatcher(gapps, simulation_id, **app_config)

    FRS_topic = app_config.get("FRS_topic", default_FRS_topic)
    rt_topic = application_output_topic(f"{FRS_topic}_RT_Control", None)
    gapps.subscribe(rt_topic, dispatcher)

    sim_time = sim_time = app_config.get("sim_time", default_sim_length)
    if sim_time == -1:
        _main_logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _main_logger.info(f"Info received from remote: sim_time {sim_time} seconds")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not dispatcher.status:
            _main_logger.error("Dispatcher failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)
    _main_logger.warning(
        "Dispatcher finished!!!-------------------------------------------------------"
    )


if __name__ == "__main__":
    _main()
