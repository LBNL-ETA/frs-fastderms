"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import argparse
import json
import logging
import sys
import time
import datetime as dt
import os
import csv
import numpy as np
import pickle
import pytz
import shutil
import re

from pathlib import Path
from itertools import chain

from gridappsd import GridAPPSD, DifferenceBuilder
from gridappsd.topics import (
    simulation_output_topic,
    application_output_topic,
    service_input_topic,
)

# App Name
app_name = "ADMS_Publisher"
__version__ = "0.9"

# The reference start time for the period of simulation, it is assumed to be in Pacific time.
default_tz = "US/Pacific"
# Timestep [s.]
default_timestep = 60
# Iteration Offset [s.]
default_iteration_offset = 0
# simulation duration (in seconds)
default_sim_length = -1

# Logger
logger_name = f"__{app_name}__"
_logger = logging.getLogger(logger_name)


class ADMSPublisher(object):
    def __init__(self, simulation_id, **kw_args):
        # Re-authenticate with GridAPPS-D (the java security token doesn't get inherited well)
        os.environ["GRIDAPPSD_APPLICATION_ID"] = logger_name
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

        self.mrid = kw_args.get("mrid", app_name)
        self._simout_topic = simulation_output_topic(simulation_id)
        self._publish_to_topic = application_output_topic(self.mrid, None)
        self._automation_topic = kw_args.get(
            "automation_topic", service_input_topic("automation", simulation_id)
        )

        _logger.warning(f"subscribing to (internal):\n {self._simout_topic}")
        _logger.warning(f"publishing to:\n {self._publish_to_topic}")

        tz_str = kw_args.get("tz", default_tz)
        self.local_tz = pytz.timezone(tz_str)
        _logger.info(f"Initializing with local timezone: {self.local_tz}")

        self.iteration_offset = kw_args.get(
            "iteration_offset", default_iteration_offset
        )
        _logger.debug(f"Iteration offset set to {self.iteration_offset} s.")

        tmstp_start = kw_args.get("tmstp_start", None)
        if tmstp_start is None:
            self.set_next_iteration(None, force=True)
            _logger.warning(f"Start time NOT provided, using first simulation output")
        else:
            self.set_next_iteration(tmstp_start, force=True)
            _logger.info(f"Start time is: {self.timestamp_to_datetime(tmstp_start)}")

        self.timestep = kw_args.get("message_period", default_timestep)
        _logger.info(f"Message period set to {self.timestep} s.")

        self.search_folder = Path(
            kw_args.get("search_folder", "./adms_input")
        ).resolve()
        self.search_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Folder for Inputs: {self.search_folder}")
        self.processed_folder = self.search_folder / "processed"
        self.processed_folder.mkdir(parents=True, exist_ok=True)
        _logger.debug(
            f"Creating folder for processed files: {self.processed_folder.name}"
        )

        # Event List:
        self.event_list = kw_args.get("event_list", [])
        self.set_next_event()

        # initialize all the variables:
        self._error_code = False
        self._message_count = 0

        _logger.info("ADMS Publisher Initialized")
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
            of ``GridAPPSD``.  Most message payloads will be serialized dictionaries, but that is
            not a requirement.
        """

        _logger.debug(f"Received message on topic: {headers['destination']}")
        message_timestamp = int(headers["timestamp"]) / 1000

        try:
            # SIMOUT Message
            if self._simout_topic in headers["destination"]:
                # Simulation Output Received
                simulation_timestamp = message["message"]["timestamp"]
                _logger.info(
                    f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}"
                )

                if self.next_iteration is None:
                    _logger.warning(
                        "Next iteration is None, starting execution at first simulation message"
                    )
                    self.next_iteration = self.timestamp_to_datetime(
                        simulation_timestamp
                    )
                    self.next_offset_timestamp = self.get_offset_timestamp(
                        self.next_iteration
                    )

                if simulation_timestamp >= self.next_offset_timestamp:
                    # SIMOUT message is for the next iteration, process it
                    _logger.debug(
                        f"Time is {self.next_iteration}, let's check for new ADMS messages"
                    )

                    # Check for new ADMS messages
                    self.check_and_publish_adms_message()

                    # Iteration is now complete, Set the next iteration
                    self.set_next_iteration(simulation_timestamp)

                else:
                    _logger.debug(
                        f"Waiting until next Iteration at {self.timestamp_to_datetime(self.next_offset_timestamp)}"
                    )

                if simulation_timestamp >= self.next_event[0]:
                    self._message_count += 1
                    _logger.info(
                        f"Simulation time: {simulation_timestamp}, event nr {int(self._message_count)}, event time: {self.next_event[0]}"
                    )
                    self.add_adms_file(self.next_event[1])
                    self.set_next_event()
                else:
                    if not np.isnan(self.next_event[0]):
                        _logger.debug(
                            f"Waiting until next Event at {self.timestamp_to_datetime(self.next_event[0])}"
                        )

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            _logger.error(
                f"Error on line {exc_tb.tb_lineno}: {repr(e)},\n{exc_type}, {exc_obj}"
            )
            self._error_code = True
            raise

    def timestamp_to_datetime(self, timestamp):
        try:
            local_dt = dt.datetime.fromtimestamp(timestamp).astimezone(self.local_tz)
        except:
            if not np.isnan(timestamp):
                _logger.error(f"Could not convert timestamp {timestamp} to datetime.")
            else:
                _logger.info(f"Received NaN timestamp.")
            local_dt = dt.datetime(1970, 1, 1, 0, 0, 0)
        return local_dt

    def set_next_iteration(self, current_time=None, force=False):

        if current_time is None:
            # Used for initialization
            self.next_iteration = None
            self.next_offset_timestamp = None

        else:
            # Regular Use Case
            if isinstance(current_time, dt.datetime):
                # Case of datetime object
                current_timestamp = current_time.timestamp()
            else:
                # Case of timestamp
                current_timestamp = current_time
                current_time = self.timestamp_to_datetime(current_time)

            if force:
                # Force the next iteration to be the current time
                self.next_iteration = current_time
            else:
                # Find the next iteration after the input timestamp
                current_iteration_timestamp = self.next_iteration.timestamp()
                time_diff = max(current_timestamp - current_iteration_timestamp, 0)
                n_timestep = np.floor(time_diff / self.timestep) + 1

                self.next_iteration += dt.timedelta(seconds=n_timestep * self.timestep)
            # Set next offset timestamp
            self.next_offset_timestamp = self.get_offset_timestamp()
            _logger.debug(f"Next iteration is: {self.next_iteration}")

    def get_offset_timestamp(self, datetime=None):
        try:
            if datetime is None:
                datetime = self.next_iteration
            offset_timestamp = (
                datetime + dt.timedelta(seconds=self.iteration_offset)
            ).timestamp()
        except:
            offset_timestamp = None
        finally:
            return offset_timestamp

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
                    raise Exception(" No GAPPS: Cannot send message")
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

    def check_and_publish_adms_message(self):
        # List files in the search folder
        files = list(self.search_folder.glob("*.csv"))
        _logger.debug(f"Found {len(files)} files in {self.search_folder}")

        current_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        for file in files:
            # Parse file and push it to Gridapps
            _logger.debug(f"Processing file: {file}")
            # Normalize by removing any secondary extension
            base_stem = re.sub(
                r"\.[^.]+$", "", file.stem
            )  # Removes anything after the last dot
            data = {}
            data["timestamp"] = int(dt.datetime.now().timestamp())
            data["datatype"] = base_stem
            data["simulation_id"] = self._simulation_id
            data["tags"] = ["simulation_id"]
            data["message"] = []

            with open(file, newline="", encoding="utf-8-sig") as csvfile:
                reader = csv.DictReader(csvfile, skipinitialspace=True)
                for row in reader:
                    data["message"].append(row)

            self.send_gapps_message(self._publish_to_topic, data)

            # Move the file to the processed folder
            shutil.move(
                file,
                self.processed_folder / f"{file.stem}_processed_{current_time_str}.csv",
            )

    def add_adms_file(self, file_path):
        try:
            file_path = Path(file_path).resolve()
            shutil.copy(file_path, self.search_folder)
            _logger.info(f"Added file {file_path.name} to Input Folder")

        except Exception as e:
            _logger.error(f"Failed to add file {file_path} to Input Folder")
            _logger.error(repr(e))

    def set_next_event(self):
        # Events are tuples of timestamp and file_path of file to copy in input folder
        self.next_event = self.event_list.pop(0) if self.event_list else (np.nan, "")
        _logger.debug(f"Next Event: \n{self.next_event}")


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
    log_filename = Path(path_to_logs) / "_log_ADMS_publisher.log"
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
        "ADMS Publisher Starting!!!-------------------------------------------"
    )

    simulation_id = opts.simulation_id

    _gapps = GridAPPSD(simulation_id=simulation_id)
    if _gapps.connected:
        _logger.debug(f"GridAPPSD connected to simulation {simulation_id}")
    else:
        _logger.error("GridAPPSD not Connected")

    adms_publisher = ADMSPublisher(simulation_id, **app_config)

    # Subscribing to Simulation Output
    sim_out_topic = simulation_output_topic(simulation_id)
    _gapps.subscribe(sim_out_topic, adms_publisher)

    _logger.warning(f"subscribing to (main):\n {sim_out_topic}")

    sim_time = app_config.get("sim_time", -1)
    if sim_time == -1:
        _logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _logger.info(f"Info received from remote: sim_time {sim_time} seconds")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not adms_publisher.running():
            if adms_publisher.error() == 2:
                _logger.warning("ADMS Publisher Terminated")
            else:
                _logger.error("ADMS Publisher Failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)

    _logger.warning(
        "ADMS Publisher finished!!!------------------------------------------"
    )


if __name__ == "__main__":
    _main()
