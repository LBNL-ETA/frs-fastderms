"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from typing import List
from itertools import chain
from pathlib import Path
from gridappsd import GridAPPSD, topics
from numpyencoder import NumpyEncoder

import datetime as dt
import pandas as pd
import numpy as np

import logging
import logging.handlers
import pickle
import types
import time
import pytz
import json
import os
import re


class fastderms:
    def __init__(self, *args, **kw_args) -> None:

        # Module Name
        self.name = kw_args.get("name", __name__)
        # Logger
        self.logger = logging.getLogger(self.name)
        self.logger.debug("Start of FASTDERMS class Initialization")
        self.logger.debug(
            f"Parent Logger {self.logger.parent}: level {self.logger.parent.level}"
        )
        self.logger.setLevel(self.logger.parent.level)
        self.logger.debug(f"Logger level {self.logger.level}")

        # mRID
        self.mrid = kw_args.get("mrid", "fastderms")

        # Timezone
        tz_str = format_tz(kw_args.get("tz", "US/Pacific"))
        self.local_tz = pytz.timezone(tz_str)
        self.logger.info(f"Initializing with local timezone: {self.local_tz}")

        self.path_to_export = Path(kw_args.get("path_to_export", "./logs/")).resolve()

        self.logger.debug("End of FASTDERMS class Initialization")

    def print_matrix(self, matrix, **kw_args):
        serialized_matrix = "\n"
        row_headers = kw_args.get("row_headers", None)
        col_headers = kw_args.get("col_headers", None)
        try:
            shape = matrix.shape
            if col_headers:
                serialized_matrix += (
                    "____|" + "   ".join([col[-4:] for col in col_headers]) + "\n"
                )

            for row in range(shape[0]):
                if row_headers:
                    serialized_matrix += f"{row_headers[row][-4:]}: "

                for col in range(shape[1]):
                    try:
                        for phase_nr in range(len(matrix[row, col])):
                            if matrix[row, col, phase_nr] == 0:
                                serialized_matrix += "_"
                            else:
                                serialized_matrix += f"{matrix[row,col,phase_nr]:.0f}"
                            if phase_nr < 2:
                                serialized_matrix += " "
                            else:
                                serialized_matrix += ".."
                    except:
                        if matrix[row, col] == 0:
                            serialized_matrix += "_.."
                        else:
                            serialized_matrix += f"{matrix[row,col]:.0f}.."
                serialized_matrix += "\n"
        except:

            for key in matrix.keys():
                serialized_matrix += f"{key}: \n"
                for bus in matrix[key].keys():
                    serialized_matrix += f"Bus: {bus}:"
                    for der in matrix[key][bus].keys():
                        serialized_matrix += f"{der}: {matrix[key][bus][der]:.2f}.. "
                    serialized_matrix += "\n"
                serialized_matrix += "\n"

        finally:
            return serialized_matrix

    def print_dict(self, dictionary, **kw_args):
        """
        Print a dictionary in a nice way
        removes empty keys
        """
        serialized_dict = "\n"
        indent = kw_args.get("indent", "")
        for key in dictionary.keys():
            if isinstance(dictionary[key], dict):
                serialized_subdict = self.print_dict(
                    dictionary[key], indent=indent + "   "
                )
                if len(serialized_subdict) > 1:
                    serialized_dict += f"{indent}{key}:"
                    serialized_dict += serialized_subdict
            elif not dictionary[key]:
                serialized_dict = ""
            else:
                serialized_dict += f"{indent}{key}: {dictionary[key]}\n"
        return serialized_dict

    def merge_phases(self, phase_list, **kw_args):
        recursion = kw_args.get("recursion", False)

        if not recursion:
            self.logger.debug(f"Merging incoming phases: {phase_list}")
        phases = []
        accepted_phases = [0, 1, 2, "s1", "s2"]

        try:
            if all(phase in accepted_phases for phase in phase_list):
                phases = [phase for phase in accepted_phases if phase in phase_list]
            elif phase_list == "s12N" or all(phase == "s12N" for phase in phase_list):
                phases = [0, 1, 2]
            else:
                for phase, phase_nr in zip(["A", "B", "C"], [0, 1, 2]):
                    if phase in phase_list:
                        phases.append(phase_nr)
                if "s1" in phase_list:
                    phases.append("s1")
                if "s2" in phase_list:
                    phases.append("s2")
            if not phases:
                raise (Exception("Trying to flatten the list"))
        except Exception as e:
            self.logger.error(e)
            # flatten the list
            phase_list = [
                phase for phase in chain.from_iterable(phases for phases in phase_list)
            ]
            self.logger.debug(f"Flatten phases: {phase_list}")
            if not recursion:
                phases = self.merge_phases(phase_list, recursion=True)
            else:
                phases = []
                self.logger.error(
                    f"Recursion limit reached, we could not parse the phases"
                )
        finally:
            if not recursion:
                self.logger.debug(f"Merged phases: {phases}")
            return phases

    def TS_phase_separator(self, Timeseries: List):

        try:
            # Separate phases / multi demensions of a timeseries
            return tuple(
                [
                    [sample[n] for sample in Timeseries]
                    for n in range(len(Timeseries[0]))
                ]
            )

        except:
            return [Timeseries]

    def timestamp_to_datetime(self, timestamp):
        try:
            local_dt = dt.datetime.fromtimestamp(timestamp).astimezone(self.local_tz)
        except:
            if not np.isnan(timestamp):
                self.logger.error(
                    f"Could not convert timestamp {timestamp} to datetime."
                )
            else:
                self.logger.info(f"Received NaN timestamp.")
            local_dt = dt.datetime(1970, 1, 1, 0, 0, 0)
        return local_dt

    def time_mod(self, time, delta, epoch=None):
        if epoch is None:
            epoch = dt.datetime(1970, 1, 1, tzinfo=time.tzinfo)
        return (time - epoch) % delta

    def time_round(self, time, delta, epoch=None, method="default"):
        mod = self.time_mod(time, delta, epoch)
        if method == "default":
            if mod < delta / 2:
                return time - mod
            else:
                return time + (delta - mod)
        elif method == "ceil":
            if mod:
                return time + (delta - mod)
            else:
                return time
        elif method == "floor":
            return time - mod
        else:
            raise (Exception(f"Unknown method {method}"))

    def load_all_pickle(self, filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def is_error(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            return False
        except:
            return True


class fastderms_app(fastderms):
    def __init__(self, *args, **kw_args) -> None:
        super().__init__(*args, **kw_args)

        # Iteration parameters
        self.timestep = kw_args.get("message_period", 0)
        self.logger.info(f"Message period set to {self.timestep} s.")
        self.iteration_offset = kw_args.get("iteration_offset", 0)
        self.logger.debug(f"Iteration offset set to {self.iteration_offset} s.")
        # Iteration Tracking
        tmstp_start = kw_args.get("tmstp_start", None)
        if tmstp_start is None:
            t_start = None
            self.logger.warning(
                f"Start time NOT provided, using first simulation output"
            )
        else:
            t_start = self.timestamp_to_datetime(tmstp_start)
            self.logger.info(f"Start time provided: {t_start}")
        self.set_next_iteration(current_time=t_start, force=True)

        # Error Code
        self._error_code = False

    def running(self):
        """
        Check if the application is running without errors.

        Returns:
            bool: True if the application is running without errors, False otherwise. The status is determined by checking if there's no error code set.
        """
        # Check if any error code
        running = not bool(self._error_code)
        return running

    def error(self):
        """
        Get the current error code of the application.

        Returns:
            int or bool: The error code value. Possible values:
                - 0 or False: No error
                - 1 or True: General error
                - 2: Controlled termination
        """
        return self._error_code

    def set_next_iteration(self, current_time=None, force=False):

        if current_time is None:
            # Used for initialization
            self.previous_iteration = None
            self.next_iteration = None
            self.next_offset_timestamp = None
            self.logger.debug('Iteration variables are initialized to "None"')

        else:
            # Regular Use Case
            # Check type of current_time
            if isinstance(current_time, dt.datetime):
                # Case of datetime object
                current_timestamp = current_time.timestamp()
            else:
                # Other Case, assuming it is a timestamp
                current_timestamp = current_time
                current_time = self.timestamp_to_datetime(current_time)

            if force:
                # Force the next iteration to be the current time
                self.next_iteration = current_time
                self.logger.debug(f"Next iteration updated to: {self.next_iteration}")
                # Arbitrarily setting previous iteration to the same
                self.previous_iteration = current_time
            else:
                # Find the next iteration after the input timestamp
                current_iteration_timestamp = self.next_iteration.timestamp()
                time_diff = current_timestamp - current_iteration_timestamp
                # Test if the next iteration is in the future
                # Using 2 iteration offsets to account for the time it takes to complete an iteration
                if time_diff < 2 * self.iteration_offset:
                    # The next iteration is already (far enough) in the future
                    self.logger.debug(f"Next iteration is not updated")
                else:
                    # Two cases are covered here
                    # time_diff > 0: The next iteration is in the past, let's compute the next one
                    # time_diff < 0: The next iteration is not far enough in the future, let's skip it and compute the next one
                    time_diff = max(time_diff, 0)
                    n_timestep = np.floor(time_diff / self.timestep) + 1
                    self.previous_iteration = self.next_iteration
                    self.next_iteration += dt.timedelta(
                        seconds=n_timestep * self.timestep
                    )

            self.logger.debug(f"Next iteration is: {self.next_iteration}")
            # Set next offset timestamp
            self.next_offset_timestamp = self.get_offset_timestamp()

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


class mRID(str):
    def __init__(self) -> None:
        super().__init__()


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


class print_time(object):
    def __init__(self):
        self.s0 = time.time()

    def start(self):
        self.s1 = time.time()

    def end(self):
        end_time = time.time()
        print(
            f"--------------------------------------------------------------\n\
Execution of this block: {(end_time - self.s1):.2f} s., total since start: {(end_time-self.s0):.2f} s."
        )


############################################################################################################
# Exceptions


class Import_Exception(Exception):
    pass


class Implementation_Exception(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "Not implemented:" + message
        super().__init__(message, *args, **kwargs)


class No_GAPPs_Exception(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "No GAPPs (Offline) " + message
        super().__init__(message, *args, **kwargs)


class Fake_GAPPs_Exception(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "Fake GAPPs: " + message
        super().__init__(message, *args, **kwargs)


class Pyomo_Exception(Exception):
    pass


class Bad_Data_Exception(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "BAD DATA DETECTED: " + message
        super().__init__(message, *args, **kwargs)


class ForceStaticException(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "FORCING STATIC DATA" + message
        super().__init__(message, *args, **kwargs)


class MissingItem_Exception(Exception):
    def __init__(self, message: str = "", *args, **kwargs):

        message = "Missing item:" + message
        super().__init__(message, *args, **kwargs)


class SkippingApp(Exception):
    def __init__(self, message: str = "", *args, **kwargs):
        message = f"Skipping {message} !"
        super().__init__(message, *args, **kwargs)


##############################


def init_logging(
    app_name: str = "app_name",
    log_level: int = logging.WARNING,
    log_level_console: int = logging.WARNING,
    path_to_logs=Path("./logs/"),
):

    # Common Logs
    path_to_logs = Path(path_to_logs)
    if not path_to_logs.exists():
        path_to_logs.mkdir(parents=True)
    filename = path_to_logs / "_log_common_demo.log"
    file_handler = logging.handlers.WatchedFileHandler(filename)
    logging.basicConfig(
        handlers=[file_handler],
        format="%(levelname)-8s %(asctime)s | %(name)-12s: %(message)s",
        level=log_level,
    )

    # Root Logger
    root_logger = logging.getLogger()

    # Individual Logger
    log_filename = path_to_logs / f"_log_{app_name}.log"
    file_handler = logging.handlers.WatchedFileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler._name = "individual_log"
    formatter = logging.Formatter(
        "%(levelname)-8s %(asctime)s | %(name)-12s: %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not any(handler._name in ["individual_log"] for handler in root_logger.handlers):
        root_logger.addHandler(file_handler)

    # Console Logger
    console = logging.StreamHandler()
    console.setLevel(log_level_console)
    console._name = "console"
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)

    if not any(handler._name in ["console"] for handler in root_logger.handlers):
        root_logger.addHandler(console)


def format_tz(tz_str) -> str:
    # Format the timezone string to allow for shortcuts (PST, EST, etc...)
    # If tz unknown or invalid, return UTC
    # If input is a timezone object, return its zone info

    if isinstance(tz_str, pytz.tzinfo.BaseTzInfo):
        return tz_str.zone
    elif tz_str == "PST":
        return "US/Pacific"
    elif tz_str == "EST":
        return "US/Eastern"
    elif tz_str in pytz.all_timezones:
        return tz_str
    else:
        return "UTC"


FRS_demo_params = {
    "model_id": "_C07972A7-600D-4AA5-B254-4CAA4263560E",  # IEEE 13 OCHRE GONOGO
    "Sbase": 2e6,
    "tz": "US/Pacific",
    "case_dt0": dt.datetime(2022, 4, 1, 0, 0, 0),
    "rt_start_dt": dt.datetime(2022, 4, 1, 14, 0, 0),
    "simulation_start_offset_minutes": 5,
    "sim_time": -1,
    "load_loadprofile_data": True,
    "reset_loadprofile_data": True,
    "load_weather_data": True,
    "reset_weather_data": True,
    "use_ochre": False,
    "mrid_orchestrator": "FRS",
    "path_to_orchestrator": "./src/FAST_DERMS_CONTROL/orchestrator.py",
    "log_orchestrator": logging.INFO,
    "mrid_iso": "ISO",
    "path_to_iso": "./src/ISO/ISO_main.py",
    "log_iso": logging.INFO,
    "dispatch_period_iso": 300,
    "use_static_data_iso": False,
    "use_rt_controller": True,
    "mrid_rt_controller": "RT_CONTROLLER",
    "path_to_rt_controller": "./src/FAST_DERMS_CONTROL/Control/DER_RT_Control.py",
    "log_rt_controller": logging.INFO,
    "period_rt_controller": 60,
    "use_dispatcher": True,
    "mrid_dispatcher": "DISPATCHER",
    "path_to_dispatcher": "./ext/dispatcher/dispatcher.py",
    "log_dispatcher": logging.INFO,
    "use_batt_aggregator": True,
    "mrid_batt_aggregator": "AGGREGATOR",
    "path_to_batt_aggregator": "./ext/battery_aggregator_app/runapp_main.py",
    "log_batt_aggregator": logging.INFO,
    "use_tmm": False,
    "use_adms_publisher": False,
    "use_mpc_dispatch_controller": False,
}


class FRS_config(fastderms):
    def __init__(self, *args, **kw_args):
        super().__init__(name="FRS Config")
        self.params = list(FRS_demo_params.keys())
        self.ignore_params = [
            "In",
            "Out",
            "get_ipython",
            "exit",
            "quit",
            "os",
            "sys",
            "params",
            "ignore_params",
            "logger",
            "logging",
            "local_tz",
            "Path",
            "dt",
            "FRS_config",
            "my_FRS_config",
            __name__,
        ]

        try:
            if len(args) > 0:
                obj = args[0]
                if not isinstance(obj, str) and not isinstance(obj, os.PathLike):
                    raise Exception("Object provided is not a path to a config file.")
                else:
                    path_to_config = obj
            else:
                path_to_config = "./FRS_config.json"
            self.import_config(obj)
        except Exception as e:
            self.logger.error(e)
            self.logger.warning(
                "Using provided data and default parameters for missing entries."
            )

            all_params = list(obj.__dict__.keys()) + self.params
            for param in all_params:
                value = getattr(obj, param, FRS_demo_params.get(param, None))
                self._set_param(param, value)

    def _set_param(self, param, value):
        try:
            # Sort params to ignore
            if not param.startswith("_") and param not in self.ignore_params:
                if type(value) not in [
                    types.ModuleType,
                    types.FunctionType,
                    types.MethodType,
                ]:
                    # Process certain types of parameters
                    if param == "tz":
                        tz_str = format_tz(value)
                        self.local_tz = pytz.timezone(tz_str)
                        self.logger.info(f"Updating timezone: {self.local_tz}")

                    if type(value) == dt.datetime:
                        value = self.local_tz.localize(value).timestamp()

                    if isinstance(value, os.PathLike):
                        value = str(value.resolve())

                    setattr(self, param, value)
                    self.params.append(param)
        except Exception as e:
            pass

    def export_config(self, path: str = "FRS_config.json"):
        config_dict = {param: getattr(self, param) for param in self.params}
        self.logger.info(f"Saving config to {path}")
        if type(path) != str:
            path = str(path.resolve())
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(config_dict, f, cls=NumpyEncoder)
        elif path.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(config_dict, f)

    def import_config(self, path: str = "FRS_config.json"):
        self.logger.info(f"Loading config from {path}")
        if path.endswith(".json"):
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                config_dict = pickle.load(f)
        for param in self.params:
            setattr(self, param, config_dict.get(param, FRS_demo_params[param]))
            if param == "tz":
                tz_str = format_tz(config_dict.get(param, FRS_demo_params[param]))
                self.local_tz = pytz.timezone(tz_str)
                self.logger.info(f"Updating timezone: {self.local_tz}")

        for param in config_dict.keys():
            if (
                not param.startswith("_")
                and param not in self.ignore_params + self.params
            ):
                setattr(self, param, config_dict[param])
                self.params.append(param)


class input_data_processor(fastderms):
    def __init__(self, *args, **kw_args) -> None:
        super().__init__(name="Input Data Processor", *args, **kw_args)
        # Re-authenticate with GridAPPS-D (the java security token doesn't get inherited well)
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

        # ---------------------- Get Model_dict.json file from the GridAPPS-D platfor,
        model_id = kw_args.get("model_id", None)
        data = self.get_CIM_dict(model_id)
        feeder_data = data["feeders"][0]
        self.logger.warning(
            f"Loaded model {feeder_data['mRID']} from GridAPPS-D platform."
        )
        self.logger.debug(f"Model data\n: {feeder_data}")
        self.switch_data = feeder_data["switches"]
        self.solar_data = feeder_data["solarpanels"]
        self.load_data = feeder_data["energyconsumers"]
        self.meas_data = feeder_data["measurements"]
        self.OCHRE_MS = None
        self.set_pv_name_regexp()

    def get_CIM_dict(self, model_id):
        message = {
            "configurationType": "CIM Dictionary",
            "parameters": {"model_id": model_id},
        }
        topic = topics.CONFIG
        _gapps = GridAPPSD()
        data_feeder = _gapps.get_response(topic, message, timeout=180)
        return data_feeder["data"]

    def get_substation_measurement_IDs(self, substation_node, subsation_line_name):
        substation_measurement_IDs = [None] * 3

        for measurement in self.meas_data:
            if (
                measurement["name"] == subsation_line_name
                and measurement["ConnectivityNode"] == substation_node
            ):
                if measurement["measurementType"] == "VA":
                    if measurement["phases"] == "A":
                        substation_measurement_IDs[0] = measurement["mRID"]
                    elif measurement["phases"] == "B":
                        substation_measurement_IDs[1] = measurement["mRID"]
                    elif measurement["phases"] == "C":
                        substation_measurement_IDs[2] = measurement["mRID"]
        return substation_measurement_IDs

    def set_pv_name_regexp(self, pv_name_regexp="(.*)(pv\d{1,2})(.*)"):
        self.pv_name_regexp = pv_name_regexp

    def get_dispatchable_PV_list(self, return_names=False):
        # searching string
        m = re.compile(self.pv_name_regexp)
        PV_list = {}
        PV_names = {}
        for data in self.solar_data:
            g = m.match(data["name"])
            if g:
                name = g.group(2).upper()
                if name in PV_list:
                    existing_meas = PV_list[name]
                    existing_names = PV_names[name]
                    if isinstance(existing_meas, list):
                        existing_meas.append(data["mRID"])
                        existing_names.append(data["name"])
                    else:
                        existing_meas = [existing_meas, data["mRID"]]
                        existing_names = [existing_names, data["name"]]

                    PV_list.update({name: existing_meas})
                    PV_names.update({name: existing_names})

                else:
                    PV_list.update({name: data["mRID"]})
                    PV_names.update({name: data["name"]})

        if return_names:
            return PV_list, PV_names

        return PV_list

    def get_non_dispatchable_PV_list(self):
        # searching string
        m = re.compile(self.pv_name_regexp)
        PV_list = {}
        for data in self.solar_data:
            g = m.match(data["name"])
            if not g:
                name = data["name"]
                if name in PV_list:
                    existing_meas = PV_list[name]
                    if isinstance(existing_meas, list):
                        existing_meas.append(data["mRID"])
                    else:
                        existing_meas = [existing_meas, data["mRID"]]

                    PV_list.update({name: existing_meas})

                else:
                    PV_list.update({name: data["mRID"]})
        return PV_list

    def get_PV_measurement_IDs(self):
        PV_list, PV_names = self.get_dispatchable_PV_list(return_names=True)

        PV_measurement_IDs = {PV_ID: [] for PV_ID in PV_list}
        all_PV_names = [item for item in chain.from_iterable(PV_names.values())]

        for measurement in self.meas_data:
            if (
                measurement["measurementType"] == "VA"
                and measurement["ConductingEquipment_type"]
                == "PowerElectronicsConnection"
            ):
                conduct_name = measurement["ConductingEquipment_name"]
                for PV_ID in PV_names:
                    if conduct_name in PV_names[PV_ID]:
                        PV_measurement_IDs[PV_ID].append(measurement["mRID"])
        return PV_measurement_IDs

    def set_OCHRE_MS(self, path_to_ms):
        self.OCHRE_MS = pd.read_excel(path_to_ms)
        self.OCHRE_MS.set_index("House_ID", inplace=True)

    def _format_OCHRE_BATT_data(self, house_dict):
        ochre_dict = {
            "State of Charge (-)": house_dict["soc"],
            "Energy Capacity (kWh)": house_dict["E"],
            "Power Capacity (kW)": house_dict["P"],
            "Charge Efficiency (-)": house_dict["nuC"],
            "Discharge Efficiency (-)": house_dict["nuD"],
        }
        return ochre_dict

    def format_OCHRE_BATT_data(self, batt_data_in):

        if self.OCHRE_MS is not None:
            # Update static data with OCHRE MS data
            for batt_id, batt_agg in batt_data_in.items():
                for house_id, house_dict in batt_agg.items():
                    try:
                        house_dict.update(
                            {"soc": self.OCHRE_MS.loc[house_id, "Initial SOC"]}
                        )
                        house_dict.update(
                            {"E": self.OCHRE_MS.loc[house_id, "Battery (kWh)"]}
                        )
                        # if data missing Battery is assumed to be 2h of storage at max power
                        house_dict.update(
                            {
                                "P": house_dict.get(
                                    "P",
                                    self.OCHRE_MS.loc[house_id, "Battery (kWh)"] / 2,
                                )
                            }
                        )
                    except:
                        self.logger.error(
                            f"House {house_id} not found in OCHRE MS data. Skipping update."
                        )

        formatted_BATT_data = {
            batt_id: {
                house_id: self._format_OCHRE_BATT_data(house_dict)
                for house_id, house_dict in batt_agg.items()
            }
            for batt_id, batt_agg in batt_data_in.items()
        }

        return formatted_BATT_data

    def get_BATT_measurement_IDs(self, batt_data_in):
        all_houses = [house for house in chain(*batt_data_in.values())]

        all_houses_loaddata = {
            house: self.OCHRE_MS.loc[house, ["Load_name", "Phase"]]
            for house in all_houses
        }
        # Process Phases
        for house in all_houses_loaddata.keys():
            if all_houses_loaddata[house]["Phase"] == 12:
                all_houses_loaddata[house]["Phase"] = ["s1", "s2"]

        BATT_meas_IDs = {
            houseID: [
                meas["mRID"]
                for meas in self.meas_data
                if meas["ConductingEquipment_name"]
                == load_data["Load_name"].split("ld_")[1]
                and meas["phases"] in load_data["Phase"]
                and meas["measurementType"] == "VA"
            ]
            for houseID, load_data in all_houses_loaddata.items()
        }
        return BATT_meas_IDs

    def get_static_data_switches(self):
        static_data = {}

        for switch in self.switch_data:
            static_data[switch["name"]] = {"status": not switch["normalOpen"]}

        return static_data

    def get_static_data_Loads(self):
        static_data = {}

        all_loads = [
            load["name"]
            for load in self.load_data
            if load["name"][0:9] not in ["tl_house_", "utility_b"]
        ]

        for meas_item in self.meas_data:
            if (
                meas_item["ConductingEquipment_name"] in all_loads
                and meas_item["measurementType"] == "VA"
            ):
                bus = meas_item["ConnectivityNode"]
                load_name = f"L-{bus}"
                static_data[load_name] = {"bus": bus}
        return static_data

    def get_static_data_PVs(self, PV_list):
        static_data = {}

        type = "PV"
        pmin = 0
        qmin = None  # Unused for single inverter
        qmax = None  # Unused for single inverter

        PV_list = [PV.upper() for PV in PV_list]

        for PV in self.solar_data:
            if PV["name"].upper() in PV_list:
                name = PV["name"].upper()
                mRID = PV["mRID"]
                bus = PV["CN1"]
                phases = PV["phases"]
                smax = PV["ratedS"]

                new_phases = []
                if "A" in phases:
                    new_phases.append(0)
                if "B" in phases:
                    new_phases.append(1)
                if "C" in phases:
                    new_phases.append(2)

                bus_phase = {bus: new_phases}

                gamma = {}
                for bus, phases in bus_phase.items():
                    gamma[bus] = [1 / len(phases) for i in phases]

                static_data[name] = {
                    "type": type,
                    "bus_phase": bus_phase,
                    "pmin": pmin,
                    "smax": smax,
                    "qmin": qmin,
                    "qmax": qmax,
                    "gamma": gamma,
                }

        return static_data

    def get_static_data_BATTs(self, BATT_list, BATT_missing_infos):
        static_data = {}

        type = "BAT"

        for BATT in BATT_list:
            name = BATT
            mRID = BATT
            eff_c = 0
            eff_d = 0
            emax = 0
            E_0 = 0
            pmax = 0
            for house_nr in BATT_list[BATT]:
                house = BATT_list[BATT][house_nr]
                eff_c += house["Charge Efficiency (-)"]
                eff_d += house["Discharge Efficiency (-)"]
                pmax += house["Power Capacity (kW)"]
                emax += house["Energy Capacity (kWh)"]
                E_0 += house["State of Charge (-)"] * house["Energy Capacity (kWh)"]

            eff_c /= len(BATT_list[BATT])
            eff_d /= len(BATT_list[BATT])
            pmax *= 1000  # convert to W
            emax *= 1000  # convert to Wh
            E_0 *= 1000  # convert to Wh

            bus_phase = BATT_missing_infos[BATT]["bus_phase"]
            gamma = BATT_missing_infos[BATT]["gamma"]

            smax = pmax
            qmin = BATT_missing_infos[BATT].get("qmin", None)
            qmax = BATT_missing_infos[BATT].get("qmax", None)

            static_data[mRID] = {
                "type": type,
                "bus_phase": bus_phase,
                "smax": smax,
                "eff_c": eff_c,
                "eff_d": eff_d,
                "emax": emax,
                "gamma": gamma,
                "qmin": qmin,
                "qmax": qmax,
                "E_0": E_0,
            }
        return static_data
