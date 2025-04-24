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
from pathlib import Path
from itertools import chain

from gridappsd import GridAPPSD, DifferenceBuilder, utils
from gridappsd.topics import (
    simulation_input_topic,
    simulation_output_topic,
    service_input_topic,
    service_output_topic,
    application_output_topic,
)

from aggregator import BatteryAggregator, BATTERY_PARAMETERS

# App Name
app_name = "Battery_Aggregator"
__version__ = "0.9"

#############################################################################################
DEFAULT_MESSAGE_PERIOD = 5

default_FRS_topic = "FRS"
default_mrid = "battery_aggregator"
logger_name = "__Main__Aggregator"
# Battery Data
# Aggregates
BAT2 = {
    "n2": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 100,
        "Power Capacity (kW)": 50,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n3": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 80,
        "Power Capacity (kW)": 40,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n4": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 120,
        "Power Capacity (kW)": 60,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
}
BAT4 = {
    "n5": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 100,
        "Power Capacity (kW)": 50,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n6": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 100,
        "Power Capacity (kW)": 50,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n16": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 180,
        "Power Capacity (kW)": 90,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n25": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 100,
        "Power Capacity (kW)": 50,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
}
BAT5 = {
    "n9": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 60,
        "Power Capacity (kW)": 30,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n11": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n31": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n13": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n14": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n15": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n23": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n37": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 40,
        "Power Capacity (kW)": 20,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
}

default_aggregate_models = {"BAT2": BAT2, "BAT4": BAT4, "BAT5": BAT5}

# Utility scale batteries
BAT1 = {
    "n28": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 466.6666666,
        "Power Capacity (kW)": 233.33333333,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n29": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 466.6666666,
        "Power Capacity (kW)": 233.33333333,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n30": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 466.6666666,
        "Power Capacity (kW)": 233.33333333,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
}
BAT3 = {
    "n22": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 140,
        "Power Capacity (kW)": 70,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
    "n38": {
        "State of Charge (-)": 0.5,
        "Energy Capacity (kWh)": 140,
        "Power Capacity (kW)": 70,
        "Charge Efficiency (-)": 0.95,
        "Discharge Efficiency (-)": 0.95,
    },
}

default_utility_models = {"BAT1": BAT1, "BAT3": BAT3}

# Measurement IDs
default_meas_ids_batt_aggregator = {
    "n2": [
        "_3c27cb50-1358-409a-9ba7-8065a6405906",
        "_93db5887-dd22-4d62-bab5-d4dd41bf005c",
    ],
    "n3": [
        "_7db90f03-6685-4c01-bde8-306858dc6cd3",
        "_462a94f9-58f2-4fab-a6d6-3755726f8be4",
    ],
    "n4": [
        "_525fae33-0ba9-47ad-860c-d2619079c272",
        "_4a8036f1-1c03-463c-ae8c-4a8e78ea5220",
    ],
    "n5": [
        "_69927cd1-1356-4978-8265-6d43bd92c6b9",
        "_67d3e9dc-1fe6-4188-88ca-55d1ae846bf7",
    ],
    "n6": [
        "_8736968c-94f5-4a9e-895a-49cc39fa0e7b",
        "_d427c158-9466-450e-ad94-60d87ab382dc",
    ],
    "n16": [
        "_8645abde-ab0e-443a-8f07-a7339d853f19",
        "_05b48fe4-7d72-4331-aa43-42cbe264a8ee",
    ],
    "n25": [
        "_1a4f37f6-45b8-41b6-b2f3-b1f3156e3352",
        "_5ddf1221-99d0-4e2f-80d0-21aa2f9aa3b0",
    ],
    "n9": [
        "_a2967aea-26f1-4a82-990a-793e89d2ea01",
        "_0e9c5eaa-2fa4-46cf-80d3-ee09420adcff",
    ],
    "n11": [
        "_8979e90f-e1b3-4976-9af3-20c6724a6622",
        "_19ddf06f-c99a-47ec-b41c-0c7a838ccd51",
    ],
    "n31": [
        "_d461ff21-8dd2-42a1-9b6c-a082ca00fe7f",
        "_895cea35-a78e-4048-88d9-9cc3023412bb",
    ],
    "n13": [
        "_b38ef1ca-ed9e-4ab9-aaf6-8b2c03612780",
        "_8302314c-b6af-42b7-b660-99c7a7426bc2",
    ],
    "n14": [
        "_002ee27e-5ec5-4c08-9a80-46f171e527a2",
        "_9e43611c-527d-4cb1-bebe-e6a874c34f70",
    ],
    "n15": [
        "_8f16e5e5-e6ba-476c-932e-4ea304830310",
        "_b1c13542-dff5-47d4-bc79-49923d60416b",
    ],
    "n23": [
        "_fc9292cc-7e3b-4512-afaf-2474cd6e01cd",
        "_29c0067e-afe0-40a4-8d24-a2492199a410",
    ],
    "n37": [
        "_1ff4b5fc-dcfc-465f-aac7-8123c13321f9",
        "_73e907d9-a88b-477e-a67a-e985d89d5907",
    ],
    "n28": [
        "_723bb5c1-88cc-4026-8d8a-c6d68d8d8247",
        "_ead6529a-efba-44dd-ad48-9e64e6c2f674",
    ],
    "n29": [
        "_8f3feed6-747e-45ff-9e0e-e3064e1e0168",
        "_f8c609ee-e44e-40ee-bbe7-beb9699bdd1a",
    ],
    "n30": [
        "_21cf8abf-7a29-4989-9ac9-187b7e0e26bb",
        "_dea20e55-9233-4a95-842e-46dc7fc65bd6",
    ],
    "n22": [
        "_cbfac12b-f72a-4085-806c-2e4aa6284c25",
        "_b2643974-286c-419d-a938-c92c54df38b8",
    ],
    "n38": [
        "_e72bc1fe-75be-431e-b606-09a2efbf16be",
        "_adb35064-6fb2-49e8-943b-0b66a8a64453",
    ],
}


_log = logging.getLogger(logger_name)


##############################################################################################
def build_csv_writers(folder, filename):
    _file = os.path.join(folder, filename)
    if os.path.exists(_file):
        os.remove(_file)
    file_handle = open(_file, "a")
    csv_writer = csv.writer(
        file_handle, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    houses_with_battery_cols = [
        "Virtual set point",
        "P_setpoint",
        "SOC",
        "Mode",
        "P Total",
        "Q Total",
        "P Uncontrolled",
    ]
    csv_writer.writerow(["time"] + houses_with_battery_cols)
    return file_handle, csv_writer


def format_agg_data(agg_data):
    return {
        "Battery_" + batt_key: batt_data for batt_key, batt_data in agg_data.items()
    }


class BatterAggregator_Object(object):
    """A simple class that handles publishing forward and reverse differences

    The object should be used as a callback from a GridAPPSD object so that the
    on_message function will get called each time a message from the simulator.  During
    the execution of on_meessage the `CapacitorToggler` object will publish a
    message to the simulation_input_topic with the forward and reverse difference specified.
    """

    def __init__(self, simulation_id, fid_select, **kw_args):
        """Create a ``BatterAggregator`` object

        This object should be used as a subscription callback from a ``GridAPPSD``
        object.

        Note
        ----
        This class does not subscribe only publishes.

        Parameters
        ----------
        simulation_id: str
            The simulation_id to use for publishing to a topic.
        gridappsd_obj: GridAPPSD
            An instatiated object that is connected to the gridappsd message bus
            usually this should be the same object which subscribes, but that
            isn't required.
        capacitor_list: list(str)
            A list of capacitors mrids to turn on/off
        """
        # Re-authenticate with GridAPPS-D (the java security token doesn't get inherited well)
        os.environ["GRIDAPPSD_APPLICATION_ID"] = logger_name
        os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

        try:
            # Connect to GridAPPS-D Platform
            gapps = GridAPPSD(simulation_id=simulation_id)
            if not gapps.connected:
                raise ConnectionError("Failed to connect to GridAPPS-D")
        except Exception as e:
            self.logger.error(e)
            gapps = None

        self._gapps = gapps
        self._simulation_id = simulation_id

        self._simout_topic = simulation_output_topic(simulation_id)
        self._ochre_topic = "/topic/goss.gridappsd.simulation.ochre.output." + str(
            simulation_id
        )
        FRS_topic = kw_args.get("FRS_topic", default_FRS_topic)
        self._frs_control_topic = application_output_topic(
            f"{FRS_topic}_RT_Control", None
        )
        self.mrid = kw_args.get("mrid", default_mrid)
        self._aggregator_topic = application_output_topic(self.mrid, None)
        self._publish_to_topic = simulation_input_topic(simulation_id)
        self._automation_topic = kw_args.get(
            "automation_topic", service_input_topic("automation", simulation_id)
        )

        _log.warning(
            f"subscribing to (internal):\n {self._simout_topic}\n {self._ochre_topic}\n {self._frs_control_topic}"
        )
        _log.warning(f"publishing to:\n {self._publish_to_topic}")

        # Aggregator Parameters
        self._time_res = dt.timedelta(minutes=5)

        self.aggregate_models = kw_args.get(
            "aggregate_models", default_aggregate_models
        )
        _log.debug(f"Aggregator models:\n{self.aggregate_models}")
        # Instantiating NREL Aggregator
        self.aggregates = {
            key: BatteryAggregator(self._time_res, format_agg_data(aggregate))
            for key, aggregate in self.aggregate_models.items()
        }
        _log.debug(
            f"Aggregates:\n{[aggregator.models for _, aggregator in self.aggregates.items()]}"
        )

        self.utility_models = kw_args.get("utility_models", default_utility_models)
        _log.debug(f"Utility models:\n{self.utility_models}")

        # Instantiating NREL Aggregator
        self.utility_batts = {
            key: BatteryAggregator(self._time_res, format_agg_data(utility_batt))
            for key, utility_batt in self.utility_models.items()
        }
        _log.debug(
            f"Utility Batteries:\n{[aggregator.models for _, aggregator in self.utility_batts.items()]}"
        )

        meas_ids_batt_aggregator = kw_args.get(
            "meas_IDs", default_meas_ids_batt_aggregator
        )
        _log.debug(
            f"Measurement IDs for battery aggregator:\n{meas_ids_batt_aggregator}"
        )
        self._house_dict = {}

        self._houses_with_battery = {
            house: meas_ids_batt_aggregator.get(f"{house}", [])
            for house in chain(
                chain.from_iterable(self.aggregate_models.values()),
                chain.from_iterable(self.utility_models.values()),
            )
        }
        self._num_houses = len(self._houses_with_battery)
        _log.debug(
            f"Number of houses managed by Battery Aggregator: {self._num_houses}"
        )
        # initializing all aggreagtors to 0
        self._agg_setpoints = {
            resourceID: 0
            for resourceID in chain.from_iterable(
                [self.aggregate_models, self.utility_models]
            )
        }
        self._batt_setpoints = {}

        # for house in self._houses_with_battery:
        #     house_name = 'House_' + house
        #     csvfile, writer = build_csv_writers('./logs', house_name + '.csv')

        #     self._house_output_dict[house_name] = {'csvfile': csvfile, 'writer': writer}

        self._error_code = False
        self.current_time = 0
        self._message_count = 0
        # self._is_initialized = True
        _log.info("Battery Aggregator Initialized")
        self.send_gapps_message(self._automation_topic, {"command": "stop_task"})

        # self._battery_names = []
        # self.vset=[]
        # self._batter_agg_csvfile, self._batter_agg_writer =None,None
        # self._do_aggregation = False

        # self.set_dict={}
        # self.vir_dict={}

    def running(self):
        # Check if any error code
        running = not bool(self._error_code)
        return running

    def error(self):
        return self._error_code

    def send_gapps_message(self, topic, message):
        out_message = {}
        try:
            for key in message.keys():
                out_message[key] = message[key]
        except:
            _log.info("Message is not a dictionary")
            out_message["message"] = message
        finally:
            try:
                if self._gapps is None:
                    raise Exception(" No GAPPS: Cannot send message")
                self._gapps.send(topic, json.dumps(out_message))
                _log.info(f"Sent message to topic {topic} at {dt.datetime.now()}")
                _log.debug(f"Message Content: {out_message}")
                return True
            except Exception as e:
                _log.error(
                    f"Failed to send message to topic {topic} at {dt.datetime.now()}"
                )
                _log.error(e)
                return False

    def send_house_command(self, house_name="House_n10", battery_p_setpoint=2.0):

        ochre_diff = DifferenceBuilder(self._simulation_id)
        forward = {"Battery": {"P Setpoint": battery_p_setpoint}}
        ochre_diff.add_difference(house_name, "Ochre.command", json.dumps(forward), "")
        msg = ochre_diff.get_message()
        _log.debug(f"Sending Ochre command to {house_name}:\n{msg}")
        self._gapps.send(self._publish_to_topic, json.dumps(msg))

    def send_aggregator_measurement(self, measurement_dict, **kw_args):
        _log.info("Sending aggregator measurements")
        message = []
        for aggregate_mRID, measurements in measurement_dict.items():
            for meas_name, meas_value in measurements.items():
                _log.debug(f"Aggregate {aggregate_mRID} has {meas_name} {meas_value}")
                message.append(
                    {
                        "measurement_mrid": f"{aggregate_mRID}_{meas_name}",
                        "value": meas_value,
                    }
                )
            pass

        timestamp = kw_args.get("timestamp", dt.datetime.now().timestamp())

        data = {}
        data["timestamp"] = int(timestamp)
        data["datatype"] = "battery_aggregator_measurements"
        data["simulation_id"] = self._simulation_id
        data["data_timestamp"] = int(timestamp)
        data["message"] = message
        data["tags"] = ["measurement_mrid", "simulation_id"]

        _log.debug(f"Measurement Payload:\n{data}")
        self._gapps.send(self._aggregator_topic, data)

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

        _log.debug(f"Received message on topic: {headers['destination']}")
        message_timestamp = int(headers["timestamp"]) / 1000

        try:
            # RT controller message
            if self._frs_control_topic in headers["destination"]:
                _log.info(f"Processing message from RT controller")
                _log.debug(f"The RT control message is:\n{message}")

                for data_dict in message["message"]:
                    # Process the message
                    resourceID = data_dict["equipment_mrid"]
                    setpoint = -float(data_dict["p_set"])
                    if resourceID in chain.from_iterable(
                        [self.aggregate_models, self.utility_models]
                    ):
                        self._agg_setpoints.update({resourceID: setpoint})
                        _log.debug(f"Updated aggregator {resourceID} to {setpoint} kW")

            elif self._simout_topic in headers["destination"]:
                _log.info(f"Processing message from simulation output")
                ########################################################################################################################
                # Process the message
                measurements = message["message"]["measurements"]
                self.current_time = message["message"]["timestamp"]

                # Get the measurements for Home Batteries & Utility Batteries
                for resourceID, aggregate in chain.from_iterable(
                    [self.aggregate_models.items(), self.utility_models.items()]
                ):

                    _log.debug(f"Processing aggregate {resourceID}")
                    aggregate_power = 0
                    for house_nr in aggregate.keys():
                        try:
                            house_power = sum(
                                np.real(
                                    measurements[meas_ID]["magnitude"]
                                    * np.exp(
                                        1j * np.deg2rad(measurements[meas_ID]["angle"])
                                    )
                                )
                                for meas_ID in measurements
                                if meas_ID in self._houses_with_battery[house_nr]
                            )
                            # Computing power in kW
                            house_power = house_power / 1000
                            aggregate_power += house_power
                            _log.debug(
                                f"House {house_nr} has power {house_power:.3f} kW"
                            )
                        except Exception as e:
                            _log.error(e)
                            _log.error(f"Could not get power for house {house_nr}")

                    _log.info(
                        f"Aggregate {resourceID} has power {aggregate_power:.3f} kW"
                    )

            elif self._ochre_topic in headers["destination"]:
                _log.info(f"Processing message from Ochre")
                _log.debug(f"The message is:\n{message}")

                for house_nr in self._houses_with_battery.keys():
                    house = "House_" + house_nr
                    if house in message.keys():
                        self._message_count += 1
                        _log.info(f"new message count is: {self._message_count}")
                # --------------------------------------------------------------------------------------------
                # Process the Aggregates
                for aggregate in self.aggregate_models.values():
                    for house_nr in aggregate.keys():
                        battery = "Battery_" + house_nr
                        if battery in message.keys():
                            aggregate[house_nr]["State of Charge (-)"] = message[
                                battery
                            ]["SOC"]

                        house = "House_" + house_nr
                        if house in message.keys():
                            values = [
                                message[battery]["SOC"],
                                message[battery]["Mode"],
                                message[house]["P Total"],
                                message[house]["Q Total"],
                                message[house]["P Uncontrolled"],
                            ]
                            self._house_dict[house] = values
                            _log.debug(f"{house} in aggregator values: {values}")
                        # else:
                        #    _log.debug(f'{house} not contained in houses in message, {message.keys()}')

                # Process the Aggregates
                for aggregate in self.utility_models.values():
                    for house_nr in aggregate.keys():
                        battery = "Battery_" + house_nr
                        if battery in message.keys():
                            aggregate[house_nr]["State of Charge (-)"] = message[
                                battery
                            ]["SOC"]

                        house = "House_" + house_nr
                        if house in message.keys():
                            values = [
                                message[battery]["SOC"],
                                message[battery]["Mode"],
                                message[house]["P Total"],
                                message[house]["Q Total"],
                                message[house]["P Uncontrolled"],
                            ]
                            self._house_dict[house] = values
                            _log.debug(f"{house} in utility Battery values: {values}")
                #                        else:
                #                            _log.debug(f'{house} not contained in houses in message, {message.keys()}')

                if (
                    self._message_count % self._num_houses == 0
                    and self._message_count > 0
                ):
                    # All Houses are up to date:
                    _agg_Data = {}
                    # let's process the aggregates
                    for key, aggregator in self.aggregates.items():
                        aggregate_model = self.aggregate_models[key]
                        # ------------ Send parameters to aggregator --------------------------------------
                        aggregator.update_models(**format_agg_data(aggregate_model))
                        # ------------ Aggregate model to create virtual battery model---------------------
                        virtual_model = aggregator.aggregate()

                        # read aggregate SOC (Code from Venkat at NREL)
                        # _agg_Data.update({key: {'SOC':virtual_model['State of Charge (-)']*virtual_model['Energy Capacity (kWh)']*1000}})
                        SOC_kwh = sum(
                            [
                                self._house_dict["House_" + imp][0]
                                * aggregate_model[imp]["Energy Capacity (kWh)"]
                                for imp in aggregate_model.keys()
                                if "House_" + imp in self._house_dict.keys()
                            ]
                        )
                        _agg_Data.update({key: {"SOC": SOC_kwh * 1000}})
                        Capacity_kwh = sum(
                            [
                                aggregate_model[imp]["Energy Capacity (kWh)"]
                                for imp in aggregate_model.keys()
                                if "House_" + imp in self._house_dict.keys()
                            ]
                        )
                        _log.debug(
                            f"Aggregate SOC for {key}: {SOC_kwh/Capacity_kwh}, Energy in Wh: {SOC_kwh*1000}"
                        )

                        # ------------ Dispatch setpoint to individual batteries --------------------------
                        try:
                            battery_setpoints = aggregator.dispatch(
                                p_setpoint=self._agg_setpoints[key]
                            )
                            _log.debug(
                                f"Individual battery setpoints from aggregator: {battery_setpoints}"
                            )
                            self._batt_setpoints.update(battery_setpoints)
                        except Exception as e:
                            _log.error(f"Error in dispatching {key} setpoints:\n{e}")
                    # let's process the utility batts
                    for key, aggregator in self.utility_batts.items():
                        aggregate_model = self.utility_models[key]
                        # ------------ Send parameters to aggregator --------------------------------------
                        aggregator.update_models(**format_agg_data(aggregate_model))
                        # ------------ Aggregate model to create virtual battery model---------------------
                        virtual_model = aggregator.aggregate()

                        # read aggregate SOC (Code from Venkat at NREL)
                        # _agg_Data.update({key: {'SOC':virtual_model['State of Charge (-)']*virtual_model['Energy Capacity (kWh)']*1000}})
                        SOC_kwh_u = sum(
                            [
                                self._house_dict["House_" + imp][0]
                                * aggregate_model[imp]["Energy Capacity (kWh)"]
                                for imp in aggregate_model.keys()
                                if "House_" + imp in self._house_dict.keys()
                            ]
                        )
                        _agg_Data.update({key: {"SOC": SOC_kwh_u * 1000}})
                        Capacity_kwh_u = sum(
                            [
                                aggregate_model[imp]["Energy Capacity (kWh)"]
                                for imp in aggregate_model.keys()
                                if "House_" + imp in self._house_dict.keys()
                            ]
                        )
                        _log.debug(
                            f"Aggregate SOC for {key}: {SOC_kwh_u/Capacity_kwh_u}, Energy in Wh: {SOC_kwh_u*1000}"
                        )

                        # ------------ Dispatch setpoint to individual batteries --------------------------
                        try:
                            battery_setpoints = aggregator.dispatch(
                                p_setpoint=self._agg_setpoints[key]
                            )
                            _log.debug(
                                f"Individual battery setpoints from aggregator: {battery_setpoints}"
                            )
                            self._batt_setpoints.update(battery_setpoints)
                        except Exception as e:
                            _log.error(f"Error in dispatching {key} setpoints:\n{e}")

                    # Dispatch the individual bateries
                    for battery, setpoint in self._batt_setpoints.items():
                        house_name = "House_" + battery.split("_")[1]
                        self.send_house_command(house_name, setpoint)
                        _log.debug(f"Setpoints for battery {house_name}: {setpoint}")

                    self.send_aggregator_measurement(
                        _agg_Data, timestamp=self.current_time
                    )

                    # -------------------------------------------------------
                    _log.warning(
                        "Start Aggregation_____________________________________________"
                    )
                    _log.debug(f"message count {self._message_count}")
                    _log.info(f"Output results\n {self._house_dict}")
                    _log.info(f"Virtual setpoint\n {self._agg_setpoints}")
                    _log.info(f"Battery setpoints\n {self._batt_setpoints}")
                    _log.info(
                        "End Aggregation_____________________________________________"
                    )
                    # -------------------------------------------------------------

                    # -------------------------- Data storege to CSV file

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            _log.error(f"Error: {e}, {exc_type}, {exc_obj}, {exc_tb.tb_lineno}")
            self._error_code = True
            raise


########################## To get node voltages
def get_meas_mrid(gapps, model_mrid, topic):

    message = {
        "modelId": model_mrid,
        "requestType": "QUERY_OBJECT_MEASUREMENTS",
        "resultFormat": "JSON",
        "objectType": "ACLineSegment",
    }
    obj_msr_ACline = gapps.get_response(topic, message, timeout=180)
    obj_msr_ACline = obj_msr_ACline["data"]
    obj_msr_ACline = [measid for measid in obj_msr_ACline if measid["type"] == "PNV"]

    message = {
        "modelId": model_mrid,
        "requestType": "QUERY_OBJECT_MEASUREMENTS",
        "resultFormat": "JSON",
        "objectType": "LoadBreakSwitch",
    }
    obj_msr_loadsw = gapps.get_response(topic, message, timeout=180)
    # print(obj_msr_loadsw)
    # print(sh)

    obj_msr_loadsw = obj_msr_loadsw["data"]

    return obj_msr_ACline, obj_msr_loadsw


########################### Main Program


def _main():
    _log.debug("Starting application")

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

    _log.setLevel(log_level)

    ############## Individual Logger
    log_filename = Path(path_to_logs) / "_log_BatteryAggregator.log"
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

    _log.warning(
        "BatteryAggregator Starting!!!-------------------------------------------"
    )

    simulation_id = opts.simulation_id
    model_id = sim_request["power_system_config"]["Line_name"]
    _log.debug(f"Model mrid is: {model_id}")

    gapps = GridAPPSD(simulation_id=simulation_id)
    if gapps.connected:
        _log.debug(f"GridAPPSD connected to simulation {simulation_id}")
    else:
        _log.error("GridAPPSD not Connected")

    battery_agg = BatterAggregator_Object(simulation_id, model_id, **app_config)

    # Subscribing to Simulation Output
    sim_out_topic = simulation_output_topic(simulation_id)
    gapps.subscribe(sim_out_topic, battery_agg)

    # Subscribing to Ochre Output
    ochre_output_topic = "/topic/goss.gridappsd.simulation.ochre.output." + str(
        simulation_id
    )
    gapps.subscribe(ochre_output_topic, battery_agg)

    # Subscribing to FRS Output
    FRS_topic = app_config.get("FRS_topic", default_FRS_topic)
    rt_topic = application_output_topic(f"{FRS_topic}_RT_Control", None)
    gapps.subscribe(rt_topic, battery_agg)

    _log.warning(
        f"subscribing to (main):\n {sim_out_topic}\n {ochre_output_topic}\n {rt_topic}"
    )

    sim_time = sim_time = app_config.get("sim_time", -1)
    if sim_time == -1:
        _log.info(f"Info received from remote: sim_time - until termination")
    else:
        _log.info(f"Info received from remote: sim_time {sim_time} seconds")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:

        if not battery_agg.running():
            if battery_agg.error() == 2:
                _log.warning("Battery Aggregator Terminated")
            else:
                _log.error("Battery Aggregator failed")
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)

    _log.warning(
        "BatteryAggregator finished!!!-------------------------------------------------------"
    )


if __name__ == "__main__":
    _main()
