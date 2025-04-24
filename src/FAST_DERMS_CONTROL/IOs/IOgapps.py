"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from gridappsd import GridAPPSD, utils
from gridappsd.timeseries import Query
from gridappsd.topics import (
    simulation_input_topic,
    simulation_output_topic,
    simulation_log_topic,
    field_output_topic,
    service_output_topic,
    service_input_topic,
    application_output_topic,
    application_input_topic,
    REQUEST_POWERGRID_DATA,
)

from ..common import *
from ..Modeling import equipmentClasses as mod
from .IOclass import IOmodule

from typing import Dict, List, Tuple, Union

import datetime as dt
import json
import pickle
import os
import tempfile
import operator
import re

fake_simulation_id = "NO_GAPPS"
test_bed_simulation_id = "12345"


class IOgapps(IOmodule):
    def __init__(self, simulation_id, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        # Re-authenticate with GridAPPS-D (the java security token doesn't get inherited well)
        os.environ["GRIDAPPSD_APPLICATION_ID"] = "FRS"
        os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
        os.environ["GRIDAPPSD_USER"] = "app_user"
        os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

        try:
            # Connect to GridAPPS-D Platform
            if simulation_id == fake_simulation_id:
                raise Exception("GridAPPS-D is not used in this simulation")

            gapps = GridAPPSD(simulation_id=simulation_id)
            if not gapps.connected:
                raise ConnectionError("Failed to connect to GridAPPS-D")

        except Exception as e:
            self.logger.error(e)
            gapps = None

        self._gapps = gapps

        self._simulation_id = simulation_id

        self.results = {}
        self.logger.warning("IOgapps Initialized")

    def get_sim_id(self) -> str:
        return self._simulation_id

    def simulation_topic(self, topic_type: str) -> str:
        if topic_type == "input":
            return simulation_input_topic(self._simulation_id)
        elif topic_type == "output":
            if self._simulation_id == test_bed_simulation_id:
                return field_output_topic(self._simulation_id)
            else:
                return simulation_output_topic(self._simulation_id)
        elif topic_type == "log":
            return simulation_log_topic(self._simulation_id)
        else:
            self.logger.error(f"Topic type {topic_type} not recognized")
            return None

    def service_topic(self, service_id: str, topic_type: str, **kw_args) -> str:
        sub_id = kw_args.get("sub_id", None)
        if sub_id is not None:
            service_id = f"{service_id}_{sub_id}"

        if topic_type == "input":
            return service_input_topic(service_id, self._simulation_id)
        elif topic_type == "output":
            # Currently not implemented ?
            # return service_output_topic(service_id, self._simulation_id, *args)
            return service_output_topic(service_id, self._simulation_id)
        else:
            self.logger.error(f"Topic type {topic_type} not recognized")
            return None

    def application_topic(self, app_id: str, topic_type: str, **kw_args) -> str:
        sub_id = kw_args.get("sub_id", None)
        if sub_id is not None:
            app_id = f"{app_id}_{sub_id}"

        if topic_type == "input":
            return application_input_topic(app_id, None)
        elif topic_type == "output":
            # Store application's data
            return application_output_topic(app_id, None)
        else:
            self.logger.error(f"Topic type {topic_type} not recognized")
            return None

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
                if self._simulation_id == fake_simulation_id:
                    raise Fake_GAPPs_Exception("Returning message")
                if self._gapps is None:
                    raise No_GAPPs_Exception("Cannot send message")
                self._gapps.send(topic, json.dumps(out_message))
                self.logger.info(
                    f"Sent message to topic {topic} at {dt.datetime.now()}"
                )
                self.logger.debug(f"Message Content: {out_message}")
                return True
            except Fake_GAPPs_Exception as e:
                self.logger.warning(e)
                return out_message
            except Exception as e:
                self.logger.error(
                    f"Failed to send message to topic {topic} at {dt.datetime.now()}"
                )
                self.logger.error(e)
                return False

    def fetch_model(self, model_id: str) -> Dict:
        self.logger.warning(f"Fetching model {model_id}")

        # Fetch Model from GridApps
        message = {
            "requestType": "QUERY_MODEL",
            "modelId": model_id,
            "resultFormat": "XML",
        }

        topic = REQUEST_POWERGRID_DATA
        if self._gapps is None:
            raise No_GAPPs_Exception("Cannot fetch model")
        response = self._gapps.get_response(topic, message, timeout=20)
        with tempfile.TemporaryFile() as fp:
            fp.write(response.encode("utf-8"))
            fp.seek(0)
            if self.logger.isEnabledFor(logging.INFO):
                try:
                    path_to_archive = self.path_to_archive / "gridapps_model.xml"
                    with open(path_to_archive.absolute(), "w") as file:
                        file.write(response)
                except:
                    self.logger.error("Failed to save model to archive")

            return super().load_model(fp)

    def query_all_nodes_gapps(self, model_mRID, timeout=180) -> List[Dict]:
        try:
            QueryNodeMessage = (
                """
            PREFIX r:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX c:  <http://iec.ch/TC57/CIM100#>
            SELECT DISTINCT ?busname ?cnid ?tpnid (group_concat(distinct ?nomu;separator="") as ?nomv ) WHERE {
            SELECT ?busname ?cnid ?tpnid ?nomu WHERE {
            VALUES ?fdrid {"%s"}
            ?fdr c:IdentifiedObject.mRID ?fdrid.
            ?bus c:ConnectivityNode.ConnectivityNodeContainer ?fdr.
            ?bus c:ConnectivityNode.TopologicalNode ?tp.
            ?bus r:type c:ConnectivityNode.
            ?bus c:IdentifiedObject.name ?busname.
            ?bus c:IdentifiedObject.mRID ?cnid.
            ?fdr c:IdentifiedObject.name ?feeder.
            ?trm c:Terminal.ConnectivityNode ?bus.
            ?trm c:Terminal.ConductingEquipment ?ce.
            
            OPTIONAL {
            ?ce  c:ConductingEquipment.BaseVoltage ?bv.
            ?bv  c:BaseVoltage.nominalVoltage ?nomu.
            }
            bind(strafter(str(?tp), str("http://localhost:8889/bigdata/namespace/kb/sparql#")) as ?tpnid)
            } ORDER by ?busname
            } 
            GROUP by ?busname ?cnid ?tpnid 
            ORDER by ?busname
            """
                % model_mRID
            )

            results = self._gapps.query_data(query=QueryNodeMessage, timeout=timeout)
            NodeQuery = results["data"]["results"]["bindings"]
        except Exception as e:
            NodeQuery = None
            raise
        finally:
            return NodeQuery

    def query_fcast_gapps(
        self, mRID, t0, n_timestep, timestep_period, **kw_args
    ) -> List[Dict]:

        try:
            start = int(t0.timestamp())
            end = int(
                (t0 + dt.timedelta(minutes=n_timestep * timestep_period)).timestamp()
            )

            f = (
                Query("forecasts")
                .select()
                .where_key("data_timestamp")
                .between(start, end)
                .where_key("equipment_mrid")
                .eq(mRID)
            )

            if self._gapps is None:
                raise No_GAPPs_Exception("Cannot query forecast")
            response = f.execute(self._gapps)
            if "error" in response.keys():
                raise Bad_Data_Exception(f"Error querying gApps for {mRID}")
            response = response["data"]
            if not response:
                raise Bad_Data_Exception(f"No data for {mRID} on GridAPPS-D")
            if isinstance(response, str):
                response = eval(response)
            response.sort(key=operator.itemgetter("data_timestamp"))
            # Formatting response into a DataFrame to facilitate processing
            # set index to 1: data_timestamp and 2: upload time
            # group_by level 1 and tail(1) to select the most recent data for each data_timestamp
            response = (
                pd.DataFrame(response)
                .set_index(["data_timestamp", "time"])
                .sort_index(level=["data_timestamp", "time"])
                .fillna(method="ffill")
                .groupby(level="data_timestamp")
                .tail(1)
                .reset_index()
                .to_dict("records")
            )

            return response
        except Exception as e:
            raise

    def query_24h_fcast_gapps(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            # Computing new start time and n_timestep to lookup full 24h of fcast
            # Starting at Midnight
            new_t0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)
            new_n_timesteps = 24 * int(60 / timestep_period)

            return self.query_fcast_gapps(
                mRID, new_t0, new_n_timesteps, timestep_period, **kw_args
            )
        except Exception as e:
            raise

    def query_measurement_gapps(
        self, mRID, t0, n_timestep, timestep_period, **kw_args
    ) -> List[Dict]:

        try:
            start = int(t0.timestamp())
            end = int(
                (t0 + dt.timedelta(minutes=n_timestep * timestep_period)).timestamp()
            )

            datatype = kw_args.get("datatype", "frs_measurement")

            f = (
                Query(datatype)
                .select()
                .where_key("data_timestamp")
                .between(start, end)
                .where_key("measurement_mrid")
                .eq(mRID)
                .where_key("simulation_id")
                .eq(str(self._simulation_id))
                .last(n_timestep)
            )
            if self._gapps is None:
                raise No_GAPPs_Exception("Cannot query measurement")
            response = f.execute(self._gapps)
            if "error" in response.keys():
                raise Bad_Data_Exception(f"Error querying gApps for {mRID}")
            response = response["data"]
            if not response:
                raise Bad_Data_Exception(f"No data for {mRID} on GridAPPS-D")
            if isinstance(response, str):
                response = eval(response)
            response.sort(key=operator.itemgetter("data_timestamp"))
            return response

        except Exception as e:
            raise

    def publish_measurements_gapps(self, measurements, **kw_args) -> None:

        timestamp = kw_args.get("timestamp", dt.datetime.now().timestamp())
        datatype = kw_args.get("datatype", "frs_data")
        topic = kw_args.get("topic", self.simulation_topic("input"))

        messages = []
        for measurement_mRID, measurement_value in measurements.items():
            messages.append(
                {"measurement_mrid": measurement_mRID, "value": measurement_value}
            )

        payload = {}
        payload["timestamp"] = timestamp
        payload["datatype"] = datatype
        payload["simulation_id"] = self._simulation_id
        payload["data_timestamp"] = timestamp
        payload["message"] = messages
        payload["tags"] = ["measurement_mrid", "simulation_id", "data_timestamp"]

        self.send_gapps_message(topic, payload)

    def publish_multiple_measurements_gapps(
        self, measurements, t0, n_timestep, timestep_period, **kw_args
    ) -> None:

        timestamp = kw_args.get("timestamp", dt.datetime.now().timestamp())
        datatype = kw_args.get("datatype", "frs_data")
        topic = kw_args.get("topic", self.simulation_topic("input"))

        messages = []
        for i in range(n_timestep):
            data_timestamp = (
                t0 + dt.timedelta(minutes=i * timestep_period)
            ).timestamp()

            for measurement_mRID, measurement_values in measurements.items():
                message = {}
                message["measurement_mrid"] = measurement_mRID
                message["data_timestamp"] = data_timestamp
                message["value"] = measurement_values[i]

                messages.append(message)

        payload = {}
        payload["timestamp"] = timestamp
        payload["datatype"] = datatype
        payload["simulation_id"] = self._simulation_id
        payload["message"] = message
        payload["tags"] = ["measurement_mrid", "simulation_id", "data_timestamp"]

        self.send_gapps_message(topic, payload)

    def publish_fcasts_gapps(
        self, mRID, fcasts, t0, n_timestep, timestep_period, **kw_args
    ) -> None:
        """
        Create a forecast message for the Gapps API.

        Parameters
        ----------
        t0 : datetime
            Timestamp of the start of the forecast (next day at midnight)
        n_timestep : int
            Number of timesteps in the forecast
        timestep_period : int
            Timestep period in seconds
        """

        timestamp = int(kw_args.get("timestamp", dt.datetime.now().timestamp()))
        topic = kw_args.get("topic", self.simulation_topic("input"))

        payload = {}
        payload["timestamp"] = timestamp
        payload["datatype"] = "forecasts"
        payload["equipment_mrid"] = mRID
        payload["message"] = []
        payload["tags"] = ["data_timestamp", "equipment_mrid"]

        # Check for single data point
        if n_timestep == 1:
            for fcast_name, fcast_values in fcasts.items():
                try:
                    len(fcast_values)
                except:
                    fcasts[fcast_name] = [fcast_values]
        else:
            if timestep_period == 0:
                raise ValueError(
                    "timestep_period must be specified for multiple timesteps"
                )

        for i in range(n_timestep):
            data_timestamp = int(
                (t0 + dt.timedelta(minutes=i * timestep_period)).timestamp()
            )

            message = {}
            message["data_timestamp"] = data_timestamp
            message.update(
                {
                    fcast_name: self.np_encoder(fcast_values[i])
                    for fcast_name, fcast_values in fcasts.items()
                    if not np.isnan(fcast_values[i])
                }
            )
            payload["message"].append(message)

        self.send_gapps_message(topic, payload)

    def get_substation_ID(self):
        raise Implementation_Exception(
            "get_substation_ID - gApps implementation missing"
        )

    def build_DERs(self, **kw_args) -> List[mod.DER]:
        raise Implementation_Exception("build_DERs - gApps implementation missing")

    def build_Loads(self) -> List[mod.Load]:
        raise Implementation_Exception("build_Loads - gApps implementation missing")

    def build_Composite_Resources(self) -> List[mod.CompositeResource]:
        raise Implementation_Exception(
            "build_Composite_Resources - gApps implementation missing"
        )

    def get_switch_status(self, mRID) -> bool:
        raise Implementation_Exception(
            "get_switch_status - gApps implementation missing"
        )

    def get_battery_SOC(self, mRID, **kw_args) -> float:
        # Get options out of kw_args
        t0 = kw_args.get("t0", dt.datetime.now())
        n_timestep = kw_args.get("n_timestep", 1)
        timestep_period = kw_args.get("timestep_period", 60)
        type = kw_args.get("type", "")
        Sbase = kw_args.get("Sbase", None)

        if type == "init":
            raise Implementation_Exception("init_SOC - gApps implementation missing")
        # Measurement Name on gapps
        meassurement_mrid = f"{mRID}_SOC"
        try:
            query_response = self.query_measurement_gapps(
                meassurement_mrid,
                t0,
                n_timestep,
                timestep_period,
                datatype="battery_aggregator_measurements",
            )
            self.logger.debug(f"query_response: {query_response}")
            # The value sent should be in Wh, so that it may be correctly perunitized as SBase is in W.
            battery_energy = [
                self.per_unitize(data["value"], base=Sbase) for data in query_response
            ]
            return battery_energy

        except:
            raise

    def get_fcast_substation(
        self, mRID, t0, n_timestep, timestep_period, **kw_args
    ) -> Dict:
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)
        Vbase = kw_args.get("Vbase", None)

        fcast = {}
        query_response = self.query_24h_fcast_gapps(
            mRID, t0, n_timestep, timestep_period
        )

        timestamps = np.array([data["data_timestamp"] for data in query_response])
        fcast["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(self.local_tz)

        timestamps = timestamps - timestamps[0]
        fcast["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # Format the ADMS data
        try:
            # ADMS data published in V, so that it may be correctly perunitized as Vbase is in V.
            fcast["V_set"] = self.per_unitize(
                [data["ADMS_V_set"] for data in query_response], base=Vbase
            )
        except:
            self.logger.warning(
                "ADMS_V_set not found in gApps, let's try ADMS_V_set_pu"
            )
            # ADMS data published in pu, no need to perunitize
            fcast["V_set"] = [data["ADMS_V_set_pu"] for data in query_response]

        # ADMS data published in W, so that it may be correctly perunitized as SBase is in W.
        fcast["P0_up_lim"] = self.per_unitize(
            [data["ADMS_P0_up_limit"] for data in query_response], base=Sbase
        )
        fcast["P0_dn_lim"] = self.per_unitize(
            [data["ADMS_P0_dn_limit"] for data in query_response], base=Sbase
        )

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def get_fcast_switch_status(self, mRID, t0, n_timestep, timestep_period) -> bool:
        raise Implementation_Exception(
            "get_fcast_switch_status - gApps implementation missing"
        )

    def get_fcast_load(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)

        fcast = {}
        query_response = self.query_24h_fcast_gapps(
            mRID, t0, n_timestep, timestep_period
        )

        timestamps = np.array([data["data_timestamp"] for data in query_response])
        fcast["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(self.local_tz)

        timestamps = timestamps - timestamps[0]
        fcast["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # Format the load data
        x_bar = self.per_unitize(
            [
                [data["PL_a_xbar"], data["PL_b_xbar"], data["PL_c_xbar"]]
                for data in query_response
            ],
            base=Sbase,
        )
        sigma = self.per_unitize(
            [
                [data["PL_a_sigma"], data["PL_b_sigma"], data["PL_c_sigma"]]
                for data in query_response
            ],
            base=Sbase,
        )

        fcast["PL"] = {"xbar": x_bar, "sigma": sigma}

        x_bar = self.per_unitize(
            [
                [data["QL_a_xbar"], data["QL_b_xbar"], data["QL_c_xbar"]]
                for data in query_response
            ],
            base=Sbase,
        )
        sigma = self.per_unitize(
            [
                [data["QL_a_sigma"], data["QL_b_sigma"], data["QL_c_sigma"]]
                for data in query_response
            ],
            base=Sbase,
        )

        fcast["QL"] = {"xbar": x_bar, "sigma": sigma}

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def get_fcast_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)

        fcast = {}
        query_response = self.query_24h_fcast_gapps(
            mRID, t0, n_timestep, timestep_period
        )

        timestamps = np.array([data["data_timestamp"] for data in query_response])
        fcast["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(self.local_tz)

        timestamps = timestamps - timestamps[0]
        fcast["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # Format the fcast data
        generic_fcast = re.compile("(.*)(_xbar|_sigma)\\b")
        flat_bid_fcast = re.compile("(tmm_bids_)(.*)(_seg_)(\d+)")
        fcasts_types = []
        tmm_bids_types = {}
        for key in query_response[0].keys():
            if key not in [
                "datatype",
                "data_timestamp",
                "time",
                "equipment_mrid",
                "data_timestamp_tag",
            ]:
                # generic fcasts
                generic_fcast_test = generic_fcast.match(key)
                if generic_fcast_test:
                    if generic_fcast_test.group(2) == "_xbar":
                        fcasts_types.append(generic_fcast_test.group(1))
                # TMM
                flat_bid_fcast_test = flat_bid_fcast.match(key)
                if flat_bid_fcast_test:
                    tmm_bids_type = flat_bid_fcast_test.group(2)
                    tmm_bids_segments = tmm_bids_types.get(tmm_bids_type, 0) + 1
                    tmm_bids_types.update({tmm_bids_type: tmm_bids_segments})

        # generic fcasts
        for fcast_type in fcasts_types:
            x_bar = [data[fcast_type + "_xbar"] for data in query_response]
            sigma = [data[fcast_type + "_sigma"] for data in query_response]
            if fcast_type in ["Pmin", "Pmax", "Qmin", "Qmax", "Emin", "Emax"]:
                x_bar = self.per_unitize(x_bar, base=Sbase)
                sigma = self.per_unitize(sigma, base=Sbase)

            fcast[fcast_type] = {"xbar": x_bar, "sigma": sigma}

        # TMM bids
        for tmm_bids_type in tmm_bids_types.keys():
            generic_fcast_test = generic_fcast.match(tmm_bids_type)
            if generic_fcast_test:
                if generic_fcast_test.group(2) == "_xbar":
                    fcast_type = generic_fcast_test.group(1)
                    if fcast_type == "TRpower":
                        # TMM sends the power in Load convention, so we need to invert the sign
                        x_bar = [
                            [
                                -data[f"tmm_bids_{fcast_type}_xbar_seg_{segment}"]
                                for segment in range(tmm_bids_types[tmm_bids_type])
                            ]
                            for data in query_response
                        ]
                        sigma = [
                            [
                                data[f"tmm_bids_{fcast_type}_sigma_seg_{segment}"]
                                for segment in range(tmm_bids_types[tmm_bids_type])
                            ]
                            for data in query_response
                        ]
                        x_bar = self.per_unitize(x_bar, base=Sbase)
                        sigma = self.per_unitize(sigma, base=Sbase)
                    else:
                        x_bar = [
                            [
                                data[f"tmm_bids_{fcast_type}_xbar_seg_{segment}"]
                                for segment in range(tmm_bids_types[tmm_bids_type])
                            ]
                            for data in query_response
                        ]
                        sigma = [
                            [
                                data[f"tmm_bids_{fcast_type}_sigma_seg_{segment}"]
                                for segment in range(tmm_bids_types[tmm_bids_type])
                            ]
                            for data in query_response
                        ]
                    fcast[fcast_type] = {"xbar": x_bar, "sigma": sigma}
            else:
                # Store the price of TMM bid as a regular fcast with xbar and sigma
                x_bar = [
                    [
                        data[f"tmm_bids_{tmm_bids_type}_seg_{segment}"]
                        for segment in range(tmm_bids_types[tmm_bids_type])
                    ]
                    for data in query_response
                ]
                sigma = [
                    [0 for segment in range(tmm_bids_types[tmm_bids_type])]
                    for data in query_response
                ]
                fcast[tmm_bids_type] = {"xbar": x_bar, "sigma": sigma}

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def get_fcast_price(self, t0, n_timestep, timestep_period) -> List[float]:
        query_response = self.query_24h_fcast_gapps(
            "price", t0, n_timestep, timestep_period
        )
        fcast = {}
        timestamps = np.array([data["data_timestamp"] for data in query_response])
        fcast["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(self.local_tz)

        timestamps = timestamps - timestamps[0]
        fcast["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # Format price data
        types = ["pi", "pi_rup", "pi_rdn"]
        subtypes = ["xbar", "sigma"]
        for fcast_type in types:
            fcast[fcast_type] = {}
            for subtype in subtypes:
                fcast[fcast_type][subtype] = [
                    data[fcast_type + "_" + subtype] for data in query_response
                ]

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def publish_DA_result(self, scenario_res: Dict) -> None:
        self.results["DA_result"] = scenario_res
        super().publish_DA_result(scenario_res)

    def get_DET_load(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # We get the latest forecast from gApps
        fcasts = self.get_fcast_load(mRID, t0, n_timestep, timestep_period, **kw_args)

        deterministic_fcast = {}
        for key in fcasts.keys():
            if key in ["t0", "timestep_period"]:
                deterministic_fcast[key] = fcasts[key]
            else:
                deterministic_fcast[key] = fcasts[key]["xbar"]
        deterministic_fcast = self.process_fcast(
            deterministic_fcast, t0, n_timestep, timestep_period
        )
        return deterministic_fcast

    def get_DET_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # We get the latest forecast from gApps
        fcasts = self.get_fcast_DER(mRID, t0, n_timestep, timestep_period, **kw_args)
        deterministic_fcast = {}
        for key in fcasts.keys():
            if key in ["t0", "timestep_period"]:
                deterministic_fcast[key] = fcasts[key]
            else:
                deterministic_fcast[key] = fcasts[key]["xbar"]
        deterministic_fcast = self.process_fcast(
            deterministic_fcast, t0, n_timestep, timestep_period
        )
        return deterministic_fcast

    def get_DET_price(self, t0, n_timestep, timestep_period, **kw_args):
        # DET price come from the ISO
        query_response = self.query_24h_fcast_gapps(
            "ISO", t0, n_timestep, timestep_period
        )
        deterministic_fcast = {}
        timestamps = np.array([data["data_timestamp"] for data in query_response])
        deterministic_fcast["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(
            self.local_tz
        )

        timestamps = timestamps - timestamps[0]
        deterministic_fcast["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # Format price data
        deterministic_fcast["pi"] = [data["MCP"] for data in query_response]
        deterministic_fcast["pi_rup"] = [data["MCP_Rup"] for data in query_response]
        deterministic_fcast["pi_rdn"] = [data["MCP_Rdn"] for data in query_response]

        deterministic_fcast = self.process_fcast(
            deterministic_fcast, t0, n_timestep, timestep_period
        )
        return deterministic_fcast

    def get_DET_input_data(self, model_ID, t0, n_timestep, timestep_period, **kw_args):
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)
        # Query Gapss for the latest data (stored as fcasts)
        query_response = self.query_24h_fcast_gapps(
            model_ID, t0, n_timestep, timestep_period
        )
        out_data = {}
        timestamps = np.array([data["data_timestamp"] for data in query_response])
        out_data["t0"] = dt.datetime.fromtimestamp(timestamps[0]).astimezone(
            self.local_tz
        )

        timestamps = timestamps - timestamps[0]
        out_data["timestep_period"] = int(
            timestamps.sum() / (sum(i for i in range(len(timestamps)))) / 60
        )

        # ISO publishes Data in MW, needs to be converted to W and per_unitized
        # Substation Data
        out_data["P0_set"] = self.per_unitize(
            [data["MW_award"] * 1000000 for data in query_response], base=Sbase
        )
        # Up / Dn Reserves
        out_data["Rup_set"] = self.per_unitize(
            [data["Rup_award"] for data in query_response], base=Sbase
        )
        out_data["Rdn_set"] = self.per_unitize(
            [data["Rdn_award"] for data in query_response], base=Sbase
        )

        out_data = self.process_fcast(out_data, t0, n_timestep, timestep_period)
        self.logger.debug(f"DA_result: {out_data}")
        return out_data

    def publish_DET_result(self, deterministic_res: Dict) -> None:
        self.results["DET_result"] = deterministic_res
        super().publish_DET_result(deterministic_res)

    def get_MPC_price(self, t0, n_timestep, timestep_period, **kw_args):
        # For now we query the DET price Data and add a 0 sigma
        DET_price = self.get_DET_price(t0, n_timestep, timestep_period, **kw_args)

        fcast = {}
        fcast["t0"] = DET_price["t0"]
        fcast["timestep_period"] = DET_price["timestep_period"]

        # Format price data
        types = ["pi", "pi_rup", "pi_rdn"]
        subtypes = ["xbar", "sigma"]
        for fcast_type in types:
            fcast[fcast_type] = {}
            fcast[fcast_type]["xbar"] = DET_price[fcast_type]
            fcast[fcast_type]["sigma"] = [0 for _ in range(len(DET_price[fcast_type]))]

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def get_MPC_input_data(self, t0, n_timestep, timestep_period):
        # Substation Data
        # Reserves
        # Battery states
        data_set = self.results["DET_result"]
        fcast_in = {
            key: data_set[key]
            for key in [
                "P0_set",
                "Rup_set",
                "Rdn_set",
                "E_All",
                "t0",
                "timestep_period",
            ]
        }
        fcast_out = self.process_fcast(fcast_in, t0, n_timestep, timestep_period)
        self.logger.debug(f"MPC input Data: {fcast_out}")
        return fcast_out

    def publish_MPC_result(
        self, MPC_data, t0, n_timestep, timestep_period, topic, **kw_args
    ) -> None:
        # Get options out of kw_args
        datatype = kw_args.get("datatype", "MPC_result")

        # Create a new message
        messages = []

        for i in range(n_timestep):
            timestamp = (t0 + dt.timedelta(minutes=i * timestep_period)).timestamp()

            for resourceID in MPC_data.keys():
                message = {}
                message["equipment_mrid"] = resourceID
                message["setpoint_timestamp"] = timestamp
                for key in MPC_data[resourceID].keys():
                    message[key] = MPC_data[resourceID][key][i]
                messages.append(message)

        message = {}
        message["timestamp"] = t0.timestamp()
        message["datatype"] = datatype
        message["message"] = messages

        # Publish the message
        status = self.send_gapps_message(topic, message)

        return status
