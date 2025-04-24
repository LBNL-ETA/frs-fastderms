"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from typing import List, Dict
from pathlib import Path

from . import IOclass as IO
from ..Modeling import equipmentClasses as mod

import numpy as np
import json
import pickle


class IOstatic(IO.IOmodule):
    def __init__(self, path_to_repo: Path, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        self.path_to_repo = path_to_repo
        self.path_to_data = self.path_to_repo / "data"
        self.path_to_CIM = self.path_to_data / "CIM_files"

        path_to_static_data = self.path_to_data / "static"

        # initialize data to None
        static_data = None
        fcasts_data = None
        deterministic_data = None

        file_data = kw_args.get("file_all_data", None)
        if file_data is not None:
            # Try the new way with 1 file
            if str(Path(file_data).parent) == ".":
                file_data_fn = path_to_static_data / file_data
            else:
                file_data_fn = Path(file_data)
            self.logger.debug(f"Loading data from {file_data_fn}")

            all_items = self.load_all_pickle(file_data_fn)
            for item in all_items:
                data_type = item.get("data_type", None)
                if data_type == "static":
                    static_data = item
                elif data_type == "fcast":
                    fcasts_data = item
                elif data_type == "deterministic":
                    deterministic_data = item
                else:
                    self.logger.error(f"Unknown data type {data_type}")
        else:
            # Try the old way with 2 files
            file_static_data = kw_args.get("file_static_data", None)
            file_fcast_data = kw_args.get("file_fcast_data", None)

            if file_static_data is not None:
                if str(Path(file_static_data).parent) == ".":
                    static_data = path_to_static_data / file_static_data
                else:
                    static_data = Path(file_static_data)
                self.logger.debug(f"Loading static data from {static_data}")
                # Load static data
                with open(static_data.absolute(), "r") as f:
                    static_data = json.load(f)
            else:
                self.logger.error("No static data file provided")

            if file_fcast_data is not None:
                if str(Path(file_fcast_data).parent) == ".":
                    fcast_data = path_to_static_data / file_fcast_data
                else:
                    fcast_data = Path(file_fcast_data)
                self.logger.debug(f"Loading forecast data from {fcast_data}")
                # Load forecast data
                with open(fcast_data.absolute(), "rb") as f:
                    fcasts_data = pickle.load(f)
            else:
                self.logger.error("No forecast data file provided")

        self.static_data = static_data
        self.fcasts = fcasts_data
        self.deterministic = deterministic_data

        self.logger.info("IO static data Initialized")

    def fetch_model(self, modelID):
        # Fetch model from local files

        if modelID == self.static_data["model_ID"]:
            path_to_model = self.path_to_CIM / self.static_data["model_file"]

        else:
            raise ValueError(f"ModelID {modelID} not recognized")

        self.logger.info(f"Opening file: {path_to_model}")
        return super().load_model(str(path_to_model.absolute()))

    def get_substation_ID(self):
        return self.static_data["substation_ID"]

    def build_DERs(self, **kw_args) -> List[mod.DER]:
        ## Example static DER Data for IEEE13, you may want to pass Sbase into here to per unitize the input.  Assume that when data is gathered off of the platform it will be in kW/kVAR/kVA and not in per unit.
        # BAT1 Utility-scale storage unit (single location)
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)
        DERs = []
        der_dict = self.static_data.get("DERs", {})

        for mRID in der_dict.keys():

            the_DER = der_dict[mRID]
            type = the_DER["type"]

            bus_phase = the_DER["bus_phase"]
            gamma = the_DER.get(
                "gamma",
                {
                    key: [int(phase in item) for phase in [0, 1, 2]]
                    for key, item in bus_phase.items()
                },
            )

            if type == "BAT":
                eff_c = the_DER.get("eff_c", 1)
                eff_d = the_DER.get("eff_d", 1)
                emax = self.per_unitize(the_DER.get("emax", None), base=Sbase)
                qmax = self.per_unitize(the_DER.get("qmax", None), base=Sbase)
                qmin = self.per_unitize(the_DER.get("qmin", None), base=Sbase)
                smax = self.per_unitize(the_DER.get("smax", None), base=Sbase)

                DER_mod = mod.Battery(
                    mRID=mRID,
                    bus_phase=bus_phase,
                    S_max=smax,
                    Q_min=qmin,
                    Q_max=qmax,
                    eff_c=eff_c,
                    eff_d=eff_d,
                    E_max=emax,
                    gamma=gamma,
                    tz=self.local_tz,
                )

            elif type == "PV":
                pmin = self.per_unitize(the_DER.get("pmin", None), base=Sbase)
                qmax = self.per_unitize(the_DER.get("qmax", None), base=Sbase)
                qmin = self.per_unitize(the_DER.get("qmin", None), base=Sbase)
                smax = self.per_unitize(the_DER.get("smax", None), base=Sbase)

                DER_mod = mod.PV(
                    mRID=mRID,
                    bus_phase=bus_phase,
                    P_min=pmin,
                    S_max=smax,
                    Q_min=qmin,
                    Q_max=qmax,
                    gamma=gamma,
                    tz=self.local_tz,
                )

            elif type == "FL":

                DER_mod = mod.FlexibleLoad(
                    mRID=mRID, bus_phase=bus_phase, gamma=gamma, tz=self.local_tz
                )

            elif type == "EV":
                eff_c = the_DER.get("eff_c", 1)
                emax = self.per_unitize(the_DER.get("emax", None), base=Sbase)
                pmin = self.per_unitize(the_DER.get("pmin", None), base=Sbase)
                qmax = self.per_unitize(the_DER.get("qmax", None), base=Sbase)
                qmin = self.per_unitize(the_DER.get("qmin", None), base=Sbase)
                smax = self.per_unitize(the_DER.get("smax", None), base=Sbase)

                DER_mod = mod.SmartChargeEV(
                    mRID=mRID,
                    bus_phase=bus_phase,
                    P_min=pmin,
                    E_max=emax,
                    S_max=smax,
                    Q_min=qmin,
                    Q_max=qmax,
                    eff_c=eff_c,
                    gamma=gamma,
                    tz=self.local_tz,
                )

            elif type == "VPP":
                pmax = self.per_unitize(the_DER.get("pmax", None), base=Sbase)
                pmin = self.per_unitize(the_DER.get("pmin", None), base=Sbase)
                qmax = self.per_unitize(the_DER.get("qmax", None), base=Sbase)
                qmin = self.per_unitize(the_DER.get("qmin", None), base=Sbase)

                DER_mod = mod.VPP(
                    mRID=mRID,
                    bus_phase=bus_phase,
                    P_max=pmax,
                    P_min=pmin,
                    Q_min=qmin,
                    Q_max=qmax,
                    gamma=gamma,
                    tz=self.local_tz,
                )

            elif type == "TR":
                pf = the_DER.get("pf", 1)

                DER_mod = mod.TransactiveResource(
                    mRID=mRID, bus_phase=bus_phase, gamma=gamma, PF=pf, tz=self.local_tz
                )

            else:
                DER_mod = None

            DERs.append(DER_mod)

        return DERs

    def build_Loads(self) -> List[mod.Load]:
        # Example Loads
        Loads = []
        loads_dict = self.static_data.get("Loads", {})
        for mRID in loads_dict.keys():

            the_Load = loads_dict[mRID]
            bus = the_Load["bus"]
            Load_mod = mod.Load(mRID=mRID, bus=bus, tz=self.local_tz)
            Loads.append(Load_mod)

        return Loads

    def build_Composite_Resources(self) -> List[mod.CompositeResource]:
        # Example Composite Resources
        Composite_Resources = []

        composite_resources_dict = self.static_data.get("Composite_Resources", {})
        # Add Composite Resources
        for mRID in composite_resources_dict.keys():

            the_Composite_Resource = composite_resources_dict[mRID]
            DER_ids = the_Composite_Resource.get("DER_ids", [])
            Load_ids = the_Composite_Resource.get("Load_ids", [])
            Composite_Resource_mod = mod.CompositeResource(
                mRID=mRID, DER_ids=DER_ids, Load_ids=Load_ids, tz=self.local_tz
            )
            Composite_Resources.append(Composite_Resource_mod)

        return Composite_Resources

    def get_switch_status(self, mRID) -> bool:
        try:
            self.logger.debug(f"Getting status for switch {mRID}")
            status = self.static_data["Switches"][mRID]["status"]
        except Exception as e:
            self.logger.error(repr(e))
            self.logger.error(f"No status found for switch {mRID}. Defaulting to True.")
            # Adding Default value
            status = True
        return status

    def get_battery_SOC(self, mRID, **kw_args) -> float:
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)
        type = kw_args.get("type", "")

        if type == "init":
            try:
                E_0 = self.static_data["DERs"][mRID]["E_0"]
                E_0 = [self.per_unitize(E_0, base=Sbase)]
            except Exception as e:
                self.logger.error(e)
                E_0 = [0]
        else:
            E_0 = None

        return E_0

    def get_fcast_substation(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        fcast = {}
        fcast["t0"] = self.fcasts["t0"]
        fcast["timestep_period"] = self.fcasts["timestep_period"]

        # Format the ADMS data
        fcast["V_set"] = [1.0 for _ in range(self.fcasts["n_timestep"])]
        fcast["P0_up_lim"] = [np.nan for _ in range(self.fcasts["n_timestep"])]
        fcast["P0_dn_lim"] = [np.nan for _ in range(self.fcasts["n_timestep"])]

        fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        return fcast

    def get_fcast_switch_status(self, mRID, t0, n_timestep, timestep_period):
        fcast = {}
        fcast["t0"] = self.fcasts["t0"]
        fcast["timestep_period"] = self.fcasts["timestep_period"]

        try:
            fcast["status"] = self.fcasts["Switches"][mRID]["status"]
        except:
            fcast["status"] = [True]
        finally:
            fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)

            return fcast

    def get_fcast_load(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)

        fcast = {}
        fcast["t0"] = self.fcasts["t0"]
        fcast["timestep_period"] = self.fcasts["timestep_period"]

        try:
            for key in self.fcasts["Loads"][mRID].keys():
                self.logger.debug(f"key: {key}")
                self.logger.debug(f'fcasts: {self.fcasts["Loads"][mRID][key]["xbar"]}')

                x_bar = self.per_unitize(
                    [
                        np.array(value)
                        for value in self.fcasts["Loads"][mRID][key]["xbar"]
                    ],
                    base=Sbase,
                )
                sigma = self.per_unitize(
                    [
                        np.array(value)
                        for value in self.fcasts["Loads"][mRID][key]["sigma"]
                    ],
                    base=Sbase,
                )

                fcast[key] = {"xbar": x_bar, "sigma": sigma}
            fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        except Exception as e:
            self.logger.error(e)
            raise
        finally:
            return fcast

    def get_fcast_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        # Get options out of kw_args
        Sbase = kw_args.get("Sbase", None)

        fcast = {}
        fcast["t0"] = self.fcasts["t0"]
        fcast["timestep_period"] = self.fcasts["timestep_period"]

        try:
            for key in self.fcasts["DERs"][mRID].keys():
                x_bar = self.per_unitize(
                    self.fcasts["DERs"][mRID][key]["xbar"], base=Sbase
                )
                sigma = self.per_unitize(
                    self.fcasts["DERs"][mRID][key]["sigma"], base=Sbase
                )
                fcast[key] = {"xbar": x_bar, "sigma": sigma}
            fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        except Exception as e:
            self.logger.error(e)
            raise
        finally:
            return fcast

    def get_fcast_price(self, t0, n_timestep, timestep_period):
        fcast = {}
        fcast["t0"] = self.fcasts["t0"]
        fcast["timestep_period"] = self.fcasts["timestep_period"]

        try:
            for key in self.fcasts["prices"].keys():
                x_bar = self.fcasts["prices"][key]["xbar"]
                sigma = self.fcasts["prices"][key]["sigma"]
                fcast[key] = {"xbar": x_bar, "sigma": sigma}
            fcast = self.process_fcast(fcast, t0, n_timestep, timestep_period)
        except Exception as e:
            self.logger.error(e)
            raise
        finally:
            return fcast

    def get_DET_load(
        self, mRID, t0, n_timestep, timestep_period, **kw_args
    ):  # needs to get the forecasts from the excel file

        # We use the same data as the forecast data for demo purposes
        fcasts = self.get_fcast_load(mRID, t0, n_timestep, timestep_period, **kw_args)

        deterministic_fcast = {}
        deterministic_fcast.update(
            {key: fcasts[key] for key in ["t0", "timestep_period"]}
        )
        deterministic_fcast.update(
            {
                key: fcasts[key]["xbar"]
                for key in fcasts.keys()
                if key != "t0" and key != "timestep_period"
            }
        )

        return deterministic_fcast

    def get_DET_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):

        # We use the same data as the forecast data for demo purposes
        fcasts = self.get_fcast_DER(mRID, t0, n_timestep, timestep_period, **kw_args)

        deterministic_fcast = {}
        deterministic_fcast.update(
            {key: fcasts[key] for key in ["t0", "timestep_period"]}
        )
        deterministic_fcast.update(
            {
                key: fcasts[key]["xbar"]
                for key in fcasts.keys()
                if key != "t0" and key != "timestep_period"
            }
        )

        return deterministic_fcast

    def get_DET_price(self, t0, n_timestep, timestep_period, **kw_args):
        deterministic_fcast = {}
        try:
            # Using deterministic data for prices
            deterministic_fcast["t0"] = self.deterministic["t0"]
            deterministic_fcast["timestep_period"] = self.deterministic[
                "timestep_period"
            ]

            for key in self.deterministic["prices"].keys():
                deterministic_fcast[key] = self.deterministic["prices"][key]

            deterministic_fcast = self.process_fcast(
                deterministic_fcast, t0, n_timestep, timestep_period
            )
        except Exception as e:
            # Missing deterministic data
            # We use the same data as the forecast data for demo purposes
            fcasts = self.get_fcast_price(t0, n_timestep, timestep_period, **kw_args)

            deterministic_fcast = {}
            deterministic_fcast.update(
                {key: fcasts[key] for key in ["t0", "timestep_period"]}
            )
            deterministic_fcast.update(
                {
                    key: fcasts[key]["xbar"]
                    for key in fcasts.keys()
                    if key != "t0" and key != "timestep_period"
                }
            )
        finally:
            return deterministic_fcast

    def publish_DA_result(
        self, scenario_res: Dict, archive: bool = False, apdx: str = ""
    ) -> None:
        self.publish_results(
            scenario_res,
            filename="_dump_scenarios_data.pkl",
            archive=archive,
            apdx=apdx,
        )

    def get_DET_input_data(self, model_ID, t0, n_timestep, timestep_period, **kw_args):
        # Data dumped from DA is already in per_unit
        data = self.load_results("_dump_scenarios_data.pkl")
        out_data = {}
        out_data["t0"] = t0
        out_data["timestep_period"] = timestep_period
        # Substation Data
        out_data["P0_set"] = [
            sum(phase_quantities) for phase_quantities in data["P0_set"]
        ]
        # Reserves
        out_data["Rup_set"] = data["Rup_set"]
        out_data["Rdn_set"] = data["Rdn_set"]

        out_data = self.process_fcast(out_data, t0, n_timestep, timestep_period)
        return out_data

    def publish_DET_result(
        self, deterministic_res: Dict, archive: bool = False, apdx: str = ""
    ) -> None:

        self.publish_results(
            deterministic_res,
            filename="_dump_deterministic_data.pkl",
            archive=archive,
            apdx=apdx,
        )

    def get_MPC_input_data(self, t0, n_timestep, timestep_period):
        # Substation Data
        # Reserves
        # Battery states
        data_set = self.load_results("_dump_deterministic_data.pkl")
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

    def ISO_gather_bids_FRS(self, t0, n_timestep, timestep_period, **kw_args):
        """
        This function is used to gather bids from the FRS.
        ### TO BE REPLACED WITH CALL TO GET DAY AHEAD BIDS FROM FRS OPTIMIZATION ###
        """
        all_files = kw_args.get("all_files", None)
        if all_files is None:
            filepath = self.path_to_export / "_dump_self_sched_bid.pkl"
            all_files = [filepath]

        bids = {}
        for file in all_files:
            with open(file, "rb") as infile:
                data_set = pickle.load(infile)
            bids.update(
                {
                    data_set["ID"]: {
                        "EnBid": data_set["EnBid"],
                        "ResBid": data_set["ResBid"],
                    }
                }
            )

        # EnBid = {(t0.replace(minute=0, second=0, microsecond=0) + dt.timedelta(minutes = i*60)).timestamp():{'s1':{'price':20, 'power':3 + (-1)**(i) * i*.05}} for i in range(n_timestep)}
        # ResBid = {(t0.replace(minute=0, second=0, microsecond=0) + dt.timedelta(minutes = i*60)).timestamp():{'Rup':{'price':10, 'power':0.1}, 'Rdn':{'price':5, 'power':0.1}} for i in range(n_timestep)}

        return bids

    def ISO_get_RT_price(self, t0, n_timestep, timestep_period, **kw_args):
        """
        This function is used to gather real-time prices for the ISO.
        """
        RT_fcast = {}
        try:
            # Using rt data for prices
            RT_fcast["t0"] = self.rt["t0"]
            RT_fcast["timestep_period"] = self.rt["timestep_period"]

            for key in self.rt["prices"].keys():
                RT_fcast[key] = self.rt["prices"][key]

            RT_fcast = self.process_fcast(RT_fcast, t0, n_timestep, timestep_period)
        except Exception as e:
            # Missing rt data
            # We use the same data as the deterministic data for demo purposes
            RT_fcast = self.get_DET_price(t0, n_timestep, timestep_period, **kw_args)

        finally:
            return RT_fcast
