"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..common import fastderms, MissingItem_Exception
from ..IOs.IOclass import IOmodule
from . import equipmentClasses as mod
from . import model_reduction as red

from .networkCase import NetworkCase
from .network import Network
from typing import List, Dict
import numpy as np
import datetime as dt
import copy

dt_0 = dt.datetime(2020, 1, 1, 0, 0, 0)


class modelHandler(fastderms):
    def __init__(
        self, modelRef, IOmodule: IOmodule, Sbase: float, *args, **kw_args
    ) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        self.IO = IOmodule
        new_self = kw_args.get("new_self", None)
        if new_self is not None:
            # IO module is not to be replaced
            _ = new_self.__dict__.pop("IO", None)
            self.__dict__.update(new_self.__dict__)
            self.logger.warning("Model Handler Initialized from previous instance")
        else:
            self.mRID = modelRef
            self.logger.debug(f"Model ID set to {modelRef}")
            self.Sbase = Sbase
            self.logger.debug(f"Model handler using Sbase = {Sbase/1000} kVA")

            # Initialize model components
            self.substation_Vbase = None
            self.Network = None
            self.full_Network = None
            self.case = None
            self.rules = []

            self.logger.info("Model Handler Initialized")

    def export_self(self):
        new_self = copy.copy(self)
        _ = new_self.__dict__.pop("IO", None)
        return new_self

    def get_network(self) -> Network:
        try:
            if not (self.case is None):
                Network = self.case.Network
            else:
                Network = self.Network

            if Network == None:
                raise Exception("Missing Network Model")

            return Network
        except:
            raise

    def get_full_network(self) -> Network:
        try:
            if self.full_Network == None:
                raise MissingItem_Exception("Full Network\nModel not reduced yet!")

            return self.full_Network
        except:
            raise

    def initialize_model(self, **kw_args):
        mRID = kw_args.get("mRID", self.mRID)
        if mRID != self.mRID:
            self.mRID = mRID
            self.logger.info(f"Model ID changed to {mRID}")

        self.import_result = self.IO.fetch_model(mRID, **kw_args)
        self.logger.info("XML file from model read")
        self.load_model_file()

    def load_model_file(self):
        substation_ID = self.IO.get_substation_ID()
        self.logger.debug(f"substation_ID: {substation_ID}")

        lines = self.IO.build_Lines(Sbase=self.Sbase)
        self.logger.debug(
            "\nlines:\n"
            + "\n".join(
                [
                    f"Line fr {line.from_bus} to {line.to_bus} on phases {line.phases}"
                    for line in lines
                ]
            )
        )

        DERs = self.IO.build_DERs(Sbase=self.Sbase)
        self.logger.debug(
            "\nDERs:\n"
            + "\n".join(
                [
                    f"{der.get_type()} of ID {der.getID()} on buses {der.bus_phase}"
                    for der in DERs
                ]
            )
        )

        Loads = self.IO.build_Loads()
        self.logger.debug(
            "\nLoads:\n"
            + "\n".join(
                [
                    f"Load {load.getID()} on bus {load.get_nodes_phases()}"
                    for load in Loads
                ]
            )
        )

        self.Network = Network(
            substation_ID, lines, DERs, Loads, self.Sbase, tz=self.local_tz
        )
        self.logger.info("Network Model created from loaded file")

    def update_model_voltage_limits(
        self, Vmin: float, Vmax: float, propagate: bool = False
    ):
        self.get_network().set_voltage_limits(Vmin, Vmax)
        if (not (self.case is None)) and propagate:
            for frame in self.case.horizon:
                frame.set_voltage_limits(Vmin, Vmax)
        self.logger.info(f"Voltage limits updated to Vmin: {Vmin} Vmax: {Vmax}")

    def per_unitize_network(self, Vbase: float = None):

        try:
            Network = self.get_network()
            # Get the lines downstream of the substation
            first_lines = Network.get_direct_downstream_lines_node(
                self.IO.get_substation_ID()
            )
            # Update substation Vbase
            if Vbase is None:
                # Use the first line nominal voltage if no Vbase is provided
                Vbase_first_line = first_lines[0].Vnom
                self.substation_Vbase = Vbase_first_line
            else:
                self.substation_Vbase = Vbase
            # per Unitize the network
            Network.per_unitize(first_lines, Vbase)

            # compute approximate impedances for network
            Network.set_approxed_imps()
            self.logger.info("Network per unitized")

        except Exception as e:
            raise (e)

    def set_reduce_model_rules(self, *args):
        self.rules = [arg() for arg in args if issubclass(arg, red.model_red_rule)]
        self.logger.info(f"Setting {len(self.rules)} rule(s) for Model reduction")

    def reduce_model(self, *args, **kw_args):
        try:
            # Check if model has been reduced
            try:
                self.logger.warning("Testing if model has been reduced")
                self.get_full_network()
                raise Exception("Model already reduced")
            except MissingItem_Exception:
                pass
            except:
                raise

            # Get original / full network
            self.full_Network = self.get_network()

            # Data to copy
            substation_ID = self.full_Network.get_substation_ID()
            DERs = self.full_Network.get_DERs()

            # Get switches
            Switches = self.full_Network.get_switches()
            Trafos = self.full_Network.get_trafos()

            # list of Nodes to preserve:
            nodes_to_keep = [substation_ID]
            assets_list = DERs + Switches + Trafos

            for asset in assets_list:
                nodes = []

                if isinstance(asset, mod.DER):
                    # Nodes with a FRS DER are preserved
                    nodes = asset.get_nodes()

                elif isinstance(asset, mod.Line3P):
                    # Nodes connected to Switches and Trafos are preserved
                    nodes = asset.get_nodes()

                # Add the nodes to the list of nodes to keep
                for node in nodes:
                    nodes_to_keep.append(node)
                self.logger.debug(f"Asset {asset.getID()} at nodes: {nodes}")

            # Keeping a list of unique nodes to keep
            nodes_to_keep = np.unique(nodes_to_keep)
            # initialize the reduced Network
            reduced_Network = red.Red_Network(
                orig_network=self.full_Network, nodes_to_keep=nodes_to_keep
            )

            # Network REduction using successive rules
            for rule in self.rules:
                reduced_Network = rule(reduced_Network)

            self.Network = reduced_Network

            self.logger.warning("Network reduced, final graph:")
            self.Network.crawl_graph(self.Network.get_substation_ID())

        except Exception as e:
            self.logger.error("Network reduction failed")
            self.logger.error(repr(e))
            raise

    def init_case(
        self,
        Network: Network = None,
        n_timestep: int = 24,
        timestep_period: int = 1,
        n_scenario: int = 10,
        t0: dt.datetime = dt_0,
    ):
        if Network == None:
            Network = self.get_network()
        self.case = NetworkCase(
            Network, n_timestep, timestep_period, t0, n_scenario, tz=self.local_tz
        )

    def update_network(self, refreshTopology=False):

        # Update line statuses
        # get line status update
        Network = self.get_network()
        switches = Network.get_switches()

        self.logger.info("Updating Switch statuses")
        for switch in switches:
            current_status = self.IO.get_switch_status(switch.getID())
            switch.set_switch_status(current_status)

        # propagate status
        Network.propagate_status(switches)
        self.logger.info("Switches status propagated to Network")
        Network.set_DER_status()
        self.logger.info("Switches status propagated to DERs")

        # Update Batteries initial Charges
        batteries = Network.get_batteries()

        for battery in batteries:
            soc_init = self.IO.get_battery_SOC(
                battery.getID(), type="init", Sbase=self.Sbase
            )[0]
            battery.set_init_SOC(soc_init)
            battery.set_final_SOC(soc_init)
        self.logger.info("Batteries initial/final SOC updated")

        if refreshTopology:
            # What to implement here ?
            self.logger.info("Topology Refreshed")
        else:
            self.logger.info("No Topology Refresh")
        self.logger.warning("Network updated")

    def update_forecasts(self, case: NetworkCase = None):

        # Current Network
        Net = self.get_network()

        # Current Case
        if case is None:
            case = self.case

        # Current Forecast Horizon
        case_settings = case.get_case_settings()
        t0 = case_settings["t0"]
        n_timestep = case_settings["n_timestep"]
        timestep_period = case_settings["timestep_period"]

        # Substation Limits
        # Substation data is stored with Feeder ID
        feeder_ID = self.mRID
        fcast = self.IO.get_fcast_substation(
            feeder_ID,
            t0,
            n_timestep,
            timestep_period,
            Sbase=self.Sbase,
            Vbase=self.substation_Vbase,
        )
        case.set_horizon(Network.set_substation_voltage, values=fcast["V_set"])
        case.set_horizon(
            Network.set_substation_power_lim, "up", values=fcast["P0_up_lim"]
        )
        case.set_horizon(
            Network.set_substation_power_lim, "dn", values=fcast["P0_dn_lim"]
        )
        self.logger.info(f"Substation limits updated")
        self.logger.debug(fcast)

        # switch forecast
        def set_switch_status(Network, mRID, value):
            current_switch = Network.get_lines(mRIDs=[mRID])
            current_switch[0].set_switch_status(value)

        switches = Net.get_switches()
        for switch in switches:
            switchID = switch.getID()
            fcast = self.IO.get_fcast_switch_status(
                switchID, t0, n_timestep, timestep_period
            )
            case.set_horizon(set_switch_status, switchID, values=fcast["status"])

            self.logger.info(f"Switch {switchID} forecast updated")
            self.logger.debug(fcast)

        # propagate status
        case.set_horizon(Network.propagate_status, switches)
        self.logger.info("Switches status propagated to Network")
        case.set_horizon(Network.set_DER_status)
        self.logger.info("Switches status propagated to DERs")

        # Load Forecasts
        loads = Net.get_loads()
        for load in loads:
            loadID = load.getID()
            self.logger.info(f"Updating forecast of load {loadID}")
            fcast = self.IO.get_fcast_load(
                loadID, t0, n_timestep, timestep_period, Sbase=self.Sbase
            )
            fcast = self._format_fcast(fcast)
            for forecastType in fcast.keys():
                one_fcast = fcast[forecastType]
                fcast_frames = [
                    {"xbar": x_bar, "sigma": sigma}
                    for x_bar, sigma in zip(one_fcast["xbar"], one_fcast["sigma"])
                ]
                case.write_horizon_fcast(
                    "load", loadID, forecastType, values=fcast_frames
                )
                self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        # DERs Forecasts
        DERs = Net.get_DERs()
        for der in DERs:
            derID = der.getID()
            self.logger.info(f"Updating forecast of DER {derID}")
            fcast = self.IO.get_fcast_DER(
                derID, t0, n_timestep, timestep_period, Sbase=self.Sbase
            )
            fcast = self._format_fcast(fcast)
            for forecastType in fcast.keys():
                one_fcast = fcast[forecastType]
                fcast_frames = [
                    {"xbar": x_bar, "sigma": sigma}
                    for x_bar, sigma in zip(one_fcast["xbar"], one_fcast["sigma"])
                ]
                case.write_horizon_fcast(
                    "DER", derID, forecastType, values=fcast_frames
                )
                self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        # Price Data
        self.logger.info("Updating price data")
        fcast = self.IO.get_fcast_price(t0, n_timestep, timestep_period)
        fcast = self._format_fcast(fcast)
        for forecastType in fcast.keys():
            one_fcast = fcast[forecastType]
            fcast_frames = [
                {"xbar": x_bar, "sigma": sigma}
                for x_bar, sigma in zip(one_fcast["xbar"], one_fcast["sigma"])
            ]
            case.write_horizon_fcast("price", None, forecastType, values=fcast_frames)
            self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        self.logger.warning("Forecasts updated")

    def updateDeterministicData(self):

        # Current Network
        Net = self.get_network()

        self.case.n_scenario = 1

        # Current Forecast Horizon
        # Current Forecast Horizon
        case_settings = self.case.get_case_settings()
        t0 = case_settings["t0"]
        n_timestep = case_settings["n_timestep"]
        timestep_period = case_settings["timestep_period"]

        # # switch forecast
        # def set_switch_status(Network, mRID, value):
        #     current_switch = Network.get_lines(mRIDs = [mRID])
        #     current_switch[0].set_switch_status(value)

        # switches = Net.get_switches()
        # for switch in switches:
        #     switchID = switch.getID()
        #     fcast = self.IO.get_fcast_switch_status(switchID, n_timestep)
        #     self.Case.set_horizon(set_switch_status, switchID, values = fcast)
        #     self.logger.debug(f'Switch {switch.getID()} with forecast:\n{fcast}')

        # # propagate status
        # self.Case.set_horizon(Network.propagate_status, switches)

        # # Update DER Status
        # self.Case.set_horizon(Network.set_DER_status)

        # Load Forecasts
        loads = Net.get_loads()
        for load in loads:
            loadID = load.getID()
            self.logger.info(f"Load {loadID} Deterministic updated to:")
            deterministic_fcast = self.IO.get_DET_load(
                loadID, t0, n_timestep, timestep_period, Sbase=self.Sbase
            )
            deterministic_fcast = self._format_fcast(deterministic_fcast)
            for forecastType in deterministic_fcast.keys():
                one_fcast = [
                    [one_value] for one_value in deterministic_fcast[forecastType]
                ]
                self.case.write_horizon_samples(
                    "load", loadID, forecastType, values=one_fcast
                )
                self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        # DERs Forecasts
        DERs = Net.get_DERs()
        for der in DERs:
            derID = der.getID()
            self.logger.info(f"DER {derID} Deterministic updated to:")
            deterministic_fcast = self.IO.get_DET_DER(
                derID, t0, n_timestep, timestep_period, Sbase=self.Sbase
            )
            deterministic_fcast = self._format_fcast(deterministic_fcast)
            for forecastType in deterministic_fcast.keys():
                one_fcast = [
                    [one_value] for one_value in deterministic_fcast[forecastType]
                ]
                self.case.write_horizon_samples(
                    "DER", derID, forecastType, values=one_fcast
                )
                self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        # Price Data
        deterministic_fcast = self.IO.get_DET_price(t0, n_timestep, timestep_period)
        deterministic_fcast = self._format_fcast(deterministic_fcast)
        self.logger.info(f"Prices Deterministic updated")
        for forecastType in deterministic_fcast.keys():
            one_fcast = [[one_value] for one_value in deterministic_fcast[forecastType]]
            self.case.write_horizon_samples(
                "price", None, forecastType, values=one_fcast
            )
            self.logger.debug(f"{forecastType}\n{one_fcast}\n\n")

        # Additional Deterministic Data
        data_set = self.IO.get_DET_input_data(
            self.mRID, t0, n_timestep, timestep_period, Sbase=self.Sbase
        )
        P0_set = data_set["P0_set"]
        self.case.set_horizon(Network.set_substation_power, values=P0_set)
        self.logger.debug(f"Substation Power updated to:\n {P0_set}")
        Rup_set = data_set["Rup_set"]
        self.case.set_horizon(Network.set_system_reserves, "up", values=Rup_set)
        self.logger.debug(f"System Up Reserves updated to:\n {Rup_set}")
        Rdn_set = data_set["Rdn_set"]
        self.case.set_horizon(Network.set_system_reserves, "dn", values=Rdn_set)
        self.logger.debug(f"System Down Reserves updated to:\n {Rdn_set}")

        self.logger.warning("Deterministic Forecasts updated")

    def init_MPC_case(
        self, n_timestep: int = 24, timestep_period: int = 5, t0: dt.datetime = dt_0
    ):
        Network = self.get_network()
        self.MPC_case = NetworkCase(
            Network, n_timestep, timestep_period, t0, n_scenario=1
        )

    def update_MPC_case(self, t0: dt.datetime, E_init: Dict = None, **kw_args):
        old_t0 = self.MPC_case.get_case_settings()["t0"]
        self.MPC_case.set_case_settings(t0=t0)
        self.update_forecasts(self.export_MPC_case())

        case_settings = self.MPC_case.get_case_settings()
        n_timestep = case_settings["n_timestep"]
        timestep_period = case_settings["timestep_period"]

        # Overriding the Price forecasts with data from ISO
        self.logger.info("Overrriding the price forecasts for MPC specific Data")
        # Price Data
        MPC_price_fcast = self.IO.get_MPC_price(t0, n_timestep, timestep_period)
        MPC_price_fcast = self._format_fcast(MPC_price_fcast)
        for forecastType in MPC_price_fcast.keys():
            one_fcast = MPC_price_fcast[forecastType]
            fcast_frames = [
                {"xbar": x_bar, "sigma": sigma}
                for x_bar, sigma in zip(one_fcast["xbar"], one_fcast["sigma"])
            ]
            self.MPC_case.write_horizon_fcast(
                "price", None, forecastType, values=fcast_frames
            )

        # Additional RT Data
        fcast = self.IO.get_MPC_input_data(t0, n_timestep, timestep_period)
        # MPC problem now uses total substation power instead of per phase values
        fcast_P0_set = [sum(P0_phase) for P0_phase in fcast["P0_set"]]
        self.MPC_case.set_horizon(Network.set_substation_power, values=fcast_P0_set)
        self.logger.debug(f'Substation Power updated to:\n {fcast["P0_set"]}')
        self.MPC_case.set_horizon(
            Network.set_system_reserves, "up", values=fcast["Rup_set"]
        )
        self.logger.debug(f'System Up Reserves updated to:\n {fcast["Rup_set"]}')
        self.MPC_case.set_horizon(
            Network.set_system_reserves, "dn", values=fcast["Rup_set"]
        )
        self.logger.debug(f'System Down Reserves updated to:\n {fcast["Rdn_set"]}')

        def propagate_uncertainty(case: NetworkCase, type: str, fcast_type: str):

            Net = case.get_network()
            if type == "load":
                loads = Net.get_loads()
                all_sigmas = sum(
                    np.array(
                        [
                            sum(sigma_phase for sigma_phase in fcast["sigma"])
                            for fcast in case.read_horizon_fcast(
                                "load", item.getID(), fcast_type
                            )
                        ]
                    )
                    for item in loads
                )
                all_sigmas_sq = sum(
                    np.array(
                        [
                            sum(sigma_phase**2 for sigma_phase in fcast["sigma"])
                            for fcast in case.read_horizon_fcast(
                                "load", item.getID(), fcast_type
                            )
                        ]
                    )
                    for item in loads
                )

            elif type == "DER":
                DERs = Net.get_DERs()
                # only take PV
                PVs = [item for item in DERs if item.get_type() == "PV"]
                all_sigmas = sum(
                    np.array(
                        [
                            fcast["sigma"]
                            for fcast in case.read_horizon_fcast(
                                "DER", item.getID(), fcast_type
                            )
                        ]
                    )
                    for item in PVs
                )
                all_sigmas_sq = sum(
                    np.array(
                        [
                            fcast["sigma"] ** 2
                            for fcast in case.read_horizon_fcast(
                                "DER", item.getID(), fcast_type
                            )
                        ]
                    )
                    for item in PVs
                )

            else:
                all_sigmas = np.array([])
                all_sigmas_sq = np.array([])

            dep_sigmas = all_sigmas
            ind_sigmas = np.sqrt(all_sigmas_sq)

            return dep_sigmas, ind_sigmas

        # MPC Additional Reserve requirements:
        opt_R_dist = kw_args.get("opt_R_dist", {})
        beta_load = opt_R_dist.get("beta_load", 0.9)
        beta_DER = opt_R_dist.get("beta_DER", 0.9)
        R_sigma_up = opt_R_dist.get("R_sigma_up", 1.5)
        R_sigma_dn = opt_R_dist.get("R_sigma_dn", 1.5)

        # load uncertainty:
        type = "load"
        fcast_type = "PL"
        dep_sigmas, ind_sigmas = propagate_uncertainty(self.MPC_case, type, fcast_type)

        total_load_sigmas = beta_load * dep_sigmas + (1 - beta_load) * ind_sigmas

        # DER uncertainty:
        type = "DER"
        fcast_type = "Pmax"
        dep_sigmas, ind_sigmas = propagate_uncertainty(self.MPC_case, type, fcast_type)

        total_DER_sigmas = beta_DER * dep_sigmas + (1 - beta_DER) * ind_sigmas

        total_uncertainty = np.sqrt(
            np.square(total_load_sigmas) + np.square(total_DER_sigmas)
        )

        R_dist_up = float(R_sigma_up) * total_uncertainty
        R_dist_dn = float(R_sigma_dn) * total_uncertainty

        # Setting R_dist
        self.MPC_case.set_horizon(
            Network.set_system_reserves, "up_dist", values=R_dist_up
        )
        self.logger.debug(f"Distrib Up Reserves updated to:\n {R_dist_up}")
        self.MPC_case.set_horizon(
            Network.set_system_reserves, "dn_dist", values=R_dist_dn
        )
        self.logger.debug(f"Distrib Down Reserves updated to:\n {R_dist_dn}")

        self.logger.info("RT Forecasts updated")

        # Update final SOC to value from DA solution
        self.logger.info(f"Updating the SoC terminal condition from DayAhead solution")
        for derID in fcast["E_All"].keys():
            final_SOC = fcast["E_All"][derID][-1]
            self.MPC_case.Network.get_DERs(mRIDs=derID)[0].set_final_SOC(final_SOC)

        # Update init SOC
        # E_init contains the results from previous MPC run
        # For first run, E_init is None
        if E_init is None:
            # not from previous run, so taken from DET results in p.u,
            E_init = {
                der_ID: fcast["E_All"][der_ID][0] for der_ID in fcast["E_All"].keys()
            }
            self.logger.info(f"E_init for the first run is [p.u.]: \n{E_init} ")

        # E_meas: Get the current SOC from G_apps
        BATT_list = self.MPC_case.get_network().get_batteries()
        E_meas = {}
        for BATT in BATT_list:
            try:
                # Query IOs for E measurement
                # We need a measurement that is more recent than the last MPC run
                # MPC stepping period in minutes
                timestep_period = (t0 - old_t0).total_seconds() / 60

                measurements = self.IO.get_battery_SOC(
                    BATT.getID(),
                    t0=old_t0,
                    n_timestep=1,
                    timestep_period=timestep_period,
                    Sbase=self.Sbase,
                )
                E_meas.update({BATT.getID(): measurements[0]})
                self.logger.debug(
                    f"Collected current SoC for {BATT.getID()}: {measurements[0]} p.u."
                )

            except:
                self.logger.error(
                    f"Error fetching the battery energy level for {BATT.getID()}"
                )

        # Update SOC
        self.logger.debug(f"E_init: {E_init}")
        self.logger.debug(f"E_meas: {E_meas}")

        # Update E_init with measured values from G_apps
        E_init.update(E_meas)

        self.logger.info(f"Updating the SoC initial condition")
        for derID in E_init.keys():
            initial_SOC = E_init[derID]
            self.MPC_case.Network.get_DERs(mRIDs=derID)[0].set_init_SOC(initial_SOC)

        self.logger.warning("RT Case Updated")

    def import_case(self, case: NetworkCase):
        self.case = case

    def import_MPC_case(self, case: NetworkCase):
        self.MPC_case = case

    def export_case(self):
        return self.case

    def export_MPC_case(self):
        return self.MPC_case

    def _format_fcast(self, fcast: Dict):
        # Format the forecast to be used in the case
        fcast_out = {}
        for fcast_type in fcast.keys():
            if fcast_type not in ["t0", "timestep_period"]:
                fcast_out.update({fcast_type: fcast[fcast_type]})
        return fcast_out
