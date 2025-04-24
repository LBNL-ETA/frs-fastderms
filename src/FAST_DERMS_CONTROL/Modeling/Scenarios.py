"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from typing import Tuple, List
from itertools import chain
from sklearn.cluster import KMeans

from .networkCase import NetworkCase
from .network import Network
from ..common import fastderms
import numpy as np


class ScenarioBuilder(fastderms):
    def __init__(self, seed: int = 1, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        new_self = kw_args.get("new_self", None)
        if new_self is not None:
            self.__dict__.update(new_self.__dict__)
            self.logger.warning("ScenarioBuilder Initialized from previous instance")

        self.logger.info(f"ScenarioBuilder initialized with seed {self.seed}")

    def build_scenarios(
        self, case: NetworkCase, n_timestep: int = 24, **kw_args
    ) -> NetworkCase:

        # Get options out of kw_args
        max_loop_nr = int(kw_args.get("max_loop_nr", 10))
        n_scenario = int(
            kw_args.get("n_scenario", case.get_case_settings(setting="n_scenario"))
        )
        case.set_case_settings(n_scenario=n_scenario)
        multiplier = kw_args.get("multiplier", 1)
        use_exp_value = bool(kw_args.get("use_exp_value", True))
        offset_scenario = int(kw_args.get("offset_scenario", -1))

        Net = case.Network
        validatedScenario = False
        loop_nr = 0

        while (not validatedScenario) and loop_nr < max_loop_nr:
            loop_nr += 1
            # Load Scenario
            loads = Net.get_loads()
            for load in loads:
                loadID = load.getID()
                self.logger.info(f"Load {loadID} Scenario being updated")
                fcast_frames = case.read_horizon_fcast("load", loadID)
                for forecastType in fcast_frames[0].keys():
                    xbar = [frame[forecastType]["xbar"] for frame in fcast_frames]
                    sigma = [
                        multiplier * frame[forecastType]["sigma"]
                        for frame in fcast_frames
                    ]
                    samples_fcast_type = self.draw_normal_samples(
                        xbar,
                        sigma,
                        self.get_samples_dims(
                            xbar,
                            n_scenario,
                            n_timestep,
                            offset_scenario=offset_scenario,
                        ),
                        add_exp_value=use_exp_value,
                    )
                    case.write_horizon_samples(
                        "load", loadID, forecastType, values=samples_fcast_type
                    )

                    self.logger.debug(f"{forecastType}\n{samples_fcast_type}\n\n")

            # DERs Scenario
            DERs = Net.get_DERs()
            for der in DERs:
                derID = der.getID()
                self.logger.info(f"DER {derID} Scenario being updated")
                is_PV = der.get_type() == "PV"
                fcast_frames = case.read_horizon_fcast("DER", derID)
                for forecastType in fcast_frames[0].keys():
                    xbar = [frame[forecastType]["xbar"] for frame in fcast_frames]
                    sigma = [
                        multiplier * frame[forecastType]["sigma"]
                        for frame in fcast_frames
                    ]
                    samples_fcast_type = self.draw_normal_samples(
                        xbar,
                        sigma,
                        self.get_samples_dims(
                            xbar,
                            n_scenario,
                            n_timestep,
                            offset_scenario=offset_scenario,
                        ),
                        add_exp_value=use_exp_value,
                    )

                    if is_PV and forecastType == "Pmax":
                        self.logger.info("Dealing with PV resource: Enforcing Pmax >=0")
                        samples_fcast_type = np.maximum(samples_fcast_type, 0)

                    if forecastType == "TRpower":
                        self.logger.info(
                            "Dealing with a Transactive Resource: Enforcing Monotonically increasing powers !"
                        )
                        samples_fcast_type = np.take_along_axis(
                            samples_fcast_type, np.argsort(samples_fcast_type), axis=-1
                        )

                    case.write_horizon_samples(
                        "DER", derID, forecastType, values=samples_fcast_type
                    )
                    self.logger.debug(f"{forecastType}\n{samples_fcast_type}\n\n")

            # Prices Scenario
            self.logger.info(f"Prices Scenario being updated")
            fcast_frames = case.read_horizon_fcast("price", None)
            for forecastType in fcast_frames[0].keys():
                xbar = [frame[forecastType]["xbar"] for frame in fcast_frames]
                sigma = [
                    multiplier * frame[forecastType]["sigma"] for frame in fcast_frames
                ]
                samples_fcast_type = self.draw_normal_samples(
                    xbar,
                    sigma,
                    self.get_samples_dims(
                        xbar, n_scenario, n_timestep, offset_scenario=offset_scenario
                    ),
                    add_exp_value=use_exp_value,
                )
                case.write_horizon_samples(
                    "price", None, forecastType, values=samples_fcast_type
                )
                self.logger.debug(f"{forecastType}\n{samples_fcast_type}\n\n")

            # Validating Scenario
            case, validatedScenario = self.validate_scenarios(case)

            if not validatedScenario:
                self.logger.error(
                    f"Batch of scenarios {loop_nr} not valid, trying again..."
                )

        # We're done
        if not validatedScenario:
            self.logger.error(
                f"Reached maximum number of tries, Scenarios NOT VALIDATED"
            )
            raise Exception("Scenarios not validated")
        else:
            self.logger.warning("Scenarios updated")

        return case

    def select_scenarios(
        self, case: NetworkCase, selected_scenarios: List[int], **kw_args
    ) -> NetworkCase:

        Net = case.Network
        # Load Scenario
        loads = Net.get_loads()
        for load in loads:
            samples_frames = case.read_horizon_samples("load", load.getID())
            for forecastType in samples_frames[0].keys():
                samples_fcast_type = [
                    [samples_frame[forecastType][i] for i in selected_scenarios]
                    for samples_frame in samples_frames
                ]
                case.write_horizon_samples(
                    "load", load.getID(), forecastType, values=samples_fcast_type
                )

        # DERs Scenario
        DERs = Net.get_DERs()
        for der in DERs:
            samples_frames = case.read_horizon_samples("DER", der.getID())
            for forecastType in samples_frames[0].keys():
                samples_fcast_type = [
                    [samples_frame[forecastType][i] for i in selected_scenarios]
                    for samples_frame in samples_frames
                ]
                case.write_horizon_samples(
                    "DER", der.getID(), forecastType, values=samples_fcast_type
                )

        # Price Scenario
        samples_frames = case.read_horizon_samples("price", None)
        for forecastType in samples_frames[0].keys():
            samples_fcast_type = [
                [samples_frame[forecastType][i] for i in selected_scenarios]
                for samples_frame in samples_frames
            ]
            case.write_horizon_samples(
                "price", None, forecastType, values=samples_fcast_type
            )

        # Update the number of scenario in case
        case.set_case_settings(n_scenario=len(selected_scenarios))

        return case

    def build_select_scenarios(
        self,
        case: NetworkCase,
        n_timestep: int = 24,
        n_scenario: int = 10,
        n_init: int = 1000,
        **kw_args,
    ) -> NetworkCase:
        """
        Build scenarios by generating a large number of scenarios, clusterizing them through k-means method and selecting the scenarios closest to the center of each clusters.
        """

        # Get options out of kw_args
        use_exp_value = kw_args.get("use_exp_value", True)
        kw_args.update({"use_exp_value": use_exp_value})

        # Generate all the scenarios
        case_with_all_scenarios = self.build_scenarios(
            case, n_timestep=n_timestep, n_scenario=n_init, **kw_args
        )

        # Build Feature Matrix
        feature_matrix = self.create_feature_matrix(case_with_all_scenarios, **kw_args)

        if use_exp_value:
            n_cluster = n_scenario - 1
        else:
            n_cluster = n_scenario

        if n_cluster == 0:
            self.logger.warning("No clusterization needed, using expected value only.")
            scenarios = [0]
        else:
            # Clusterize Scenarios with K-Means
            # Perform k-means classification
            kmeans = KMeans(n_clusters=n_cluster, random_state=self.seed).fit(
                feature_matrix
            )
            kmeans_labels = kmeans.labels_
            kmeans_dists = kmeans.transform(feature_matrix)

            # Within each cluster, Select the closest scenario to the center of the cluster
            scenarios = [0] * use_exp_value + [
                np.asarray(kmeans_labels == cluster).nonzero()[0][
                    kmeans_dists[kmeans_labels == cluster, cluster].argmin()
                ]
                + (1 * use_exp_value)
                for cluster in range(n_cluster)
            ]
            scenarios.sort()

        self.logger.info(f"Scenarios selected: {scenarios}")

        # Update the case with the selected scenarios
        case_selected_scenarios = self.select_scenarios(
            case_with_all_scenarios, scenarios
        )

        return case_selected_scenarios

    def draw_normal_samples(
        self, x_bar, sigma, dims: Tuple[int], add_exp_value: bool = True
    ):

        if dims[0] == 0 and add_exp_value:
            self.logger.info("No scenario to draw, only using expected Value")
            result = np.expand_dims(np.array(x_bar), axis=0)
        else:
            draw = self.rng.normal(x_bar, sigma, dims)
            if add_exp_value:
                exp_value = np.expand_dims(np.array(x_bar), axis=0)
                result = np.concatenate([exp_value, draw], axis=0)
            else:
                result = draw
        result = np.swapaxes(result, 0, 1)
        return result

    def get_samples_dims(
        self, xbar, n_scenario: int, n_timestep: int, offset_scenario=-1
    ) -> Tuple[int]:
        """
        Return the dimensions for drawing samples out of the distribution
        """
        try:
            element_dim = len(xbar[0])
            dims = (n_scenario + offset_scenario, n_timestep, element_dim)
        except:
            dims = (n_scenario + offset_scenario, n_timestep)

        self.logger.debug(f"Dimensions: {dims}")

        return dims

    def increasing_bids_test(self, bids_samples, name: str) -> bool:
        """
        Check whether Transactive Resource Bid power realization is monotonically increasing at each scenario and time period.
        """
        status = all(
            [
                all(
                    [
                        all(
                            data_i <= data_j
                            for data_i, data_j in zip(realisation, realisation[1:])
                        )
                        for realisation in timesteps
                    ]
                )
                for timesteps in bids_samples
            ]
        )

        if status:
            self.logger.info(f"TransactiveResource Data {name} OK !")
        else:
            self.logger.error(
                f"TransactiveResource Data {name} Error: Bid realizations are not increasing."
            )
        return status

    def validate_scenarios(self, case: NetworkCase):
        Net = case.Network

        elec_prices_samples = case.read_horizon_samples("price", None, "pi")

        # Initialize test as passing
        increasing_bids_result = True
        increasing_power_result = True

        # DERs Scenario
        DERs = Net.get_DERs()
        for der in DERs:
            derID = der.getID()
            if der.get_type() == "TransactiveResource":
                # Check that the bids are monotonously increasing
                bids_samples = case.read_horizon_samples("DER", derID, "TRprice")
                increasing_bids_result = self.increasing_bids_test(
                    bids_samples, f"{derID} TRprice"
                )
                power_samples = case.read_horizon_samples("DER", derID, "TRpower")
                increasing_power_result = self.increasing_bids_test(
                    power_samples, f"{derID} TRpower"
                )

                if not increasing_bids_result or not increasing_power_result:
                    break
                # Pmin is set to the Max load of the ressource in the PQ pair
                Pmin_samples = [
                    [TRpowers[0] for TRpowers in TRpowers_scenarios]
                    for TRpowers_scenarios in power_samples
                ]
                case.write_horizon_samples("DER", derID, "Pmin", values=Pmin_samples)
                # Pmax is set to the Min load of the ressource in the PQ pair
                Pmax_samples = [
                    [TRpowers[2] for TRpowers in TRpowers_scenarios]
                    for TRpowers_scenarios in power_samples
                ]
                case.write_horizon_samples("DER", derID, "Pmax", values=Pmax_samples)

        validatedScenario = increasing_bids_result and increasing_power_result

        # We're done
        if not validatedScenario:
            self.logger.error(f"Scenarios not validated !")
        else:
            self.logger.warning("Scenario Validated")

        return case, validatedScenario

    def create_feature_matrix(
        self,
        case: NetworkCase,
        add_prices: bool = False,
        TS_metrics: bool = False,
        **kw_args,
    ):
        """
        Create a feature matrix for the given feature and scenario and timestep
        """

        def get_load_samples(load, scenario_nr):
            load_samples = case.read_horizon_samples(
                "load", load.getID(), fcast_type="PL", scenario_nr=scenario_nr
            )
            return load_samples

        def get_DER_samples(DER, fcast_type: str, scenario_nr):
            DER_samples = case.read_horizon_samples(
                "DER", DER.getID(), fcast_type=fcast_type, scenario_nr=scenario_nr
            )
            return DER_samples

        def scenario_vector(
            scenario_nr, loads, DERs, add_prices: bool, TS_metrics: bool
        ):
            self.logger.debug(f"Features of scenario {scenario_nr}")

            if TS_metrics:
                load_data = (
                    data
                    for data in chain.from_iterable(
                        chain.from_iterable(
                            self.compute_TS_metrics(load_phase)
                            for load_phase in self.TS_phase_separator(
                                get_load_samples(load, scenario_nr)
                            )
                            if any(data_point != 0 for data_point in load_phase)
                        )
                        for load in loads
                    )
                )
                DER_data = (
                    data
                    for data in chain.from_iterable(
                        chain.from_iterable(
                            chain.from_iterable(
                                self.compute_TS_metrics(DER_fcast)
                                for DER_fcast in self.TS_phase_separator(
                                    get_DER_samples(DER, fcast_type, scenario_nr)
                                )
                                if any(data_point != 0 for data_point in DER_fcast)
                            )
                            for fcast_type in DER.get_forecast_types(source="samples")
                        )
                        for DER in DERs
                    )
                )
                if add_prices:
                    elec_price_data = self.compute_TS_metrics(
                        case.read_horizon_samples(
                            "price", None, "pi", scenario_nr=scenario_nr
                        )
                    )
                else:
                    elec_price_data = ()

            else:
                load_data = (
                    data
                    for data in chain.from_iterable(
                        chain.from_iterable(
                            load_phase
                            for load_phase in self.TS_phase_separator(
                                get_load_samples(load, scenario_nr)
                            )
                            if any(data_point != 0 for data_point in load_phase)
                        )
                        for load in loads
                    )
                )
                DER_data = (
                    data
                    for data in chain.from_iterable(
                        chain.from_iterable(
                            chain.from_iterable(
                                DER_fcast
                                for DER_fcast in self.TS_phase_separator(
                                    get_DER_samples(DER, fcast_type, scenario_nr)
                                )
                                if any(data_point != 0 for data_point in DER_fcast)
                            )
                            for fcast_type in DER.get_forecast_types(source="samples")
                        )
                        for DER in DERs
                    )
                )
                if add_prices:
                    elec_price_data = (
                        elec_price
                        for elec_price in case.read_horizon_samples(
                            "price", None, "pi", scenario_nr=scenario_nr
                        )
                    )
                else:
                    elec_price_data = ()

            scenario_vector = tuple(
                data for data in chain(load_data, DER_data, elec_price_data)
            )

            return scenario_vector

        n_scenario = case.get_case_settings(setting="n_scenario")

        # Get Networrk Resources
        loads = case.get_horizon_asset("load", frame_idx=0)
        DERs = case.get_horizon_asset("DER", frame_idx=0)
        # ONly selecting active DERs
        DERs = [der for der in DERs if der.get_active_status()]

        feature_matrix = [
            scenario_vector(scenario_nr, loads, DERs, add_prices, TS_metrics)
            for scenario_nr in range(1, n_scenario)
        ]
        # making feature matrix a numpy array instead of list of lists
        feature_matrix = np.array(feature_matrix)

        self.logger.info(f"Generated Feature Matrix: of size{feature_matrix.shape}")

        return feature_matrix

    def compute_TS_metrics(self, Timeseries):
        """
        Computes a set of metrics for the given Timeseries that are used for the classification
        """

        try:
            TS = np.asarray(Timeseries)

            out = {}
            out["len"] = len(TS)
            out["var"] = TS.std() ** 2
            out["energy"] = TS.sum()
            out["max"] = TS.max()
            out["min"] = TS.min()
            out["peak_int"] = TS.argmax()
            TS_ramps = TS[1:] - TS[:-1]
            out["mileage"] = np.abs(TS_ramps).sum()
            # Max Min ramp between 1 interval
            out["max_ramp"] = TS_ramps.max()
            out["min_ramp"] = TS_ramps.min()
            TS_3ramps = TS[3] - TS[:-3]
            # Max Min ramp between 3 interval
            out["max_3ramp"] = TS_3ramps.max()
            out["min_3ramp"] = TS_3ramps.min()

            output = (
                out["energy"],
                out["mileage"],
                out["max"],
                out["min"],
                out["max_ramp"],
                out["min_ramp"],
                out["max_3ramp"],
                out["min_3ramp"],
                out["var"],
                out["peak_int"],
            )

        except:
            self.logger.error(
                f"Error: input data of type {type(TS)} is not recognized as time series"
            )
            self.logger.error(f"Input data: {TS}")

            output = Tuple([0] * 10)

        finally:

            return output
