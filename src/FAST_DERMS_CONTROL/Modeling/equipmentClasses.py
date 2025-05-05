"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..common import fastderms, mRID

from typing import List
from pprint import pformat

import numpy as np

import math


class CIMcomponent(fastderms):
    def __init__(self, mRID: mRID, active: bool = True, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        self.mRID = mRID
        self.active = active
        super().__init__(*args, **kw_args)

    def set_active_status(self, status: bool):
        """
        Method to set active [True] / inactive [False] status of the resource
        """
        self.active = status

    def get_active_status(self) -> bool:
        """
        Method to get active [True] / inactive [False] status of the resource
        """
        return self.active

    def getID(self) -> mRID:

        return self.mRID

    def get_type(self) -> str:
        return self.__class__.__name__


class Forecast(fastderms):
    def __init__(self, forecast_type, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        self.forecast_type = forecast_type
        self.forecast = {}
        self.samples = {}
        super().__init__(*args, **kw_args)

    def set_forecast(self, forecast_type, fcast):
        if forecast_type in self.forecast_type:
            self.forecast[forecast_type] = fcast
        else:
            self.logger.error("Incorrect Forecast type")

    def get_forecast(self, forecast_type=None):

        if forecast_type is None:
            fcast = self.forecast
        elif forecast_type in self.forecast_type:
            fcast = self.forecast[forecast_type]
        else:
            self.logger.error("Incorrect Forecast type")
            fcast = None

        return fcast

    def get_forecast_types(self, source: str = "accepted"):

        if source == "accepted":
            fcast_types = self.forecast_type
        elif source == "fcast":
            fcast_types = self.forecast.keys()
        elif source == "samples":
            fcast_types = self.samples.keys()
        else:
            fcast_types = None
            self.logger.error("Incorrect source")
        return fcast_types

    def set_samples(self, forecast_type, samples):
        if forecast_type in self.forecast_type:
            self.samples[forecast_type] = samples
        else:
            self.logger.error("Incorrect Forecast type")

    def get_samples(self, forecast_type=None, scenario_nr: int = None):
        if forecast_type is None:
            samples = self.samples
        elif forecast_type in self.forecast_type:
            if scenario_nr is None:
                samples = self.samples[forecast_type]
            else:
                samples = self.samples[forecast_type][scenario_nr]
        else:
            samples = None
            self.logger.error("Incorrect Forecast type")
        return samples


class Line3P(CIMcomponent):
    def __init__(self, from_bus, to_bus, phases, R, X, Vnom, Sbase, *args, **kw_args):
        # Forward module name to parent class
        local_name = kw_args.get("name", __name__)
        kw_args.update({"name": local_name})

        # the parent in the parent-child relationship
        self.from_bus = from_bus
        # the child in the parent-child-relationship
        self.to_bus = to_bus
        self.phases = phases
        self.Vnom = Vnom
        self.Sbase = Sbase
        self.Raa = R[0][
            0
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Rab = R[0][
            1
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Rac = R[0][
            2
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Rbb = R[1][
            1
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Rbc = R[1][
            2
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Rcc = R[2][
            2
        ]  # R is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xaa = X[0][
            0
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xab = X[0][
            1
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xac = X[0][
            2
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xbb = X[1][
            1
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xbc = X[1][
            2
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.
        self.Xcc = X[2][
            2
        ]  # X is a numpy array input, or array of arrays of numeric values, units of p.u.

        pu = kw_args.get("pu", None)
        # List of BUS IDs in the network
        if pu == None:
            self.pu = False
        else:
            self.pu = pu

        # Maximum apparent power of line, Smax in W
        self.populate_capacity(kw_args.get("Smax", 10000000000))

        super().__init__(*args, **kw_args)

        self.logger.warning(
            f"Creating {local_name} fr {from_bus} to {to_bus} on phases: {self.phases}"
        )
        self.logger.debug(
            f"\nR matrix:\n{self.Raa:.4f}, {self.Rab:.4f}, {self.Rac:.4f}\n....., {self.Rbb:.4f}, {self.Rbc:.4f},\n....., .....,  {self.Rcc:.4f}"
        )
        self.logger.debug(
            f"\nX matrix:\n{self.Xaa:.4f}, {self.Xab:.4f}, {self.Xac:.4f}\n....., {self.Xbb:.4f}, {self.Xbc:.4f},\n....., .....,  {self.Xcc:.4f}"
        )

    def get_nodes(self):
        return [self.from_bus, self.to_bus]

    def impedance_approx(self):
        """Create MP and MQ matrices for each line from the network impedances. (Lin3DistFlow)"""

        self.MP = [
            [
                -2 * self.Raa,
                self.Rab - np.sqrt(3) * self.Xab,
                self.Rac + np.sqrt(3) * self.Xac,
            ],
            [
                self.Rab + np.sqrt(3) * self.Xab,
                -2 * self.Rbb,
                self.Rbc - np.sqrt(3) * self.Xbc,
            ],
            [
                self.Rac - np.sqrt(3) * self.Xac,
                self.Rbc + np.sqrt(3) * self.Xbc,
                -2 * self.Rcc,
            ],
        ]

        self.MQ = [
            [
                -2 * self.Xaa,
                np.sqrt(3) * self.Rab + self.Xab,
                -np.sqrt(3) * self.Rac + self.Xac,
            ],
            [
                -np.sqrt(3) * self.Rab + self.Xab,
                -2 * self.Xbb,
                np.sqrt(3) * self.Rbc + self.Xbc,
            ],
            [
                np.sqrt(3) * self.Rac + self.Xac,
                -np.sqrt(3) * self.Rbc + self.Xbc,
                -2 * self.Xcc,
            ],
        ]

        self.logger.debug(f"MP:\n {self.MP}")
        self.logger.debug(f"MQ:\n {self.MQ}")
        if not self.pu:
            self.logger.error(f"Line was NOT per-unitized !")

    def populate_capacity(self, Smax: float = 10000000000):
        """Populate the capacity of the line."""
        if self.pu:
            Smax = Smax / self.Sbase
        self.Smax = Smax

    def per_unitize(self, Vbase: float = None):

        if not self.pu:
            if Vbase is None:
                self.Vbase = self.Vnom
            else:
                self.Vbase = Vbase
            self.logger.info(
                f"Per Unitizing {self.__class__.__name__} {self.mRID} fr {self.from_bus} to {self.to_bus} with Vbase {self.Vbase}V"
            )

            self.Zbase = self.Vbase**2 / self.Sbase
            try:
                self.Smax = self.Smax / self.Sbase

                self.Raa = self.Raa / self.Zbase
                self.Rab = self.Rab / self.Zbase
                self.Rac = self.Rac / self.Zbase
                self.Rbb = self.Rbb / self.Zbase
                self.Rbc = self.Rbc / self.Zbase
                self.Rcc = self.Rcc / self.Zbase
                self.Xaa = self.Xaa / self.Zbase
                self.Xab = self.Xab / self.Zbase
                self.Xac = self.Xac / self.Zbase
                self.Xbb = self.Xbb / self.Zbase
                self.Xbc = self.Xbc / self.Zbase
                self.Xcc = self.Xcc / self.Zbase

                self.pu = True

            except Exception as e:
                self.logger.error(e)
        else:
            # implement change of Base ?
            pass

        return self.Vbase


class Switch3P(Line3P):
    def __init__(self, *args, **kw_args):
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        super().__init__(*args, **kw_args)

    def set_switch_status(self, switch_forecast: bool):
        """
        Method to set switch status True: switch is closed / False: switch is open
        """
        self.switch_closed = switch_forecast  # list indicating whether switch is closed [True] or open [False] at all time periods

    def get_switch_status(self) -> bool:
        return self.switch_closed


class Trafo3P(Line3P):
    def __init__(self, *args, **kw_args):
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        # Vnom for secondary
        self.Vnom2 = kw_args.get("Vnom2", None)
        super().__init__(*args, **kw_args)

    def per_unitize(self, Vbase: float):

        Vbase = super().per_unitize(Vbase=Vbase)

        if Vbase is None:
            Vbase2 = self.Vnom2
        else:
            Vbase2 = Vbase * self.Vnom2 / self.Vnom
        if Vbase2 != Vbase:
            self.logger.warning(
                f"Change of voltage base from {Vbase}V to {Vbase2}V downstream of Trafo {self.mRID}"
            )
        return Vbase2


class DER(CIMcomponent, Forecast):
    def __init__(
        self, bus_phase, gamma, S_max=None, Q_min=None, Q_max=None, *args, **kw_args
    ) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        # dictionnary of Bus ID / phase where the DER(s) resides
        self.bus_phase = bus_phase
        # Maximum apparent power [pu]
        self.S_max = S_max
        # Minimum Reactive Power (generator convention) [pu]
        self.Q_min = Q_min
        # Maximum Reactive Power (generator convention) [pu]
        self.Q_max = Q_max
        # Allocation percentage dictionnary for the aggregation, keys are buses
        # Check that it adds up to 1 and reallocate if necessary
        total_weight = sum(
            sum(gamma[busID][phase] for phase in [0, 1, 2]) for busID in gamma.keys()
        )
        if total_weight == 1:
            self.gamma = gamma
        else:
            self.gamma = {
                busID: [gamma[busID][phase] / total_weight for phase in [0, 1, 2]]
                for busID in gamma.keys()
            }

        additional_forecast_type = kw_args.pop("forecast_type", [])
        forecast_type = [
            "Pmax",
            "Pmin",
            "Pclear",
            "Qmin",
            "Qmax",
            "Emin",
            "Emax",
            "EVperc",
            "Del_Eev",
            "alpha",
        ] + additional_forecast_type

        super().__init__(forecast_type=forecast_type, *args, **kw_args)

    def set_DER_status(self, active, gamma):
        self.set_active_status(active)
        self.gamma = gamma

    def get_nodes(self):
        return [*self.bus_phase.keys()]


class Battery(DER):
    def __init__(
        self,
        eff_c: float = 1,
        eff_d: float = 1,
        P_min: float = 0,
        E_max: float = None,
        *args,
        **kw_args,
    ) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        # Charging Efficiency
        self.eff_c = eff_c
        # Discharging Efficiency
        self.eff_d = eff_d

        self.P_min = P_min

        self.E_max = E_max
        # Initializing the SOC to 0 by default
        self.E_0 = 0
        # Initializing the final SOC to None by default
        self.E_F = None
        super().__init__(*args, **kw_args)

    def set_init_SOC(self, SOC: float):
        self.logger.info(f"{self.mRID}: initial SOC {SOC} p.u.")
        self.E_0 = SOC

    def set_final_SOC(self, SOC: float):
        self.logger.info(f"{self.mRID}: final SOC {SOC} p.u.")
        self.E_F = SOC


class FlexibleLoad(Battery):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        super().__init__(eff_c=1, eff_d=1, *args, **kw_args)


class SmartChargeEV(Battery):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        super().__init__(eff_d=1, *args, **kw_args)


class TransactiveResource(DER):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        # Power Factor
        self.PF = kw_args.get("PF", 1)

        # Additional forecast types for TR
        forecast_type = ["TRpower", "TRprice"]

        super().__init__(forecast_type=forecast_type, *args, **kw_args)

    def get_number_pairs(self):
        # sample TRprice
        prices_sample = self.get_samples("TRprice")
        try:
            return len(prices_sample[0])
        except:
            # in case there are no scenarios
            return len(prices_sample)


class PV(DER):
    def __init__(self, P_min, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        # Minimum real power parameter (in generator convention) [pu]
        self.P_min = P_min
        super().__init__(*args, **kw_args)


class VPP(DER):
    def __init__(self, P_max, P_min, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        self.P_min = P_min  # This should be forecast
        self.P_max = P_max  # This should be forecast
        super().__init__(*args, **kw_args)


class Load(CIMcomponent, Forecast):
    def __init__(self, bus, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        # dictionnary of Bus ID / phase where the DER(s) resides
        self.set_nodes(bus=bus)
        forecast_type = ["PL", "QL"]
        super().__init__(forecast_type=forecast_type, *args, **kw_args)

    def get_nodes(self):
        return [*self.bus_phase.keys()]

    def get_nodes_phases(self, **kw_args):
        bus = kw_args.get("bus", None)
        if bus is not None:
            phases = self.bus_phase.get(bus, None)
            return {bus: phases}
        else:
            return self.bus_phase

    def set_nodes(self, **kw_args):

        bus = kw_args.get("bus", None)
        if bus is not None:
            self.bus_phase = {bus: [0, 1, 2]}

        bus_phase = kw_args.get("bus_phase", None)
        if bus_phase is not None:
            self.bus_phase = bus_phase

        # Check that each phase is only connected to one bus
        allowed_phases = [0, 1, 2]
        self.bus_phase = {
            bus: [
                phase
                for phase in self.bus_phase[bus]
                if not self.is_error(list.remove, allowed_phases, phase)
            ]
            for bus in self.bus_phase.keys()
        }


class CompositeResource(CIMcomponent, Forecast):
    def __init__(self, DER_ids, Load_ids, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        # For now we don't include any forecast type in the composite resource itself
        kw_args.update({"forecast_type": []})

        super().__init__(*args, **kw_args)

        # DERs
        self.DER_ids = DER_ids
        # Loads
        self.Load_ids = Load_ids

    def get_DER_ids(self):
        return self.DER_ids

    def get_Load_ids(self):
        return self.Load_ids


class Price(Forecast):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        forecast_type = ["pi", "pi_rup", "pi_rdn"]
        super().__init__(forecast_type=forecast_type, *args, **kw_args)
