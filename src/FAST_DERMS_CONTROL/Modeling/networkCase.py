"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from typing import List
from FAST_DERMS_CONTROL.common import fastderms
from .network import Network

import copy as cp
import datetime as dt

dt_0 = dt.datetime(2020, 1, 1, 0, 0, 0)


class NetworkCase(fastderms):
    def __init__(
        self,
        Network: Network,
        n_timestep: int = 24,
        timestep_period: int = 60,
        t0: dt.datetime = dt_0,
        n_scenario: int = 10,
        *args,
        **kw_args,
    ) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        self.Network = Network
        self.n_timestep = n_timestep
        self.timestep_period = timestep_period
        self.n_scenario = n_scenario
        self.t0 = t0
        self.horizon = [cp.deepcopy(Network) for n in range(n_timestep)]

        super().__init__(*args, **kw_args)

        self.logger.warning(f"Case initialized")

    def set_case_settings(self, **kw_args):
        """
        Set case setting(s)
        """

        t0 = kw_args.get("t0", None)
        if t0 is not None:
            self.t0 = t0

        n_scenario = kw_args.get("n_scenario", None)
        if n_scenario is not None:
            self.n_scenario = n_scenario

        timestep_period = kw_args.get("timestep_period", None)
        if timestep_period is not None:
            self.timestep_period = timestep_period

        n_timestep = kw_args.get("n_timestep", None)
        if n_timestep is not None:
            self.n_timestep = n_timestep

            if len(self.horizon) >= n_timestep:
                self.horizon = self.horizon[:n_timestep]
            else:
                horizon = [cp.deepcopy(Network) for n in range(n_timestep)]
                horizon[: len(self.horizon)] = self.horizon
                self.horizon = horizon

    def get_case_settings(self, **kw_args):

        setting = kw_args.get("setting", None)

        settings = ["n_timestep", "timestep_period", "n_scenario", "t0"]

        if setting in settings:
            return getattr(self, setting)
        else:
            case_settings = {setting: getattr(self, setting) for setting in settings}
            return case_settings

    def get_network(self, frame_idx: int = None):
        if frame_idx is None:
            Network = self.Network
        else:
            Network = self.horizon[frame_idx]
        return Network

    def get_horizon(self, func, *args, frame_idx: int = None, **kw_args):
        try:
            if frame_idx is None:
                all_results = [func(frame, *args, **kw_args) for frame in self.horizon]
                return all_results
            else:
                result = func(self.horizon[frame_idx], *args, **kw_args)
                return result
        except:
            raise

    def set_horizon(self, func, *args, values=None, frame_idx: int = None, **kw_args):
        try:
            if frame_idx == None:
                if values is None:
                    for frame in self.horizon:
                        func(frame, *args, **kw_args)
                else:
                    for value, frame in zip(values, self.horizon):
                        func(frame, *args, value, **kw_args)

            else:
                func(self.horizon[frame_idx], *args, values, **kw_args)
        except:
            raise

    def get_horizon_asset(self, obj_type: str, mRID=None, frame_idx: int = None):
        def get_asset(Network: Network, func, mRID):
            if mRID is None:
                # Prices
                current_obj = func(Network)
            else:
                current_obj = func(Network, mRIDs=[mRID])[0]
            return current_obj

        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            horizon_asset = self.get_horizon(get_asset, func, mRID, frame_idx=frame_idx)

        except Exception as e:
            self.logger.error(e)
            horizon_asset = []
        finally:
            return horizon_asset

    def read_horizon_fcast(
        self, obj_type: str, mRID, fcast_type=None, frame_idx: int = None
    ):

        def read_fcast(Network: Network, func, mRID, fcast_type=None):
            if mRID is None:
                # Prices
                current_obj = func(Network)
            else:
                current_obj = func(Network, mRIDs=[mRID])[0]
            fcast = current_obj.get_forecast(fcast_type)
            return fcast

        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            elif obj_type == "price":
                func = Network.get_prices
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            horizon_fcast = self.get_horizon(
                read_fcast, func, mRID, fcast_type, frame_idx=frame_idx
            )

        except Exception as e:
            self.logger.error(e)
            horizon_fcast = []

        finally:
            return horizon_fcast

    def write_horizon_fcast(self, obj_type: str, mRID, fcast_type, values):

        def write_fcast(Network: Network, func, mRID, fcast_type, value):
            if mRID is None:
                # Prices
                current_obj = func(Network)
            else:
                current_obj = func(Network, mRIDs=[mRID])[0]
            current_obj.set_forecast(fcast_type, value)

        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            elif obj_type == "price":
                func = Network.get_prices
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            self.set_horizon(write_fcast, func, mRID, fcast_type, values=values)

        except Exception as e:
            self.logger.error(e)

    def read_horizon_samples(
        self,
        obj_type: str,
        mRID,
        fcast_type=None,
        frame_idx: int = None,
        scenario_nr: int = None,
    ):

        def read_samples(
            Network: Network, func, mRID, fcast_type=None, scenario_nr: int = None
        ):
            if mRID is None:
                # Prices
                current_obj = func(Network)
            else:
                current_obj = func(Network, mRIDs=[mRID])[0]
            samples = current_obj.get_samples(fcast_type, scenario_nr)
            return samples

        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            elif obj_type == "price":
                func = Network.get_prices
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            horizon_samples = self.get_horizon(
                read_samples,
                func,
                mRID,
                fcast_type,
                frame_idx=frame_idx,
                scenario_nr=scenario_nr,
            )

        except Exception as e:
            self.logger.error(e)
            horizon_samples = []

        finally:
            return horizon_samples

    def write_horizon_samples(self, obj_type: str, mRID, fcast_type, values):

        def write_samples(Network: Network, func, mRID, fcast_type, value):
            if mRID is None:
                # Prices
                current_obj = func(Network)
            else:
                current_obj = func(Network, mRIDs=[mRID])[0]
            current_obj.set_samples(fcast_type, value)

        try:
            if obj_type == "load":
                func = Network.get_loads
            elif obj_type == "DER":
                func = Network.get_DERs
            elif obj_type == "price":
                func = Network.get_prices
            else:
                raise Exception(f"Object type {obj_type} not recognized")

            self.set_horizon(write_samples, func, mRID, fcast_type, values=values)

        except Exception as e:
            self.logger.error(e)
