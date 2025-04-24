"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from .IOstatic_data import IOstatic
from .IOgapps import IOgapps

from ..common import *
from ..Modeling import equipmentClasses as mod

from typing import Dict, List, Tuple, Union


class IObackup(IOgapps, IOstatic):

    def fetch_model(self, model_id: str, **kw_args) -> Dict:
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().fetch_model(model_id)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(f"Failed to fetch model {model_id} through gridappsd")
            # Falling back onto IOstatic
            return super(IOgapps, self).fetch_model(model_id)

    def get_substation_ID(self):
        try:
            # Trying the gapps version
            return super().get_substation_ID()
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_substation_ID()

    def build_DERs(self, **kw_args) -> List[mod.DER]:
        try:
            # Trying the gapps version
            return super().build_DERs(**kw_args)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).build_DERs(**kw_args)

    def build_Loads(self) -> List[mod.Load]:
        try:
            # Trying the gapps version
            return super().build_Loads()
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).build_Loads()

    def build_Composite_Resources(self) -> List[mod.CompositeResource]:
        try:
            # Trying the gapps version
            return super().build_Composite_Resources()
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).build_Composite_Resources()

    def get_switch_status(self, mRID) -> bool:
        try:
            # Trying the gapps version
            return super().get_switch_status(mRID)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_switch_status(mRID)

    def get_battery_SOC(self, mRID, **kw_args) -> float:
        """
        Returns the state of charge of the battery in Wh or p.u. if Sbase provided

        input:
            mRID: str
                mRID of the battery
            kw_args: dict
                additional arguments to be passed to the function
        output:
            SOC: float
        """

        try:
            # Trying the gapps version
            return super().get_battery_SOC(mRID, **kw_args)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_battery_SOC(mRID, **kw_args)

    def get_fcast_substation(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            # Trying the gapps version
            return super().get_fcast_substation(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_fcast_substation(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_fcast_switch_status(self, mRID, t0, n_timestep, timestep_period):
        try:
            # Trying the gapps version
            return super().get_fcast_switch_status(
                mRID, t0, n_timestep, timestep_period
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_fcast_switch_status(
                mRID, t0, n_timestep, timestep_period
            )

    def get_fcast_load(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_fcast_load(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_fcast_load(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_fcast_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_fcast_DER(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_fcast_DER(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_fcast_price(self, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_fcast_price(t0, n_timestep, timestep_period)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_fcast_price(t0, n_timestep, timestep_period)

    def get_DET_load(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_DET_load(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_DET_load(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_DET_DER(self, mRID, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_DET_DER(mRID, t0, n_timestep, timestep_period, **kw_args)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_DET_DER(
                mRID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_DET_price(self, t0, n_timestep, timestep_period, **kw_args):
        try:
            if kw_args.get("force_static", False):
                raise ForceStaticException()
            # Trying the gapps version
            return super().get_DET_price(t0, n_timestep, timestep_period, **kw_args)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_DET_price(
                t0, n_timestep, timestep_period, **kw_args
            )

    def get_DET_input_data(self, model_ID, t0, n_timestep, timestep_period, **kw_args):
        try:
            # Trying the gapps version
            return super().get_DET_input_data(
                model_ID, t0, n_timestep, timestep_period, **kw_args
            )
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_DET_input_data(
                model_ID, t0, n_timestep, timestep_period, **kw_args
            )

    def get_MPC_input_data(self, t0, n_timestep, timestep_period):
        try:
            # Trying the gapps version
            return super().get_MPC_input_data(t0, n_timestep, timestep_period)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).get_MPC_input_data(
                t0, n_timestep, timestep_period
            )

    def publish_DA_result(self, scenario_res: Dict) -> None:
        try:
            # Trying the gapps version
            return super().publish_DA_result(scenario_res)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).publish_DA_result(scenario_res)

    def publish_DET_result(self, deterministic_res: Dict) -> None:
        try:
            # Trying the gapps version
            return super().publish_DET_result(deterministic_res)
        except Exception as e:
            self.logger.error(e)
            # Falling back onto IOstatic
            return super(IOgapps, self).publish_DET_result(deterministic_res)
