"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..common import *
from ..Modeling import equipmentClasses as mod

from pathlib import Path
from typing import List, Dict
from itertools import chain

import PyCIM as cim
import numpy as np

import pickle


class IOmodule(fastderms):
    def __init__(self, *args, model_id=None, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})

        self._model_id = model_id

        self.path_to_archive = Path(
            kw_args.get("path_to_archive", Path(".") / "archive")
        )

        if not self.path_to_archive.exists():
            self.path_to_archive.mkdir()

        super().__init__(*args, **kw_args)
        self.logger.warning("IOmodule Initialized")

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()
        else:
            return object

    def get_model_id(self) -> str:
        return self._model_id

    def per_unitize(self, value_in, **kw_args) -> float:
        # Convert to per unit
        base = kw_args.get("base", None)
        return_base = kw_args.get("return_base", False)

        if value_in is None:
            value_out = None
        elif type(value_in) == list:
            value_out = [
                self.per_unitize(val, base=base, return_base=False) for val in value_in
            ]
        elif base is not None:
            value_out = value_in / base
        else:
            value_out = value_in

        if return_base:
            return value_out, base
        else:
            return value_out

    def load_model(self, source) -> None:
        try:
            if not source:
                raise Import_Exception("Path to model is empty!")

            import_result = cim.cimread(source)

        except Import_Exception as err:
            self.logger.exception(err)
            import_result = None

        except Exception as err:
            self.logger.exception(err)
            import_result = None
            raise err

        finally:
            if not import_result:
                self.logger.error("Failed to import CIM model")

            self.import_result = import_result

            return import_result

    def build_Lines(self, **kw_args) -> List[mod.Line3P]:

        Sbase = kw_args.get("Sbase", 1)

        # list of Devices that will become 'lines'
        AClines = ["ACLineSegment"]
        switches = ["Recloser", "Fuse", "Breaker", "LoadBreakSwitch"]
        trafos = ["PowerTransformer"]

        # list of CIM objects
        ACline_list = []
        Switch_list = []
        Trafo_list = []

        for key, value in self.import_result.items():
            class_name = value.__class__.__name__
            if class_name in AClines:
                ACline_list.append(key)
            elif class_name in switches:
                Switch_list.append(key)
            elif class_name in trafos:
                Trafo_list.append(key)

        AClines = [
            line
            for line in (self.parse_ACline(ACline, Sbase) for ACline in ACline_list)
            if line is not None
        ]
        switches = [
            line
            for line in (self.parse_switch(switch, Sbase) for switch in Switch_list)
            if line is not None
        ]
        trafos = [
            line
            for line in (self.parse_trafo(trafo, Sbase) for trafo in Trafo_list)
            if line is not None
        ]
        lines = AClines + switches + trafos

        return lines

    def parse_ACline(self, lineID: mRID, Sbase) -> mod.Line3P:

        new_line = None

        try:
            AClineSegment = self.import_result[lineID]

            self.logger.info(f"Parsing Line {lineID}")

            # Buses are extracted from connectivity node attached to the terminals of the line
            for terminal in AClineSegment.Terminals:
                if terminal.sequenceNumber == 1:
                    from_bus = terminal.ConnectivityNode.name
                elif terminal.sequenceNumber == 2:
                    to_bus = terminal.ConnectivityNode.name

            # Length of the line
            length = AClineSegment.length

            # Base Voltage
            Vnom = AClineSegment.BaseVoltage.nominalVoltage

            # Extract the line data
            phases = [x.phase for x in AClineSegment.ACLineSegmentPhases]
            phases = self.merge_phases(phases)

            if all(x in phases for x in ["s1", "s2"]):
                # secondary of splitphase trafo, we don't implement it
                raise Implementation_Exception(
                    f"Phases are {phases}: Line on secondary of splitphase trafo; not needed"
                )

            nbConductor = AClineSegment.PerLengthImpedance.conductorCount

            R_mat_in = np.zeros((nbConductor, nbConductor))
            X_mat_in = np.zeros((nbConductor, nbConductor))

            # Collect the perlength impedance Data
            for (
                PhaseImpedanceData
            ) in AClineSegment.PerLengthImpedance.PhaseImpedanceData:
                row = PhaseImpedanceData.row - 1
                col = PhaseImpedanceData.column - 1

                r = PhaseImpedanceData.r
                x = PhaseImpedanceData.x

                # populate the pre formatted 2D array
                R_mat_in[row, col] = r * length
                X_mat_in[row, col] = x * length

            R_mat = np.zeros((3, 3))
            X_mat = np.zeros((3, 3))

            # Depending on the nb of conductor and phase present
            if nbConductor == 3:
                R_mat = np.transpose(R_mat_in)
                X_mat = np.transpose(X_mat_in)

            elif nbConductor == 2:
                try:
                    if all(x in phases for x in [0, 2]):
                        R_mat[0, 0] = R_mat_in[0, 0]
                        R_mat[0, 2] = R_mat_in[1, 0]
                        R_mat[2, 0] = R_mat_in[0, 1]
                        R_mat[2, 2] = R_mat_in[1, 1]

                        X_mat[0, 0] = X_mat_in[0, 0]
                        X_mat[0, 2] = X_mat_in[1, 0]
                        X_mat[2, 0] = X_mat_in[0, 1]
                        X_mat[2, 2] = X_mat_in[1, 1]

                    else:
                        R_mat[phases[0] : phases[1] + 1, phases[0] : phases[1] + 1] = (
                            np.transpose(R_mat_in)
                        )
                        X_mat[phases[0] : phases[1] + 1, phases[0] : phases[1] + 1] = (
                            np.transpose(X_mat_in)
                        )
                except Exception as e:
                    self.logger.error(f"Could not find {phases}")
                    self.logger.error(e)

            elif nbConductor == 1:
                try:
                    phase = phases[0]
                    R_mat[phase, phase] = R_mat_in[0][0]
                    X_mat[phase, phase] = X_mat_in[0][0]
                except Exception as e:
                    self.logger.error(f"Could not find {phases}")
                    self.logger.error(e)
            else:
                self.logger.error(
                    f"Number of Conductor is not supported: {nbConductor}"
                )
                pass

            R_mat = R_mat.tolist()
            X_mat = X_mat.tolist()

            # Create new line and add to network
            new_line = mod.Line3P(
                from_bus,
                to_bus,
                phases,
                R_mat,
                X_mat,
                Vnom,
                Sbase,
                active=True,
                mRID=lineID,
                name="Line",
                tz=self.local_tz,
            )

        except Implementation_Exception as e:
            self.logger.warning(e)
            pass

        except Exception as e:
            self.logger.error(f"Error in parsing Line {lineID}")
            self.logger.error(e)

        finally:
            return new_line

    def parse_switch(self, switchID: mRID, Sbase) -> mod.Line3P:
        new_line = None

        try:
            Switch = self.import_result[switchID]
            self.logger.info(f"Parsing Switch {switchID}")

            phases = []
            # Buses are extracted from connectivity node attached to the terminals of the line
            for terminal in Switch.Terminals:
                if terminal.sequenceNumber == 1:
                    from_bus = terminal.ConnectivityNode.name
                    phases.append(terminal.phases)
                elif terminal.sequenceNumber == 2:
                    to_bus = terminal.ConnectivityNode.name

            # Base Voltage
            Vnom = Switch.BaseVoltage.nominalVoltage

            # Initialize the data as 0s
            R_mat = np.zeros((3, 3))
            X_mat = np.zeros((3, 3))

            # Merge phases and create line
            phases = self.merge_phases(phases)

            new_line = mod.Switch3P(
                from_bus,
                to_bus,
                phases,
                R_mat,
                X_mat,
                Vnom,
                Sbase,
                active=True,
                mRID=switchID,
                name="Switch",
                tz=self.local_tz,
            )

        except Exception as e:
            self.logger.error(f"Error in parsing Switch {switchID}")
            self.logger.error(e)

        finally:
            return new_line

    def parse_trafo(self, trafoID: mRID, Sbase) -> List[mod.Line3P]:

        try:
            Trafo = self.import_result[trafoID]
            self.logger.info(f"Parsing Trafo {trafoID}")

            # Buses are extracted from the connectivity nodes associated to the terminals
            # bus_list = []
            # for terminal in Trafo.Terminals:
            #    bus_list.append(terminal.ConnectivityNode.name)

            # /!\ bus_list values arenot unique, let's correct that
            # bus_list = list(set(bus_list))

            # line_list = list(combinations(bus_list,2))

            if len(Trafo.TransformerTanks) > 0:

                try:
                    # Initialize the data as 0s
                    R_mat = np.zeros((3, 3))
                    X_mat = np.zeros((3, 3))

                    phases = []
                    Vnom = [0, 0]

                    for tank in Trafo.TransformerTanks:
                        nb_ends = len(tank.TransformerTankEnds)

                        if nb_ends < 2:
                            raise Implementation_Exception(
                                f"Number of ends is not supported: {nb_ends}"
                            )
                        elif nb_ends > 3:
                            raise Implementation_Exception(
                                f"Number of ends is not supported: {nb_ends}"
                            )
                        else:
                            pass

                        tank_phases = self.merge_phases(
                            [end.phases for end in tank.TransformerTankEnds]
                        )

                        if all(x in tank_phases for x in ["s1", "s2"]):
                            # secondary of splitphase trafo, we don't implement it
                            raise Implementation_Exception(
                                f"Phases are {tank_phases}: Splitphase trafo; not needed"
                            )

                        for end in tank.TransformerTankEnds:

                            if end.endNumber == 1:
                                from_bus = end.Terminal.ConnectivityNode.name
                                Vnom[0] = end.BaseVoltage.nominalVoltage
                            elif end.endNumber == 2:
                                to_bus = end.Terminal.ConnectivityNode.name
                                Vnom[1] = end.BaseVoltage.nominalVoltage

                        try:
                            if 0 in tank_phases:
                                R_mat[0][0] = (
                                    tank.Assets[0].AssetInfo.TransformerEndInfos[0].r
                                )
                                X_mat[0][0] = 0
                            elif 1 in tank_phases:
                                R_mat[1][1] = (
                                    tank.Assets[0].AssetInfo.TransformerEndInfos[0].r
                                )
                                X_mat[1][1] = 0
                            elif 2 in tank_phases:
                                R_mat[2][2] = (
                                    tank.Assets[0].AssetInfo.TransformerEndInfos[0].r
                                )
                                X_mat[2][2] = 0
                            else:
                                self.logger.error(
                                    f"Could not find phases {tank_phases}"
                                )
                        except:
                            for info in tank.TransformerTankInfo.TransformerEndInfos:
                                indice = info.endNumber - 1
                                R_mat[indice][indice] = info.r
                                X_mat[indice][indice] = 0

                        phases.append(tank_phases)

                    # Merge the phases and create the line
                    phases = self.merge_phases(phases)
                    new_line = mod.Trafo3P(
                        from_bus,
                        to_bus,
                        phases,
                        R_mat,
                        X_mat,
                        Vnom=Vnom[0],
                        Vnom2=Vnom[1],
                        Sbase=Sbase,
                        active=True,
                        mRID=trafoID,
                        name="Trafo",
                        tz=self.local_tz,
                    )

                except Exception as e:
                    self.logger.error(f"Error in parsing Trafo {trafoID}")
                    self.logger.error(e)
                    new_line = None
                    raise (e)

            elif len(Trafo.PowerTransformerEnd) > 0:
                # Simple transformer model

                for PTend in Trafo.PowerTransformerEnd:
                    Vnom = []
                    from_bus = PTend.Terminal.ConnectivityNode.name
                    Vnom.append(PTend.BaseVoltage.nominalVoltage)

                    phases = PTend.Terminal.phases
                    phases = self.merge_phases(phases)

                    for meshImpedance in PTend.FromMeshImpedance:
                        to_bus = meshImpedance.ToTransformerEnd[
                            0
                        ].Terminal.ConnectivityNode.name
                        Vnom.append(
                            meshImpedance.ToTransformerEnd[0].BaseVoltage.nominalVoltage
                        )

                        # Initialize the data as 0s
                        R_mat = np.zeros((3, 3))
                        X_mat = np.zeros((3, 3))

                        R_mat[0][0] = meshImpedance.r
                        R_mat[1][1] = meshImpedance.r
                        R_mat[2][2] = meshImpedance.r

                        X_mat[0][0] = meshImpedance.x
                        X_mat[1][1] = meshImpedance.x
                        X_mat[2][2] = meshImpedance.x

                        new_line = mod.Trafo3P(
                            from_bus,
                            to_bus,
                            phases,
                            R_mat,
                            X_mat,
                            Vnom=Vnom[0],
                            Vnom2=Vnom[1],
                            Sbase=Sbase,
                            active=True,
                            mRID=trafoID,
                            name="Trafo",
                            tz=self.local_tz,
                        )
            else:
                raise (
                    Implementation_Exception(
                        "Trafo with no TransformerTanks or PowerTransformerEnd: Not yet implemented"
                    )
                )

        except Exception as e:
            self.logger.error(f"Error in parsing Trafo {trafoID}")
            self.logger.error(e)
            new_line = None

        finally:
            return new_line

    def loadDERs(self) -> List[mod.DER]:
        DERs = []
        return DERs

    def loadLoads(self) -> List[mod.Load]:
        Loads = []
        return Loads

    def process_fcast(
        self,
        fcast_in,
        t0: dt.datetime,
        n_timestep: int,
        timestep_period: int,
        **kwargs,
    ):
        t0_fcast = fcast_in.pop("t0")
        timestep_period_fcast = fcast_in.pop("timestep_period")

        # Extend incoming Fcast to cover 24h
        n_timesteps_fcast = int(24 * 60 / timestep_period_fcast)
        self.logger.info(
            f"Enforcing incoming forecast length of 24h ({n_timesteps_fcast} timesteps)"
        )
        fcast_in = self.extend_fcast(fcast_in, n_timesteps_fcast)

        self.logger.info(
            f"Processing to match requested t0: {t0} and period = {timestep_period} min."
        )
        # Resample Fcast
        if timestep_period_fcast != timestep_period:
            self.logger.info(
                f"Resampling forecast period from {timestep_period_fcast} min. to {timestep_period} min."
            )
            for forecastType in fcast_in.keys():
                # if forecastType not in ['t0', 'timestep_period']:
                if forecastType == "Del_Eev":
                    method = "even"
                elif forecastType == "alpha":
                    method = "power"
                elif forecastType == "E_All":
                    method = "interpolate"
                else:
                    method = ""
                # overriding method if specified in kwargs
                method = kwargs.get("method", method)
                fcast_in[forecastType] = self.resample_fcast(
                    fcast_in[forecastType],
                    timestep_period_fcast,
                    timestep_period,
                    method,
                )
        else:
            self.logger.info(f"Matching forecast timestep period")

        # Check t0 align
        if t0.tzinfo is None:
            self.logger.warning(
                f"No timezone information provided, assuming local timezone {self.local_tz}"
            )
            t0 = self.local_tz.localize(t0)
        if t0_fcast != t0:
            self.logger.info(f"Updating t0 from {t0_fcast} to {t0}")
            n_skip = int((t0 - t0_fcast).total_seconds() / 60 / timestep_period)
            if n_skip < 0:
                self.logger.critical(
                    "Requested t0 is before the start time of the forecast data available to us, we will assume the forecast starts at requested t0"
                )
                n_skip = 0
        else:
            self.logger.info(f"Matching start time")
            n_skip = 0

        self.logger.debug(f"Skipping {n_skip} timesteps")

        # Select correct samples
        fcast_out = {}
        for forecastType in fcast_in.keys():
            # if forecastType not in ['t0', 'timestep_period']:
            try:
                fcast_out[forecastType] = {}
                for forecast_subtype in fcast_in[forecastType].keys():
                    self.logger.debug(
                        f"Subtype {forecast_subtype} for {forecastType}, selecting [{n_skip}:{n_skip+n_timestep}]"
                    )
                    fcast_out[forecastType][forecast_subtype] = fcast_in[forecastType][
                        forecast_subtype
                    ][n_skip : n_skip + n_timestep]

            except:
                self.logger.debug(
                    f"No subtypes for {forecastType}, selecting [{n_skip}:{n_skip+n_timestep}]"
                )
                fcast_out[forecastType] = fcast_in[forecastType][
                    n_skip : n_skip + n_timestep
                ]

        fcast_out = self.extend_fcast(fcast_out, n_timestep)
        # Add t0 and timestep_period
        fcast_out["t0"] = t0
        fcast_out["timestep_period"] = timestep_period
        return fcast_out

    def resample_fcast(
        self, fcast_in, deltaT_in, deltaT_out, method: str = "interpolate"
    ):
        """
        Resample data to deltaT minutes.
        """
        if deltaT_in == deltaT_out:
            self.logger.info("No resampling required")
            return fcast_in
        else:
            try:
                fcast_out = {}
                for fcast_subtype in fcast_in.keys():
                    self.logger.debug(f"Resampling {fcast_subtype}")
                    fcast_out[fcast_subtype] = self.resample_fcast(
                        fcast_in[fcast_subtype], deltaT_in, deltaT_out, method
                    )
                return fcast_out

            except:
                samples = fcast_in
                if deltaT_in > deltaT_out:
                    nb_intervals = int(deltaT_in / deltaT_out)
                    if method == "":
                        method = "default - ffill"
                    self.logger.debug(
                        f"Resampling with method: {method} and the incoming data: \n{samples}"
                    )
                    if method == "even":
                        new_samples = [
                            sample / nb_intervals
                            for sample in samples
                            for interval in range(nb_intervals)
                        ]
                    elif method == "power":
                        new_samples = [
                            sample ** (1 / nb_intervals)
                            for sample in samples
                            for interval in range(nb_intervals)
                        ]
                    elif method == "interpolate":
                        new_samples = np.concatenate(
                            [
                                np.linspace(sample, sample_1, nb_intervals)
                                for sample, sample_1 in zip(samples, samples[1:])
                            ]
                        )
                    else:
                        new_samples = [
                            sample
                            for sample in samples
                            for interval in range(nb_intervals)
                        ]
                else:
                    # Case where we need to downsample
                    n_dn_sampling = int(deltaT_out / deltaT_in)
                    self.logger.debug(
                        f"Downsampling by factor {n_dn_sampling} and the incoming data: \n{samples}"
                    )
                    samples = np.array(samples).reshape(-1, n_dn_sampling)
                    new_samples = np.nanmean(samples, axis=1)
                self.logger.debug(f"Resampled data:\n {new_samples}")

            return new_samples

    def extend_fcast(self, fcast_in, n_timestep: int, **kwargs):
        """
        Extend forecast to n_timestep
        """
        try:
            fcast_out = {}
            for fcast_subtype in fcast_in.keys():
                self.logger.debug(f"Checking {fcast_subtype}")
                fcast_out[fcast_subtype] = self.extend_fcast(
                    fcast_in[fcast_subtype], n_timestep
                )
            return fcast_out
        except:
            if len(fcast_in) == n_timestep:
                self.logger.debug("No extension required")
                return fcast_in
            else:
                if len(fcast_in) > n_timestep:
                    self.logger.info(
                        f"Trimming forecast from {len(fcast_in)} to {n_timestep} timesteps"
                    )
                    fcast_out = fcast_in[:n_timestep]
                else:
                    self.logger.info(
                        f"Extending forecast from {len(fcast_in)} to {n_timestep} timesteps"
                    )
                    last_value = fcast_in[-1]
                    fcast_out = fcast_in + [last_value] * (n_timestep - len(fcast_in))
                    fcast_out = fcast_out[:n_timestep]
                return fcast_out

    def publish_results(self, results: Dict, **kw_args) -> None:

        filename = kw_args.get("filename", "results.pkl")
        override_path = kw_args.get("override_path", False)
        archive = kw_args.get("archive", False)
        apdx = kw_args.get("apdx", "")

        if override_path:
            path_to_file = Path(filename)
        else:
            path_to_file = self.path_to_export / filename

        try:
            with open(str(path_to_file.absolute()), "wb") as outfile:
                pickle.dump(results, outfile)
        except Exception as e:
            self.logger.error(
                f"Error in publishing results to {path_to_file.absolute()}"
            )
            self.logger.error(e)

        if archive:
            try:
                if apdx == "":
                    path_to_archive = self.path_to_archive / filename
                else:
                    path_to_archive = self.path_to_archive / (
                        path_to_file.stem + "_" + apdx + path_to_file.suffix
                    )
                with open(str(path_to_archive.absolute()), "wb") as outfile:
                    pickle.dump(results, outfile)
            except Exception as e:
                self.logger.error(
                    f"Error in archiving results to {path_to_archive.absolute()}"
                )
                self.logger.error(e)

    def load_results(self, filename: str, override_path: bool = False) -> Dict:
        if override_path:
            path_to_file = Path(filename)
        else:
            path_to_file = self.path_to_export / filename

        self.logger.debug(f"Loading results from {path_to_file.absolute()}")

        with open(str(path_to_file.absolute()), "rb") as infile:
            results = pickle.load(infile)
        return results
