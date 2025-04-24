"""
Part of the FAST-DERMS Flexible Resource Scheduler
"""

from ..common import fastderms
from ..OPT.Constraints.DERs import TR_ComputePrices
from ..OPT.Rules.Objective import (
    power_cost,
    transactive_cost,
    loss_cost,
    reserve_cost,
    reverse_pf_cost,
    substation_deviation_cost,
    reserves_deviation_cost,
    battery_deviation_cost,
    battery_charge_discharge_cost,
    PV_curtailment_cost,
)

import pyomo.environ as pe
import numpy as np


class OptEngine(fastderms):
    def __init__(self, *args, **kw_args) -> None:
        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", __name__)})
        super().__init__(*args, **kw_args)

    def export_full_pyomo_data(self, model: pe.Model, **kw_args):
        try:
            # Get options out of kw_args
            power_weight = kw_args.get("power_weight", 0)
            transactive_weight = kw_args.get("transactive_weight", 0)
            reserve_weight = kw_args.get("reserve_weight", 0)
            loss_weight = kw_args.get("loss_weight", 0)
            reverse_pf_weight = kw_args.get("reverse_pf_weight", None)
            substation_deviation_weight = kw_args.get("substation_deviation_weight", 0)
            substation_deviation_price_override = kw_args.get(
                "substation_deviation_price_override", None
            )
            reserves_deviation_weight = kw_args.get("reserves_deviation_weight", 0)
            reserves_deviation_price_override = kw_args.get(
                "reserves_deviation_price_override", None
            )
            pv_curtailment_weight = kw_args.get("pv_curtailment_weight", 0)
            battery_charge_discharge_weight = kw_args.get(
                "battery_charge_discharge_weight", 0
            )
            battery_deviation_weight = kw_args.get("battery_deviation_weight", 0)

            Net_case = pe.value(model.networkCase)
            # P0 same over all scenarios
            substation_ID = Net_case.Network.get_substation_ID()
            P0_set = [
                [
                    pe.value(model.Pkt[substation_ID, phi, frame_idx, "sched", 1])
                    for phi in model.Phases
                ]
                for frame_idx in model.T
            ]
            Rup_set = [pe.value(model.Rup[t]) for t in model.T]
            Rdn_set = [pe.value(model.Rdn[t]) for t in model.T]
            MainResDict = {
                "Substation_Power": P0_set,
                "Reserves_Up": Rup_set,
                "Reserves_Down": Rdn_set,
            }

            # Collecting Nodal Flows and Voltage
            # Prepare indices
            nodes = np.unique([pkt_ind[0] for pkt_ind in model.Pkt.keys()])
            phases = [ph for ph in model.Phases]
            frames = [t for t in model.T]
            cases = [case for case in model.cases]
            scens = [scen for scen in model.Scenarios]
            # Collecting DER Setpoints
            PVs = [PV for PV in model.PV_list]
            Batts = [Batt for Batt in model.Battery_list]
            TRs = [TR for TR in model.TR_list]

            IndexDictPF = {
                "nodes": nodes,
                "phases": phases,
                "frames": frames,
                "cases": cases,
                "scens": scens,
            }
            # indexList = [(node,ph,frame,case,scen) for node in nodes for ph in phases for frame in frames for case in cases for scen in scens]
            # Collect Data

            PFDict = {}
            for scen in scens:
                for node in nodes:
                    for ph in phases:
                        PCaseDict = {}
                        YCaseDict = {}
                        for case in cases:
                            PCaseDict = {
                                **PCaseDict,
                                case: [
                                    pe.value(model.Pkt[(node, ph, frame, case, scen)])
                                    for frame in frames
                                ],
                            }
                            YCaseDict = {
                                **YCaseDict,
                                case: [
                                    pe.value(model.Ykt[(node, ph, frame, case, scen)])
                                    for frame in frames
                                ],
                            }
                        PFDict = {
                            **PFDict,
                            (scen, node, ph): {
                                "P": PCaseDict,
                                "Q": [
                                    pe.value(model.Qkt[(node, ph, frame, scen)])
                                    for frame in frames
                                ],
                                "PL": [
                                    pe.value(model.PL[(node, ph, frame, scen)])
                                    for frame in frames
                                ],
                                "Y": YCaseDict,
                            },
                        }
            # ETA variable (Soft Constraints)
            EtaDict = {}
            try:
                EtaDict["eta"] = {
                    (scen, ph, case): [
                        pe.value(model.eta[(ph, frame, case, scen)]) for frame in frames
                    ]
                    for scen in scens
                    for ph in phases
                    for case in np.unique([val[2] for val in model.eta.keys()])
                }
            except Exception as err:
                self.logger.error(f"Error in collecting eta: {err}")

            try:
                EtaDict["eta_up_reserves"] = [
                    pe.value(model.eta_up_reserves[frame]) for frame in frames
                ]
            except Exception as err:
                self.logger.error(f"Error in collecting eta up reserves: {err}")

            try:
                EtaDict["eta_dn_reserves"] = [
                    pe.value(model.eta_dn_reserves[frame]) for frame in frames
                ]
            except Exception as err:
                self.logger.error(f"Error in collecting eta dn reserves: {err}")

            try:
                EtaDict["eta_energy"] = [
                    pe.value(model.eta_energy[derID]) for derID in Batts
                ]
            except Exception as err:
                self.logger.error(f"Error in collecting eta batteries: {err}")

            # Objective Subcomponents
            ObjSubsDict = {}
            try:
                if power_weight != 0:
                    ObjSubsDict["power_cost"] = {
                        scen: [
                            pe.value(
                                power_weight
                                * power_cost(model, Net_case, frame_idx, scen)
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting power_cost: {err}")
            try:
                if transactive_weight != 0:
                    ObjSubsDict["transactive_cost"] = {
                        scen: [
                            pe.value(
                                transactive_weight
                                * transactive_cost(model, Net_case, frame_idx, scen)
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting transactive_cost: {err}")

            try:
                if loss_weight != 0:
                    ObjSubsDict["loss_cost"] = {
                        scen: [
                            pe.value(
                                loss_weight
                                * loss_cost(model, Net_case, frame_idx, scen)
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting loss_cost: {err}")

            try:
                if reserve_weight != 0:
                    ObjSubsDict["reserve_cost"] = {
                        scen: [
                            pe.value(
                                reserve_weight
                                * reserve_cost(model, Net_case, frame_idx, scen)
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting reserve_cost: {err}")

            try:
                if not (reverse_pf_weight is None or reverse_pf_weight == 0):
                    ObjSubsDict["reverse_pf_cost"] = {
                        scen: [
                            pe.value(
                                reverse_pf_weight
                                * reverse_pf_cost(model, Net_case, frame_idx, scen)
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting reverse_pf_cost: {err}")

            try:
                if substation_deviation_weight != 0:
                    try:
                        ObjSubsDict["substation_deviation_cost"] = {
                            scen: [
                                substation_deviation_weight
                                * pe.value(
                                    substation_deviation_cost(
                                        model,
                                        frame_idx,
                                        scen,
                                        cases,
                                        substation_deviation_price_override,
                                    )
                                )
                                for frame_idx in frames
                            ]
                            for scen in scens
                        }
                    except:
                        ObjSubsDict["substation_deviation_cost"] = {
                            scen: [
                                substation_deviation_weight
                                * pe.value(
                                    substation_deviation_cost(
                                        model,
                                        frame_idx,
                                        scen,
                                        ["sched"],
                                        substation_deviation_price_override,
                                    )
                                )
                                for frame_idx in frames
                            ]
                            for scen in scens
                        }
            except Exception as err:
                self.logger.error(
                    f"Error in collecting substation_deviation_cost: {err}"
                )

            try:
                if reserves_deviation_weight != 0:
                    ObjSubsDict["reserves_deviation_cost"] = {
                        scen: [
                            reserves_deviation_weight
                            * pe.value(
                                reserves_deviation_cost(
                                    model,
                                    frame_idx,
                                    scen,
                                    reserves_deviation_price_override,
                                )
                            )
                            for frame_idx in frames
                        ]
                        for scen in scens
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting reserves_deviation_cost: {err}")

            try:
                if battery_deviation_weight != 0:
                    ObjSubsDict["battery_deviation_cost"] = {
                        derID: pe.value(
                            battery_deviation_weight
                            * battery_deviation_cost(model, derID)
                        )
                        for derID in Batts
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting battery_deviation_cost: {err}")

            try:
                if battery_charge_discharge_weight != 0:
                    ObjSubsDict["battery_charge_discharge_cost"] = {
                        derID: {
                            scen: [
                                pe.value(
                                    battery_charge_discharge_weight
                                    * battery_charge_discharge_cost(
                                        model, derID, frame_idx, scen
                                    )
                                )
                                for frame_idx in frames
                            ]
                            for scen in scens
                        }
                        for derID in Batts
                    }
            except Exception as err:
                self.logger.error(
                    f"Error in collecting battery_charge_discharge_cost: {err}"
                )

            try:
                if pv_curtailment_weight != 0:
                    ObjSubsDict["PV_curtailment_cost"] = {
                        derID: {
                            scen: [
                                pe.value(
                                    pv_curtailment_weight
                                    * PV_curtailment_cost(model, derID, frame_idx, scen)
                                )
                                for frame_idx in frames
                            ]
                            for scen in scens
                        }
                        for derID in PVs
                    }
            except Exception as err:
                self.logger.error(f"Error in collecting PV_curtailment_cost: {err}")

            PiDict = {}
            PiDict["pi"] = {
                scen: [
                    Net_case.read_horizon_samples(
                        "price", None, "pi", frame_idx=frame - 1, scenario_nr=scen - 1
                    )
                    for frame in frames
                ]
                for scen in scens
            }
            PiDict["pi_rup"] = {
                scen: [
                    Net_case.read_horizon_samples(
                        "price",
                        None,
                        "pi_rup",
                        frame_idx=frame - 1,
                        scenario_nr=scen - 1,
                    )
                    for frame in frames
                ]
                for scen in scens
            }
            PiDict["pi_rdn"] = {
                scen: [
                    Net_case.read_horizon_samples(
                        "price",
                        None,
                        "pi_rdn",
                        frame_idx=frame - 1,
                        scenario_nr=scen - 1,
                    )
                    for frame in frames
                ]
                for scen in scens
            }

            IndexDictDER = {"PVs": PVs, "Batteries": Batts, "TRs": TRs}

            # PV
            PVDict = {}
            for PV in PVs:
                for scen in scens:
                    P = [
                        pe.value(model.PDER[("PV", PV, frame, scen)])
                        for frame in frames
                    ]
                    Q = [
                        pe.value(model.QDER[("PV", PV, frame, scen)])
                        for frame in frames
                    ]
                    Rup = [
                        pe.value(model.rDER_up[("PV", PV, frame, scen)])
                        for frame in frames
                    ]
                    Rdn = [
                        pe.value(model.rDER_dn[("PV", PV, frame, scen)])
                        for frame in frames
                    ]
                    PMax = Net_case.read_horizon_samples(
                        "DER", PV, fcast_type="Pmax", scenario_nr=scen - 1
                    )
                    PVDict = {
                        **PVDict,
                        (PV, scen): {
                            "P": P,
                            "Q": Q,
                            "Rup": Rup,
                            "Rdn": Rdn,
                            "Pmax": PMax,
                        },
                    }

            # Batteries:
            BattDict = {}
            for batt in Batts:
                for scen in scens:
                    P = [
                        pe.value(model.PDER[("Battery", batt, frame, scen)])
                        for frame in frames
                    ]
                    Q = [
                        pe.value(model.QDER[("Battery", batt, frame, scen)])
                        for frame in frames
                    ]
                    Rup = [
                        pe.value(model.rDER_up[("Battery", batt, frame, scen)])
                        for frame in frames
                    ]
                    Rdn = [
                        pe.value(model.rDER_dn[("Battery", batt, frame, scen)])
                        for frame in frames
                    ]
                    E = [pe.value(model.E_BAT[(batt, frame, scen)]) for frame in frames]
                    E0 = Net_case.Network.get_DERs(mRIDs=batt)[0].E_0
                    Pc = [
                        pe.value(model.Pc_BAT[(batt, frame, scen)]) for frame in frames
                    ]
                    Pd = [
                        pe.value(model.Pd_BAT[(batt, frame, scen)]) for frame in frames
                    ]
                    BattDict = {
                        **BattDict,
                        (batt, scen): {
                            "P": P,
                            "Q": Q,
                            "Rup": Rup,
                            "Rdn": Rdn,
                            "E": E,
                            "E0": E0,
                            "Pc": Pc,
                            "Pd": Pd,
                        },
                    }

            # Transactive REsources
            TRDict = {}
            for scen in scens:
                TR_pi = TR_ComputePrices(model, scenario=scen)
                for TR in TRs:
                    P = [
                        pe.value(model.PDER["TransactiveResource", TR, frame, scen])
                        for frame in frames
                    ]
                    Q = [
                        pe.value(model.QDER["TransactiveResource", TR, frame, scen])
                        for frame in frames
                    ]
                    TRDict = {
                        **TRDict,
                        (TR, scen): {"P": P, "Q": Q, "Price": TR_pi[TR]},
                    }

            DERDict = {**PVDict, **BattDict, **TRDict}
            IndexDict = {**IndexDictPF, **IndexDictDER}

            exportDict = {
                "MainResDict": MainResDict,
                "PFDict": PFDict,
                "EtaDict": EtaDict,
                "DERDict": DERDict,
                "IndexDict": IndexDict,
                "Objective": pe.value(model.obj),
                "ObjSubsDict": ObjSubsDict,
                "Prices": PiDict,
            }

        except Exception as err:
            self.logger.error(f"Error in exporting data: {err}")
            exportDict = None
        finally:
            return exportDict
