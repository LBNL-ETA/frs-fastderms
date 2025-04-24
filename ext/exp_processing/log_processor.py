"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import datetime as dt
import pandas as pd

import json
import pytz
import pickle

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

default_tz_str = "US/Pacific"


class LogProcessor:
    def __init__(self, case_name: str = "", path_to_aws: str = "../AWS/", **kw_args):
        self.case_name = case_name
        path_to_aws = Path(path_to_aws).resolve()

        if self.case_name:
            self.path_to_main = path_to_aws / self.case_name
        else:
            self.path_to_main = path_to_aws

        self.set_all_filenames(**kw_args)

        all_input_data = self.load_all_pickle(self.all_filenames["input_data"])
        for item in all_input_data:
            data_type = item.get("data_type", None)
            if data_type == "static":
                self.static_data = item
            elif data_type == "fcast":
                self.fcasts_data = item
            elif data_type == "deterministic":
                self.deterministic_data = item

        # Feeder Data
        self.substation_ID = self.static_data["substation_ID"]
        # DERs list
        BATTs = []
        PVs = []
        TRs = []
        for DER_ID, DER in self.static_data["DERs"].items():
            if DER["type"] == "BAT":
                BATTs.append(DER_ID)
            elif DER["type"] == "PV":
                PVs.append(DER_ID)
            elif DER["type"] == "TR":
                TRs.append(DER_ID)

        self.BATTs = BATTs
        self.PVs = PVs
        self.TRs = TRs
        self.DERs = BATTs + PVs + TRs
        self.dispatchableDERs = BATTs + PVs

        # TMM House list
        try:
            if self.config["use_tmm"]:
                TMM_houses = []
                TMM_models = self.config[
                    "models_tmm"
                ]  # all of the house variable initializations for the TMM
                for house_param in TMM_models.keys():
                    house_num = house_param.split("_")[1]
                    TMM_houses.append(house_num)
                TMM_houses = set(TMM_houses)
                self.TMM_houses = TMM_houses
                self.TMM_no_cooling_list = []
            else:
                print("No TMM active in simulation")
        except:
            pass

        # Loads list
        uncontrolled_PVs = []
        regular_Loads = []
        for Load_ID in self.fcasts_data["Loads"].keys():
            if "PV" in Load_ID:
                uncontrolled_PVs.append(Load_ID)
            else:
                regular_Loads.append(Load_ID)

        self.uncontrolled_PVs = uncontrolled_PVs
        self.regular_Loads = regular_Loads
        self.Loads = uncontrolled_PVs + regular_Loads

        self.figs_export = []

    def set_all_filenames(self, **kw_args):
        path_to_archive = self.path_to_main / "archive"
        path_to_log = self.path_to_main / "logs"
        path_to_inputs = self.path_to_main / "inputs"

        try:
            config_fn = kw_args.get("config_fn", None)
            if config_fn is None:
                config_fn = path_to_archive / "FRS_config.json"
            # Load the config file
            with open(config_fn, "r") as f:
                self.config = json.load(f)

            loadprofile_fn = kw_args.get("loadprofile_fn", None)
            if loadprofile_fn is None:
                root_folder = self.path_to_main.parents[
                    self.path_to_main.parts[::-1].index("Shared_Data_notinGIT")
                ]
                loadprofile_fn_other_fs = Path(self.config["path_to_loadprofile_data"])
                loadprofile_fn_relative = loadprofile_fn_other_fs.relative_to(
                    loadprofile_fn_other_fs.parents[
                        loadprofile_fn_other_fs.parts[::-1].index(
                            "Shared_Data_notinGIT"
                        )
                    ]
                )
                loadprofile_fn = root_folder / loadprofile_fn_relative

            # Load the config file
            with open(loadprofile_fn, "r") as f:
                pass
        except Exception as e:
            print(f"Error loading a file: {e}")
            raise e

        # Get all filenames in the case_name folder
        tz_str = self.config.get("tz", default_tz_str)
        tz = pytz.timezone(tz_str)
        self.dt0 = dt.datetime.fromtimestamp(self.config["case_dt0"]).astimezone(tz)
        try:
            config_rt_dt0 = self.config["rt_dt0"]
        except:
            # Fall back on old name
            config_rt_dt0 = self.config["rt_start_dt"]
        self.rt_dt0 = dt.datetime.fromtimestamp(config_rt_dt0).astimezone(tz)

        all_filenames = {}
        all_filenames["config"] = config_fn
        all_filenames["loadprofile"] = loadprofile_fn
        all_filenames["DA_log"] = path_to_archive / "DayAhead_log_common_demo.log"
        all_filenames["DA_data"] = path_to_archive / "DA_data.pkl"
        all_filenames["DET_data"] = path_to_archive / "DET_data.pkl"
        all_filenames["MPC_data"] = (
            path_to_archive / f'MPC_data_{self.rt_dt0.strftime("%Y%m%d_%H%M")}.pkl'
        )

        all_filenames["common_log"] = path_to_log / "_log_common_demo.log"

        all_filenames["dump_DA"] = path_to_log / "_dump_scenarios_data.pkl"
        all_filenames["dump_DET"] = path_to_log / "_dump_deterministic_data.pkl"
        all_filenames["dump_bid"] = path_to_log / "_dump_self_sched_bid.pkl"

        try:
            all_filenames["input_data"] = path_to_inputs / self.config["data_filename"]
        except:
            full_path = Path(self.config["file_all_data"])
            all_filenames["input_data"] = path_to_inputs / full_path.name

        self.all_filenames = all_filenames

    def load_all_pickle(self, filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def load_loadprofile(self, feeder_base_load):
        load_profile = pd.read_csv(
            self.all_filenames["loadprofile"],
            header=None,
            names=["time info", "load scale"],
        )
        load_profile["timestamp"] = pd.date_range(
            start=self.dt0,
            periods=len(load_profile),
            freq=pd.Timedelta(load_profile["time info"][1]),
        )
        load_profile = load_profile.set_index("timestamp")
        load_profile = load_profile.resample("1min").interpolate()
        load_profile["load kW"] = load_profile["load scale"] * feeder_base_load
        self.load_profile = load_profile
        return load_profile

    def parse_data_in_line(self, log_line, string_before, string_after=None):
        if string_after:
            text = log_line.split(string_before)[1].split(string_after)[0].strip()
        else:
            text = log_line.split(string_before)[1].strip()
        try:
            value = eval(text)
        except:
            value = text
        return value

    def add_data_to_df(self, df, col_name, timestamp, value):
        if timestamp is None:
            print(f"timestamp is None, skipping")
            return df
        df.loc[timestamp, col_name] = value
        return df

    def parse_common_log(self, **kw_args):

        mpc_timestep = kw_args.get("mpc_timestep", 5)  # in minutes
        common_log_fn = self.all_filenames["common_log"]
        with open(common_log_fn, "r") as f:
            log_lines = f.readlines()

        df = pd.DataFrame()
        t = None
        timestamp = None
        line_index = 0
        # Initializing Values for TMM parsing
        # house_message_no = 1
        # TMM_total_power = float(0)
        # TMM_unmanaged_power = float(0)
        print(f"Parsing {len(log_lines)} lines")
        for line in log_lines:
            line_index += 1
            try:
                if "MPC timestep_period is" in line:
                    # DEBUG    2024-06-13 22:03:51,165 | __Main__Orchestrator: MPC timestep_period is: 15
                    mpc_timestep = self.parse_data_in_line(
                        line, "MPC timestep_period is:"
                    )
                    print(f"MPC timestep is {mpc_timestep} minutes")

                elif "SIMOUT" in line:
                    #  __Main__RT_Control: SIMOUT message received at 2022-09-24 15:11:23-07:00, waiting until 2022-09-24 15:12:00-07:00
                    t = self.parse_data_in_line(line, "received at", ", waiting until")
                    t = t.split(", but RT dispatch is OFF")[0]
                    t = t.split(", MPC is not running")[0]
                    timestamp = pd.Timestamp(t)

                elif "Substation power received at t0" in line:
                    # Substation Power - '2022-07-14 18:12:03,825: Substation power received at t0 = 2022-04-01 14:01:02-07:00'
                    t = self.parse_data_in_line(line, "received at t0 =")
                    timestamp = pd.Timestamp(t)

                elif "ISO Dispatch received at t0" in line:
                    # ISO Dispatch received at t0 = 2022-09-24 15:15:00-07:00
                    t = self.parse_data_in_line(line, "received at t0 =")
                    timestamp = pd.Timestamp(t)

                elif "__Main__Aggregator: Aggregate SOC for" in line:
                    # DEBUG    2024-03-28 23:06:09,820 | __Main__Aggregator: Aggregate SOC for BAT1: 0.5999097743845324, Energy in Wh: 1049841.5052631572
                    DER_ID = self.parse_data_in_line(line, "SOC for", ":")
                    soc = self.parse_data_in_line(
                        line, f"SOC for {DER_ID}:", ", Energy"
                    )
                    e_wh = self.parse_data_in_line(line, "Energy in Wh: ")
                    df = self.add_data_to_df(df, f"{DER_ID}_soc", timestamp, soc)
                    df = self.add_data_to_df(df, f"{DER_ID}_E", timestamp, e_wh)

                elif "__Main__Aggregator: Aggregates:" in line:
                    pass
                elif "__Main__Aggregator: Aggregate" in line:
                    if "SOC" in line:
                        pass
                    else:
                        #  __Main__Aggregator: Aggregate BAT1 has power 119.357 kW
                        DER_ID = self.parse_data_in_line(line, "Aggregate", "has power")
                        value = self.parse_data_in_line(line, "has power", "kW")
                        df = self.add_data_to_df(df, f"{DER_ID}_P", timestamp, value)

                elif "__Main__RT_Control: Subs_p0" in line:
                    # __Main__RT_Control: Subs_p0 = 8408.403 kW
                    value = self.parse_data_in_line(line, "Subs_p0 =", "kW")
                    df = self.add_data_to_df(df, "Subs_p0", timestamp, value)

                elif any(
                    [f"__Main__RT_Control: {PV_ID}" in line for PV_ID in self.PVs]
                ):
                    # DEBUG    2024-01-12 17:08:21,673 | __Main__RT_Control: PV_UTILITY_1 power -155.410 kW
                    Power = self.parse_data_in_line(line, "power ", " kW")
                    DER_ID = self.parse_data_in_line(
                        line, "__Main__RT_Control: ", " power"
                    )
                    df = self.add_data_to_df(df, f"{DER_ID}_P", timestamp, Power)

                elif "MPC Data received at t0" in line:
                    #'2022-07-14 18:09:01,556: MPC Data received at t0 = 2022-04-01 14:00:00-07:00'
                    t = self.parse_data_in_line(line, "received at t0 = ")

                    tpd = pd.Timestamp(t)
                    MPC_fn = self.all_filenames["MPC_data"]
                    MPC_fn = (
                        MPC_fn.parent / f'MPC_data_{tpd.strftime("%Y%m%d_%H%M")}.pkl'
                    )

                    with open(MPC_fn, "rb") as file:
                        print(f"Loading MPC data from {MPC_fn.name}")
                        MPC_data = pickle.load(file)

                    scenario = 1
                    for frame_idx in MPC_data["IndexDict"]["frames"]:
                        t_mpc = tpd + pd.Timedelta(
                            minutes=(frame_idx - 1) * mpc_timestep
                        )

                        # Substation Power
                        value = (
                            self.config["Sbase"]
                            / 1000
                            * sum(
                                [
                                    MPC_data["PFDict"][(1, self.substation_ID, phase)][
                                        "P"
                                    ]["sched"][frame_idx - 1]
                                    for phase in range(3)
                                ]
                            )
                        )
                        df = self.add_data_to_df(
                            df, "Subs_p0_MPC", t_mpc, round(value, 2)
                        )
                        # Prices
                        df = self.add_data_to_df(
                            df,
                            "DA ISO Energy Price",
                            t_mpc,
                            MPC_data["Prices"]["pi"][scenario][frame_idx - 1],
                        )
                        df = self.add_data_to_df(
                            df,
                            "DA ISO Rup Price",
                            t_mpc,
                            MPC_data["Prices"]["pi_rup"][scenario][frame_idx - 1],
                        )
                        df = self.add_data_to_df(
                            df,
                            "DA ISO Rdn Price",
                            t_mpc,
                            MPC_data["Prices"]["pi_rdn"][scenario][frame_idx - 1],
                        )
                        # Objective Subcomponents

                        try:
                            obj_subcomponents = MPC_data["ObjSubsDict"].keys()
                            for obj_subcomponent in obj_subcomponents:
                                if obj_subcomponent in [
                                    "power_cost",
                                    "transactive_cost",
                                    "loss_cost",
                                    "reserve_cost",
                                    "substation_deviation_cost",
                                    "reserves_deviation_cost",
                                ]:
                                    try:
                                        value = MPC_data["ObjSubsDict"][
                                            f"{obj_subcomponent}"
                                        ][scenario][frame - 1]
                                        df = self.add_data_to_df(
                                            df, f"OBJ_{obj_subcomponent}", t_mpc, value
                                        )
                                    except:
                                        pass
                                elif obj_subcomponent == "battery_deviation_cost":
                                    for derID in self.BATTs:
                                        try:
                                            value = MPC_data["ObjSubsDict"][
                                                f"{obj_subcomponent}"
                                            ][derID]
                                            df = self.add_data_to_df(
                                                df,
                                                f"OBJ_{obj_subcomponent}_{derID}",
                                                t_mpc,
                                                value,
                                            )
                                        except:
                                            pass
                                elif (
                                    obj_subcomponent == "battery_charge_discharge_cost"
                                ):
                                    for derID in self.BATTs:
                                        try:
                                            value = MPC_data["ObjSubsDict"][
                                                f"{obj_subcomponent}"
                                            ][derID][scenario][frame - 1]
                                            df = self.add_data_to_df(
                                                df,
                                                f"OBJ_{obj_subcomponent}_{derID}",
                                                t_mpc,
                                                value,
                                            )
                                        except:
                                            pass
                                elif obj_subcomponent == "PV_curtailment_cost":
                                    for derID in self.PVs:
                                        try:
                                            value = MPC_data["ObjSubsDict"][
                                                f"{obj_subcomponent}"
                                            ][derID][scenario][frame - 1]
                                            df = self.add_data_to_df(
                                                df,
                                                f"OBJ_{obj_subcomponent}_{derID}",
                                                t_mpc,
                                                value,
                                            )
                                        except:
                                            pass
                        except Exception as e:
                            print(e)
                            print("Cannot find Objective subcomponent Data")

                        for k in MPC_data["DERDict"].keys():
                            DER_ID = k[0]
                            DERPower = (
                                (MPC_data["DERDict"][k]["P"][frame_idx - 1])
                                * self.config["Sbase"]
                                / 1000
                            )
                            df = self.add_data_to_df(
                                df, f"{DER_ID}_P_MPC", t_mpc, round(DERPower, 2)
                            )

                            try:
                                DERMax = (
                                    (
                                        MPC_data["DERDict"][k]["P"][frame_idx - 1]
                                        + MPC_data["DERDict"][k]["Rup"][frame_idx - 1]
                                    )
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                DERMin = (
                                    (
                                        MPC_data["DERDict"][k]["P"][frame_idx - 1]
                                        - MPC_data["DERDict"][k]["Rdn"][frame_idx - 1]
                                    )
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                DERRup = (
                                    (MPC_data["DERDict"][k]["Rup"][frame_idx - 1])
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                DERRdn = (
                                    (MPC_data["DERDict"][k]["Rdn"][frame_idx - 1])
                                    * self.config["Sbase"]
                                    / 1000
                                )
                            except:
                                DERMax = (
                                    (MPC_data["DERDict"][k]["P"][frame_idx - 1] + 0)
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                DERMin = (
                                    (MPC_data["DERDict"][k]["P"][frame_idx - 1] - 0)
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                DERRup = 0
                                DERRdn = 0
                                if DER_ID not in self.TRs:
                                    print(f"Resource {k} does not have reserves.")
                            # Write DF
                            if DER_ID not in self.TRs:
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_P_max", t_mpc, round(DERMax, 2)
                                )
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_P_min", t_mpc, round(DERMin, 2)
                                )
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_Rup_MPC", t_mpc, round(DERRup, 2)
                                )
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_Rdn_MPC", t_mpc, round(DERRdn, 2)
                                )

                            try:
                                DERPrice = MPC_data["DERDict"][k]["Price"][
                                    frame_idx - 1
                                ]
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_price", t_mpc, round(DERPrice, 3)
                                )
                            except:
                                pass

                            # For Battery
                            if DER_ID in self.BATTs:
                                try:
                                    DER_E0 = (
                                        MPC_data["DERDict"][k]["E0"]
                                        * self.config["Sbase"]
                                        / 1000
                                    )
                                    t_E0 = tpd + pd.Timedelta(seconds=1)
                                    df = self.add_data_to_df(
                                        df, f"{DER_ID}_E_MPC", t_E0, round(DER_E0, 2)
                                    )
                                except:
                                    pass
                                DER_E = (
                                    MPC_data["DERDict"][k]["E"][frame_idx - 1]
                                    * self.config["Sbase"]
                                    / 1000
                                )
                                t_energy = tpd + pd.Timedelta(
                                    minutes=(frame_idx) * mpc_timestep
                                )
                                df = self.add_data_to_df(
                                    df, f"{DER_ID}_E_MPC", t_energy, round(DER_E, 2)
                                )

                elif "Current Dispatch Error" in line:
                    #'2022-07-14 18:09:01,653: Current Dispatch Error: 0'
                    Disp_err = self.parse_data_in_line(line, "Current Dispatch Error: ")
                    df = self.add_data_to_df(df, "Dispatch_Error", timestamp, Disp_err)

                elif "setpoints:" in line:
                    #'2022-07-14 18:11:03,772: New setpoints: {'BAT1': 188.386, 'BAT2': -86.359, 'BAT3': 10.016, 'PV1': 159.052, 'PV2': 176.725, 'PV3': 18.255, 'BAT4': 25.789, 'BAT5': -55.901}'
                    setpts = self.parse_data_in_line(line, "setpoints: ")
                    for k in setpts.keys():
                        df = self.add_data_to_df(
                            df, f"{k}_P_setpt", timestamp, setpts[k]
                        )

                elif "Dispatch is" in line:
                    #'Dispatch is 4195.4539613724155 and Substation Power is 4056.1888882529997'
                    ISO_Dispatch = self.parse_data_in_line(
                        line, "Dispatch is", "and Substation Power is"
                    )
                    df = self.add_data_to_df(df, "Subs_p0_ISO", timestamp, ISO_Dispatch)

                elif (
                    "__Main__TMM : Number of houses managed by TMM: " in line
                ):  # This makes sure the we know how many houses are under TMM management
                    TMM_no_houses = int(
                        self.parse_data_in_line(line, "managed by TMM: ")
                    )
                    # df = self.add_data_to_df(df, '')

                elif (
                    "__Main__TMM : Message:" in line
                ):  # This will indicate that the next line is the message that contains the house power for the TMM
                    Next_line = log_lines[line_index]
                    house_message = eval(Next_line)
                    house_num = list(house_message.keys())[0].split("_")[
                        1
                    ]  # each message is for a specific house number, need to strip that out to get to the value in the dict.
                    total_power = house_message["House_" + house_num]["P Total"]
                    unmanaged_power = house_message["House_" + house_num][
                        "P Uncontrolled"
                    ]
                    try:
                        Cooling_setpoint = house_message["HVAC Cooling_" + house_num][
                            "T Setpoint"
                        ]
                        df = self.add_data_to_df(
                            df, house_num + "T_set", timestamp, Cooling_setpoint
                        )
                    except:
                        # print(f"House_{house_num} does not appear to have cooling.")
                        self.TMM_no_cooling_list.append(house_num)
                    #'HVAC Cooling_n36': {'T Setpoint': 20.0,
                    # if round(house_message_no/TMM_no_houses)==house_message_no/TMM_no_houses:
                    df = self.add_data_to_df(
                        df, house_num + "_total_power", timestamp, total_power
                    )
                    df = self.add_data_to_df(
                        df, house_num + "_unmanaged_power", timestamp, unmanaged_power
                    )

                    #    TMM_total_power = 0
                    #    TMM_unmanaged_power = 0
                    # house_message_no+=1 #This value should be initialized as equal to 1, such that when we receive our 40th message, we'll

                elif "Local Price: " in line:
                    local_prices = self.parse_data_in_line(line, "Local Price: ")
                    df = self.add_data_to_df(
                        df, "Price to TMM", timestamp, local_prices[0]
                    )

                elif "__TMM : Sending Ochre command" in line:
                    Next_line = log_lines[line_index]
                    message = eval(Next_line)
                    house_num = message["input"]["message"]["forward_differences"][0][
                        "object"
                    ].split("_")[
                        1
                    ]  # list(house_message.keys())[0].split('_')[1]
                    value = eval(
                        message["input"]["message"]["forward_differences"][0]["value"]
                    )
                    Setpoint = value["HVAC Cooling"]["Cooling Setpoint"]
                    df = self.add_data_to_df(
                        df, house_num + "T_set_command", timestamp, Setpoint
                    )

                elif "RT_Control: Price" in line:
                    iso_price = self.parse_data_in_line(
                        line, ": Price ", " USD/MWh"
                    )  # RT_Control: Price 13.62295 USD/MWh
                    df = self.add_data_to_df(
                        df, "RT ISO Energy Price", timestamp, iso_price
                    )

                elif "Aggregator: Sending Ochre command to House" in line:
                    Next_line = log_lines[line_index]
                    message = eval(Next_line)
                    house_num = message["input"]["message"]["forward_differences"][0][
                        "object"
                    ].split("_")[
                        1
                    ]  # list(house_message.keys())[0].split('_')[1]
                    value = eval(
                        message["input"]["message"]["forward_differences"][0]["value"]
                    )
                    Setpoint = value["Battery"]["P Setpoint"]
                    df = self.add_data_to_df(
                        df, house_num + "_Batt_set_command", timestamp, Setpoint
                    )

                else:
                    pass
                # Add the parsing for the House data

            except Exception as e:
                print(e)
                print(f"Error in line {line_index}:\n{line}")

        if len(df) == 0:
            raise Exception("Empty Dataframe, Stopping the execution")
        else:
            print(f"Current Datafram: {df.head()}")
        df = df.copy()
        df = df.sort_index()

        # SOC / Energy in battery is interpolated
        for BATT_ID in self.BATTs:
            try:
                df[f"{BATT_ID}_soc"] = df[f"{BATT_ID}_soc"].interpolate()
                df[f"{BATT_ID}_E"] = df[f"{BATT_ID}_E"].interpolate()
                df[f"{BATT_ID}_E_MPC"] = df[f"{BATT_ID}_E_MPC"].interpolate()
            except:
                print(f"Error in interpolating {BATT_ID} SOC/Energy")

        df = df.ffill()
        try:
            df["total_dispatch"] = df[
                [f"{DER_ID}_P_setpt" for DER_ID in self.dispatchableDERs]
            ].sum(axis=1)
        except:
            print("Error in calculating total dispatch")
            pass

        try:
            df["MPC_dispatch"] = df[
                [f"{DER_ID}_P_MPC" for DER_ID in self.dispatchableDERs]
            ].sum(axis=1)
            df["MPC_Rup"] = df[
                [f"{DER_ID}_Rup_MPC" for DER_ID in self.dispatchableDERs]
            ].sum(axis=1)
            df["MPC_Rdn"] = df[
                [f"{DER_ID}_Rdn_MPC" for DER_ID in self.dispatchableDERs]
            ].sum(axis=1)
        except:
            print("Error in calculating MPC dispatch")
            pass

        try:
            if self.config["use_tmm"]:
                print(
                    f"These houses did not have HVAC_Cooling dicts: {sorted(set(self.TMM_no_cooling_list))}"
                )
                try:
                    df["TMM_unmanaged_power"] = df[
                        [f"{house}_unmanaged_power" for house in self.TMM_houses]
                    ].sum(axis=1)
                    df["TMM_total_power"] = df[
                        [f"{house}_total_power" for house in self.TMM_houses]
                    ].sum(axis=1)
                except:
                    print("Error in summing house powers for TMM")
        except:
            pass

        for DER in self.dispatchableDERs:
            try:
                df[f"{DER}_P_corrected"] = pd.concat(
                    [
                        df[f"{DER}_P_min"],
                        df[[f"{DER}_P_setpt", f"{DER}_P_max"]].min(axis=1),
                    ],
                    axis=1,
                ).max(axis=1)
            except:
                print(f"Error in calculating corrected dispatch for {DER}")
                pass
        try:
            df["total_dispatch_corrected"] = df[
                [f"{DER}_P_corrected" for DER in self.dispatchableDERs]
            ].sum(axis=1)
        except:
            print("Error in calculating corrected total dispatch")
            pass

        print(f"Selecting only data after start of Simulation: {self.rt_dt0}")
        df = df[df.index > self.rt_dt0]
        if len(df) == 0:
            print("Empty Dataframe, check that you loaded the correct FRS_config")
        return df

    def parse_DA_data(self):

        tz_str = self.config.get("tz", default_tz_str)
        tz = pytz.timezone(tz_str)
        self.dt0 = dt.datetime.fromtimestamp(self.config["case_dt0"]).astimezone(tz)

        DA_data_fn = self.all_filenames["DA_data"]
        with open(DA_data_fn, "rb") as f:
            DA_data = pickle.load(f)
        self.DA_index = {}
        nodes = DA_data["IndexDict"]["nodes"]
        self.DA_index["nodes"] = nodes
        phases = DA_data["IndexDict"]["phases"]
        self.DA_index["phases"] = phases
        scenarios = DA_data["IndexDict"]["scens"]
        self.DA_index["scenarios"] = scenarios
        frames = DA_data["IndexDict"]["frames"]
        self.DA_index["frames"] = frames

        time_index = [self.dt0 + dt.timedelta(minutes=60 * (i - 1)) for i in frames]
        time_index.append(self.dt0 + dt.timedelta(minutes=60 * frames[-1]))

        fcast_fields = ["DA_xbar", "DA_sigma"]
        multi_index = pd.MultiIndex.from_product(
            [fcast_fields, time_index], names=["field", "time"]
        )
        df = pd.DataFrame(index=multi_index)

        DA_common_log_fn = self.all_filenames["DA_log"]
        with open(DA_common_log_fn, "r") as f:
            log_lines = f.readlines()

        grab_next_line = False
        line_index = 0
        for line in log_lines:
            line_index += 1
            if grab_next_line:
                grab_next_line = False
                try:
                    fcast_data = eval(line)
                    df.loc[("DA_xbar"), data_name] = fcast_data["xbar"] + [
                        fcast_data["xbar"][-1]
                    ]
                    df.loc[("DA_sigma"), data_name] = fcast_data["sigma"] + [
                        fcast_data["sigma"][-1]
                    ]
                except:
                    pass
            elif "FAST_DERMS_CONTROL.Modeling.modelHandler: pi" in line:
                data_name = self.parse_data_in_line(
                    line, "FAST_DERMS_CONTROL.Modeling.modelHandler:"
                )
                grab_next_line = True
            elif "_Main__TMM : tmm_bids:" in line:
                self.TMM_bids = eval(log_lines[line_index])

        self.querry_data = df.copy()

        multi_index = pd.MultiIndex.from_product(
            [scenarios, time_index], names=["scenario", "time"]
        )
        cols = (
            [f"{node}_V_{phase}" for node in nodes for phase in phases]
            + [f"{node}_P" for node in nodes]
            + [f"{node}_PL" for node in nodes]
        )
        df = pd.DataFrame(index=multi_index, columns=cols)

        for time, frame in zip(time_index, frames):
            for scenario in scenarios:
                # Nodal Data
                for node in nodes:
                    try:
                        power = 0
                        load = 0
                        for phase in phases:
                            df.loc[(scenario, time), f"{node}_V_{phase}"] = DA_data[
                                "PFDict"
                            ][(scenario, node, phase)]["Y"]["sched"][frame - 1]
                            power += DA_data["PFDict"][(scenario, node, phase)]["P"][
                                "sched"
                            ][frame - 1]
                            load += DA_data["PFDict"][(scenario, node, phase)]["PL"][
                                frame - 1
                            ]
                        df.loc[(scenario, time), f"{node}_P"] = (
                            power * self.config["Sbase"] / 1000
                        )
                        df.loc[(scenario, time), f"{node}_PL"] = (
                            load * self.config["Sbase"] / 1000
                        )
                    except Exception as e:
                        print(e)
                        print(f"Error at {time} {scenario} {node}")

                # Substation Data
                subs_p0 = 0
                for phase in phases:
                    subs_p0 += DA_data["MainResDict"]["Substation_Power"][frame - 1][
                        phase
                    ]
                df.loc[(scenario, time), f"Subs_p0"] = (
                    subs_p0 * self.config["Sbase"] / 1000
                )
                df.loc[(scenario, time), f"Subs_rup"] = (
                    DA_data["MainResDict"]["Reserves_Up"][frame - 1]
                    * self.config["Sbase"]
                    / 1000
                )
                df.loc[(scenario, time), f"Subs_rdn"] = (
                    DA_data["MainResDict"]["Reserves_Down"][frame - 1]
                    * self.config["Sbase"]
                    / 1000
                )

                # DER data
                for k in DA_data["DERDict"].keys():
                    DER_ID = k[0]
                    for data in DA_data["DERDict"][(DER_ID, scenario)].keys():
                        try:
                            df.loc[(scenario, time), f"{DER_ID}_{data}"] = (
                                DA_data["DERDict"][(DER_ID, scenario)][data][frame - 1]
                                * self.config["Sbase"]
                                / 1000
                            )
                        except:
                            df.loc[(scenario, time), f"{DER_ID}_{data}"] = (
                                DA_data["DERDict"][(DER_ID, scenario)][data]
                                * self.config["Sbase"]
                                / 1000
                            )

                # Price Data
                try:
                    df.loc[(scenario, time), "DA ISO Energy Price"] = DA_data["Prices"][
                        "pi"
                    ][scenario][frame - 1]
                    df.loc[(scenario, time), "DA ISO Rup Price"] = DA_data["Prices"][
                        "pi_rup"
                    ][scenario][frame - 1]
                    df.loc[(scenario, time), "DA ISO Rdn Price"] = DA_data["Prices"][
                        "pi_rdn"
                    ][scenario][frame - 1]
                except:
                    print(
                        f"Error parsing the price data for scenario {scenario} at time: {frame-1}"
                    )

                ##Objective Subcomponents
                try:
                    obj_subcomponents = DA_data["ObjSubsDict"].keys()
                    for obj_subcomponent in obj_subcomponents:
                        if obj_subcomponent in [
                            "power_cost",
                            "transactive_cost",
                            "loss_cost",
                            "reserve_cost",
                            "substation_deviation_cost",
                            "reserves_deviation_cost",
                        ]:
                            try:
                                value = DA_data["ObjSubsDict"][f"{obj_subcomponent}"][
                                    scenario
                                ][frame - 1]
                                df.loc[(scenario, time), f"OBJ_{obj_subcomponent}"] = (
                                    value
                                )
                            except:
                                pass
                        elif obj_subcomponent == "battery_deviation_cost":
                            for derID in self.BATTs:
                                try:
                                    value = DA_data["ObjSubsDict"][
                                        f"{obj_subcomponent}"
                                    ][derID]
                                    df.loc[
                                        (scenario, time),
                                        f"OBJ_{obj_subcomponent}_{derID}",
                                    ] = value
                                except:
                                    pass
                        elif obj_subcomponent == "battery_charge_discharge_cost":
                            for derID in self.BATTs:
                                try:
                                    value = DA_data["ObjSubsDict"][
                                        f"{obj_subcomponent}"
                                    ][derID][scenario][frame - 1]
                                    df.loc[
                                        (scenario, time),
                                        f"OBJ_{obj_subcomponent}_{derID}",
                                    ] = value
                                except:
                                    pass
                        elif obj_subcomponent == "PV_curtailment_cost":
                            for derID in self.PVs:
                                try:
                                    value = DA_data["ObjSubsDict"][
                                        f"{obj_subcomponent}"
                                    ][derID][scenario][frame - 1]
                                    df.loc[
                                        (scenario, time),
                                        f"OBJ_{obj_subcomponent}_{derID}",
                                    ] = value
                                except:
                                    pass
                except Exception as e:
                    print(e)
                    print("Cannot find Objective subcomponent Data")

                # ETA variable
                total_eta = 0
                for phase in phases:
                    total_eta += DA_data["EtaDict"]["eta"][(scenario, phase, "sched")][
                        frame - 1
                    ]
                df.loc[(scenario, time), "eta"] = total_eta

        # Copying the last frame to the last time to facilitate plots
        for scenario in scenarios:
            df.loc[(scenario, df.index.get_level_values(1)[-1]), :] = df.loc[
                (scenario, df.index.get_level_values(1)[-2]), :
            ]

        df["total_load"] = df[[f"{node}_PL" for node in nodes]].sum(axis=1)

        # Post Processing the Battery Energy Data
        # Energy value represent the energy at the end of the time interval, so shifting these value by 1 (1 hour) and adding E0 at the start
        for BATT_ID in self.BATTs:
            for scenario in scenarios:
                df.loc[scenario, f"{BATT_ID}_E"] = (
                    df.loc[scenario, f"{BATT_ID}_E"]
                    .shift(
                        1, fill_value=self.static_data["DERs"][BATT_ID]["E_0"] / 1000
                    )
                    .values
                )

        df = df.copy()
        return df

    def parse_DET_data(self):

        DET_data_fn = self.all_filenames["DET_data"]
        with open(DET_data_fn, "rb") as f:
            DET_data = pickle.load(f)

        self.DET_index = {}
        nodes = DET_data["IndexDict"]["nodes"]
        self.DET_index["nodes"] = nodes
        phases = DET_data["IndexDict"]["phases"]
        self.DET_index["phases"] = phases
        frames = DET_data["IndexDict"]["frames"]
        self.DET_index["frames"] = frames
        # Only one scenario in deterministic data
        scenario = 1

        time_index = [self.dt0 + dt.timedelta(minutes=60 * (i - 1)) for i in frames]
        time_index.append(self.dt0 + dt.timedelta(minutes=60 * frames[-1]))

        fields = ["DET"]
        multi_index = pd.MultiIndex.from_product(
            [fields, time_index], names=["field", "time"]
        )
        df = pd.DataFrame(index=multi_index)

        DA_common_log_fn = self.all_filenames["DA_log"]
        with open(DA_common_log_fn, "r") as f:
            log_lines = f.readlines()

        grab_next_line = False
        for line in log_lines:
            if grab_next_line:
                grab_next_line = False
                try:
                    fcast_data = eval(line)
                    df.loc[("DET"), data_name] = fcast_data + [fcast_data[-1]]
                except:
                    pass
            elif "FAST_DERMS_CONTROL.Modeling.modelHandler: pi" in line:
                data_name = self.parse_data_in_line(
                    line, "FAST_DERMS_CONTROL.Modeling.modelHandler:"
                )
                grab_next_line = True
        try:
            new_df = pd.concat([self.querry_data, df])
            self.querry_data = new_df
        except:
            pass

        cols = (
            [f"{node}_V_{phase}" for node in nodes for phase in phases]
            + [f"{node}_P" for node in nodes]
            + [f"{node}_PL" for node in nodes]
        )
        df = pd.DataFrame(index=time_index, columns=cols)

        for time, frame in zip(time_index, frames):
            # Nodal Data
            for node in nodes:
                try:
                    power = 0
                    load = 0
                    for phase in phases:
                        df.loc[time, f"{node}_V_{phase}"] = DET_data["PFDict"][
                            (scenario, node, phase)
                        ]["Y"]["sched"][frame - 1]
                        power += DET_data["PFDict"][(scenario, node, phase)]["P"][
                            "sched"
                        ][frame - 1]
                        load += DET_data["PFDict"][(scenario, node, phase)]["PL"][
                            frame - 1
                        ]
                    df.loc[time, f"{node}_P"] = power * self.config["Sbase"] / 1000
                    df.loc[time, f"{node}_PL"] = load * self.config["Sbase"] / 1000
                except Exception as e:
                    print(e)
                    print(f"Error at {time} {node}")

            # Substation Data
            subs_p0 = 0
            for phase in phases:
                subs_p0 += DET_data["MainResDict"]["Substation_Power"][frame - 1][phase]
            df.loc[time, f"Subs_p0"] = subs_p0 * self.config["Sbase"] / 1000
            df.loc[time, f"Subs_rup"] = (
                DET_data["MainResDict"]["Reserves_Up"][frame - 1]
                * self.config["Sbase"]
                / 1000
            )
            df.loc[time, f"Subs_rdn"] = (
                DET_data["MainResDict"]["Reserves_Down"][frame - 1]
                * self.config["Sbase"]
                / 1000
            )

            # Price Data
            df.loc[time, f"DA ISO Energy Price"] = DET_data["Prices"]["pi"][scenario][
                frame - 1
            ]
            df = self.add_data_to_df(
                df,
                "DA ISO Rup Price",
                time,
                DET_data["Prices"]["pi_rup"][scenario][frame - 1],
            )
            df = self.add_data_to_df(
                df,
                "DA ISO Rdn Price",
                time,
                DET_data["Prices"]["pi_rdn"][scenario][frame - 1],
            )

            # DER data
            for k in DET_data["DERDict"].keys():
                DER_ID = k[0]
                print(DER_ID)
                for data in DET_data["DERDict"][(DER_ID, scenario)].keys():
                    try:
                        if data != "Price":
                            df.loc[time, f"{DER_ID}_{data}"] = (
                                DET_data["DERDict"][(DER_ID, scenario)][data][frame - 1]
                                * self.config["Sbase"]
                                / 1000
                            )
                    except:
                        if data != "Price":
                            df.loc[time, f"{DER_ID}_{data}"] = (
                                DET_data["DERDict"][(DER_ID, scenario)][data]
                                * self.config["Sbase"]
                                / 1000
                            )
                try:
                    DERPrice = DET_data["DERDict"][(DER_ID, scenario)]["Price"][
                        frame - 1
                    ]
                    df = self.add_data_to_df(
                        df, f"{DER_ID}_price", time, round(DERPrice, 3)
                    )
                except:
                    print(f"No Price Data for {DER_ID}")
                    pass

            # Objective Subcomponents
            try:
                obj_subcomponents = DET_data["ObjSubsDict"].keys()
                for obj_subcomponent in obj_subcomponents:
                    if obj_subcomponent in [
                        "power_cost",
                        "transactive_cost",
                        "loss_cost",
                        "reserve_cost",
                        "substation_deviation_cost",
                        "reserves_deviation_cost",
                    ]:
                        try:
                            value = DET_data["ObjSubsDict"][f"{obj_subcomponent}"][
                                scenario
                            ][frame - 1]
                            df.loc[time, f"OBJ_{obj_subcomponent}"] = value
                        except:
                            pass
                    elif obj_subcomponent == "battery_deviation_cost":
                        for derID in self.BATTs:
                            try:
                                value = DET_data["ObjSubsDict"][f"{obj_subcomponent}"][
                                    derID
                                ]
                                df.loc[time, f"OBJ_{obj_subcomponent}_{derID}"] = value
                            except:
                                pass
                    elif obj_subcomponent == "battery_charge_discharge_cost":
                        for derID in self.BATTs:
                            try:
                                value = DET_data["ObjSubsDict"][f"{obj_subcomponent}"][
                                    derID
                                ][scenario][frame - 1]
                                df.loc[time, f"OBJ_{obj_subcomponent}_{derID}"] = value
                            except:
                                pass
                    elif obj_subcomponent == "PV_curtailment_cost":
                        for derID in self.PVs:
                            try:
                                value = DET_data["ObjSubsDict"][f"{obj_subcomponent}"][
                                    derID
                                ][scenario][frame - 1]
                                df.loc[time, f"OBJ_{obj_subcomponent}_{derID}"] = value
                            except:
                                pass
            except Exception as e:
                print(e)
                print("Cannot find Objective subcomponent Data")

            # ETA Variable
            for eta in DET_data["EtaDict"].keys():
                if eta == "eta":
                    total_eta = 0
                    for phase in phases:
                        total_eta += DET_data["EtaDict"][eta][
                            (scenario, phase, "sched")
                        ][frame - 1]
                    df.loc[time, eta] = total_eta
                else:
                    df.loc[time, eta] = DET_data["EtaDict"][eta][frame - 1]

        df.loc[df.index[-1], :] = df.loc[df.index[-2], :]

        df["total_load"] = df[[f"{node}_PL" for node in nodes]].sum(axis=1)

        # Post Processing the Battery Energy Data
        for BATT_ID in self.BATTs:
            df[f"{BATT_ID}_E"] = (
                df[f"{BATT_ID}_E"]
                .shift(1, fill_value=self.static_data["DERs"][BATT_ID]["E_0"] / 1000)
                .values
            )

        df = df.copy()
        return df

    def add_to_export_list(self, fig):
        self.figs_export.append(fig)

    def export_figs(self, filename: str = "figures_export.pdf"):
        path = self.path_to_main / filename
        with PdfPages(path) as pp:
            for fig in self.figs_export:
                fig.savefig(pp, format="pdf", bbox_inches="tight")
