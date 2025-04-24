"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

from pathlib import Path

import datetime as dt
import pandas as pd

import logging
import sys
import time
import pytz
import os
import pickle
import json
import shutil
import subprocess
import argparse

from gridappsd import GridAPPSD, utils
from gridappsd.simulation import Simulation

from Helpers import Demo_Handler

path_to_here = Path(__file__).resolve().parent
path_to_repo = path_to_here / ".."
path_to_src = path_to_repo / "src"
path_to_utils = path_to_repo / "ext" / "gapps_data_feed"

# Add repo to Python Path
sys.path.append(str(path_to_here))
sys.path.append(str(path_to_src))
# Import Demo Handler
from Helpers import Demo_Handler

# Import FAST DERMS code
from FAST_DERMS_CONTROL.common import init_logging, FRS_config, SkippingApp
from FAST_DERMS_CONTROL.IOs.IObackup import IObackup
from ISO.ISO_main import mockISO

# Console level when script is executed
display_logging_level = logging.WARNING
# Script logging level
script_logging_level = logging.DEBUG

# Fake Simulation ID for Day Ahead
fake_simulation_id = "123456789"


def copy_files(files_to_copy={}):
    for fn, file in files_to_copy.items():
        try:
            shutil.copyfile(file["src"], file["dest"])
            print(f"Copied {file}")
        except:
            print(f"Error with {file}")


def run(cases={}, baselines={}, path_to_folder=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dayahead", "-da", help="Run Day Ahead", action="store_true")
    parser.add_argument(
        "--no-dayahead",
        "-nda",
        help="Do not Run Day Ahead",
        dest="dayahead",
        action="store_false",
    )
    parser.add_argument(
        "--simulation", "-s", help="Run Simulation", action="store_true"
    )
    parser.add_argument(
        "--no-simulation",
        "-ns",
        help="Do not Run Simulation",
        dest="simulation",
        action="store_false",
    )
    parser.add_argument(
        "--baseline", "-b", help="Run Baseline Simulations", action="store_true"
    )
    parser.add_argument(
        "--no-baseline",
        "-nb",
        help="Do not Run Baseline Simulations",
        dest="baseline",
        action="store_false",
    )
    parser.add_argument("--case", "-c", help="Which Case to Run")
    parser.set_defaults(dayahead=True, simulation=True, baseline=False, case=0)

    opts = parser.parse_args()
    case_nr = opts.case
    run_dayahead = opts.dayahead
    run_baseline = opts.baseline
    run_simulation = opts.simulation

    # Get the specific case to run
    case = cases.get(case_nr, "")
    if case:
        case = f"_{case}"

    # Paths
    if path_to_folder is None:
        path_to_folder = Path(__file__).resolve().parent
    else:
        path_to_folder = Path(path_to_folder).resolve()
    paths = {}
    paths["root"] = path_to_folder
    paths["inputs"] = path_to_folder / "inputs"
    paths["logs"] = path_to_folder / "logs"
    paths["archive"] = path_to_folder / "archive"
    paths["backup"] = path_to_folder / "backup"
    paths["export"] = path_to_folder / f"results{case}"

    # Creating Folders if they don't exist
    for name, path in paths.items():
        if not path.exists() and name != "export":
            path.mkdir(parents=True)

    # Notebook Logger
    logger = logging.getLogger("DEMO_script")

    # Specify baselines to run
    if not baselines:
        run_baselines = {}
        if run_baseline:
            logger.warning("Baseline run requested, but No Baseline specified")
    else:
        run_baselines = baselines if run_baseline else {}

    # Setup the automator
    automator = Demo_Handler(logger)

    config_fn = f"FRS_config{case}.json"
    config_path = paths["inputs"] / config_fn
    if not config_path.exists():
        logger.error(f"Config file {config_fn} not found")
        logger.critical("Using default config")
        config_path = paths["root"] / "FRS_config.json"

    FULL_DEMO_single_day(
        automator,
        config_path,
        script_logging_level,
        paths,
        run_dayahead=run_dayahead,
        run_simulation=run_simulation,
        run_baselines=run_baselines,
        case=case,
    )


##############################################################################################################
def FULL_DEMO_single_day(
    automator,
    config_path,
    script_logging_level,
    paths,
    run_dayahead=True,
    run_simulation=True,
    run_baselines={},
    case="",
):
    logger = automator.logger

    # Gridapps-D Environment Variables
    os.environ["GRIDAPPSD_APPLICATION_ID"] = "demo-remote-notebook"
    os.environ["GRIDAPPSD_APPLICATION_STATUS"] = "STARTED"
    os.environ["GRIDAPPSD_USER"] = "app_user"
    os.environ["GRIDAPPSD_PASSWORD"] = "1234App"

    path_to_logs = paths["logs"]
    path_to_archive = paths["archive"]
    path_to_export = paths["export"]
    path_to_backup = paths["backup"]

    # setting up all the logging facility
    init_logging(
        app_name="DEMO_script",
        log_level=script_logging_level,
        log_level_console=display_logging_level,
        path_to_logs=path_to_logs,
    )
    logger.setLevel(script_logging_level)

    try:
        # Emptying Archive Folder
        automator.clean_folder(path_to_archive)
        # Logging Facility
        automator.clean_folder(path_to_logs)
        # Config
        config = FRS_config(str(config_path))

        path_to_tmm_bids = getattr(config, "path_to_bids_tmm", None)

        if path_to_tmm_bids is not None:
            if config.use_tmm:
                paths.update({"TMM_bids": path_to_tmm_bids})
        else:
            if not run_dayahead and config.use_tmm:
                paths.update({"TMM_bids": path_to_backup / f"TMM_bids{case}.pkl"})
        # Forecasts
        UPLOAD_Forecasts(automator, config_path, paths)
        # Archive Logs for initialization of the Demo
        files_to_copy = {}
        files_to_copy["logs"] = {
            "src": path_to_logs / "_log_common_demo.log",
            "dest": path_to_archive / "Setup_log_common_demo.log",
        }
        files_to_copy["logs_backup"] = {
            "src": path_to_logs / "_log_common_demo.log",
            "dest": path_to_backup / f"Setup_log_common_demo{case}.log",
        }
        copy_files(files_to_copy)

        # Start of the Simulation
        if run_dayahead:
            automator.clean_folder(path_to_logs)
            # # DA
            RUN_DayAhead(automator, config_path, paths)
            # Add some delay to wrap things up
            time.sleep(60)
            # Archive Day Ahead Logs and Backup DayAhead
            files_to_copy = {}
            files_to_copy["logs"] = {
                "src": path_to_logs / "_log_common_demo.log",
                "dest": path_to_archive / "DayAhead_log_common_demo.log",
            }
            files_to_copy["logs_backup"] = {
                "src": path_to_logs / "_log_common_demo.log",
                "dest": path_to_backup / f"DayAhead_log_common_demo{case}.log",
            }
            files_to_copy["DayAhead_pkl"] = {
                "src": path_to_logs / "DayAhead_Process.pkl",
                "dest": path_to_backup / f"DayAhead_Process{case}.pkl",
            }
            files_to_copy["DA_data"] = {
                "src": path_to_archive / "DA_data.pkl",
                "dest": path_to_backup / f"DA_data{case}.pkl",
            }
            files_to_copy["DET_data"] = {
                "src": path_to_archive / "DET_data.pkl",
                "dest": path_to_backup / f"DET_data{case}.pkl",
            }
            files_to_copy["self_sched"] = {
                "src": path_to_logs / "_dump_self_sched_bid.pkl",
                "dest": path_to_backup / f"_dump_self_sched_bid{case}.pkl",
            }
            files_to_copy["det_dump"] = {
                "src": path_to_logs / "_dump_deterministic_data.pkl",
                "dest": path_to_backup / f"_dump_deterministic_data{case}.pkl",
            }
            if config.use_tmm and path_to_tmm_bids is None:
                files_to_copy["TMM_bids"] = {
                    "src": path_to_logs / "TMM_bids.pkl",
                    "dest": path_to_backup / f"TMM_bids{case}.pkl",
                }
            # Adding Hotstart file with correct name
            files_to_copy["hotstart"] = {
                "src": path_to_logs / "DayAhead_Process.pkl",
                "dest": path_to_archive / f"DayAhead_Process{case}.pkl",
            }
        else:
            # Put Back Backed up files
            files_to_copy = {}
            files_to_copy["setup_logs_backup"] = {
                "dest": path_to_archive / "Setup_log_common_demo.log",
                "src": path_to_backup / f"Setup_log_common_demo{case}.log",
            }
            files_to_copy["logs_backup"] = {
                "dest": path_to_archive / "DayAhead_log_common_demo.log",
                "src": path_to_backup / f"DayAhead_log_common_demo{case}.log",
            }
            files_to_copy["DA_data"] = {
                "dest": path_to_archive / "DA_data.pkl",
                "src": path_to_backup / f"DA_data{case}.pkl",
            }
            files_to_copy["DET_data"] = {
                "dest": path_to_archive / "DET_data.pkl",
                "src": path_to_backup / f"DET_data{case}.pkl",
            }
            files_to_copy["DayAhead_pkl"] = {
                "dest": path_to_archive / f"DayAhead_Process{case}.pkl",
                "src": path_to_backup / f"DayAhead_Process{case}.pkl",
            }
            files_to_copy["self_sched"] = {
                "dest": path_to_logs / "_dump_self_sched_bid.pkl",
                "src": path_to_backup / f"_dump_self_sched_bid{case}.pkl",
            }
            files_to_copy["det_dump"] = {
                "dest": path_to_logs / "_dump_deterministic_data.pkl",
                "src": path_to_backup / f"_dump_deterministic_data{case}.pkl",
            }

        copy_files(files_to_copy)

        if run_simulation:
            # # MPC RT Simulation
            paths["export"] = path_to_export / "Simulation"
            RUN_RT(automator, config_path, paths)
            # Add some delay to wrap things up
            time.sleep(60)

        if run_baselines:

            for baseline_name, params in run_baselines.items():
                cancelled_DERs = params["cancelled_DERs"]
                ignored_DERs = params["ignored_DERs"]
                paths["export"] = path_to_export / baseline_name
                RUN_RT(automator, config_path, paths, cancelled_DERs, ignored_DERs)
                # Add some delay to wrap things up
                time.sleep(60)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logger.critical("Script Crashed")
        logger.critical(
            f"Error on line {exc_tb.tb_lineno}: {repr(e)},\n{exc_type}, {exc_obj}"
        )


##############################################################################################################
def UPLOAD_Forecasts(automator, config_path, paths):

    path_to_logs = paths["logs"]
    path_to_archive = paths["archive"]

    # # Forecasts Upload
    logger = automator.logger
    logger.critical("Starting Forecasts Upload")
    logger.critical(
        f"Notebook Logging Level: {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    logger.warning(f"Running Python: {sys.version}")
    logger.warning(f"PYTHONPATH: \n {sys.path}")
    logger.warning(f"PATH: \n {os.environ['PATH']}")
    # ## Static Parameters
    config = FRS_config(str(config_path))
    logger.info(f"FRS_config: \n {config.__dict__}")
    # Fake request to get the model_id
    request = {"power_system_config": {"Line_name": config.model_id}}

    ##########################################################################################
    #  GridAPPS Connection
    try:
        # Connect to GridAPPS-D Platform
        gapps = GridAPPSD()
        if not gapps.connected:
            raise Exception("GridAPPSD not connected")

        myIOs = IObackup(
            simulation_id=fake_simulation_id,
            path_to_repo=path_to_repo,
            file_all_data=config.file_all_data,
            path_to_export=path_to_logs,
            path_to_archive=path_to_archive,
            tz=config.tz,
        )

        # Setup Automator
        automation_topic = myIOs.service_topic("automation", "input")
        automator.setup_automator(myIOs, input_topic=automation_topic)

    except Exception as e:
        logger.error(repr(e))
        raise e

    ##########################################################################################
    # # Upload Forecasts to GridApps-D

    # ## Clear Existing Fcasts from Database
    container_id = subprocess.getoutput("docker ps -qf name=influxdb")
    try:
        logger.warning("Erasing Existing Forecast Data")
        _ = subprocess.getoutput(
            f'docker exec -t {container_id} influx -database proven -execute "drop measurement forecasts"'
        )
        # Giving the database 5 seconds to catch up
        time.sleep(5)
    except Exception as e:
        logger.error("Something went wrong when erasing the forecasts")
        logger.error(repr(e))

    # ## Push Fcasts to GridApps
    automator.start_task("Forecasts Upload")
    tz = pytz.timezone(config.tz)
    dt0 = dt.datetime.fromtimestamp(config.case_dt0).astimezone(tz)
    forecaster_topic = myIOs.application_topic("forecaster", "output")

    if config.use_adms_publisher:
        # ### Push Substation Limits
        # Substation data is provided with the same timestep as the MPC for the entire horizon
        substation_ID = config.model_id
        n_timestep = config.DA_n_timestep * int(
            config.DA_timestep_period / config.MPC_timestep_period
        )
        timestep_period = config.MPC_timestep_period
        dt0_fcast = dt0
        substation_fcasts = {}
        substation_fcasts["ADMS_V_set_pu"] = [1.0] * n_timestep
        substation_fcasts["ADMS_P0_up_limit"] = [999999999999] * n_timestep
        substation_fcasts["ADMS_P0_dn_limit"] = [-999999999999] * n_timestep

        myIOs.publish_fcasts_gapps(
            substation_ID,
            substation_fcasts,
            dt0_fcast,
            n_timestep,
            timestep_period,
            topic=forecaster_topic,
        )

    # ### Push Load Fcasts
    load_IDs = myIOs.fcasts["Loads"].keys()
    n_timestep = config.DA_n_timestep
    timestep_period = config.DA_timestep_period
    for load_ID in load_IDs:
        load_fcast = myIOs.get_fcast_load(
            load_ID, dt0, n_timestep, timestep_period, force_static=True
        )

        # Flatten the price forecast
        dt0_fcast = load_fcast["t0"]
        timestep_period = load_fcast["timestep_period"]
        flat_load_fcast = {}
        for fcast_type in load_fcast.keys():
            if fcast_type == "t0" or fcast_type == "timestep_period":
                continue
            if type(load_fcast[fcast_type]) is dict:
                for key in load_fcast[fcast_type].keys():
                    flat_load_fcast[fcast_type + "_a_" + key] = [
                        load_fcast[fcast_type][key][i][0] for i in range(n_timestep)
                    ]
                    flat_load_fcast[fcast_type + "_b_" + key] = [
                        load_fcast[fcast_type][key][i][1] for i in range(n_timestep)
                    ]
                    flat_load_fcast[fcast_type + "_c_" + key] = [
                        load_fcast[fcast_type][key][i][2] for i in range(n_timestep)
                    ]
            else:
                flat_load_fcast[fcast_type] = load_fcast[fcast_type]
        myIOs.publish_fcasts_gapps(
            load_ID,
            flat_load_fcast,
            dt0_fcast,
            n_timestep,
            timestep_period,
            topic=forecaster_topic,
        )

    # ### Push Price Fcasts
    n_timestep = config.DA_n_timestep
    timestep_period = config.DA_timestep_period

    if config.use_static_data_iso:
        # Using data from static file
        price_fcast = myIOs.get_fcast_price(
            dt0, n_timestep, timestep_period, force_static=True
        )

        # Flatten the price forecast
        dt0_fcast = price_fcast["t0"]
        timestep_period_fcast = price_fcast["timestep_period"]

        flat_price_fcast = {}
        for fcast_type in price_fcast.keys():
            if fcast_type == "t0" or fcast_type == "timestep_period":
                continue
            if type(price_fcast[fcast_type]) is dict:
                for key in price_fcast[fcast_type].keys():
                    flat_price_fcast[fcast_type + "_" + key] = price_fcast[fcast_type][
                        key
                    ]
            else:
                flat_price_fcast[fcast_type] = price_fcast[fcast_type]
    else:
        # Using data from CAISO
        iso = mockISO(myIOs)
        price_DF = iso.get_CAISO_prices(
            dt0, duration=24, Market="DAM", date_override=config.case_dt0_iso
        )
        static_sigma = 0.25

        dt0_fcast = dt0
        timestep_period_fcast = 60
        # Package Price fcast
        flat_price_fcast = {}
        flat_price_fcast["pi_xbar"] = price_DF["MW"].astype("float").tolist()
        flat_price_fcast["pi_sigma"] = [
            static_sigma * price for price in flat_price_fcast["pi_xbar"]
        ]
        flat_price_fcast["pi_rup_xbar"] = price_DF["RU"].astype("float").tolist()
        flat_price_fcast["pi_rup_sigma"] = [
            static_sigma * price for price in flat_price_fcast["pi_rup_xbar"]
        ]
        flat_price_fcast["pi_rdn_xbar"] = price_DF["RD"].astype("float").tolist()
        flat_price_fcast["pi_rdn_sigma"] = [
            static_sigma * price for price in flat_price_fcast["pi_rdn_xbar"]
        ]

    myIOs.publish_fcasts_gapps(
        "price",
        flat_price_fcast,
        dt0_fcast,
        n_timestep,
        timestep_period_fcast,
        topic=forecaster_topic,
    )

    # TMM forecasts
    path_to_tmm_bids = paths.get("TMM_bids", None)
    if path_to_tmm_bids is not None:
        with open(path_to_tmm_bids, "rb") as f:
            tmm_fcast = pickle.load(f)
        myIOs.publish_fcasts_gapps(
            tmm_fcast["mRID"],
            tmm_fcast["fcast_message"],
            tmm_fcast["t0"],
            tmm_fcast["n_timestep"],
            tmm_fcast["timestep_period"],
            topic=forecaster_topic,
        )

    # Giving the platform some time to process the forecast
    time.sleep(5)
    automator.stop_task()

    # ## Clean UP
    automator.clean_processes()
    logger.critical("End of Forecasts Upload")
    # ## Archive Forecasts Logs
    files_to_copy = {}
    files_to_copy["logs"] = {
        "src": path_to_logs / "_log_common_demo.log",
        "dest": path_to_archive / "Fcasts_Upload_log_common_demo.log",
    }
    copy_files(files_to_copy)


##############################################################################################################
def RUN_DayAhead(automator, config_path, paths):
    path_to_logs = paths["logs"]
    path_to_archive = paths["archive"]
    # # DAY AHEAD
    logger = automator.logger
    logger.critical("Starting Day Ahead Simulation")
    logger.critical(
        f"Notebook Logging Level: {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    logger.warning(f"Running Python: {sys.version}")
    logger.warning(f"PYTHONPATH: \n {sys.path}")
    logger.warning(f"PATH: \n {os.environ['PATH']}")
    # ## Static Parameters
    config = FRS_config(str(config_path))
    logger.info(f"FRS_config: \n {config.__dict__}")
    # Fake request to get the model_id
    request = {"power_system_config": {"Line_name": config.model_id}}

    ##########################################################################################
    #  GridAPPS Connection
    try:
        # Connect to GridAPPS-D Platform
        gapps = GridAPPSD()
        if not gapps.connected:
            raise Exception("GridAPPSD not connected")

        myIOs = IObackup(
            simulation_id=fake_simulation_id,
            path_to_repo=path_to_repo,
            file_all_data=config.file_all_data,
            path_to_export=path_to_logs,
            path_to_archive=path_to_archive,
            tz=config.tz,
        )

        # Setup Automator
        automation_topic = myIOs.service_topic("automation", "input")
        automator.setup_automator(myIOs, input_topic=automation_topic)

    except Exception as e:
        logger.error(repr(e))
        raise e

    if config.use_tmm:
        logger.info("TMM is not included in this release, using provided data")

        # ##########################################################################################
        # # ## TMM Day Ahead
        # automator.start_task('TMM Startup')

        # app_config = {}
        # app_config['mrid'] = config.mrid_tmm
        # app_config['log_level'] = config.log_tmm
        # app_config['path_to_export'] = str(path_to_logs)
        # app_config['tz'] = config.tz
        # app_config['Sbase'] = config.Sbase
        # app_config['sim_time'] = -1
        # app_config['automation_topic'] = automation_topic
        # app_config['models'] = config.models_tmm

        # try:
        #     _ = automator.add_process(config.path_to_tmm, fake_simulation_id, request, app_config)
        # except Exception as e:
        #     logger.error('Something went wrong launching TMM')
        #     logger.error(repr(e))
        # # Wait till the TMM flags the end of its initialization
        # automator.wait_for_task()

        # automator.start_task('TMM Day Ahead')
        # # Orchestrator input for commands
        # TMM_topic = myIOs.service_topic(config.mrid_tmm, 'input')
        # opts = {'t0' : int(config.DA_t0)}
        # automator.send_command(TMM_topic, 'DA', opts = opts)
        # # Wait till the TMM flags the end of its DA process
        # automator.wait_for_task()

    ##########################################################################################
    # ## Orchestrator
    # ### Starting Process
    automator.start_task("FRS Orchestrator Startup")

    app_config = {}
    app_config["mrid"] = config.mrid_orchestrator
    app_config["tz"] = config.tz
    app_config["file_all_data"] = str(config.file_all_data)
    app_config["Sbase"] = config.Sbase
    app_config["sim_time"] = -1
    app_config["log_level"] = config.log_orchestrator
    app_config["path_to_repo"] = str(path_to_repo)
    app_config["path_to_export"] = str(path_to_logs)
    app_config["path_to_archive"] = str(path_to_archive)
    app_config["automation_topic"] = automation_topic

    try:
        _ = automator.add_process(
            config.path_to_orchestrator, fake_simulation_id, request, app_config
        )
    except Exception as e:
        logger.error("Something went wrong launching Orchestrator")
        logger.error(repr(e))

    # Orchestrator input for commands
    orchestrator_topic = myIOs.service_topic(config.mrid_orchestrator, "input")

    # Wait till the Orchestrator flags the end of its initialization
    automator.wait_for_task()

    # ### Run FRS Day Ahead Scenarios
    automator.start_task("FRS Model Initialization")
    # #################################################
    #  INIT PARAMS
    options = {}
    options.update({"Vmin": config.INIT_Vmin})
    options.update({"Vmax": config.INIT_Vmax})
    options.update({"force_static": True})

    automator.send_command(orchestrator_topic, "init", options)
    # Wait till the Orchestrator flags the end of its initialization
    automator.wait_for_task()
    automator.start_task("FRS Day Ahead Scenarios")
    # #################################################
    #  DA PARAMS
    options = {}
    options.update({"t0": int(config.DA_t0)})
    options.update({"n_timestep": config.DA_n_timestep})
    options.update({"timestep_period": config.DA_timestep_period})
    options.update({"n_scenario": config.DA_n_scenario})
    options.update({"n_init": config.DA_n_init})
    options.update({"add_prices": config.DA_add_prices})
    options.update({"TS_metrics": config.DA_TS_metrics})
    options.update({"use_exp_value": config.DA_use_exp_value})
    options.update({"max_loop_nr": config.DA_max_loop_nr})
    options.update({"sigma_multiplier": config.DA_sigma_multiplier})
    options.update({"n_thermal": config.DA_n_thermal})
    options.update({"remove_reserves": config.DA_remove_reserves})
    options.update({"unity_powerfactor": config.DA_unity_powerfactor})
    options.update({"power_weight": config.DA_power_weight})
    options.update({"transactive_weight": config.DA_transactive_weight})
    options.update({"reserve_weight": config.DA_reserve_weight})
    options.update({"loss_weight": config.DA_loss_weight})
    options.update({"reverse_pf_weight": config.DA_reverse_pf_weight})
    options.update(
        {"substation_deviation_weight": config.DA_substation_deviation_weight}
    )
    options.update(
        {
            "substation_deviation_price_override": config.DA_substation_deviation_price_override
        }
    )
    options.update({"solver": config.DA_solver})

    automator.send_command(orchestrator_topic, "DA", options)
    # Wait till the Orchestrator flags the end of its DA process
    automator.wait_for_task()

    ##########################################################################################
    # ## Run ISO Day Ahead
    automator.start_task("ISO Startup")
    app_config = {}
    app_config["mrid"] = config.mrid_iso
    app_config["tz"] = config.tz
    app_config["log_level"] = config.log_iso
    app_config["path_to_repo"] = str(path_to_repo)
    app_config["path_to_export"] = str(path_to_logs)
    app_config["path_to_archive"] = str(path_to_archive)
    app_config["file_all_data"] = str(config.file_all_data)
    app_config["sim_time"] = -1

    try:
        _ = automator.add_process(
            config.path_to_iso, fake_simulation_id, request, app_config
        )
    except Exception as e:
        logger.error("Something went wrong launching ISO")
        logger.error(repr(e))

    # Wait till the ISO flags the end of its initialization
    automator.wait_for_task()

    # ISO input topic
    iso_topic = myIOs.service_topic("ISO", "input")

    automator.start_task("ISO Day Ahead Market")
    options = {}
    options.update({"t0": int(config.DA_t0)})
    options.update({"force_static": config.use_static_data_iso})
    options.update({"date_override": config.case_dt0_iso})

    automator.send_command(iso_topic, "DA", options)
    # Wait till the Orchestrator flags the end of its DA process
    automator.wait_for_task()

    # Give a few seconds for the forecasts to be processed
    time.sleep(5)

    ##########################################################################################
    # ## Run FRS Day Ahead Deterministic
    automator.start_task("FRS Day Ahead Deterministic")
    # #################################################
    #  DET PARAMS
    options = {}
    options.update({"n_thermal": config.DA_n_thermal})
    options.update({"unity_powerfactor": config.DET_unity_powerfactor})
    options.update({"power_weight": config.DET_power_weight})
    options.update({"transactive_weight": config.DET_transactive_weight})
    options.update({"reserve_weight": config.DET_reserve_weight})
    options.update({"loss_weight": config.DET_loss_weight})
    options.update({"reverse_pf_weight": config.DET_reverse_pf_weight})
    options.update(
        {"substation_deviation_weight": config.DET_substation_deviation_weight}
    )
    options.update(
        {
            "substation_deviation_price_override": config.DET_substation_deviation_price_override
        }
    )
    options.update({"reserves_deviation_weight": config.DET_reserves_deviation_weight})
    options.update(
        {
            "reserves_deviation_price_override": config.DET_reserves_deviation_price_override
        }
    )
    options.update({"solver": config.DET_solver})

    automator.send_command(orchestrator_topic, "DET", options)
    # Wait till the Orchestrator flags the end of its DET process
    automator.wait_for_task()

    ##########################################################################################
    # ## Export Day Ahead Data for RT Simulation
    automator.start_task("FRS Export Day Ahead Results")
    # #################################################
    #  EXPORT PARAMS
    options = {}
    options["filename"] = "DayAhead_Process.pkl"

    automator.send_command(orchestrator_topic, "export_DA", options)
    # Wait till the Orchestrator flags the end of its DA process
    automator.wait_for_task()

    # ## Clean UP
    automator.clean_processes()
    logger.critical("End of Day Ahead Simulation")


##############################################################################################################
def RUN_RT(automator, config_path, paths, cancelled_DERs=[], ignored_DERs=[]):
    path_to_logs = paths["logs"]
    path_to_archive = paths["archive"]
    path_to_inputs = paths["inputs"]
    ## RT Simulation
    logger = automator.logger

    automator.reset()
    # Logging Facility
    list_of_files_to_keep = [
        "_dump_self_sched_bid.pkl",
        "DayAhead_Process.pkl",
        "_dump_deterministic_data.pkl",
        "_dump_scenarios_data.pkl",
    ]
    automator.clean_folder(path_to_logs, list_of_files_to_keep=list_of_files_to_keep)
    # Clean MPC files
    mpc_files = [
        f
        for f in path_to_archive.iterdir()
        if f.is_file() and f.name.startswith("MPC_data_")
    ]
    automator.clean_folder(path_to_archive, list_of_files_to_delete=mpc_files)

    logger.critical("Starting MPC / RT Simulation")
    logger.critical(
        f"Notebook Logging Level: {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    logger.warning(f"Running Python: {sys.version}")
    logger.warning(f"PYTHONPATH: \n {sys.path}")
    logger.warning(f"PATH: \n {os.environ['PATH']}")

    config = FRS_config(str(config_path))
    logger.info(f"FRS_config: \n {config.__dict__}")
    path_to_config_export = path_to_archive / "FRS_config.json"
    config.export_config(path_to_config_export)
    # Fake request to get the model_id
    request = {"power_system_config": {"Line_name": config.model_id}}

    # ## Load Timeseries for Load and weather
    tz = pytz.timezone(config.tz)

    ##########################################################################################
    # Load Data
    raw_input_load_file = config.path_to_loadprofile_data
    load_file_name = "out_loadprofile_data.txt"
    load_tz = config.tz

    # Weather Data
    raw_input_weather_file = config.path_to_weather_data
    weather_file_name = "out_weather_data.txt"
    weather_tz = None

    try:
        container_id = subprocess.getoutput("docker ps -qf name=influxdb")

        if config.load_loadprofile_data:
            try:
                logger.warning("Erasing Existing Load Data")
                _ = subprocess.getoutput(
                    f'docker exec -t {container_id} influx -database proven -execute "drop measurement ieeezipload"'
                )

                logger.warning("Loading loadprofile data")
                path_to_load_file = (path_to_inputs / load_file_name).resolve()

                if (not path_to_load_file.exists()) or config.reset_loadprofile_data:
                    if config.reset_loadprofile_data:
                        logger.warning("Resetting loadprofile data")
                    path_to_loaddata_converter = path_to_utils / "process_load_data.py"
                    subprocess.run(
                        [
                            "python",
                            str(path_to_loaddata_converter),
                            "--input",
                            raw_input_load_file,
                            "--output",
                            str(path_to_load_file),
                            "--tz",
                            str(load_tz),
                        ]
                    )

                # load file into influx database
                path_to_load_file = str(path_to_load_file)
                _ = subprocess.getoutput(
                    f"docker cp {path_to_load_file} {container_id}:/{load_file_name}"
                )
                output = subprocess.getoutput(
                    f"docker exec -t {container_id} influx -import -path {load_file_name} -precision s"
                )
                logger.info(output)
            except:
                logger.error("Failed to load loadprofile data")
                raise
        else:
            logger.warning("Not loading loadprofile data")

        if config.load_weather_data:
            try:
                logger.warning("Erasing Existing Weather Data")
                _ = subprocess.getoutput(
                    f'docker exec -t {container_id} influx -database proven -execute "drop measurement weather"'
                )

                logger.warning("Loading weather data")
                path_to_weather_file = (path_to_inputs / weather_file_name).resolve()

                if (not path_to_weather_file.exists()) or config.reset_weather_data:
                    if config.reset_weather_data:
                        logger.warning("Resetting weather data")
                    path_to_weatherdata_converter = (
                        path_to_utils / "process_weather_data.py"
                    )
                    subprocess.run(
                        [
                            "python",
                            str(path_to_weatherdata_converter),
                            "--input",
                            raw_input_weather_file,
                            "--output",
                            str(path_to_weather_file),
                            "--tz",
                            str(weather_tz),
                        ]
                    )

                # Load weather data into influx database
                path_to_weather_file = str(path_to_weather_file)
                _ = subprocess.getoutput(
                    f"docker cp {path_to_weather_file} {container_id}:/{weather_file_name}"
                )
                output = subprocess.getoutput(
                    f"docker exec -t {container_id} influx -import -path {weather_file_name} -precision s"
                )
                logger.info(output)
            except:
                logger.error("Failed to load weather data")
                raise
        else:
            logger.warning("Not loading weather data")

        try:
            logger.warning("Erasing battery aggregator measurements")
            _ = subprocess.getoutput(
                f'docker exec -t {container_id} influx -database proven -execute "delete battery_aggregator_measurements"'
            )
        except:
            logger.error("Failed to erase battery aggregator measurements")
    except Exception as e:
        logger.error("The platform is not running it seems !")
        raise

    # ## Battery Initial SOC for Aggregator
    # Load the MS spreadsheet
    OCHRE_MS = pd.read_excel(config.ms_file).set_index("House_ID")

    if config.load_battery_E0_data:
        # Load DET results
        path_to_file = path_to_logs / "_dump_deterministic_data.pkl"

        with open(str(path_to_file), "rb") as infile:
            export_Dict = pickle.load(infile)

        # Which interval is active
        active_interval = int(
            (
                dt.datetime.fromtimestamp(config.rt_dt0).astimezone(tz)
                - dt.datetime.fromtimestamp(config.case_dt0).astimezone(tz)
            )
            / dt.timedelta(hours=1)
        )

        # export SOC in Wh from DET result for Batt Aggregator
        Batt_SOC = {
            der_ID: {
                "E_init": export_Dict["E_All"][der_ID][active_interval] * config.Sbase
            }
            for der_ID in export_Dict["E_All"].keys()
        }
        # Fix the Data in Spreadsheet

        # Aggregate Models
        for agg_name, aggregate in config.aggregate_models_batt_aggregator.items():
            total_cap = (
                sum(
                    aggregate[house_nr]["Energy Capacity (kWh)"]
                    for house_nr in aggregate.keys()
                )
                * 1000
            )
            # SOC rounded to 2 decimal
            soc = round(Batt_SOC[agg_name]["E_init"] / total_cap, 2)
            soc_kwh = round(soc * total_cap / 1000, 2)
            logger.info(
                f"Aggregate {agg_name} SOC initial condition for Simulation: {soc} p.u. ({soc_kwh} kWh)"
            )
            for house_nr in aggregate.keys():
                OCHRE_MS.loc[house_nr, "Initial SOC"] = soc
                aggregate[house_nr]["State of Charge (-)"] = soc

        # Utility Models
        for utility_name, utility in config.utility_models_batt_aggregator.items():
            total_cap = (
                sum(
                    utility[house_nr]["Energy Capacity (kWh)"]
                    for house_nr in utility.keys()
                )
                * 1000
            )
            # SOC rounded to 2 decimal
            soc = round(Batt_SOC[utility_name]["E_init"] / total_cap, 2)
            soc_kwh = round(soc * total_cap / 1000, 2)
            logger.info(
                f"Aggregate {utility_name} SOC initial condition for Simulation: {soc} p.u. ({soc_kwh} kWh)"
            )
            for house_nr in utility.keys():
                OCHRE_MS.loc[house_nr, "Initial SOC"] = soc
                utility[house_nr]["State of Charge (-)"] = soc

    # Export new MS
    export_path = path_to_archive / Path(config.ms_file).name
    export_path = str(export_path)
    OCHRE_MS.reset_index().set_index("ID").reset_index().to_excel(
        export_path, index=False, sheet_name="DOOM Input file data"
    )

    # Copy MS to the platform
    try:
        output = subprocess.getoutput("docker ps -qf name=gridappsd")
        container_id = output
        path_to_ms_file = export_path
        _ = subprocess.getoutput(
            f"docker cp -a {path_to_ms_file} {container_id}:gridappsd/services/gridappsd-ochre/inputs/MS/"
        )
        logger.info(
            "Successfully loaded the updated master spreadsheet for OCHRE into the GridAPPS-D Docker"
        )
    except Exception as e:
        logger.error(repr(e))
        logger.error("Failed to copy MS to the platform")

    ##########################################################################################
    # ## Launch Simulation in GridApps
    try:
        # Connect to GridAPPS-D Platform
        gapps = GridAPPSD()
        if not gapps.connected:
            raise Exception("GridAPPSD not connected")

        if config.use_ochre:
            logger.warning("Using OCHRE simulation")
            run_config = json.load(open(config.sim_file_ochre))
        else:
            logger.warning("Using GRIDLAB simulation")
            run_config = json.load(open(config.sim_file_gapps))

        start_timestamp = int(
            (
                dt.datetime.fromtimestamp(config.rt_dt0).astimezone(tz)
                - dt.timedelta(minutes=config.simulation_start_offset_minutes)
            ).timestamp()
        )
        if run_config["simulation_config"]["start_time"] != start_timestamp:
            logger.warning(
                f'Simulation start time does not match the config file.\n\tConfig: {run_config["simulation_config"]["start_time"]}\n\tNotebook: {start_timestamp} <---- Fixing the configuration to use this one.'
            )
            run_config["simulation_config"]["start_time"] = start_timestamp

        start_datetime_tz = dt.datetime.fromtimestamp(start_timestamp).astimezone(tz)
        start_datetime_utc = dt.datetime.fromtimestamp(start_timestamp).astimezone(
            pytz.utc
        )
        logger.critical(
            f"Simulation starts at {start_datetime_tz}({start_datetime_utc} in UTC)[ timestamp:  {start_timestamp}]"
        )

        # We give a buffer to the Simulation to account for the delay in starting the apps after the start of the simulation
        if config.sim_time != -1:
            total_offset = (
                config.simulation_start_offset_minutes
                + config.simulation_end_offset_minutes
            )
            sim_duration = config.sim_time + total_offset * 60
            sim_time = config.sim_time + 60 * config.simulation_end_offset_minutes
            logger.warning(
                f"We give a buffer of {config.simulation_start_offset_minutes} min. ({config.simulation_start_offset_minutes*60} s.) to the Simulation to account for the delay in starting the apps after the start of the simulation"
            )
            logger.warning(
                f"We add an extra {config.simulation_end_offset_minutes} min. of Simulation"
            )
        else:
            sim_duration = config.sim_time
        logger.critical(
            f"FRS Simulation duration: {sim_time}s ({dt.timedelta(seconds=sim_time)})"
        )

        if run_config["simulation_config"]["duration"] != sim_duration:
            logger.warning(
                f'Simulation duration does not match the config file.\n\tConfig: {run_config["simulation_config"]["duration"]}\n\tNotebook: {sim_duration} <---- Fixing the configuration to use this one.'
            )
            run_config["simulation_config"]["duration"] = sim_duration
        logger.critical(
            f"GridLab Simulation duration: {sim_duration}s ({dt.timedelta(seconds=sim_duration)})"
        )

        # Create Simulation object
        simulation_obj = Simulation(gapps, run_config)
        # Start Simulation
        simulation_obj.start_simulation(timeout=60)
        # Obtain Simulation ID
        simulation_id = simulation_obj.simulation_id
        logger.warning(
            f"Successfully started simulation with simulation_id: {simulation_id}"
        )

        myIOs = IObackup(
            simulation_id=simulation_id,
            model_id=config.model_id,
            path_to_repo=path_to_repo,
            file_all_data=config.file_all_data,
            path_to_export=path_to_logs,
            tz=config.tz,
        )

        # Topic for automation
        automation_topic = myIOs.service_topic("automation", "input")
        automator.setup_automator(myIOs, input_topic=automation_topic)

    except Exception as e:
        logger.error(repr(e))
        raise e

    # ## Waiting for GridLab-D / OCHRE to Start
    # We should sleep for a minute to let GridAPPS-D spinup the OCHRE simulation
    if config.use_ochre:
        time.sleep(60)
    else:
        time.sleep(10)

    # ## Copy Weather and Load files
    # copying weather file
    if config.log_orchestrator <= logging.INFO:
        output = subprocess.getoutput("docker ps -qf name=gridappsd")
        container_id = output
        try:
            logger.info("Copying weather file")
            _ = subprocess.getoutput(
                f'docker cp {container_id}:/tmp/gridappsd_tmp/{simulation_id}/model_weather.csv {path_to_archive / "weather_data_gridapps.csv"}'
            )
        except:
            logger.error("Failed to copy weather file")
        try:
            logger.info("Copying load file")
            _ = subprocess.getoutput(
                f'docker cp {container_id}:/tmp/gridappsd_tmp/{simulation_id}/ieeezipload.player {path_to_archive / "ieeezipload.player"}'
            )
        except:
            logger.error("Failed to copy load file")

    ##########################################################################################
    # ## Start Orchestrator
    automator.start_task("FRS Orchestrator Startup")

    app_config = {}
    app_config["mrid"] = config.mrid_orchestrator
    app_config["tz"] = config.tz
    app_config["file_all_data"] = config.file_all_data
    app_config["Sbase"] = config.Sbase
    app_config["sim_time"] = sim_time
    app_config["log_level"] = config.log_orchestrator
    app_config["path_to_repo"] = str(path_to_repo)
    app_config["path_to_export"] = str(path_to_logs)
    app_config["path_to_archive"] = str(path_to_archive)

    try:
        _ = automator.add_process(
            config.path_to_orchestrator, simulation_id, request, app_config
        )
        # Orchestrator input for commands
        orchestrator_topic = myIOs.service_topic(config.mrid_orchestrator, "input")

    except Exception as e:
        logger.error("Something went wrong launching Orchestrator")
        logger.error(repr(e))
        raise e

    # Wait till the Orchestrator flags the end of its initialization
    automator.wait_for_task()

    # ## Orchestrator Hotstart
    automator.start_task("FRS Orchestrator Hotstart")

    options = {"path_to_file": config.path_to_hotstart_file}

    automator.send_command(orchestrator_topic, "hotstart", options)

    # Wait till the Orchestrator flags the end of its hotstart
    automator.wait_for_task()

    ##########################################################################################
    # ## Startup the other Apps
    app_config = {}
    app_config["tz"] = config.tz
    app_config["Sbase"] = config.Sbase
    app_config["tmstp_start"] = int(config.rt_dt0)
    app_config["sim_time"] = sim_time
    app_config["path_to_repo"] = str(path_to_repo)
    app_config["path_to_export"] = str(path_to_logs)
    app_config["path_to_archive"] = str(path_to_archive)
    app_config["file_all_data"] = config.file_all_data
    app_config["automation_topic"] = automation_topic

    try:
        automator.start_task("ISO Startup")
        app_config["mrid"] = config.mrid_iso
        app_config["log_level"] = config.log_iso
        app_config["message_period"] = config.dispatch_period_iso

        automator.add_process(config.path_to_iso, simulation_id, request, app_config)
        # Wait till the ISO flags the end of its initialization
        automator.wait_for_task()
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching ISO")
        logger.error(repr(e))
        raise e

    try:
        automator.start_task("RT Controller Startup")
        if not config.use_rt_controller:
            raise (SkippingApp("RT Controller"))

        app_config["mrid"] = config.mrid_rt_controller
        app_config["log_level"] = config.log_rt_controller
        app_config["message_period"] = config.period_rt_controller
        app_config["substation_ID"] = myIOs.get_substation_ID()
        app_config["substation_measurement_IDs"] = (
            config.substation_measurement_IDs_rt_controller
        )
        app_config["PV_measurement_IDs"] = config.PV_measurement_IDs_rt_controller
        app_config["Kp"] = config.Kp_rt_controller
        app_config["Ki"] = config.Ki_rt_controller
        app_config["Kd"] = config.Kd_rt_controller
        app_config["MPC_error_rollover"] = config.rollover_error_rt_controller
        app_config["ISO_ramp"] = config.iso_ramp_rt_controller
        app_config["ISO_ramp_n_steps"] = config.iso_ramp_n_step_rt_controller
        app_config["cancelled_DERs"] = cancelled_DERs
        app_config["ignored_DERs"] = ignored_DERs

        automator.add_process(
            config.path_to_rt_controller, simulation_id, request, app_config
        )
        # Wait till the RT controller flags the end of its initialization
        automator.wait_for_task()
    except SkippingApp as e:
        automator.stop_task(e, bypass_stats=True)
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching RT Controller")
        logger.error(repr(e))

    try:
        automator.start_task("Dispatcher Startup")
        if not config.use_dispatcher:
            raise (SkippingApp("Dispatcher"))

        app_config["mrid"] = config.mrid_dispatcher
        app_config["log_level"] = config.log_dispatcher
        app_config["PV_list"] = config.PV_list_dispatcher

        automator.add_process(
            config.path_to_dispatcher, simulation_id, request, app_config
        )
        # Wait till the Dispatcher flags the end of its initialization
        automator.wait_for_task()
    except SkippingApp as e:
        automator.stop_task(e, bypass_stats=True)
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching Dispatcher")
        logger.error(repr(e))

    try:
        automator.start_task("Battery Aggregator Startup")
        if not config.use_batt_aggregator:
            raise (SkippingApp("Battery Aggregator"))

        app_config["mrid"] = config.mrid_batt_aggregator
        app_config["log_level"] = config.log_batt_aggregator
        app_config["aggregate_models"] = config.aggregate_models_batt_aggregator
        app_config["utility_models"] = config.utility_models_batt_aggregator
        app_config["meas_IDs"] = config.meas_ids_batt_aggregator

        automator.add_process(
            config.path_to_batt_aggregator, simulation_id, request, app_config
        )
        # Wait till the Batt Aggregator flags the end of its initialization
        automator.wait_for_task()
    except SkippingApp as e:
        automator.stop_task(e, bypass_stats=True)
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching Battery Aggregator")
        logger.error(repr(e))

    try:
        automator.start_task("ADMS Publisher Startup")
        if not config.use_adms_publisher:
            raise (SkippingApp("ADMS Publisher"))
        app_config["mrid"] = config.mrid_adms_publisher
        app_config["log_level"] = config.log_adms_publisher
        adms_event_list = [(config.adms_constraint_time, config.adms_constraint_file)]
        app_config["event_list"] = adms_event_list

        automator.add_process(
            config.path_to_adms_publisher, simulation_id, request, app_config
        )
        # Wait till the Batt Aggregator flags the end of its initialization
        automator.wait_for_task()
    except SkippingApp as e:
        automator.stop_task(e, bypass_stats=True)
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching ADMS Publisher")
        logger.error(repr(e))

    try:
        automator.start_task("TMM Startup")
        # In this Open Source Release, TMM is not included
        raise (SkippingApp("TMM/THEMS"))

        # Regular code for when the TMM is included
        if not config.use_tmm:
            raise (SkippingApp("TMM/THEMS"))

        app_config["log_level"] = config.log_tmm
        app_config["mrid"] = config.mrid_tmm
        app_config["timestep"] = config.timestep_tmm
        app_config["iteration_offset"] = config.iteration_offset_tmm
        app_config["MPC_n_timesteps"] = config.MPC_n_timestep
        app_config["MPC_timestep_period"] = config.MPC_timestep_period
        app_config["models"] = config.models_tmm

        automator.add_process(config.path_to_tmm, simulation_id, request, app_config)
        # Wait till the TMM flags the end of its initialization
        automator.wait_for_task()
    except SkippingApp as e:
        automator.stop_task(e, bypass_stats=True)
    except Exception as e:
        automator.stop_task()
        logger.error("Something went wrong launching TMM")
        logger.error(repr(e))

    ##########################################################################################
    # ## Start all commands
    automator.start_task("FRS Orchestrator Init MPC")
    # Orchestrator: Start MPC Process
    options = {}
    options.update({"tmstp_start": int(config.MPC_t0)})
    options.update({"iteration_offset": config.MPC_iteration_offset})
    options.update({"n_timestep": config.MPC_n_timestep})
    options.update({"timestep_period": config.MPC_timestep_period})
    options.update({"n_skip": config.MPC_n_skip})
    options.update({"opt_R_dist": config.MPC_opt_R_dist})
    options.update({"sigma_multiplier": config.MPC_sigma_multiplier})
    options.update({"n_thermal": config.MPC_n_thermal})
    options.update({"unity_powerfactor": config.MPC_unity_powerfactor})
    options.update({"power_weight": config.MPC_power_weight})
    options.update({"transactive_weight": config.MPC_transactive_weight})
    options.update({"reserve_weight": config.MPC_reserve_weight})
    options.update(
        {"substation_deviation_weight": config.MPC_substation_deviation_weight}
    )
    options.update(
        {
            "substation_deviation_price_override": config.MPC_substation_deviation_price_override
        }
    )
    options.update({"reserves_deviation_weight": config.MPC_reserves_deviation_weight})
    options.update(
        {
            "reserves_deviation_price_override": config.MPC_reserves_deviation_price_override
        }
    )
    options.update({"pv_curtailment_weight": config.MPC_pv_curtailment_weight})
    options.update(
        {"battery_charge_discharge_weight": config.MPC_battery_charge_discharge_weight}
    )
    options.update({"battery_deviation_weight": config.MPC_battery_deviation_weight})
    options.update({"E_BAT_final_flag": config.MPC_E_BAT_final_flag})
    options.update({"E_EV_final_flag": config.MPC_E_EV_final_flag})
    options.update({"E_FL_final_flag": config.MPC_E_FL_final_flag})

    automator.send_command(orchestrator_topic, "MPC", options)
    # Wait till the Orchestrator flags the end of its initialization
    automator.wait_for_task()

    automator.start_task("ISO Turn RT Dispatch ON")
    iso_topic = myIOs.service_topic(config.mrid_iso, "input")
    # Turn on the RT dispatch of ISO
    options = {}
    options.update({"date_override": config.case_dt0_iso})
    options.update({"force_static": config.use_static_data_iso})
    automator.send_command(iso_topic, "RT_dispatch_ON", options)
    # Wait till ISO flags the end of its process
    automator.wait_for_task()

    automator.start_task("Simulation is ongoing")
    # Wait till the simulation is completed
    automator.wait_for_task()

    # ## CLEAN UP
    automator.clean_processes(simulation_obj=simulation_obj)
    logger.critical("End of MPC / RT Simulation")

    # ## Archive Results
    path_to_exp_results = paths["export"]
    path_to_exp_results.mkdir(parents=True, exist_ok=True)

    folders_to_copy = {}
    folders_to_copy["archive"] = {
        "src": path_to_archive,
        "dest": path_to_exp_results / "archive",
    }
    folders_to_copy["logs"] = {
        "src": path_to_logs,
        "dest": path_to_exp_results / "logs",
    }
    folders_to_copy["inputs"] = {
        "src": path_to_inputs,
        "dest": path_to_exp_results / "inputs",
    }

    # Archive the Results
    for fn, folder in folders_to_copy.items():
        try:
            shutil.copytree(folder["src"], folder["dest"])
            logger.info(f"{fn} folder archived to {folder['dest']}")
        except Exception as e:
            logger.error(f"{fn} folder could not be archived to {folder['dest']}")
            logger.error(repr(e))


if __name__ == "__main__":
    run()
