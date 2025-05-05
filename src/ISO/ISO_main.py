"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

from FAST_DERMS_CONTROL.common import fastderms_app, init_logging
from FAST_DERMS_CONTROL.IOs.IObackup import IObackup
from FAST_DERMS_CONTROL.IOs.IOclass import IOmodule

from gridappsd import GridAPPSD
from io import BytesIO
from pathlib import Path

import pandas as pd
import datetime as dt
import numpy as np

import pytz
import time
import requests
import zipfile
import argparse
import logging
import json
import sys

# App Name
app_name = "ISO"
__version__ = "0.9"

default_ISO_mrid = "ISO"

## Default values
# The reference start time for the period of simulation, it is assumed to be in Pacific time.
default_tz = "US/Pacific"
default_sim_start = dt.datetime(2022, 4, 1, 14, 0)

# Offset to run the ISO in seconds ahead of the timesteps
ISO_offset = -60

# A method to toggle on and off reserve dispatches. For now we are using just 5 minute energy.
Add_reserve_dispatch = False

# ISO Settings
# Day Ahead settings:
# I don't think we will ever use current time, but we'll leave the option open.
# Real_Time_Ops = False

# Dispatch settings
# timestep [s.]
default_timestep = 60
# simulation duration (in seconds)
default_sim_length = 24 * 60 * 60

# Logger
logger_name = f"__Main__{app_name}"
_main_logger = logging.getLogger(logger_name)


class mockISO(fastderms_app):
    def __init__(self, IO: IOmodule = None, **kw_args):

        # Forward module name to parent class
        kw_args.update({"name": kw_args.get("name", logger_name)})
        kw_args.update({"mrid": kw_args.get("mrid", default_ISO_mrid)})
        kw_args.update(
            {"message_period": kw_args.get("message_period", default_timestep)}
        )
        kw_args.update(
            {"iteration_offset": kw_args.get("iteration_offset", ISO_offset)}
        )

        super().__init__(**kw_args)

        self.IOs = IO
        self._simout_topic = self.IOs.simulation_topic("output")
        self._iso_topic = self.IOs.service_topic(self.mrid, "input")
        self.logger.warning(
            f"subscribing to:\n {self._simout_topic}\n {self._iso_topic}"
        )

        self._publish_to_topic = self.IOs.application_topic(self.mrid, "output")
        self.logger.warning(f"publishing to:\n {self._publish_to_topic}")

        ## Mock ISO for Day Ahead Market Awards
        # Establishing the Day for Simulation (input)
        # Mock ISO for Dispatch
        # Contains the code to inject mock ISO messages into GridApps to dispatch
        sim_time = kw_args.get("sim_time", default_sim_length)
        if sim_time == -1:
            sim_time = default_sim_length
        self.sim_duration = sim_time / 60 / 60

        # Inittializing App variables
        self.RTPriceDF = None
        self.RT_ON = False
        self._message_count = 0

        self.logger.info(f"Mock ISO initialized")
        self._automation_topic = kw_args.get(
            "automation_topic", self.IOs.service_topic("automation", "input")
        )
        self.IOs.send_gapps_message(self._automation_topic, {"command": "stop_task"})

    def on_message(self, headers, message):
        """Handle incoming messages on the simulation_output_topic for the simulation_id
        Parameters
        ----------
        headers: dict
            A dictionary of headers that could be used to determine topic of origin and
            other attributes.
        message: object
            A data structure following the protocol defined in the message structure
            of ``GridAPPSD``. Most message payloads will be serialized dictionaries, but that is
            not a requirement.
        """

        try:

            # Remote Control message
            if self._iso_topic in headers["destination"]:
                # Used to trigger what to run
                command = message["command"]
                if command == "DA":

                    force_static = message.get("force_static", False)
                    all_files = message.get("all_files", None)
                    date_override = message.get("date_override", None)
                    t0 = message.get("t0", None)
                    if t0 is not None:
                        self.set_next_iteration(t0, force=True)

                    t_start = self.next_iteration
                    if t_start is None:
                        raise ValueError("Start Timestamp missing")

                    # run the DA market
                    self.logger.info(
                        f'Running Day Ahead Market for {t_start.strftime("%Y-%m-%d")}'
                    )
                    # DA market is published at midnight of the day of the simulation
                    DA_market_timestamp = t_start.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )

                    # Code to generate the market award based on the day ahead optimization schedule/bids
                    # Collect the Day Ahead market results (this is may also be used by the ):
                    DA_prices, DA_awards = self.solve_DA_market(
                        RefDate=DA_market_timestamp,
                        RefNode="SDG_LNODE12A",
                        RefRegion="South",
                        force_static=force_static,
                        all_files=all_files,
                        date_override=date_override,
                    )

                    # Publish the market award to GridAPPS-D
                    status = self.publish_DA_market(
                        DA_market_timestamp, DA_prices, DA_awards
                    )

                    # In case it hasn't been 5 seconds since the Day-Ahead Call was made (CAISO API doesn't allow calls more often than 5 seconds)
                    # time.sleep(5)
                    self.IOs.send_gapps_message(
                        self._automation_topic, {"command": "stop_task"}
                    )
                    quit_after = message.get("quit_after", False)
                    if quit_after:
                        # Terminate the ISO
                        self._error_code = 2

                elif command == "RT_dispatch_ON":
                    try:
                        force_static = message.get("force_static", False)
                        date_override = message.get("date_override", None)
                        tmstp_start = message.get("tmstp_start", None)
                        if tmstp_start is not None:
                            self.set_next_iteration(tmstp_start, force=True)

                        t_start = self.next_iteration
                        if t_start is None:
                            raise ValueError("Start Timestamp missing")

                        # For now we load the Bids from the Day Ahead Market
                        DA_date = dt.datetime(t_start.year, t_start.month, t_start.day)
                        self.bids = self.IOs.ISO_gather_bids_FRS(DA_date, 24, 60)

                        self.logger.info("Turning on RT dispatch")
                        # Collect the RT Price Data that is used to generate dispatch (do this before loop):
                        # Gather the Real-time Market Prices that will be used for the Dispatch Loop:
                        self.RTPriceDF = self.get_CAISO_prices(
                            t_start,
                            duration=self.sim_duration,
                            Market="RTM",
                            node="SDG_LNODE12A",
                            AS_Region="South",
                            force_static=force_static,
                            date_override=date_override,
                        )
                    except Exception as e:
                        self.logger.error(f"Error getting real-time CAISO prices: {e}")
                    finally:
                        self.RT_ON = True
                        self.IOs.send_gapps_message(
                            self._automation_topic, {"command": "stop_task"}
                        )

                else:
                    self.logger.warning(f"Unknown command: {command}")

            # SIMOUT message
            elif self._simout_topic in headers["destination"]:
                # Simulation Output Received
                simulation_timestamp = message["message"]["timestamp"]
                self.logger.info(
                    f"SIMOUT message received at {self.timestamp_to_datetime(simulation_timestamp)}"
                )

                # Case to run the ISO dispatch
                # starting after t_start and iterating every timestep
                if self.RT_ON and (simulation_timestamp >= self.next_offset_timestamp):

                    self._message_count += 1
                    # Every message_period messages we are going to iterate the ISO
                    self.logger.info(f"ISO iteration {int(self._message_count)}")
                    ################################################################
                    #     INSERT MOCK ISO CODE HERE
                    ################################################################
                    timestamp = self.next_iteration.timestamp()
                    self.logger.debug(
                        f"Finding Bid for {self.next_iteration} (timestamp: {timestamp})"
                    )
                    Bids = self.get_current_bids(timestamp)
                    self.logger.debug(f"Bids are: {Bids}")
                    [Dispatch, MCPe] = self.solve_RT_dispatch(
                        Bids, timestamp, Add_reserve_dispatch=False
                    )

                    status = self.publish_RT_dispatch(timestamp, Dispatch, MCPe)

                    self.set_next_iteration(simulation_timestamp)
                    self.logger.debug(f"Next iteration is: {self.next_iteration}")
                    self.next_offset_timestamp = self.get_offset_timestamp(
                        self.next_iteration
                    )

                else:
                    if self.RT_ON:
                        self.logger.debug(
                            f"Waiting until next Iteration at {self.timestamp_to_datetime(self.next_offset_timestamp)}"
                        )
                    else:
                        self.logger.debug(f"RT dispatch is OFF")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logger.error(
                f"Error on line {exc_tb.tb_lineno}: {repr(e)},\n{exc_type}, {exc_obj}"
            )
            self._error_code = True
            raise

    def tz_PST_to_UTC(self, input_datetime):
        # Check if naive datetime object
        if input_datetime.tzinfo is None:
            local_datetime = self.local_tz.localize(input_datetime, is_dst=True)
        else:
            local_datetime = input_datetime
        # Update naive datetime object with local timezone
        utc_datetime = local_datetime.astimezone(pytz.utc)
        # Convert to UTC
        return utc_datetime

    def get_CAISO_prices(
        self,
        StartTime,
        duration=24,
        Market="DAM",
        node="SDG_LNODE12A",
        AS_Region="South",
        **kw_args,
    ):
        """
        prices from CAISO API
        """

        # Basic Inputs
        StrtTime = self.tz_PST_to_UTC(StartTime)
        EndTime = StrtTime + dt.timedelta(hours=duration)
        if Market == "DAM":
            market_id = "DAM"
            query_type = "PRC_LMP"
            ASquery_type = "PRC_AS"
            version = "12"
            ASversion = "12"
        elif Market == "RTM":
            market_id = "RTM"
            query_type = "PRC_INTVL_LMP"
            ASquery_type = "PRC_INTVL_AS"
            version = "2"
            ASversion = "1"
        else:
            print(
                "ERROR: Neither DAM or RTM was provided for market type.  Defaulting to DAM."
            )
            market_id = "DAM"
            query_type = "PRC_LMP"
            ASquery_type = "PRC_AS"
            version = "12"
            ASversion = "12"

        force_static = kw_args.get("force_static", False)
        if force_static:
            self.logger.info("Using static data")
            duration = int(np.ceil(duration))
            if Market == "DAM":
                prices = self.IOs.get_DET_price(
                    StartTime, duration, 60, force_static=force_static
                )
            elif Market == "RTM":
                prices = self.IOs.ISO_get_RT_price(
                    StartTime, duration, 60, force_static=force_static
                )

            col_starttime = [
                StrtTime + dt.timedelta(hours=hour) for hour in range(duration)
            ]
            col_endtime = [
                StrtTime + dt.timedelta(hours=hour + 1) for hour in range(duration)
            ]
            col_opr_dt = [date.date() for date in col_starttime]
            col_opr_hr = [date.hour for date in col_starttime]
            col_opr_inter = [0] * duration
            col_node = [f"{node}"] * duration
            col_market_id = [f"{market_id}"] * duration
            # Day Ahead price is stored in Col MW, while RT price is stored in col VALUE
            col_MW = prices["pi"]
            col_VALUE = prices["pi"]
            col_RU = prices["pi_rup"]
            col_RD = prices["pi_rdn"]

            FinalPriceDF = pd.DataFrame(
                np.array(
                    [
                        col_starttime,
                        col_endtime,
                        col_opr_dt,
                        col_opr_hr,
                        col_opr_inter,
                        col_node,
                        col_market_id,
                        col_MW,
                        col_VALUE,
                        col_RU,
                        col_RD,
                    ]
                ).transpose(),
                index=col_starttime,
                columns=[
                    "INTERVALSTARTTIME_GMT",
                    "INTERVALENDTIME_GMT",
                    "OPR_DT",
                    "OPR_HR",
                    "OPR_INTERVAL",
                    "NODE",
                    "MARKET_RUN_ID",
                    "MW",
                    "VALUE",
                    "RU",
                    "RD",
                ],
            )

            FinalPriceDF.index.name = "IntStartTime_GMT"

        else:
            self.logger.info("Using CAISO API")
            date_override = kw_args.get("date_override", None)
            if date_override is not None:
                query_date = dt.datetime.fromtimestamp(int(date_override)).astimezone(
                    self.local_tz
                )
                query_StartTime = StartTime.replace(
                    year=query_date.year, month=query_date.month, day=query_date.day
                )
                self.logger.warning(
                    f"Querying Data using date override: {query_StartTime}"
                )
                query_StrtTime = self.tz_PST_to_UTC(query_StartTime)
                query_EndTime = query_StrtTime + dt.timedelta(hours=duration)
            else:
                query_StrtTime = StrtTime
                query_EndTime = EndTime

            # Compute the time difference between requested datetime and query datetime
            time_diff = (StrtTime - query_StrtTime).total_seconds()
            if time_diff != 0:
                self.logger.warning(
                    f"Requested datetime is {time_diff} seconds different from query datetime"
                )

            # prepare the time in the right format (assumes it entered as a naive datetime that was meant to be in pacific time)
            strttime = query_StrtTime.strftime(
                "%Y%m%dT%H:%M-0000"
            )  #'20220227T07:00-0000' #times are in GMT
            endtime = query_EndTime.strftime(
                "%Y%m%dT%H:%M-0000"
            )  #'20220228T07:00-0000' #times are in GMT

            # prepare the query for LMP
            CAISOQuery = (
                "http://oasis.caiso.com/oasisapi/SingleZip?queryname="
                + query_type
                + "&startdatetime="
                + strttime
                + "&enddatetime="
                + endtime
                + "&version="
                + version
                + "&resultformat=6&market_run_id="
                + market_id
                + "&node="
                + node
            )

            # Perform Query to CAISO OASIS
            # Send HTTP GET request via requests
            self.logger.debug(f"Querying CAISO for prices:\n{CAISOQuery}")
            req = requests.get(CAISOQuery)
            # pull out the zip
            zipdata = zipfile.ZipFile(BytesIO(req.content))
            # read the content of the zipfile
            reader = zipdata.open(zipdata.namelist()[0]).readlines()
            # define headers
            headers = reader[0].decode("utf-8").replace("\n", "").split(",")
            # Create the data lists
            data = []
            for line in reader[1 : len(reader)]:
                data.append(line.decode("utf-8").replace("\n", "").split(","))

            # Create a pandas dataframe from the data, then drop all the extra stuff and index on time
            priceInDF = pd.DataFrame(data, columns=headers)
            priceDF = priceInDF[priceInDF["LMP_TYPE"] == "LMP"]
            priceDF = priceDF.drop(
                labels=[
                    "NODE_ID_XML",
                    "NODE_ID",
                    "XML_DATA_ITEM",
                    "PNODE_RESMRID",
                    "GRP_TYPE",
                    "POS",
                    "GROUP",
                ],
                axis=1,
            )
            priceDF["IntStartTime_GMT"] = pd.to_datetime(
                priceDF["INTERVALSTARTTIME_GMT"]
            )
            priceDF = priceDF.set_index("IntStartTime_GMT").sort_index()

            # wait a couple of seconds before sending a request
            time.sleep(5)

            # Perform Query to get the AS Services Prices
            CAISOQuery = (
                "http://oasis.caiso.com/oasisapi/SingleZip?queryname="
                + ASquery_type
                + "&startdatetime="
                + strttime
                + "&enddatetime="
                + endtime
                + "&version="
                + ASversion
                + "&resultformat=6&market_run_id="
                + market_id
            )

            # Perform Query to CAISO OASIS
            # Send HTTP GET request via requests
            req = requests.get(CAISOQuery)
            # pull out the zip
            zipdata = zipfile.ZipFile(BytesIO(req.content))
            # read the content of the zipfile
            reader = zipdata.open(zipdata.namelist()[0]).readlines()
            # define headers
            headers = reader[0].decode("utf-8").replace("\n", "").split(",")
            # Create the data lists
            data = []
            for line in reader[1 : len(reader)]:
                data.append(line.decode("utf-8").replace("\n", "").split(","))

            # Create a pandas dataframe from the data
            ASpriceInDF = pd.DataFrame(data, columns=headers)

            # check the Region
            Southern_AS_Regions = [
                "AS_CAISO",
                "AS_CAISO_EXP",
                "AS_SP26",
                "AS_SP26_EXP",
                "AS_SP15",
                "AS_SP15_EXP",
                "AS_CAISO_SP15",
                "AS_CAISO_SP26",
            ]
            Northern_AS_Regions = [
                "AS_CAISO",
                "AS_CAISO_EXP",
                "AS_NP26",
                "AS_NP26_EXP",
                "AS_NP15",
                "AS_NP15_EXP",
                "AS_CAISO_NP15",
                "AS_CAISO_NP26",
            ]

            if AS_Region == "South":
                Regions = Southern_AS_Regions
            elif AS_Region == "North":
                Regions = Northern_AS_Regions
            else:
                print(
                    "The AS Region provided was neither North or South, providing AS prices for the South."
                )
                Regions = Southern_AS_Regions

            # compute the actual prices from the shadow prices in the appropriate region
            Starts = set(ASpriceInDF["INTERVALSTARTTIME_GMT"].to_list())
            Services = set(ASpriceInDF["ANC_TYPE"].to_list())
            ASPriceHeaders = ["IntStartTime_GMT"]
            [ASPriceHeaders.append(Service) for Service in Services]
            # print(ASPriceHeaders)
            data = []
            for start in Starts:
                DFofInterest = ASpriceInDF[
                    ASpriceInDF["INTERVALSTARTTIME_GMT"] == start
                ]
                DFofInterest = DFofInterest[DFofInterest["ANC_REGION"].isin(Regions)]
                rowdata = [pd.to_datetime(start)]
                for Service in Services:
                    NewDF = DFofInterest[DFofInterest["ANC_TYPE"] == Service]
                    TotPrice = NewDF["MW"].astype(float).sum()
                    # print(NewDF)
                    rowdata.append(TotPrice)
                data.append(rowdata)
            ASPriceDF = pd.DataFrame(data, columns=ASPriceHeaders)
            ASPriceDF = ASPriceDF.set_index("IntStartTime_GMT").sort_index()
            FinalPriceDF = priceDF.join(ASPriceDF)

            # Fix the timestamp to be the requested time (and not the override)
            FinalPriceDF.index = FinalPriceDF.index + pd.Timedelta(seconds=time_diff)

        return FinalPriceDF

    def solve_DA_market(
        self,
        RefDate=dt.datetime(2019, 8, 16, 0, 0),
        RefNode="SDG_LNODE12A",
        RefRegion="South",
        **kw_args,
    ):
        # Grab the Resolved Prices for DAM
        StartTime = RefDate.replace(
            hour=0, minute=0, second=0, microsecond=0
        )  # Does it this way so the ref date is always starting at midnight local
        nd = RefNode  # A random SDG&E node
        dur = 24  # Number of hours we'd like the forecast to include
        ASR = RefRegion  # A string, either 'North' or 'South' that indicates a AS region we are creating prices for
        Mket = "DAM"  # A string, either "DAM" or "RTM" that indicates the market that we are pulling prices for.
        MCPs = self.get_CAISO_prices(
            StartTime, duration=dur, Market=Mket, node=nd, AS_Region=ASR, **kw_args
        )

        prices = pd.DataFrame(
            np.array(
                [
                    MCPs["MW"].astype(float).to_list(),
                    MCPs["RU"].to_list(),
                    MCPs["RD"].to_list(),
                ]
            ).transpose(),
            index=MCPs.index,
            columns=["MCP Energy", "MCP Rup", "MCP Rdn"],
        )

        # Gather the Bid from DA optim results
        bids = self.IOs.ISO_gather_bids_FRS(StartTime, dur, 60, **kw_args)

        awards = {}
        for feeder_ID in bids.keys():
            self.logger.info(f"Received Bid for Feeder ID: {feeder_ID}")
            EnBid = bids[feeder_ID]["EnBid"]
            self.logger.debug(f"EnBid:\n {EnBid}")
            ResBid = bids[feeder_ID]["ResBid"]
            self.logger.debug(f"ResBid:\n {ResBid}")
            # resolve the market outcome for the resource (compare bids to the market clearing prices)
            Dispatch = []
            Rup = []
            Rdn = []
            all_timestamps = list(EnBid.keys())
            all_timestamps.sort()
            for hr in range(24):
                MCPe = float(
                    MCPs["MW"].to_list()[hr]
                )  # Energy Market Clearing Price for the current hour
                MCPrup = float(
                    MCPs["RU"].to_list()[hr]
                )  # Regulation Up Market Clearing price for the current hour
                MCPrdn = float(
                    MCPs["RD"].to_list()[hr]
                )  # Regulation down market clearing price for the current hour
                # Comparison for Energy
                if (
                    len(EnBid[all_timestamps[hr]]) == 1
                ):  # If the bid has one segment, treat it as a self schedule
                    Dispatch.append(EnBid[all_timestamps[hr]]["s1"]["power"])
                else:
                    Prices = [
                        EnBid[all_timestamps[hr]][segment]["price"]
                        for segment in list(EnBid[all_timestamps[hr]].keys())
                    ]
                    Powers = [
                        EnBid[all_timestamps[hr]][segment]["power"]
                        for segment in list(EnBid[all_timestamps[hr]].keys())
                    ]
                    # if price is less than price indicated in the segment, then I will consume segment power
                    prbool = [MCPe < price for price in Prices]
                    Dispatch.append(
                        np.array(Powers)[prbool].min()
                    )  # Assumes that the bid is monotonically increasing, I have not instituted error checking on the inputs.
                # Ancillary Services bids are a single price-quantity pair, no segments
                # R up
                if (
                    ResBid[all_timestamps[hr]]["Rup"]["price"] == 0
                ):  # treat this as a self schedule
                    Rup.append(ResBid[all_timestamps[hr]]["Rup"]["power"])
                elif ResBid[all_timestamps[hr]]["Rup"]["price"] <= MCPrup:
                    Rup.append(ResBid[all_timestamps[hr]]["Rup"]["power"])
                else:
                    Rup.append(0)
                # R down
                if (
                    ResBid[all_timestamps[hr]]["Rdn"]["price"] == 0
                ):  # treat this as a self schedule
                    Rdn.append(ResBid[all_timestamps[hr]]["Rup"]["power"])
                elif ResBid[all_timestamps[hr]]["Rdn"]["price"] <= MCPrdn:
                    Rdn.append(ResBid[all_timestamps[hr]]["Rdn"]["power"])
                else:
                    Rdn.append(0)

            awards[feeder_ID] = pd.DataFrame(
                np.array([Dispatch, Rup, Rdn]).transpose(),
                index=MCPs.index,
                columns=["MW", "Rup Award", "Rdn Award"],
            )

        # right now, this returns the market results in a pandas dataframe, we can change this to make those results look like the dictionary that should be posted to GridAPPs=D

        return prices, awards

    def publish_DA_market(self, RefDate, DA_prices, DA_awards):

        dt0_fcast = RefDate
        timestep_period_fcast = 60

        # Market Prices published to GridAPPS-D
        # Package Price fcast
        flat_price_fcast = {}
        flat_price_fcast["MCP"] = DA_prices["MCP Energy"].astype("float").tolist()
        flat_price_fcast["MCP_Rup"] = DA_prices["MCP Rup"].astype("float").tolist()
        flat_price_fcast["MCP_Rdn"] = DA_prices["MCP Rdn"].astype("float").tolist()

        self.IOs.publish_fcasts_gapps(
            "ISO",
            flat_price_fcast,
            dt0_fcast,
            24,
            timestep_period_fcast,
            topic=self._publish_to_topic,
        )

        # Market Dispatch published to GridAPPS-D
        for feeder_ID in DA_awards.keys():
            DA_results = DA_awards[feeder_ID]
            # Package Dispatch fcast
            flat_dispatch_fcast = {}
            flat_dispatch_fcast["MW_award"] = DA_results["MW"].astype("float").tolist()
            flat_dispatch_fcast["Rup_award"] = (
                DA_results["Rup Award"].astype("float").tolist()
            )
            flat_dispatch_fcast["Rdn_award"] = (
                DA_results["Rdn Award"].astype("float").tolist()
            )
            self.IOs.publish_fcasts_gapps(
                feeder_ID,
                flat_dispatch_fcast,
                dt0_fcast,
                24,
                timestep_period_fcast,
                topic=self._publish_to_topic,
            )

    def get_current_bids(self, next_timestamp):

        all_feeders = self.bids.keys()
        bids = {}
        for feeder in all_feeders:
            all_timestamps = list(self.bids[feeder]["EnBid"].keys())
            all_timestamps.sort()
            # First let's check if the next timestamp is before the first timestamp in the bids or after the last timestamp in the bids
            if next_timestamp < all_timestamps[0]:
                bids[feeder] = self.bids[feeder]["EnBid"][all_timestamps[0]]
            else:
                bids[feeder] = self.bids[feeder]["EnBid"][all_timestamps[-1]]
            # Then let's check if the next timestamp is in between two timestamps in the bids
            for current, next in zip(all_timestamps, all_timestamps[1:]):
                if current <= next_timestamp and next_timestamp < next:
                    bids[feeder] = self.bids[feeder]["EnBid"][current]
                    break

            # current_index = self.awards[feeder].index.get_indexer([next_timestamp], method='pad')[0]
            # if current_index == -1:
            #     self.logger.debug(f'Feeder: {feeder}, no awards for {next_timestamp}')
            #     if next_timestamp < self.awards[feeder].index[0]:
            #         self.logger.debug(f'Feeder: {feeder}, using first award for {next_timestamp}')
            #         current_index = 0
            #     else:
            #         self.logger.debug(f'Feeder: {feeder}, using last award for {next_timestamp}')
            # award = self.awards[feeder].iloc[current_index]['MW']
            # self.logger.debug(f'Feeder: {feeder}, awards for {next_timestamp}: {award} MW')
            # disptach[feeder] = award

        return bids

    def get_current_time_interval(self, next_timestamp):

        next_datetime = self.timestamp_to_datetime(next_timestamp)
        current_datetime = self.RTPriceDF.loc[
            self.RTPriceDF.index <= next_datetime
        ].last_valid_index()

        if current_datetime is None:
            current_datetime = self.RTPriceDF.index[0]

        return current_datetime

    def solve_RT_dispatch(self, Bids, timestamp, Add_reserve_dispatch=False):

        # resolve the market outcome for the resource (compare bids to the market clearing prices)
        DTtime = self.get_current_time_interval(timestamp)

        # Energy Market Clearing Price for the current hour
        MCPe = float(self.RTPriceDF.loc[DTtime]["VALUE"])

        Dispatch = {}
        for feeder_ID in Bids.keys():
            Bid = Bids[feeder_ID]
            # Comparison for Energy
            if len(Bid) == 1:  # If the bid has one segment, treat it as a self schedule
                Dispatch[feeder_ID] = Bid["s1"]["power"]
            else:
                Prices = [Bid[segment]["price"] for segment in list(Bid.keys())]
                Powers = [Bid[segment]["power"] for segment in list(Bid.keys())]
                # if price is less than price indicated in the segment, then I will consume segment power
                prbool = [MCPe < price for price in Prices]
                Dispatch[feeder_ID] = np.array(Powers)[
                    prbool
                ].min()  # Assumes that the bid is monotonically increasing, I have not instituted error checking on the inputs.
            # Ancillary Services bids are a single price-quantity pair, no segments

            if Add_reserve_dispatch:
                self.logger.warning(
                    "Unable to dispatch regulation reserves at the current time. Further dev needed."
                )

                # Need to add a value for the reserve dispatch.  Requires:
                # - loading a normalized dispatch value for the 4 second interval
                # - evaluating if it is a reserve up or down dispatch (positive or negative)
                # - scaling that by the reserve award for the resource in the correct direction.
                # Rup = float(DAResults.loc[DTtime]['R-Up Award'])
                # Rdn = float(DAResults.loc[DTtime]['R-Dn Award'])
                # - adding that value to the 5 minute dispatch value to get the 4 second dispatch

                # This does not re-evaluate the reserve offer provided day ahead to see if there is an increase in reserve quantities awarded.
                # We'd need to change the bid gathering to provide the hourly bid for reserves as well. If we did:
                # MCPrup = float(RTPriceDF.loc[DTtime]['RU'])#Regulation Up Market Clearing price for the current hour
                # MCPrdn = float(RTPriceDF.loc[DTtime]['RD'])#Regulation down market clearing price for the current hour
                ##R up
                # if ResBid['Rup']['price'] == 0: #treat this as a self schedule
                #    Rup=ResBid['Rup']['power']
                # elif ResBid['Rup']['price']<=MCPrup:
                #    Rup=ResBid['Rup']['power']
                # else:
                #    Rup=0
                ##R down
                # if ResBid['Rdn']['price'] == 0: #treat this as a self schedule
                #    Rdn = ResBid['Rup']['power']
                # elif ResBid['Rdn']['price']<=MCPrdn:
                #    Rdn = ResBid['Rdn']['power']
                # else:
                #    Rdn = 0
        return Dispatch, MCPe

    def publish_RT_dispatch(self, timestamp, dispatch, price):

        payload = {}
        payload["datatype"] = "dispatch"
        payload["message"] = []
        for feeder_ID in dispatch.keys():
            # Create a new message
            message = {}
            message["dispatch_timestamp"] = timestamp
            message["model_mrid"] = feeder_ID
            # Dispatch [MW]
            message["dispatch"] = dispatch[feeder_ID]
            #  Energy Price [$/MW]
            message["cleared_price"] = price

            payload["message"].append(message)
        # Publish the Dispatch to GridAPPS-D
        status = self.IOs.send_gapps_message(self._publish_to_topic, payload)

        return status


def _main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "simulation_id", help="Simulation id to use for responses on the message bus."
    )
    parser.add_argument("request", help="Path to the simulation request file")
    parser.add_argument("config", help="App Config")

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("'", ""))
    app_config = json.loads(opts.config.replace("'", ""))

    # Logging Facility
    log_level = app_config.get("log_level", logging.INFO)
    path_to_export = app_config.get("path_to_export", "./logs")
    path_to_export = Path(path_to_export).resolve()

    init_logging(app_name="ISO", log_level=log_level, path_to_logs=path_to_export)

    _main_logger.warning(
        "Mock ISO starting!!!-------------------------------------------------------"
    )

    simulation_id = opts.simulation_id
    _main_logger.debug(f"Info received from remote: Simulation ID {simulation_id}")

    time_multiplier = app_config.get("time_multiplier", 1)
    if time_multiplier != 1:
        _main_logger.warning(f"Time multiplier is set to {time_multiplier}")
        message_period = app_config.get("message_period", default_timestep)
        message_period = int(message_period / time_multiplier)
        _main_logger.warning(f"New message period is {message_period}")
        app_config.update({"message_period": message_period})

    _gapps = GridAPPSD()
    file_all_data = app_config.get("file_all_data", None)
    kw_args = {}
    if file_all_data is not None:
        # new way with 1 file
        _main_logger.debug(f"Info received from remote: All data file {file_all_data}")
        kw_args.update({"file_all_data": file_all_data})
    else:
        # old way with 2 files
        file_static_data = app_config.get("static_data", None)
        _main_logger.debug(
            f"Info received from remote: Static data file {file_static_data}"
        )
        kw_args.update({"file_static_data": file_static_data})

        file_fcast_data = app_config.get("fcast_data", None)
        _main_logger.debug(
            f"Info received from remote: Forecast data file {file_fcast_data}"
        )
        kw_args.update({"file_fcast_data": file_fcast_data})

    path_to_repo = app_config.get("path_to_repo", "../../")
    path_to_repo = Path(path_to_repo).resolve()

    # Instantiate IO module
    IO = IObackup(
        simulation_id=simulation_id,
        path_to_repo=path_to_repo,
        path_to_export=path_to_export,
        **kw_args,
    )

    mock_iso = mockISO(IO, **app_config)

    # All ther subscriptions:
    # SIM Output
    simout_topic = IO.simulation_topic("output")
    _gapps.subscribe(simout_topic, mock_iso)

    # ISO input for commands
    ISO_mrid = app_config.get("mrid", default_ISO_mrid)
    input_topic = IO.service_topic(ISO_mrid, "input")
    _gapps.subscribe(input_topic, mock_iso)

    sim_time = app_config.get("sim_time", default_sim_length)

    if sim_time == -1:
        _main_logger.info(f"Info received from remote: sim_time - until termination")
    else:
        _main_logger.info(f"Info received from remote: sim_time {sim_time} seconds")

    elapsed_time = 0
    time_to_sleep = 0.1
    while elapsed_time < sim_time or sim_time == -1:
        if not mock_iso.running():
            if mock_iso.error() == 2:
                _main_logger.warning("Mock ISO Terminated")
                mock_iso.IOs.send_gapps_message(
                    mock_iso._automation_topic, {"command": "stop_task"}
                )
            else:
                _main_logger.error("Mock ISO failed")
                mock_iso.IOs.send_gapps_message(
                    mock_iso._automation_topic, {"command": "error"}
                )
            break

        elapsed_time += time_to_sleep
        time.sleep(time_to_sleep)

    _main_logger.warning(
        "Mock ISO finished!!!-------------------------------------------------------"
    )


if __name__ == "__main__":
    _main()
