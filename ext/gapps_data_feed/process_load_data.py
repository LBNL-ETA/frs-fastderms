"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import datetime as dt
import pandas as pd

import argparse
import pytz
import time
import os

raw_input_file = "../../data/Shared_Data_notinGIT/DEMO_data/ieeezipload.player"
bulk_output_file = "out_loadprofile_data.txt"
measurement = "ieeezipload"
database = "proven"
output_timestep = "1min"


def create_import_header_lines(database):
    s = "CREATE DATABASE " + database + "\n"
    s = s + "# DML\n"
    s = s + "# CONTEXT-DATABASE: " + database + "\n"
    s = s + "# CONTEXT-RETENTION-POLICY: autogen \n\n"
    return s


def strip_extra_chars(s):
    s = s.rstrip(" ")
    s = s.rstrip("\t")
    s = s.rstrip("\m")
    s = s.rstrip("\n")
    s = s.rstrip("\r")
    return s


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file name", default=raw_input_file)
parser.add_argument("--output", help="output file name", default=bulk_output_file)
parser.add_argument("--tz_in", help="Timezone of input data", default="UTC")

opts = parser.parse_args()
raw_input_file = opts.input
print(f"raw_input_file: {raw_input_file}")
# Load the input file
load_profile = pd.read_csv(
    raw_input_file, header=None, names=["time info", "load scale"]
)
load_profile["timedelta"] = pd.timedelta_range(
    start=0, periods=len(load_profile), freq=pd.Timedelta(load_profile["time info"][1])
)
load_profile = load_profile.drop(columns=["time info"]).set_index("timedelta")
load_profile = load_profile.resample(output_timestep).interpolate()

bulk_output_file = opts.output
print(f"exporting to {bulk_output_file}")
# Jan/1/2022 00:00:00 PST
tz_in_str = opts.tz_in
print(f"TZ str: {tz_in_str}")
tz_in = pytz.timezone(tz_in_str)
# Load Data must be in Year 2018 for GridApps !
seed_datetime = dt.datetime(2018, 1, 1, 0, 0, 0)
# seed_epoch_date = 1514764800

with open(bulk_output_file, "w+") as f:
    f.write(create_import_header_lines(database) + "\n")
    for day in range(0, 366):
        for index, row in load_profile.iterrows():
            epoch_index = int(
                tz_in.localize(
                    seed_datetime + dt.timedelta(days=day) + index
                ).timestamp()
            )
            f.write(
                measurement
                + " value="
                + str(row["load scale"])
                + " "
                + str(epoch_index)
                + "\n"
            )

print(f"Bulk load file created!")
