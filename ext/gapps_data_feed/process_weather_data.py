"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import datetime as dt
import pytz
import argparse

raw_input_file = "../../data/Shared_Data_notinGIT/DEMO_data/weather_data.csv"
bulk_output_file = "out_weather_data.txt"
measurement = "weather"
database = "proven"
# GridApps only processes data from year 2013 regardless of requested start date
year_override = 2013


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def process_value(s):
    if is_number(s):
        return s
    else:
        s = '"' + s + '"'
    return s


def strip_extra_chars(string_in):
    string_out = string_in.strip(" #\m\n\r\t")
    return string_out


def create_import_header_lines(database):
    s = "# DML \n"
    s = s + f"# CONTEXT-DATABASE: {database} \n"
    s = s + "# CONTEXT-RETENTION-POLICY: autogen \n\n"
    # s = "# DROP DATABASE " + database + "\n"
    # s = s + "# CREATE DATABASE " + database + "\n"
    # s = s + "# DML\n"
    # s = s + "# CONTEXT-DATABASE: " + database + "\n"
    # s = s + "# CONTEXT-RETENTION-POLICY: autogen \n"
    # s = s + "USE proven\n\n"
    return s


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file name", default=raw_input_file)
parser.add_argument("--output", help="output file name", default=bulk_output_file)
parser.add_argument("--tz_in", help="Timezone of input data", default="None")

opts = parser.parse_args()
raw_input_file = opts.input
print(f"raw_input_file: {raw_input_file}")
bulk_output_file = opts.output
print(f"bulk_output_file: {bulk_output_file}")
# Jan/1/2022 00:00:00 PST
tz_in_str = opts.tz_in
if tz_in_str == "None":
    print("No Timezone Override provided")
    tz_in = None
else:
    print(f"tz_in_str: {tz_in_str}")
    tz_in = pytz.timezone(tz_in_str)
    print(f"Timezone Used: {tz_in}")

count = 0
tags = ""
title_columns = 0
title_tokens = []
data_columns = 0
data_tokens = ""
with open(bulk_output_file, "w+") as file_out:
    file_out.write(create_import_header_lines(database))
    with open(raw_input_file) as fin:
        for line in fin:
            line = strip_extra_chars(line)
            count += 1
            if count == 1:
                # First line with Meta Data
                metadata = line.split(",")
                place = strip_extra_chars(metadata[0])
                place = place.lstrip("Ddata from").replace(" ", "\ ")
                lat = strip_extra_chars(metadata[1])
                lat = lat.strip("Llatitude ").replace(" ", "\ ")
                long = strip_extra_chars(metadata[2])
                long = long.strip("Llongitude ").replace(" ", "\ ")

                tags = f'place="{place}",lat="{lat}",long="{long}" '

                utc_offset = strip_extra_chars(metadata[4]).strip(
                    "Ttime Zzone UTCutc GMTgmt"
                )
                utc_minus_offset = dt.timezone(dt.timedelta(hours=int(utc_offset)))
                if tz_in is None:
                    print(f"Timezone Used: {utc_minus_offset}")
            elif count == 2:
                title_tokens = line.split(",")
                for token in title_tokens:
                    token = token.replace(" ", "\\ ")
                title_columns = len(title_tokens)

            elif count > 2:
                data_tokens = line.split(",")
                data_columns = len(data_tokens)
                if (data_columns) != title_columns:
                    print("ERROR:  " + line)
                    print(
                        "ERROR: " + str(data_columns) + " " + str(title_columns) + "\n"
                    )
                    break
                else:
                    column_counter = 0
                    newline = ""
                    while column_counter < title_columns:
                        if column_counter == 0:
                            newline = (
                                title_tokens[column_counter]
                                + "="
                                + process_value(data_tokens[column_counter])
                            )
                        else:
                            newline = (
                                newline
                                + ","
                                + title_tokens[column_counter]
                                + "="
                                + process_value(data_tokens[column_counter])
                            )
                        column_counter = column_counter + 1

                        d = data_tokens[0] + " " + data_tokens[1]
                        p = "%m/%d/%y %H:%M"

                    d_datetime = dt.datetime.strptime(d, p)
                    d_datetime = d_datetime.replace(year=year_override)
                    if tz_in is None:
                        # Dealing with static UTC offset
                        d_datetime = d_datetime.replace(tzinfo=utc_minus_offset)
                        epoch = int(d_datetime.timestamp())
                    else:
                        epoch = int(tz_in.localize(d_datetime).timestamp())
                    file_out.write(
                        measurement
                        + ","
                        + tags
                        + " "
                        + newline
                        + " "
                        + str(epoch)
                        + "\n"
                    )

print(f"Bulk weather file created!")
