import openstudio
import os
import pandas as pd
import numpy as np
import re

from config import ILLUMINANCE_MAP_SAMPLES_PER_AXIS

LINES_PER_HOUR_BLOCK = ILLUMINANCE_MAP_SAMPLES_PER_AXIS + 1 #+1 for the hour header line
HOUR_REGEX_PATTERN = re.compile(r'^\s*\d{2}/\d{2}\s+(\d{2}):\d{2}')
DATE_REGEX_PATTERN = re.compile(r'^\s*(\d{2}/\d{2})')

#encountered a rare issue where a design will get calculated illuminance values of ridiculously high amounts, more than is posisble from full sunlight
#so if any signle reading goes above this value below, an illuminance value of 0 is returned for that design
MAX_REASONABLE_LUX = 150_000 #daylight is about 120,000 lux max
class _IlluminanceOutlierError(Exception):
    pass


def parse_output(simulation_output_directory: str) -> tuple[float, list[list[float]], list[list[float]]]:
    #parse the simulation outputs
    sql_output_path = os.path.join(simulation_output_directory, "eplusout.sql")
    heating_energy_dataframe = _fetch_report_variable(sql_output_path, "DistrictHeatingWater:Facility")
    ideal_annual_heating_load_J = heating_energy_dataframe["DistrictHeatingWater:Facility"].sum()
    cooling_energy_dataframe = _fetch_report_variable(sql_output_path, "DistrictCooling:Facility")
    ideal_annual_cooling_load_J = cooling_energy_dataframe["DistrictCooling:Facility"].sum()

    illuminance_map_output_path = os.path.join(simulation_output_directory, "eplusmap.csv")
    floor_daily_hourly_mean_illuminance_values, centroid_daily_hourly_mean_illuminance_values = _parse_illuminance_map_csv(illuminance_map_output_path)

    return ideal_annual_heating_load_J, ideal_annual_cooling_load_J, floor_daily_hourly_mean_illuminance_values, centroid_daily_hourly_mean_illuminance_values


def _fetch_report_variable(sql_path: str, variable_name: str) -> pd.DataFrame:
    #fetches data for a given variable from the ReportData table.
    sql_file = openstudio.SqlFile(openstudio.path(sql_path))
    if not sql_file.connectionOpen():
        raise Exception(f"unable to open SQL file at {sql_path}")

    #get time indices
    time_query = f"""
        SELECT ReportData.TimeIndex 
        FROM ReportData 
        INNER JOIN ReportDataDictionary 
        ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex 
        WHERE ReportDataDictionary.Name = '{variable_name}';
    """
    results_time = sql_file.execAndReturnVectorOfInt(time_query)

    #get variable values
    value_query = f"""
        SELECT ReportData.Value 
        FROM ReportData 
        INNER JOIN ReportDataDictionary 
        ON ReportData.ReportDataDictionaryIndex = ReportDataDictionary.ReportDataDictionaryIndex 
        WHERE ReportDataDictionary.Name = '{variable_name}';
    """
    results_value = sql_file.execAndReturnVectorOfDouble(value_query)

    sql_file.close()

    if results_time.is_initialized() and results_value.is_initialized():
        time_indices = results_time.get()
        variable_values = results_value.get()
        return pd.DataFrame({"TimeIndex": time_indices, variable_name: variable_values})
    else:
        raise ValueError(f"No results found for variable: {variable_name}")
    

def _parse_illuminance_map_csv(csv_path: str) -> tuple[list[list[float]], list[list[float]]]:
    #parse the illuminance map CSV file and print the contents
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Illuminance map CSV file not found: {csv_path}")

    try:
        with open(csv_path, "r") as file:
            lines = file.readlines()
    except Exception as e:
        raise IOError(f"Error reading file {csv_path}: {e}")
    
    #check for error message I want to ignore
    #this one appears when a design is generated without any windows, this is a completely valid design so i remove the error message
    if lines[-1].strip().startswith("CloseReportIllumMaps"):
        lines = lines[:-1]

    #split data for the two illuminance maps
    search_start_index = (len(lines) // 2) - 100 #100 lines seems like good tolerance
    centroid_map_start_index = None
    for i in range(search_start_index, len(lines)):
        line = lines[i]
        if line.strip().startswith("Date/Time"):
            if "CENTROID" in line:
                centroid_map_start_index = i
                break

    if centroid_map_start_index is None:
        raise ValueError(f"CENTROID map not found in the csv file {csv_path}")

    floor_map_lines = lines[:centroid_map_start_index]
    centroid_map_lines = lines[centroid_map_start_index:]

    floor_map_day_blocks = _partition_csv_lines_to_day_blocks(floor_map_lines)
    centroid_map_day_blocks = _partition_csv_lines_to_day_blocks(centroid_map_lines)    

    try:
        first_floor_day_block = floor_map_day_blocks[0]
        #skip the day header line and grab the first hour block lines
        first_floor_hour_block = first_floor_day_block[1:(1 + LINES_PER_HOUR_BLOCK)]
        floor_inside_point_mask = _get_inside_point_mask(first_floor_hour_block)

        first_centroid_day_block = centroid_map_day_blocks[0]
        first_centroid_hour_block = first_centroid_day_block[1:(1 + LINES_PER_HOUR_BLOCK)]
        centroid_inside_point_mask = _get_inside_point_mask(first_centroid_hour_block)

        floor_daily_hourly_mean_illuminance_values = _get_daily_hourly_mean_illuminance_values(floor_map_day_blocks, floor_inside_point_mask)
        centroid_daily_hourly_mean_illuminance_values = _get_daily_hourly_mean_illuminance_values(centroid_map_day_blocks, centroid_inside_point_mask)
    except _IlluminanceOutlierError:
        #found an illuminance value which is far above what is physically possible, clearly broken so invalidates this design
        #set all values to 0.0 if any outlier is found
        floor_daily_hourly_mean_illuminance_values = [[0.0] for _ in range(365)]
        centroid_daily_hourly_mean_illuminance_values = [[0.0] for _ in range(365)]

    return floor_daily_hourly_mean_illuminance_values, centroid_daily_hourly_mean_illuminance_values


def _partition_csv_lines_to_day_blocks(lines: list[str]) -> list[list[str]]:
    day_header_indices = [i for i, line in enumerate(lines) if line.strip().startswith("Date/Time")]

    #split lines into blocks based on the header indices
    day_blocks = []
    for day_start_index, day_end_index in zip(day_header_indices, day_header_indices[1:]):
        day_blocks.append(lines[day_start_index:day_end_index])
    
    #add the last block, which extends to the end of the file
    last_day_header_index = day_header_indices[-1]
    day_blocks.append(lines[last_day_header_index:])

    return day_blocks


def _get_inside_point_mask(hour_block_lines: list[str]) -> np.ndarray:
    illuminance_value_grid_np, _ = _hour_block_to_illuminance_grid_np(hour_block_lines)
    #all points outside of the building yield identical illuminance values, which are also the minimum values in the grid
    #once we have this mask we can use it on all other illuminance map readings to filter out invalid points
    min_illuminance_value = illuminance_value_grid_np.min()
    inside_point_mask = (illuminance_value_grid_np != min_illuminance_value)
    return inside_point_mask


def _get_daily_hourly_mean_illuminance_values(day_blocks: list[list[str]], inside_point_mask: np.ndarray) -> list[list[float]]:
    daily_hourly_mean_illuminance_values = []
    for day_block_lines in day_blocks:
        first_hour_header_line = day_block_lines[1]
        date_match = DATE_REGEX_PATTERN.match(first_hour_header_line)
        if date_match:
            date = date_match.group(1)
        else:
            raise ValueError(f"invalid date found in header line: {first_hour_header_line}")

        hourly_mean_illuminance_values = []
        #first line is the day header, then the hour blocks follow
        #hour_header_indices = np.linspace(1, len(day_block_lines), num=LINES_PER_HOUR_BLOCK, dtype=int)
        #for hour_head_index in hour_header_indices:
        for hour_head_index in range(1, len(day_block_lines), LINES_PER_HOUR_BLOCK):
            hour_block_lines = day_block_lines[hour_head_index:(hour_head_index + LINES_PER_HOUR_BLOCK)]
            illuminance_value_grid_np, hour = _hour_block_to_illuminance_grid_np(hour_block_lines)
            inside_illuminance_values = illuminance_value_grid_np[inside_point_mask]
            if inside_illuminance_values.size == 0:
                mean_inside_illuminance_value = 0.0
            else:
                mean_inside_illuminance_value = inside_illuminance_values.mean()
            hourly_mean_illuminance_values.append(mean_inside_illuminance_value)
        
        daily_hourly_mean_illuminance_values.append(hourly_mean_illuminance_values)
    
    return daily_hourly_mean_illuminance_values


def _hour_block_to_illuminance_grid_np(hour_block_lines: list[str]) -> np.ndarray:
    #hour block includes the header line, extract the time from this line as an integer from 0-23
    hour_header_line = hour_block_lines[0]
    hour_match = HOUR_REGEX_PATTERN.match(hour_header_line)
    if hour_match:
        hour = int(hour_match.group(1))
    else:
        raise ValueError(f"invalid hour found in header line: {hour_header_line}")

    illuminance_value_lines = hour_block_lines[1:]
    illuminance_value_grid_np = _illuminance_lines_to_integer_grid_np(illuminance_value_lines)

    return illuminance_value_grid_np, hour


def _illuminance_lines_to_integer_grid_np(illuminance_value_lines: list[str]) -> np.ndarray:
    illuminance_value_grid = []
    for line in illuminance_value_lines:
        #strip line and seperate on comma
        comma_seperated_chunks = line.strip().split(",")
        illuminance_values = [int(value) for value in comma_seperated_chunks[1:]]
        illuminance_value_grid.append(illuminance_values)

    illuminance_value_grid_np = np.array(illuminance_value_grid)  
    #check for illuminance outliers, there is a rare bug which results in huge values in here
    #propogate this error up to _parse_illuminance_map_csv, where it is caught, and illuminance is set to 0.0 for that design
    if np.any(illuminance_value_grid_np > MAX_REASONABLE_LUX):
        #found an illuminance value which is far above what is physically possible, clearly broken so invalidates this design
        raise _IlluminanceOutlierError

    return illuminance_value_grid_np
