import os
from concurrent.futures import ProcessPoolExecutor
import shutil
from itertools import repeat
import numpy as np
import trimesh

from candidate import Candidate
from crop import Crop
from energyplus_model_construction import build_model
from energyplus_simulation import run_simulation
from energyplus_output_handling import parse_output
from crop_model_simulation import simulate_annual_expected_yields
from config import GROWING_LEVEL_INITIAL_FLOOR_OFFSET, GROWING_LEVEL_Z_INTERVAL, USABLE_GROWING_VOLUME_PROPORTION, LED_EFFICIENCY_UMOL_PER_J, ELECTRICITY_PRICE_USD_PER_KWH, HEATING_ENERGY_PRICE_USD_PER_KWH, ANNUAL_AVERAGE_HEATING_SYSTEM_EFFICIENCY, ANNUAL_AVERAGE_COOLING_SYSTEM_COP

# https://www.apogeeinstruments.com/conversion-ppfd-to-lux/#:~:text=Multiply%20the%20Lux%20by%20the,1%20(108%2C000%20%E2%88%97%200.0185).
#PAR (phtosynthetic active radiation) only counts photons in roughly the 400-700nm range, as these are the wavelengths that plants use for photosynthesis
#this conversion factor is a typical estimate for daylight, from the above site
DAYLIGHT_LUX_TO_PAR_FACTOR = 0.0185 # µmol / s / lm
# https://www.gigahertz-optik.com/en-us/service-and-support/knowledge-base/measurement-of-par/
JOULES_PER_UMOL_PAR = 1 / 4.6 # J / µmol
#3600 for seconds per hour
DAYLIGHT_LUMEN_HOUR_TO_JOULE_PAR = 3600 * DAYLIGHT_LUX_TO_PAR_FACTOR * JOULES_PER_UMOL_PAR # J / lumen hour

#increasing this value increases the effective yield value target of the simulation
#this is not 1 to 1, so experiment with it to find a good value
SQRT_YIELD_MULTIPLIER = 350


def assess_candidate_fitness_parallel(
    candidates: list[Candidate], 
    simulation_crop: Crop,
    base_directory: str, 
    weather_file_path: str, 
    delete_after_evaluation: bool = True, 
    max_workers: int = 8
) -> None:
    if not candidates:
        return
    
    #run energyplus simulations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _evaluate_single_candidate, 
            candidates, 
            repeat(simulation_crop),
            repeat(base_directory), 
            repeat(weather_file_path), 
            repeat(delete_after_evaluation)
        )

    #update fitness values, also give extra info to the candidate object
    for candidate, return_info in zip(candidates, results):
        candidate.fitness = return_info["fitness"]
        candidate.annual_yield_value_USD = return_info["annual_yield_value_USD"]
        candidate.annual_lighting_cost_USD = return_info["annual_lighting_cost_USD"]
        candidate.annual_heating_cost_USD = return_info["annual_heating_cost_USD"]
        candidate.annual_cooling_cost_USD = return_info["annual_cooling_cost_USD"]
        candidate.annual_crop_yield_kg = return_info["annual_crop_yield_kg"]
        candidate.mean_daily_total_PAR_energy_per_area_MJ = return_info["mean_daily_total_PAR_energy_per_area_MJ"]


#identical to assess_candidate_fitness_parallel, but only for a single candidate
def assess_candidate_fitness_single(
    candidate: Candidate,
    simulation_crop: Crop,
    base_directory: str,
    weather_file_path: str,
    delete_after_evaluation: bool = True
) -> None:
    if not candidate:
        return
    
    return_info = _evaluate_single_candidate(
        candidate, 
        simulation_crop,
        base_directory, 
        weather_file_path, 
        delete_after_evaluation
    )

    #update fitness values, also give extra info to the candidate object
    candidate.fitness = return_info["fitness"]
    candidate.annual_yield_value_USD = return_info["annual_yield_value_USD"]
    candidate.annual_lighting_cost_USD = return_info["annual_lighting_cost_USD"]
    candidate.annual_heating_cost_USD = return_info["annual_heating_cost_USD"]
    candidate.annual_cooling_cost_USD = return_info["annual_cooling_cost_USD"]
    candidate.annual_crop_yield_kg = return_info["annual_crop_yield_kg"]
    candidate.mean_daily_total_PAR_energy_per_area_MJ = return_info["mean_daily_total_PAR_energy_per_area_MJ"]


def _evaluate_single_candidate(
    candidate: Candidate, 
    simulation_crop: Crop,
    base_directory: str, 
    weather_file_path: str, 
    delete_after_evaluation: bool = True
) -> float:
    #create the candidate directory, all files related to this candidate will be stored here
    candidate_directory = os.path.join(base_directory, candidate.id)
    if os.path.exists(candidate_directory):
        raise FileExistsError(f"candidate directory {candidate_directory} already exists")
    os.makedirs(candidate_directory, exist_ok=True)    

    #create the output directory for the energyplus simulation ofr this candidate
    sim_output_directory = os.path.join(candidate_directory, "sim_output")
    if os.path.exists(sim_output_directory):
        raise FileExistsError(f"simulation output directory {sim_output_directory} already exists")
    os.makedirs(sim_output_directory, exist_ok=True)

    #run the candidate through the entire energyplus simulation pipeline
    low_illuminance_map_slice_area, high_illuminance_map_slice_area = build_model(candidate, candidate_directory, weather_file_path)
    run_simulation(candidate_directory, sim_output_directory, weather_file_path)
    ideal_annual_heating_load_J, ideal_annual_cooling_load_J, floor_daily_hourly_mean_illuminance_values, centroid_daily_hourly_mean_illuminance_values = parse_output(sim_output_directory)

    return_info = _calculate_fitness(
        candidate, 
        simulation_crop,
        ideal_annual_heating_load_J,
        ideal_annual_cooling_load_J,
        floor_daily_hourly_mean_illuminance_values,
        centroid_daily_hourly_mean_illuminance_values,
        low_illuminance_map_slice_area,
        high_illuminance_map_slice_area
    )

    if delete_after_evaluation:
        #clean up files for the candidate as they are no longer needed
        _delete_candidate_files(candidate, base_directory)
    
    return return_info


def _calculate_fitness(
    candidate: Candidate, 
    simulation_crop: Crop,
    ideal_annual_heating_load_J: float, # Joules
    ideal_annual_cooling_load_J: float, # Joules
    floor_daily_hourly_mean_illuminance_values: list[list[float]],
    centroid_daily_hourly_mean_illuminance_values: list[list[float]],
    floor_illuminance_map_slice_area: float,
    centroid_illuminance_map_slice_area: float
) -> float:
    #do a bunch of checks here
    if len(floor_daily_hourly_mean_illuminance_values) != len(centroid_daily_hourly_mean_illuminance_values):
        raise ValueError("Floor and centroid illuminance maps have different number of days recordings")

    num_days = len(floor_daily_hourly_mean_illuminance_values)
    daily_total_lumen_hours = []
    for day_index in range(num_days):
        floor_hourly_mean_lux = floor_daily_hourly_mean_illuminance_values[day_index]
        floor_total_lux_hours = np.sum(floor_hourly_mean_lux)
        floor_total_lumen_hours = floor_total_lux_hours * floor_illuminance_map_slice_area

        centroid_hourly_mean_lux = centroid_daily_hourly_mean_illuminance_values[day_index]
        centroid_total_lux_hours = np.sum(centroid_hourly_mean_lux)
        centroid_total_lumen_hours = centroid_total_lux_hours * centroid_illuminance_map_slice_area

        #each illuminance map captures approximately half of the total volume (growing area), so they have equal weighting
        blended_total_lumen_hours_estimation = (floor_total_lumen_hours + centroid_total_lumen_hours) / 2.0

        #estimated_interior_total_lux_hours = area_weighted_floor_total_lux_hours + area_weighted_centroid_total_lux_hours
        daily_total_lumen_hours.append(blended_total_lumen_hours_estimation)

    mean_daily_total_lumen_hours = np.mean(daily_total_lumen_hours)
    mean_daily_total_PAR_energy = mean_daily_total_lumen_hours * DAYLIGHT_LUMEN_HOUR_TO_JOULE_PAR

    usable_growing_area = _calculate_usable_growing_area(candidate.building_structure.mesh, GROWING_LEVEL_INITIAL_FLOOR_OFFSET, GROWING_LEVEL_Z_INTERVAL, USABLE_GROWING_VOLUME_PROPORTION)
    mean_daily_total_PAR_energy_per_area = mean_daily_total_PAR_energy / usable_growing_area #J/m^2
    #convert J to MJ, this is the unit used in the crop model
    mean_daily_total_PAR_energy_per_area_MJ = mean_daily_total_PAR_energy_per_area / 1e6 #MJ/m^2

    temperature_setpoint = candidate.setpoints.temperature #C
    radiation_setpoint = candidate.setpoints.radiation #MJ/m^2
    if radiation_setpoint < mean_daily_total_PAR_energy_per_area_MJ:
        effective_radiation_setpoint = mean_daily_total_PAR_energy_per_area_MJ #light level can never fall below the ambient light level
    else:
        effective_radiation_setpoint = radiation_setpoint

    #expected yield value calculation
    annual_crop_yield_kg_per_m2 = simulate_annual_expected_yields(simulation_crop, temperature_setpoint, effective_radiation_setpoint) ######################### this should not be able to go below the ambient light level
    annual_crop_yield_kg = annual_crop_yield_kg_per_m2 * usable_growing_area #kg
    annual_yield_value_USD = annual_crop_yield_kg * simulation_crop.price_USD_per_kg

    #lighting cost calculation
    daily_lighting_intensity_shortfall = effective_radiation_setpoint - mean_daily_total_PAR_energy_per_area_MJ #MJ/m^2, if the setpoint is less than the ambient light level then it gets set equal to this, so shortfall will never be negative
    annual_lighting_intensity_shortfall = daily_lighting_intensity_shortfall * 365.0 #MJ/m^2
    annual_lighting_energy_shortfall_MJ = annual_lighting_intensity_shortfall * usable_growing_area #MJ
    annual_lighting_energy_shortfall_J = annual_lighting_energy_shortfall_MJ * 1e6 #J
    required_annual_umol_PAR = annual_lighting_energy_shortfall_J / JOULES_PER_UMOL_PAR #µmol PAR
    required_annual_lighting_electricity_J = required_annual_umol_PAR / LED_EFFICIENCY_UMOL_PER_J #J
    required_annual_lighting_electricity_kWh = required_annual_lighting_electricity_J / 3.6e6 #kWh
    annual_lighting_cost_USD = required_annual_lighting_electricity_kWh * ELECTRICITY_PRICE_USD_PER_KWH #USD

    #heating cost calculation
    #annual_heating_energy is in joules
    required_annual_heating_energy_J = ideal_annual_heating_load_J / ANNUAL_AVERAGE_HEATING_SYSTEM_EFFICIENCY #J
    required_annual_heating_energy_kWh = required_annual_heating_energy_J / 3.6e6 #kWh
    annual_heating_cost_USD = required_annual_heating_energy_kWh * HEATING_ENERGY_PRICE_USD_PER_KWH #USD

    #cooling cost calculation
    required_annual_cooling_electricity_J = ideal_annual_cooling_load_J / ANNUAL_AVERAGE_COOLING_SYSTEM_COP #J
    required_annual_cooling_electricity_kWh = required_annual_cooling_electricity_J / 3.6e6 #kWh
    annual_cooling_cost_USD = required_annual_cooling_electricity_kWh * ELECTRICITY_PRICE_USD_PER_KWH #USD

  
    fitness = SQRT_YIELD_MULTIPLIER * np.sqrt(annual_yield_value_USD) - annual_lighting_cost_USD - annual_heating_cost_USD - annual_cooling_cost_USD

    ###################################################### Alternative fitness functions
    # fitness = annual_yield_value_USD - annual_lighting_cost_USD - annual_heating_cost_USD - annual_cooling_cost_USD
    #
    # fitness = annual_yield_value_USD / (annual_lighting_cost_USD + annual_heating_cost_USD + annual_cooling_cost_USD + 100_000)
    #
    # yield_target = 10_000_000
    # if annual_crop_yield_kg < yield_target:
    #     fitness_yield_component = annual_crop_yield_kg
    # else:
    #     fitness_yield_component = yield_target + np.sqrt(annual_crop_yield_kg - yield_target)
    # fitness = fitness_yield_component - annual_lighting_cost_USD - annual_heating_cost_USD - annual_cooling_cost_USD
    # print(f"annual_crop_yield_kg: {annual_crop_yield_kg}, fitness_yield_component {fitness_yield_component}, fitness: {fitness}")
    ######################################################

    #print(f"yield_value: {annual_yield_value_USD}, lighting_cost: {annual_lighting_cost_USD}, heating_cost: {annual_heating_cost_USD}, cooling_cost: {annual_cooling_cost_USD}, fitness: {fitness}")
    return_info = {
        "fitness": fitness,
        "annual_yield_value_USD": annual_yield_value_USD,
        "annual_lighting_cost_USD": annual_lighting_cost_USD,
        "annual_heating_cost_USD": annual_heating_cost_USD,
        "annual_cooling_cost_USD": annual_cooling_cost_USD,
        "annual_crop_yield_kg": annual_crop_yield_kg,
        "mean_daily_total_PAR_energy_per_area_MJ": mean_daily_total_PAR_energy_per_area_MJ,
    }
    return return_info


def _calculate_usable_growing_area(mesh: trimesh.Trimesh, growing_rack_ground_offset: float, growing_rack_height_interval: float, usable_volume_proportion: float) -> float:
    mesh_bounds = mesh.bounds
    floor_height = mesh_bounds[0][2]
    ceiling_height = mesh_bounds[1][2]
    #remove the last level as it will have less than the required clearance above
    growing_level_heights = np.arange(floor_height + growing_rack_ground_offset, ceiling_height, growing_rack_height_interval)[:-1]

    available_area_per_level = []
    growing_plane_normal = [0, 0, 1] #points upward in the z direction
    for growing_level_z in growing_level_heights:
        growing_plane_origin = [0, 0, growing_level_z]
        section = mesh.section(plane_origin=growing_plane_origin, plane_normal=growing_plane_normal)

        if section is None:
            print(f"no section found at height {growing_level_z}")
            available_area_per_level.append(0)
            continue

        slice_2D, _ = section.to_2D()
        available_area = slice_2D.area
        available_area_per_level.append(available_area)
    
    total_usable_growing_area = sum(available_area_per_level) * usable_volume_proportion
    return total_usable_growing_area


def _delete_candidate_files(candidate: Candidate, base_directory: str) -> None:
    candidate_directory = os.path.join(base_directory, candidate.id)
    if not os.path.exists(candidate_directory):
        raise FileNotFoundError(f"Failed to delete candidate {candidate.id}: Directory '{candidate_directory}' not found.")

    try:
        shutil.rmtree(candidate_directory)
    except OSError as e:
        print(f"Failed to delete candidate {candidate.id}: {e}")
