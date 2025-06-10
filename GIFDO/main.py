import os
from collections import Counter
import pickle
import matplotlib
matplotlib.use("Agg") #tkinter was causing some obscure threading related crashes, this is supposed to fix that

from candidate_evaluation import assess_candidate_fitness_parallel, assess_candidate_fitness_single
from population import Population
from candidate import Candidate
from material import WALL_MATERIAL_SET, FLOOR_MATERIAL_SET, GlazingMaterial, OpaqueMaterial
from crop import SIMULATION_CROP
from plotting import save_top_half_generic_history_plot, save_top_fitness_history_plot, save_top_half_material_use_percentage_history_plot
from config import MIN_DIMENSIONS, MAX_DIMENSIONS, WEATHER_FILE_NAME, NUM_WORKER_THREADS, TOTAL_GENERATIONS, SUBDIVISION_GENERATION, SNAPSHOT_EVERY_X_GENERATIONS, POPULATION_SIZE, TEMPERATURE_RANGE, RADIATION_RANGE, INITIAL_TEMPERATURE_MUTATION_STD, INITIAL_RADIATION_MUTATION_STD

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..")

CANDIDATE_EVALUATION_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "candidate_files")
RUN_SUMMARIES_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "run_summaries")
WEATHER_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data", "weather_files", WEATHER_FILE_NAME)

CHECKPOINT_EVERY_X_GENERATIONS = 100
FIRST_CHECKPOINT_GENERATION = 5

#set to true and provide a filepath to load from chackpoint, false starts a new simulation run
#the pickle files generated during a run are the checkpoints
LOAD_FROM_CHECKPOINT = False
CHECKPOINT_RUN = ""
CHECKPOINT_FILE_NAME = ""
CHECKPOINT_PATH = os.path.join(RUN_SUMMARIES_DIRECTORY, CHECKPOINT_RUN, CHECKPOINT_FILE_NAME)


def main():
    if not os.path.exists(CANDIDATE_EVALUATION_DIRECTORY):
        os.makedirs(CANDIDATE_EVALUATION_DIRECTORY, exist_ok=True)
    if not os.path.exists(RUN_SUMMARIES_DIRECTORY):
        os.makedirs(RUN_SUMMARIES_DIRECTORY, exist_ok=True)  

    if LOAD_FROM_CHECKPOINT:
        try:
            checkpoint_data = _load_checkpoint(CHECKPOINT_PATH)
        except FileNotFoundError:
            print(f"checkpoint file not found at {CHECKPOINT_PATH}, change path or set LOAD_FROM_CHECKPOINT to False in main.py")
            return

        population = checkpoint_data["population"]

        top_fitness_history = checkpoint_data["top_fitness_history"]
        top_half_candidates_material_use_percentage_history = checkpoint_data["top_half_candidates_material_use_percentage_history"]
        top_half_candidates_fitness_history = checkpoint_data["top_half_candidates_fitness_history"]
        top_half_candidates_volume_history = checkpoint_data["top_half_candidates_volume_history"]
        top_half_candidates_temperature_setpoint_history = checkpoint_data["top_half_candidates_temperature_setpoint_history"]
        top_half_candidates_radiation_setpoint_history = checkpoint_data["top_half_candidates_radiation_setpoint_history"]
        top_half_candidates_natural_radiation_history = checkpoint_data["top_half_candidates_natural_radiation_history"]
        top_half_candidates_area_volume_ratio_history = checkpoint_data["top_half_candidates_area_volume_ratio_history"]
        top_half_candidates_yield_value_history = checkpoint_data["top_half_candidates_yield_value_history"]
        top_half_candidates_heating_cost_history = checkpoint_data["top_half_candidates_heating_cost_history"]
        top_half_candidates_cooling_cost_history = checkpoint_data["top_half_candidates_cooling_cost_history"]
        top_half_candidates_lighting_cost_history = checkpoint_data["top_half_candidates_lighting_cost_history"]
        top_half_candidates_window_wall_ratio_history = checkpoint_data["top_half_candidates_window_wall_ratio_history"]

        this_run_directory = os.path.join(RUN_SUMMARIES_DIRECTORY, CHECKPOINT_RUN)
        start_generation = population.generation_number + 1
    else:
        population = Population(
            population_size=POPULATION_SIZE, 
            available_wall_materials=WALL_MATERIAL_SET, 
            available_floor_materials=FLOOR_MATERIAL_SET,
            min_dimensions=MIN_DIMENSIONS, 
            max_dimensions=MAX_DIMENSIONS,
            temperature_range=TEMPERATURE_RANGE,
            radiation_range=RADIATION_RANGE,
            temperature_mutation_std=INITIAL_TEMPERATURE_MUTATION_STD,
            radiation_mutation_std=INITIAL_RADIATION_MUTATION_STD,
        )
        population.generate_initial_population()
        #calculate fitness for each candidate initially
        assess_candidate_fitness_parallel(population.members, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH, max_workers=NUM_WORKER_THREADS)

        #lists for saving various candidate info for plotting later
        top_fitness_history = []
        top_half_candidates_material_use_percentage_history = []
        top_half_candidates_fitness_history = []
        top_half_candidates_volume_history = []
        top_half_candidates_temperature_setpoint_history = []
        top_half_candidates_radiation_setpoint_history = []
        top_half_candidates_natural_radiation_history = []
        top_half_candidates_area_volume_ratio_history = []
        top_half_candidates_yield_value_history = []
        top_half_candidates_heating_cost_history = []
        top_half_candidates_cooling_cost_history = []
        top_half_candidates_lighting_cost_history = []
        top_half_candidates_window_wall_ratio_history = []

        this_run_directory = _find_next_run_output_directory(base_directory=RUN_SUMMARIES_DIRECTORY)  
        start_generation = 1

#========================================== MAIN GENETIC ALGORITHM LOOP ==========================================
    for generation in range(start_generation, TOTAL_GENERATIONS + 1):
        print(f"Generation {generation}")

        #subdivide after a certain numer of generations, allows design to be more fine tuned
        if generation == SUBDIVISION_GENERATION:
            population.increase_subdivision_level()
            #fitness values get changed slightly after subdivision. Must recalculate to put all candidates on the same scale
            assess_candidate_fitness_parallel(population.members, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH, max_workers=NUM_WORKER_THREADS)
            print(f"Subdivision level increased to {population.subdivision_level}")

        #while there is no crossover taking place, the survivor rate has to be higher to preserve good traits
        if generation <= 100:
            survivor_rate = 0.4
        elif generation <= 200:
            survivor_rate = 0.3
        elif generation <= 300:
            survivor_rate = 0.2
        else:
            #8 survivors
            survivor_rate = 0.16

        #short period of completely random exploration at the start, to get a good spread of candidates
        if generation <= 100:
            crossover_rate = 0
        elif generation <= 200:
            crossover_rate = 0.2
        elif generation <= 300:
            crossover_rate = 0.4
        elif generation <= 400:
            crossover_rate = 0.6
        else:
            #38 crossover
            crossover_rate = 0.76

        #lower mutation probability slightly after an initial period
        #remains relatively high to encourage exploration, especially important after subdivision
        if generation <= 400:
            geometry_mutation_probability = 1
        else:
            geometry_mutation_probability = 0.6

        #progressively decrease the setpoint mutation standard deviations, as they usually converge on optimal values pretty early
        #the minimum std dev is set to 0.1, so that exploration never completely stops
        if generation % 100 == 0:
            if population.temperature_mutation_std > 0.1:
                population.temperature_mutation_std *= 0.8
                print("temperature setpoint mutation standard deviation reduced")
            if population.radiation_mutation_std > 0.1:
                population.radiation_mutation_std *= 0.8
                print("radiation setpoint mutation standard deviation reduced")

        population.advance_generation(
            survivor_rate=survivor_rate, 
            crossover_rate=crossover_rate, 
            geometry_mutation_probability=geometry_mutation_probability, 
            setpoint_mutation_probability=0.2
        )

        #only evaluate new candidates without a fitness score
        candidates_to_evaluate = [candidate for candidate in population.members if candidate.fitness is None]
        assess_candidate_fitness_parallel(candidates_to_evaluate, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH, max_workers=NUM_WORKER_THREADS)
#=============================================================================================================

        #give running update on the best candidate fitness
        #update top_fitness_history with the best candidate fitness for this generation
        population.sort_population_descending_fitness()
        top_candidate = population.members[0]
        top_fitness_history.append(top_candidate.fitness)
        if generation % 10 == 0:
            print(f"Best candidate fitness: {top_candidate.fitness}")

        top_half_candidates = population.members[:int(0.5 * len(population.members))]

        #save all the top half candidates info for later plotting
        top_half_candidates_fitness_history.append([candidate.fitness for candidate in top_half_candidates])
        top_half_candidates_volume_history.append([candidate.building_structure.mesh.volume for candidate in top_half_candidates])
        top_half_candidates_temperature_setpoint_history.append([candidate.setpoints.temperature for candidate in top_half_candidates])
        top_half_candidates_radiation_setpoint_history.append([candidate.setpoints.radiation for candidate in top_half_candidates])
        top_half_candidates_natural_radiation_history.append([candidate.mean_daily_total_PAR_energy_per_area_MJ for candidate in top_half_candidates])
        top_half_candidates_area_volume_ratio_history.append([candidate.building_structure.mesh.area / candidate.building_structure.mesh.volume for candidate in top_half_candidates])
        top_half_candidates_yield_value_history.append([candidate.annual_yield_value_USD for candidate in top_half_candidates])
        top_half_candidates_heating_cost_history.append([candidate.annual_heating_cost_USD for candidate in top_half_candidates])
        top_half_candidates_cooling_cost_history.append([candidate.annual_cooling_cost_USD for candidate in top_half_candidates])
        top_half_candidates_lighting_cost_history.append([candidate.annual_lighting_cost_USD for candidate in top_half_candidates])
        top_half_candidates_material_use_percentage_history.append(_get_material_percentages(top_half_candidates))
        top_half_candidates_window_wall_ratio_history.append(_get_window_wall_ratios(top_half_candidates))

        #save the best candidates and population wide info every 100 generations, avoid duplicate saving if the final generation is a multiple of 100
        if generation % SNAPSHOT_EVERY_X_GENERATIONS == 0 and generation != TOTAL_GENERATIONS:
            snapshot_directory = os.path.join(this_run_directory, f"snapshot_generation_{generation}")
            if not os.path.exists(snapshot_directory):
                os.makedirs(snapshot_directory)

            _save_population_wide_info(population.members, snapshot_directory)
            _save_detailed_info_top_candidates(population.members[:5], snapshot_directory)
            _create_and_save_all_history_plots(
                save_directory=snapshot_directory,
                top_fitness_history=top_fitness_history,
                top_half_candidates_material_use_percentage_history=top_half_candidates_material_use_percentage_history,
                top_half_candidates_fitness_history=top_half_candidates_fitness_history,
                top_half_candidates_volume_history=top_half_candidates_volume_history,
                top_half_candidates_temperature_setpoint_history=top_half_candidates_temperature_setpoint_history,
                top_half_candidates_radiation_setpoint_history=top_half_candidates_radiation_setpoint_history,
                top_half_candidates_natural_radiation_history=top_half_candidates_natural_radiation_history,
                top_half_candidates_area_volume_ratio_history=top_half_candidates_area_volume_ratio_history,
                top_half_candidates_yield_value_history=top_half_candidates_yield_value_history,
                top_half_candidates_heating_cost_history=top_half_candidates_heating_cost_history,
                top_half_candidates_cooling_cost_history=top_half_candidates_cooling_cost_history,
                top_half_candidates_lighting_cost_history=top_half_candidates_lighting_cost_history,
                top_half_candidates_window_wall_ratio_history=top_half_candidates_window_wall_ratio_history
            )
            print(f"summary data for {generation} saved to {snapshot_directory}")

        #save a checkpoint at generation 105, 205, 305 etc.
        if (generation - FIRST_CHECKPOINT_GENERATION) % CHECKPOINT_EVERY_X_GENERATIONS == 0:
            _save_checkpoint(
                population=population,
                save_directory=this_run_directory,
                top_fitness_history=top_fitness_history,
                top_half_candidates_material_use_percentage_history=top_half_candidates_material_use_percentage_history,
                top_half_candidates_fitness_history=top_half_candidates_fitness_history,
                top_half_candidates_volume_history=top_half_candidates_volume_history,
                top_half_candidates_temperature_setpoint_history=top_half_candidates_temperature_setpoint_history,
                top_half_candidates_radiation_setpoint_history=top_half_candidates_radiation_setpoint_history,
                top_half_candidates_natural_radiation_history=top_half_candidates_natural_radiation_history,
                top_half_candidates_area_volume_ratio_history=top_half_candidates_area_volume_ratio_history,
                top_half_candidates_yield_value_history=top_half_candidates_yield_value_history,
                top_half_candidates_heating_cost_history=top_half_candidates_heating_cost_history,
                top_half_candidates_cooling_cost_history=top_half_candidates_cooling_cost_history,
                top_half_candidates_lighting_cost_history=top_half_candidates_lighting_cost_history,
                top_half_candidates_window_wall_ratio_history=top_half_candidates_window_wall_ratio_history
            )

    #save the best candidates at the end of the run, as well as some population wide info
    population.sort_population_descending_fitness()
    final_results_directory = os.path.join(this_run_directory, "final_simulation_results")
    if not os.path.exists(final_results_directory):
        os.makedirs(final_results_directory)

    _save_population_wide_info(population.members, final_results_directory)
    _save_detailed_info_top_candidates(population.members[:5], final_results_directory)
    _create_and_save_all_history_plots(
        save_directory=final_results_directory,
        top_fitness_history=top_fitness_history,
        top_half_candidates_material_use_percentage_history=top_half_candidates_material_use_percentage_history,
        top_half_candidates_fitness_history=top_half_candidates_fitness_history,
        top_half_candidates_volume_history=top_half_candidates_volume_history,
        top_half_candidates_temperature_setpoint_history=top_half_candidates_temperature_setpoint_history,
        top_half_candidates_radiation_setpoint_history=top_half_candidates_radiation_setpoint_history,
        top_half_candidates_natural_radiation_history=top_half_candidates_natural_radiation_history,
        top_half_candidates_area_volume_ratio_history=top_half_candidates_area_volume_ratio_history,
        top_half_candidates_yield_value_history=top_half_candidates_yield_value_history,
        top_half_candidates_heating_cost_history=top_half_candidates_heating_cost_history,
        top_half_candidates_cooling_cost_history=top_half_candidates_cooling_cost_history,
        top_half_candidates_lighting_cost_history=top_half_candidates_lighting_cost_history,
        top_half_candidates_window_wall_ratio_history=top_half_candidates_window_wall_ratio_history
    )
    print(f"final results saved to {final_results_directory}")


def _load_checkpoint(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    return checkpoint_data


def _save_checkpoint(
    population: Population,
    save_directory: str,
    top_fitness_history: list[float],
    top_half_candidates_material_use_percentage_history: list[Counter],
    top_half_candidates_fitness_history: list[list[float]],
    top_half_candidates_volume_history: list[list[float]],
    top_half_candidates_temperature_setpoint_history: list[list[float]],
    top_half_candidates_radiation_setpoint_history: list[list[float]],
    top_half_candidates_natural_radiation_history: list[list[float]],
    top_half_candidates_area_volume_ratio_history: list[list[float]],
    top_half_candidates_yield_value_history: list[list[float]],
    top_half_candidates_heating_cost_history: list[list[float]],
    top_half_candidates_cooling_cost_history: list[list[float]],
    top_half_candidates_lighting_cost_history: list[list[float]],
    top_half_candidates_window_wall_ratio_history: list[list[float]],
) -> None:
    checkpoint_data = {
        "population": population,
        "top_fitness_history": top_fitness_history,
        "top_half_candidates_material_use_percentage_history": top_half_candidates_material_use_percentage_history,
        "top_half_candidates_fitness_history": top_half_candidates_fitness_history,
        "top_half_candidates_volume_history": top_half_candidates_volume_history,
        "top_half_candidates_temperature_setpoint_history": top_half_candidates_temperature_setpoint_history,
        "top_half_candidates_radiation_setpoint_history": top_half_candidates_radiation_setpoint_history,
        "top_half_candidates_natural_radiation_history": top_half_candidates_natural_radiation_history,
        "top_half_candidates_area_volume_ratio_history": top_half_candidates_area_volume_ratio_history,
        "top_half_candidates_yield_value_history": top_half_candidates_yield_value_history,
        "top_half_candidates_heating_cost_history": top_half_candidates_heating_cost_history,
        "top_half_candidates_cooling_cost_history": top_half_candidates_cooling_cost_history,
        "top_half_candidates_lighting_cost_history": top_half_candidates_lighting_cost_history,
        "top_half_candidates_window_wall_ratio_history": top_half_candidates_window_wall_ratio_history
    }

    checkpoint_path = os.path.join(save_directory, f"checkpoint_generation_{population.generation_number}.pickle")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"checkpoint saved to {checkpoint_path}")


def _find_next_run_output_directory(base_directory: str) -> str:
    #find the next available directory for saving the info for a given run
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    x = 1
    while True:
        output_directory = os.path.join(base_directory, f"run_{x}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            return output_directory
        x += 1


def _get_material_percentages(candidates: list[Candidate]) -> Counter[str, float]:
    material_counter = Counter()
    total_faces = 0
    for candidate in candidates:
        material_names = (material.name for material in candidate.building_structure.face_materials)
        material_counter.update(material_names)
        total_faces += len(candidate.building_structure.face_materials)

    if total_faces > 0:
        for material in material_counter:
            material_counter[material] = (material_counter[material] / total_faces) * 100
        return material_counter
    else:
        return material_counter


def _get_window_wall_ratios(candidates: list[Candidate]) -> list[float]:
    window_wall_ratios = []
    for candidate in candidates:
        total_window_area = 0
        total_wall_area = 0
        for i, material in enumerate(candidate.building_structure.face_materials):
            if isinstance(material, GlazingMaterial):
                window_area = candidate.building_structure.mesh.area_faces[i]
                total_window_area += window_area
            elif isinstance(material, OpaqueMaterial):
                wall_area = candidate.building_structure.mesh.area_faces[i]
                total_wall_area += wall_area

        if total_wall_area == 0:
            window_wall_ratio = 0
        else:
            window_wall_ratio = total_window_area / total_wall_area
        window_wall_ratios.append(window_wall_ratio)

    return window_wall_ratios


def _save_population_wide_info(candidates: list[Candidate], save_directory: str) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")

    #save the setpoints of each candidate to a text file, in descending fitness order
    setpoints_list = [f"Candidate {i + 1}: {candidate.setpoints}" for i, candidate in enumerate(candidates)] 
    setpoint_save_path = os.path.join(save_directory, "setpoints_list.txt")
    _save_list_to_text_file(setpoints_list, setpoint_save_path)
    print(f"Setpoints list saved to {setpoint_save_path}")

    #save the dimensions of each candidate to a text file, in descending fitness order
    dimensions_list = [f"Candidate {i + 1}: {candidate.building_structure.mesh.bounding_box.extents}" for i, candidate in enumerate(candidates)]
    dimensions_save_path = os.path.join(save_directory, "dimensions_list.txt")
    _save_list_to_text_file(dimensions_list, dimensions_save_path)
    print(f"Dimensions list saved to {dimensions_save_path}")

    #save the fitness of each candidate to a text file, in descending fitness order
    fitness_list = [f"Candidate {i + 1}: {candidate.fitness}" for i, candidate in enumerate(candidates)]
    fitness_save_path = os.path.join(save_directory, "fitness_list.txt")
    _save_list_to_text_file(fitness_list, fitness_save_path)
    print(f"Fitness list saved to {fitness_save_path}")

    yield_value_list = [f"Candidate {i + 1}: {candidate.annual_yield_value_USD}" for i, candidate in enumerate(candidates)]
    yield_value_save_path = os.path.join(save_directory, "yield_value_list.txt")
    _save_list_to_text_file(yield_value_list, yield_value_save_path)
    print(f"Yield value list saved to {yield_value_save_path}")

    lighitng_cost_list = [f"Candidate {i + 1}: {candidate.annual_lighting_cost_USD}" for i, candidate in enumerate(candidates)]
    lighting_cost_save_path = os.path.join(save_directory, "lighting_cost_list.txt")
    _save_list_to_text_file(lighitng_cost_list, lighting_cost_save_path)
    print(f"Lighting cost list saved to {lighting_cost_save_path}")

    heating_cost_list = [f"Candidate {i + 1}: {candidate.annual_heating_cost_USD}" for i, candidate in enumerate(candidates)]
    heating_cost_save_path = os.path.join(save_directory, "heating_cost_list.txt")
    _save_list_to_text_file(heating_cost_list, heating_cost_save_path)
    print(f"Heating cost list saved to {heating_cost_save_path}")

    cooling_cost_list = [f"Candidate {i + 1}: {candidate.annual_cooling_cost_USD}" for i, candidate in enumerate(candidates)]
    cooling_cost_save_path = os.path.join(save_directory, "cooling_cost_list.txt")
    _save_list_to_text_file(cooling_cost_list, cooling_cost_save_path)
    print(f"Cooling cost list saved to {cooling_cost_save_path}")    

    yield_weight_list = [f"Candidate {i + 1}: {candidate.annual_crop_yield_kg}" for i, candidate in enumerate(candidates)]
    yield_weight_save_path = os.path.join(save_directory, "yield_weight_list.txt")
    _save_list_to_text_file(yield_weight_list, yield_weight_save_path)
    print(f"Yield weight list saved to {yield_weight_save_path}")

    ambient_light_list = [f"Candidate {i + 1}: {candidate.mean_daily_total_PAR_energy_per_area_MJ}" for i, candidate in enumerate(candidates)]
    ambient_light_save_path = os.path.join(save_directory, "ambient_light_list.txt")
    _save_list_to_text_file(ambient_light_list, ambient_light_save_path)
    print(f"Ambient light list saved to {ambient_light_save_path}")


def _save_list_to_text_file(data_list: list, save_path: str) -> None:
    with open(save_path, "w") as f:
        for line in data_list:
            f.write(line + "\n")


def _save_detailed_info_top_candidates(top_candidates_descending_fitness: list[Candidate], save_directory: str) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")

    for i, candidate in enumerate(top_candidates_descending_fitness):
        candidate_save_directory = os.path.join(save_directory, f"top_candidate_{i + 1}")
        if not os.path.exists(candidate_save_directory):
            os.makedirs(candidate_save_directory)

        _save_detailed_candidate_info(candidate, candidate_save_directory)
        

def _save_detailed_candidate_info(candidate: Candidate, save_directory: str) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")

    #regenerate the files for the candidate, not deleting after evaluation so they persist
    assess_candidate_fitness_single(candidate, SIMULATION_CROP, save_directory, WEATHER_FILE_PATH, delete_after_evaluation=False)

    #save the entire candidate object, inclusing mesh and materials
    candidate_pickle_save_path = os.path.join(save_directory, "candidate.pickle")
    candidate.save_pickle(candidate_pickle_save_path)

    #save a coloured version of the mesh, but no transparency available as its a .ply file
    coloured_mesh = candidate.building_structure.get_coloured_mesh()
    mesh_save_path = os.path.join(save_directory, "coloured_mesh.ply")
    coloured_mesh.export(mesh_save_path)    
    
    #gather and save candidate info
    candidate_info = {
        "id": candidate.id,
        "fitness": candidate.fitness,
        "setpoints": {
            "temperature": candidate.setpoints.temperature,
            "radiation": candidate.setpoints.radiation
        },
        "building_structure": {
            "dimensions": candidate.building_structure.mesh.bounding_box.extents,
            "face materials": candidate.building_structure.face_materials
        },
        "annual_yield_value_USD": candidate.annual_yield_value_USD,
        "annual_lighting_cost_USD": candidate.annual_lighting_cost_USD,
        "annual_heating_cost_USD": candidate.annual_heating_cost_USD,
        "annual_cooling_cost_USD": candidate.annual_cooling_cost_USD,
        "annual_crop_yield_kg": candidate.annual_crop_yield_kg,
        "mean_daily_total_PAR_energy_per_area_MJ": candidate.mean_daily_total_PAR_energy_per_area_MJ
    }
    info_save_path = os.path.join(save_directory, "candidate_info.txt")
    with open(info_save_path, "w") as f:
        for key, value in candidate_info.items():
            f.write(f"{key}: {value}\n")

    print(f"Candidate {candidate.id} data saved to {save_directory}")


def _create_and_save_all_history_plots(
    save_directory: str,        
    top_fitness_history: list[float],
    top_half_candidates_material_use_percentage_history: list[Counter],
    top_half_candidates_fitness_history: list[list[float]],
    top_half_candidates_volume_history: list[list[float]],
    top_half_candidates_temperature_setpoint_history: list[list[float]],
    top_half_candidates_radiation_setpoint_history: list[list[float]],
    top_half_candidates_natural_radiation_history: list[list[float]],
    top_half_candidates_area_volume_ratio_history: list[list[float]],
    top_half_candidates_yield_value_history: list[list[float]],
    top_half_candidates_heating_cost_history: list[list[float]],
    top_half_candidates_cooling_cost_history: list[list[float]],
    top_half_candidates_lighting_cost_history: list[list[float]],
    top_half_candidates_window_wall_ratio_history: list[list[float]]
) -> None:
    if not os.path.exists(save_directory):
        raise FileNotFoundError(f"save directory not found: {save_directory}")

    save_top_fitness_history_plot(top_fitness_history, save_directory)
    save_top_half_material_use_percentage_history_plot(top_half_candidates_material_use_percentage_history, save_directory)

    save_top_half_generic_history_plot(
        top_half_candidates_fitness_history,
        save_directory,
        colour="green",
        y_label="Fitness",
        y_units=None
    )
    save_top_half_generic_history_plot(
        top_half_candidates_volume_history,
        save_directory,
        colour="orange",
        y_label="Volume",
        y_units="m$^3$"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_temperature_setpoint_history,
        save_directory,
        colour="red",
        y_label="Temperature Setpoint",
        y_units="C"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_radiation_setpoint_history,
        save_directory,
        colour="#393b79", #dark indigo
        y_label="PAR Setpoint",
        y_units="MJ m$^{-2}$ day$^{-1}$"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_natural_radiation_history,
        save_directory,
        colour="purple",
        y_label="Natural PAR",
        y_units="MJ m$^{-2}$ day$^{-1}$"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_area_volume_ratio_history,
        save_directory,
        colour="blue",
        y_label="Area/Volume Ratio",
        y_units=None
    )
    save_top_half_generic_history_plot(
        top_half_candidates_yield_value_history,
        save_directory,
        colour="cyan",
        y_label="Yield Value",
        y_units="USD"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_heating_cost_history,
        save_directory,
        colour="pink",
        y_label="Heating Cost",
        y_units="USD"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_cooling_cost_history,
        save_directory,
        colour="brown",
        y_label="Cooling Cost",
        y_units="USD"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_lighting_cost_history,
        save_directory,
        colour="grey",
        y_label="Lighting Cost",
        y_units="USD"
    )
    save_top_half_generic_history_plot(
        top_half_candidates_window_wall_ratio_history,
        save_directory,
        colour="olive",
        y_label="Window/Wall Ratio",
        y_units=None
    )


if __name__ == "__main__":
    main()
