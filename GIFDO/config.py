import numpy as np

#====== plotting.py ======
LOCATION_NAME = "Oslo, Norway"

#====== main.py ======
#set the weather file for the location of the simulation
WEATHER_FILE_NAME = "NOR_Oslo.Fornebu.014880_IWEC.epw"
#parameters related to the overall genetic algorithm
NUM_WORKER_THREADS = 12
TOTAL_GENERATIONS = 1200
POPULATION_SIZE = 50
#confine the dimensions of the generated building designs
#the x and y bounds are not specifically limitations on their respective dimension, but the footprint area of the building is constrained by their product
MIN_DIMENSIONS = np.array([5, 5, 5], dtype=float) #meters
MAX_DIMENSIONS = np.array([50, 50, 25], dtype=float) #meters
#tightening these ranges will improve speed of convergence, but clearly reduce the total search space
TEMPERATURE_RANGE = (18.0, 30.0) #degrees C, typical range encompasing most indoor farming setups
INITIAL_TEMPERATURE_MUTATION_STD = 1 #degrees C
RADIATION_RANGE = (0.0, 12.0) #MJ/(m**2 * day), typical range encompasing most indoor farming setups
INITIAL_RADIATION_MUTATION_STD = 1 #MJ/(m**2 * day)
#subdivide the mesh after a certian number of generations, increasing simulation time but improving detail in the design as there are more vertices to work with
#if set below 1 or above TOTAL_GENERATIONS it will have no effect
SUBDIVISION_GENERATION = 701
#saves detailed data  on top candidates and the population as a whole every X generations, as well as creating some plots
SNAPSHOT_EVERY_X_GENERATIONS = 100
#changing default file paths must be done in main.py directly

#====== energyplus_model_construction.py ======
#controls the number of points in the grid for each illuminance map, larger values improve lighting simulation accuracy
#WARNING: increasing this parameter n will increase total running time with roughly n^2 scaling, lighitng simulation dominates the total run time
ILLUMINANCE_MAP_SAMPLES_PER_AXIS = 5
#sets how far inside the building bounds the edge of the illuminance map are placed
#recommended value for this is min(MIN_DIMENSIONS) / (2 * ILLUMINANCE_MAP_SAMPLES_PER_AXIS) 
ILLUMINANCE_GRID_INSET_DISTANCE = 0.5 #meters

#====== candidate_evaluation.py ======
USABLE_GROWING_VOLUME_PROPORTION = 0.5
#crop specific parameters, change these to reflect the required height clearance required above the crop in question
GROWING_LEVEL_INITIAL_FLOOR_OFFSET = 0.3 #meters
GROWING_LEVEL_Z_INTERVAL = 1.5 #meters
#efficiency value is for CREE XQ-A Red 625nm LED
#from "Light use efficiency for vegetables production in protected and indoor environments"
#this should be changed based on the LEDs used in the system
LED_EFFICIENCY_UMOL_PER_J = 2.5
#energy prices are set in USD per kWh, and are used to calculate the cost of heating and cooling the building
ELECTRICITY_PRICE_USD_PER_KWH = 0.23 #0.15
HEATING_ENERGY_PRICE_USD_PER_KWH = 0.18 #0.10
#heating efficiency must be between 0 and 1
ANNUAL_AVERAGE_HEATING_SYSTEM_EFFICIENCY = 0.9 #typical for a natural gas boiler
#COP can often be greater than 1, as you dont need say 100 joules of energy to remove 100 joules of heat from a space: you can do it with less
ANNUAL_AVERAGE_COOLING_SYSTEM_COP = 3.8 #COP = Coefficient of Performance

#====== material.py ======
#available materials and their properties are set in material.py

#====== crop.py ======
#crop used in the simulation, and its properties are set in crop.py

#====== energyplus_simulation.py ======
ENERGYPLUS_EXECUTABLE_PATH = "C:/EnergyPlusV24-1-0/energyplus.exe"
