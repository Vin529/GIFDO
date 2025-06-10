import os
import subprocess

from config import ENERGYPLUS_EXECUTABLE_PATH


def run_simulation(candidate_directory: str, simulation_output_directory: str, weather_file_path: str) -> None:
    idf_path = os.path.join(candidate_directory, "model.idf")
    if not os.path.exists(idf_path):
        raise FileNotFoundError(f"IDF file not found at {idf_path}")
    
    _execute_simulation_commands(ENERGYPLUS_EXECUTABLE_PATH, weather_file_path, idf_path, simulation_output_directory)
    

def _execute_simulation_commands(energyplus_exe: str, weather_file_path: str, idf_path: str, simulation_output_directory: str) -> None:
    #confirm existance of necessary files and directories
    if not os.path.exists(simulation_output_directory):
        raise FileNotFoundError(f"simulation output directory not found: {simulation_output_directory}")
    if not os.path.exists(energyplus_exe):
        raise FileNotFoundError(f"EnergyPlus executable not found at {energyplus_exe}")
    if not os.path.exists(weather_file_path):
        raise FileNotFoundError(f"weather file not found at {weather_file_path}")

    command = [
        energyplus_exe,
        "-w", weather_file_path,
        "-d", simulation_output_directory,
        idf_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("EnergyPlus simulation failed. Check esplusout.err for details.")
        raise RuntimeError("EnergyPlus simulation failed.")
