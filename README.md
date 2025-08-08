This repository contains the source code, dissertation, and additional resources which were created as a part of my 4th year master's project for Computer Science at Durham University. The dissertation included here was awarded 91% by the university, which was tied for the highest grade in my year.

## Requirements
The following versions have been tested. Other versions may work but are unverified:

1. **Python 3.11.9**  
2. **OpenStudio Python package** v3.9.0 
3. **EnergyPlus v24.1.0** — download from the [NREL EnergyPlus Releases](https://github.com/NREL/EnergyPlusRelease/releases) page  
4. **Python dependencies** — install all packages listed in requirements.txt

## Setup
In config.py, set ENERGYPLUS_EXECUTABLE_PATH to point at your EnergyPlus v24.1.0 executable.

## Usage
Run main.py to begin a simulation. The run time for a full simulation is very long (over 24 hours on my machine). Set LOAD_FROM_CHECKPOINT=False at the top of main.py to begin a new simulation. Alternatively, set it to True and provide a path to a snapshot_generation_X.pickle file to resume from a previous checkpoint.

The application expects the data folder to be in the directory above main.py. This behaviour can be changed by modifying the paths at the top of main.py.

The data/candidate_files directory is used to evaluate candidates. It is recommended to make sure this is empty before starting a run. All simulation data and results are saved to the data/run_summaries directory. Both of these directories will be automatically created in the data folder if they do not exist at the start of a run.

Key simulation parameters are in config.py.
File path related parameters are located in main.py, can be changed to specify which directories the program should save files to, and look for files in.
Material parameters, and specifying available materials for the simulation to use should be done in material.py.
Specifying new crops, and changing the crop used in the fitness calculations, can be done in crop.py.

The croprl subdirectory inside the GIFDO directory contains code from https://github.com/iscoe/croprl which implements the SIMPLE crop model in python. It was easier for me to work with the raw code as opposed to packaging it up and importing it with pip. None of the code in the croprl folder is my own: I am simply using it in line with the licence provided. Only select files from the croprl repo have been included, therefore following the instructions in README_croprl will not yield expected behaviour. In addition, the scripts in croprl have been modified with gymnasium imports as opposed to the outdated gym package.

Copy a candidate.pickle file into the data/visualisation_files directory, point and run visualise_candidate_mesh.py to see a 3D render of a given design.

## Compatibility
Due to improvements made to the checkpointing system since writing the report, the checkpoints in oslo_run and singapore_run cannot be resumed from using the current application version. Checkpoints generated on the current version can be resumed from as normal.

