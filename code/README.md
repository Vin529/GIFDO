The GIFDO directory contains all source code for this project.

Before using this application install all required packages listed in the requirements.txt file. Run main.py to begin a simulation. The run time for a full simulation is very long (over 24 hours). Set LOAD_FROM_CHECKPOINT=False at the top of main.py to begin a new simulation. Alternatively, set it to True and provide a path to a snapshot_generation_X.pickle file to resume from a previous checkpoint.

The application expects the data folder to be in the directory above main.py. This behaviour can be changed by modifying the paths at the top of main.py.

The data/candidate_files directory is used to evaluate candidates. It is recommended to make sure this is empty before starting a run. All simulation data and results are saved to the data/run_summaries directory.

Key simulation parameters are in config.py.
File path related parameters are located in main.py, can be changed to specify which directories the program should save files to, and look for files in.
Material parameters, and specifying available materials for the simulation to use should be done in material.py.
Specifying new crops, and changing the crop used in the fitness calculations, can be done in crop.py.

The croprl subdirectory inside the GIFDO directory contains code from https://github.com/iscoe/croprl which implements the SIMPLE crop model in python. It was easier for me to work with the raw code as opposed to packaging it up and importing it with pip. None of the code in the croprl folder is my own: I am simply using it in line with the licence provided.

Copy a candidate.pickle file into the data/visualisation_files directory and run visualise_candidate_mesh.py to see a 3D render of a given design.
