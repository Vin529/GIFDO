from __future__ import annotations

import uuid
import pickle

from candidate_building_structure import BuildingStructure
from candidate_setpoints import SetPoints

class Candidate:
    def __init__(self, generation_number: int, building_structure: BuildingStructure, setpoints: SetPoints):
        self._id = f"gen_{generation_number}_{uuid.uuid4().hex[:16]}"
        self.fitness = None
        self.building_structure = building_structure
        self.setpoints = setpoints

        #extra info that gets returned when calculating fitness
        self.annual_yield_value_USD = None
        self.annual_lighting_cost_USD = None
        self.annual_heating_cost_USD = None
        self.annual_cooling_cost_USD = None
        self.annual_crop_yield_kg = None
        self.mean_daily_total_PAR_energy_per_area_MJ = None

    @classmethod
    #using kwargs here isnt ideal, but there are alot of optional parameters for the BuildingStructure class, and i havent got round to creating some sort of parameter dictionary object yet
    def crossover(cls, parent1: "Candidate", parent2: "Candidate", generation_number: int, **kwargs) -> Candidate | None:
        child_building_structure = BuildingStructure.crossover(parent1.building_structure, parent2.building_structure, **kwargs)
        #crossover can fail, the above function in population.py will handle this
        if child_building_structure is None:
            return None
        
        child_setpoints = SetPoints.crossover(parent1.setpoints, parent2.setpoints)
        return cls(generation_number, child_building_structure, child_setpoints)

    @property
    def id(self) -> str:
        return self._id

    def mutate_geometry(self, **kwargs) -> None:
        self.building_structure.mutate(**kwargs)

    def mutate_setpoints(self, temperature_std: float, radiation_std: float) -> None:
        self.setpoints.mutate(temperature_std, radiation_std)

    def save_pickle(self, file_path: str) -> None:
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def __repr__(self):
        return (f"Candidate(ID={self.id}, Fitness={self.fitness}, Setpoints={self.setpoints})")
    