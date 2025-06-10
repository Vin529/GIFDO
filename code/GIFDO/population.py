import random
import numpy as np

from candidate import Candidate
from candidate_building_structure import BuildingStructure
from material import Material
from candidate_setpoints import SetPoints

MAX_RANDOM_CANDIDATE_CREATION_ATTEMPTS = 100
MAX_CROSSOVER_ATTEMPTS = 1000


class Population:
    def __init__(
        self, 
        population_size: int, 
        available_wall_materials: set[Material], 
        available_floor_materials: set[Material],
        min_dimensions: np.ndarray[float], #meters
        max_dimensions: np.ndarray[float], #meters
        temperature_range: tuple[float, float], #degrees C,
        radiation_range: tuple[float, float], #MJ/(m**2 * day)
        temperature_mutation_std: float, #degrees C
        radiation_mutation_std: float, #MJ/(m**2 * day)
        selection_pressure: float = 1.8, #between 0.0 and 2.0 for expected behaviour
        subdivision_level: int = 0
    ):
        if population_size <= 0:
            raise ValueError(f"Population size must be greater than 0, but it was set to {population_size}.")
        if min_dimensions.shape != (3,):
            raise ValueError(f"min_dimensions must be a numpy array of shape (3,), but it was set to {min_dimensions}.")
        if max_dimensions.shape != (3,):
            raise ValueError(f"max_dimensions must be a numpy array of shape (3,), but it was set to {max_dimensions}.")

        self.population_size = population_size
        self.available_wall_materials = available_wall_materials
        self.available_floor_materials = available_floor_materials
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        self.temperature_range = temperature_range
        self.radiation_range = radiation_range
        self.temperature_mutation_std = temperature_mutation_std
        self.radiation_mutation_std = radiation_mutation_std
        self.selection_pressure = selection_pressure
        self.subdivision_level = subdivision_level

        self.generation_number = 0
        self.members = []

    def generate_initial_population(self) -> None:
        if len(self.members) > 0:
            raise ValueError("Cannot generate initial population: Population already exists.")

        while len(self.members) < self.population_size:
            attempts = 0
            while True:
                try:
                    random_building_structure = BuildingStructure.from_random(self.available_wall_materials, self.available_floor_materials, self.min_dimensions, self.max_dimensions, self.subdivision_level)
                    random_setpoints = SetPoints.from_random(self.temperature_range, self.radiation_range)
                    candidate = Candidate(self.generation_number, random_building_structure, random_setpoints)
                    self.members.append(candidate)
                    break
                except ValueError:
                    attempts += 1
                    if attempts >= MAX_RANDOM_CANDIDATE_CREATION_ATTEMPTS:
                        raise RuntimeError(f"unable to create a valid random candidate after {MAX_RANDOM_CANDIDATE_CREATION_ATTEMPTS} attempts.")

    def sort_population_descending_fitness(self) -> None:
        for candidate in self.members:
            if not isinstance(candidate.fitness, (int, float)):
                raise ValueError(f"Cannot sort population: Candidate {candidate} has an invalid fitness: {candidate.fitness}")
        
        self.members.sort(key=lambda candidate: candidate.fitness, reverse=True)

    def increase_subdivision_level(self) -> None:
        self.subdivision_level += 1
        for candidate in self.members:
            candidate.building_structure.subdivide()

    def advance_generation(
        self, 
        survivor_rate: float, 
        crossover_rate: float, 
        geometry_mutation_probability: float, 
        setpoint_mutation_probability: float
    ) -> None:
        if len(self.members) == 0:
            raise ValueError("Cannot advance generation: Population is empty. Consider calling generate_initial_population() first.")
        
        self.generation_number += 1
        next_generation_members = []

        #first fill next generation with survivors, then children from crossover
        survivors = self._survival_selection_operator(survivor_rate)
        children = self._crossover_population(crossover_rate, geometry_mutation_probability, setpoint_mutation_probability)
        next_generation_members.extend(survivors)
        next_generation_members.extend(children)

        if len(next_generation_members) > self.population_size:
            #cull extra candidates
            next_generation_members = next_generation_members[:self.population_size]
        else:
            #fill the rest of the population with random candidates
            num_random_candidates = self.population_size - len(next_generation_members)
            for _ in range(num_random_candidates):
                attempts = 0
                while True:
                    try:
                        random_building_structure = BuildingStructure.from_random(self.available_wall_materials, self.available_floor_materials, self.min_dimensions, self.max_dimensions, self.subdivision_level)
                        random_setpoints = SetPoints.from_random(self.temperature_range, self.radiation_range)
                        candidate = Candidate(self.generation_number, random_building_structure, random_setpoints)
                        next_generation_members.append(candidate)
                        break
                    except ValueError:
                        attempts += 1
                        if attempts >= MAX_RANDOM_CANDIDATE_CREATION_ATTEMPTS:
                            raise RuntimeError(f"unable to create a valid random candidate after {MAX_RANDOM_CANDIDATE_CREATION_ATTEMPTS} attempts.")

        self.members = next_generation_members 

    def _survival_selection_operator(self, survivor_rate: float) -> list[Candidate]:
        #just choose the candidates with the highest fitness
        num_survivors = np.round(self.population_size * survivor_rate).astype(int)
        self.sort_population_descending_fitness()
        survivors = self.members[:num_survivors]
        return survivors
    
    def _crossover_population(self, crossover_rate: float, geometry_mutation_probability: float, setpoint_mutation_probability: float) -> list[Candidate]:
        ranks = np.arange(self.population_size, 0, -1)
        selection_probabilities = ((2 - self.selection_pressure) + (2 * (self.selection_pressure - 1) * (ranks - 1) / (self.population_size - 1)))
        #normalise probabilities
        selection_probabilities /= np.sum(selection_probabilities)
        #sort population by fitness, selectino operator assumes the population is sorted in this way
        self.sort_population_descending_fitness()

        children = []
        num_children_target = np.round(self.population_size * crossover_rate).astype(int)

        #if enough unique parent pairs result in invalid children (None returned from Candidate.crossover), then we break the loop
        crossover_attempt_count = 0
        while len(children) < num_children_target and crossover_attempt_count < MAX_CROSSOVER_ATTEMPTS:
            crossover_attempt_count += 1
            parent_1, parent_2 = self._crossover_selection_operator(self.members, selection_probabilities)
            if parent_1 is parent_2:
                print("both selected parents are the same candidate, this should not be possible")
                continue
            
            child = Candidate.crossover(parent_1, parent_2, self.generation_number) #this can take extra kwargs for geometry crossover
            if child is None:
                continue
            else:
                children.append(child)
        
        if crossover_attempt_count >= MAX_CROSSOVER_ATTEMPTS:
            print(f"unable to find enough viable parent pairs for crossover after {MAX_CROSSOVER_ATTEMPTS} attempts, only generated {len(children)} children")

        children = self._mutate_children(
                children, 
                geometry_mutation_probability, 
                setpoint_mutation_probability, 
                self.temperature_mutation_std, 
                self.radiation_mutation_std
            )
        return children

    @staticmethod
    def _crossover_selection_operator(candidates: list[Candidate], selection_probabilities: np.ndarray) -> tuple[Candidate, Candidate]:
        if not len(candidates) == len(selection_probabilities):
            raise ValueError(f"candidates and selection_probabilities must be the same length, but they are {len(candidates)} and {len(selection_probabilities)}")
        if len(candidates) < 2:
            raise ValueError(f"candidates must have at least 2 members, but it has {len(candidates)}")

        #replace is set to False to ensure that the same candidate is not selected twice
        parent_1, parent_2 = np.random.choice(candidates, size=2, p=selection_probabilities, replace=False)
        return parent_1, parent_2   
    
    @staticmethod
    def _mutate_children(
        children: list[Candidate], 
        geometry_mutation_probability: float, 
        setpoint_mutation_probability: float,
        temperature_mutation_std: float,
        radiation_mutation_std: float
    ) -> list[Candidate]:
        mutated_children = children.copy()
        for child in mutated_children:
            if random.random() < geometry_mutation_probability:
                child.mutate_geometry() #this can take additional kwargs for geometry mutation
            if random.random() < setpoint_mutation_probability:
                child.mutate_setpoints(temperature_mutation_std, radiation_mutation_std)
        return mutated_children
