import numpy as np


class SetPoints:
    def __init__(
        self, 
        temperature: float, 
        radiation: float, 
        temperature_range: tuple[float, float], 
        radiation_range: tuple[float, float]
    ):
        if not (temperature_range[0] <= temperature <= temperature_range[1]):
            raise ValueError(f"temperature {temperature} is out of range {temperature_range}")
        if not (radiation_range[0] <= radiation <= radiation_range[1]):
            raise ValueError(f"radiation {radiation} is out of range {radiation_range}")
        
        self.temperature = temperature # degrees C
        self.radiation = radiation # MJ/(m**2 * day)

        self.temperature_range = temperature_range
        self.radiation_range = radiation_range


    @classmethod
    def from_random(cls, temperature_range: tuple[float, float], radiation_range: tuple[float, float]) -> "SetPoints":
        if temperature_range[0] >= temperature_range[1]:
            raise ValueError(f"temperature range {temperature_range} is invalid")
        if radiation_range[0] >= radiation_range[1]:
            raise ValueError(f"radiation range {radiation_range} is invalid")

        temperature = np.random.uniform(*temperature_range)
        radiation = np.random.uniform(*radiation_range)
        return cls(temperature, radiation, temperature_range, radiation_range)
    
    @classmethod 
    def crossover(cls, parent1: "SetPoints", parent2: "SetPoints", temperature_alpha: float | None = None, radiation_alpha: float | None = None) -> "SetPoints":
        if temperature_alpha is None:
            temperature_alpha = np.random.uniform(0, 1)
        if radiation_alpha is None:
            radiation_alpha = np.random.uniform(0, 1)

        #temperature_alpha is the blending factor between parent1 and parent2 temperature, 1 is fully parent1, 0 is fully parent2, 0.5 is their mean
        if temperature_alpha < 0 or temperature_alpha > 1:
            raise ValueError(f"temperature_alpha must be between 0 and 1, but it was set to {temperature_alpha}")
        #radiation_alpha is the blending factor between parent1 and parent2 radiation, 1 is fully parent1, 0 is fully parent2, 0.5 is their mean
        if radiation_alpha < 0 or radiation_alpha > 1:
            raise ValueError(f"radiation_alpha must be between 0 and 1, but it was set to {radiation_alpha}")
        
        child_temperature = temperature_alpha * parent1.temperature + (1 - temperature_alpha) * parent2.temperature
        child_radiation = radiation_alpha * parent1.radiation + (1 - radiation_alpha) * parent2.radiation

        child_temperature_range = (
            min(parent1.temperature_range[0], parent2.temperature_range[0]),
            max(parent1.temperature_range[1], parent2.temperature_range[1])
        )
        child_radiation_range = (
            min(parent1.radiation_range[0], parent2.radiation_range[0]),
            max(parent1.radiation_range[1], parent2.radiation_range[1])
        )

        child = cls(
            temperature=child_temperature, 
            radiation=child_radiation, 
            temperature_range=child_temperature_range, 
            radiation_range=child_radiation_range
        )
        return child
    
    def mutate(self, temperature_std: float, radiation_std: float) -> None:
        temperature_change = np.random.normal(0, temperature_std)
        new_temperature = self.temperature + temperature_change
        if new_temperature < self.temperature_range[0]:
            new_temperature = self.temperature_range[0]
        elif new_temperature > self.temperature_range[1]:
            new_temperature = self.temperature_range[1]
        self.temperature = new_temperature

        radiation_change = np.random.normal(0, radiation_std)
        new_radiation = self.radiation + radiation_change
        if new_radiation < self.radiation_range[0]:
            new_radiation = self.radiation_range[0]
        elif new_radiation > self.radiation_range[1]:
            new_radiation = self.radiation_range[1]
        self.radiation = new_radiation

    def __repr__(self):
        return f"SetPoints(Temperature={self.temperature}, Radiation={self.radiation})"
    