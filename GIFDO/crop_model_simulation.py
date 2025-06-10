import numpy as np
import datetime

from croprl.simple_model_env import SimpleCropModelEnv
from crop import Crop

#used for outdoor simulation. Irrelevant for indoor farming as temp and sunlight are controlled
#assume a location at sea level on the equator, should be default values
LAT = 0
ELEV = 0

#   https://www.weather.gov/epz/wxcalc_vaporpressure
CO2 = 1000 # PPM
AVG_VAPOR_PRESSURE = 20 # hPa #20


class IndoorClimateConditions:
    def __init__(
        self, 
        num_days: int, 
        daily_min_temp: np.ndarray[float], 
        daily_max_temp: np.ndarray[float], 
        daily_mean_temp: np.ndarray[float], 
        daily_radiation: np.ndarray[float],
        co2: float,
        avg_vapor_pressure: float,
    ):
        self.num_days = num_days

        self.min_temp = daily_min_temp # degrees C
        self.max_temp = daily_max_temp # degrees C
        self.mean_temp = daily_mean_temp # degrees C
        self.radiation = daily_radiation # MJ/(m**2 * day)

        if not len(daily_min_temp) == num_days:
            raise ValueError(f"daily_min_temp must be of length {num_days}, but it was set to {len(daily_min_temp)}")
        if not len(daily_max_temp) == num_days:
            raise ValueError(f"daily_max_temp must be of length {num_days}, but it was set to {len(daily_max_temp)}")
        if not len(daily_mean_temp) == num_days:
            raise ValueError(f"daily_mean_temp must be of length {num_days}, but it was set to {len(daily_mean_temp)}")
        if not len(daily_radiation) == num_days:
            raise ValueError(f"daily_radiation must be of length {num_days}, but it was set to {len(daily_radiation)}")

        self.precipitation = np.zeros(self.num_days, dtype=np.float64) # mm (not applicable in indoor farming), use irrigation instead
        self.avg_wind = np.zeros(self.num_days, dtype=np.float64) # m/s (not applicable for indoor farming)

        self.co2 = np.ones(num_days, dtype=np.float64) * co2  # PPM 
        self.avg_vapor_pressure = np.ones(num_days, dtype=np.float64) * avg_vapor_pressure  # hPa


#this object is required to pass to the constructor for the SimpleCropModelEnv class
#for a fitness function, random variation is not useful, so all standard deviation values are set to 0
class ZeroVariationClimateSTDs:
    def __init__(self):
        self.max_temp = 0               # degrees C
        self.min_temp = 0               # degrees C
        self.precipitation = 0          # mm
        self.radiation = 0              # MJ/(m**2 * day)
        self.co2 = 0                    # PPM
        self.avg_vapor_pressure = 0     # hPa
        self.mean_temp = 0              # degrees C; not in SIMPLE model
        self.avg_wind = 0               # m/s; not in SIMPLE model


def simulate_annual_expected_yields(
    crop: Crop, 
    temperature_setpoint: float,    # degrees C
    radiation_setpoint: float       # MJ/(m**2 * day)
):
    #these parameters are uniform throughout the growing period
    #in reality there will be slight variation even in a controlled indoor environment, but this gives a good approximation
    daily_temperature_setpoints = np.ones(crop.growth_cycle_days) * temperature_setpoint
    daily_radiation_setpoints = np.ones(crop.growth_cycle_days) * radiation_setpoint

    #min, max and mean temperature are all the same in this case, modelling a perfectly controlled environment
    climate_conditions = IndoorClimateConditions(
        num_days=crop.growth_cycle_days,
        daily_min_temp=daily_temperature_setpoints,
        daily_max_temp=daily_temperature_setpoints,
        daily_mean_temp=daily_temperature_setpoints,
        daily_radiation=daily_radiation_setpoints,
        co2=CO2,
        avg_vapor_pressure=AVG_VAPOR_PRESSURE
    )

    #this is not used, just a placeholder as the SimpleCropModelEnv constructor requires a sowing date
    sowing_date = datetime.datetime(day=1, month=1, year=2025) 

    env = SimpleCropModelEnv(
        sowing_date=sowing_date,
        num_growing_days=crop.growth_cycle_days,
        weather_schedule=climate_conditions,
        weather_forecast_stds=ZeroVariationClimateSTDs(),
        latitude=LAT,
        elevation=ELEV,
        crop_parameters=crop.crop_parameters,
        seed=0, #ensure reproducibility for any stochastic elements
        cumulative_biomass_std=0,
        plant_available_water_std=0
    )
    
    env.reset()
    done = False
    iter = 0
    while not done and iter < 1_000_000:
        #this number corresponds to available irrigation capacity
        #as i am not considering irrigation needs in my applicaiton, this is set to a very high value so as to not be a limiting factor
        action = [1_000_000]
        s, r, done, info = env.step(action)

        #so for every iteration before the end here, reward is just oging to give me back some negative constant multiplied by whatever irrigation value i feed it in action
        #on the last iteration, it will give me self.cumulative_biomass * self.crop_harvest_index plus this negative irrigaiton reward
        #in both cases the irrigation value in question is not a total, just what it gets passed in that iteration

        iter += 1       

    #the final reward and cumulative reward include a small negative penalty per unit of irrigation
    #for my purposes I am not considering irrigation, so i set the amount of available irrigation sufficiently high so as to not be a limiting factor, and calculate yield directly from the cumulative biomass
    #so the reward is not used

    #yield is in units of metric tons per hectare in paper
    growth_period_yield_ton_per_hectare = env.cumulative_biomass * env.crop_harvest_index
    growth_period_yield_kg_per_m2 = growth_period_yield_ton_per_hectare * 1000 / 10_000 #convert to kg/m^2
    annual_yield_kg_per_m2 = growth_period_yield_kg_per_m2 * (365 / crop.growth_cycle_days) #convert to annual yield
    return annual_yield_kg_per_m2
