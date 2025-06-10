class CropParametersSpec:    
    def __init__(
        self,
        temp_base,
        temp_opt,
        RUE,
        rad_50p_growth,
        rad_50p_senescence,
        maturity_temp,
        rad_50p_max_heat,
        rad_50p_max_water,
        heat_stress_thresh,
        heat_stress_extreme,
        drought_stress_sensitivity,
        deep_drainage_coef,
        water_holding_capacity,
        runoff_curve_number,
        root_zone_depth,
        co2_sensitivity,
        harvest_index
    ):          
        self.temp_base = temp_base                                       # T_base
        self.temp_opt = temp_opt                                         # T_opt
        self.RUE = RUE                                                   # RUE
        self.rad_50p_growth = rad_50p_growth                             # I_50A
        self.rad_50p_senescence = rad_50p_senescence                     # I_50B
        self.maturity_temp = maturity_temp                               # T_sum
        self.rad_50p_max_heat = rad_50p_max_heat                         # I_50maxH
        self.rad_50p_max_water = rad_50p_max_water                       # I_50maxW
        self.heat_stress_thresh = heat_stress_thresh                     # T_max
        self.heat_stress_extreme = heat_stress_extreme                   # T_ext
        self.drought_stress_sensitivity = drought_stress_sensitivity     # S_water
        self.deep_drainage_coef = deep_drainage_coef                     # DDC
        self.water_holding_capacity = water_holding_capacity             # AWC
        self.runoff_curve_number = runoff_curve_number                   # RCN
        self.root_zone_depth = root_zone_depth                           # RZD
        self.co2_sensitivity = co2_sensitivity                           # S_CO2
        self.harvest_index = harvest_index                               # HI


class Crop:
    def __init__(self, crop_name: str, crop_parameters: CropParametersSpec, growth_cycle_days: int, price_USD_per_kg: float):
        self.crop_name = crop_name
        self.crop_parameters = crop_parameters
        self.growth_cycle_days = growth_cycle_days
        #self.photoperiod_hours = photoperiod_hours
        self.price_USD_per_kg =  price_USD_per_kg


#these potato parameters were provided in the example script for croprl
potato_russet_usa_params = CropParametersSpec(
    temp_base=4,                        # T_base
    temp_opt=22,                        # T_opt
    RUE=1.30,                           # RUE
    rad_50p_growth=500,                 # I_50A
    rad_50p_senescence=400,             # I_50B
    maturity_temp=2300,                 # T_sum
    rad_50p_max_heat=50,                # I_50maxH
    rad_50p_max_water=30,               # I_50maxW
    heat_stress_thresh=34,              # T_max
    heat_stress_extreme=45,             # T_ext
    drought_stress_sensitivity=0.4,     # S_water
    deep_drainage_coef=0.8,             # DDC
    water_holding_capacity=0.1,         # AWC
    runoff_curve_number=64,             # RCN
    root_zone_depth=1200,               # RZD
    co2_sensitivity=0.10,               # S_CO2
    harvest_index=0.9                   # HI    
)
potato_crop = Crop(
    crop_name="Potato",
    crop_parameters=potato_russet_usa_params,
    growth_cycle_days=120,
    price_USD_per_kg=1 #2
)

#from SIMPLE paper, rearranged order of parameters to match their order in the table
tomato_params = CropParametersSpec(
    maturity_temp=2800,                 # T_sum
    harvest_index=0.68,                 # HI  
    rad_50p_growth=520,                 # I_50A
    rad_50p_senescence=400,             # I_50B
    temp_base=6,                        # T_base
    temp_opt=26,                        # T_opt
    RUE=1.00,                           # RUE
    rad_50p_max_heat=100,               # I_50maxH
    rad_50p_max_water=5,                # I_50maxW
    heat_stress_thresh=32,              # T_max
    heat_stress_extreme=45,             # T_ext
    co2_sensitivity=0.07,               # S_CO2
    drought_stress_sensitivity=2.5,     # S_water
    #these parameters are for the soil
    water_holding_capacity=0.19,        # AWC
    runoff_curve_number=85,             # RCN
    deep_drainage_coef=0.1,             # DDC
    root_zone_depth=800,                # RZD
)
tomato_crop = Crop(
    crop_name="Tomato",
    crop_parameters=tomato_params,
    growth_cycle_days=120,
    price_USD_per_kg=4
)


SIMULATION_CROP = tomato_crop
