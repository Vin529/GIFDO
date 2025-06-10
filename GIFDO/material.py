import openstudio
import numpy as np


class Material:
    def __init__(
        self, 
        name: str,                   # used as a unique identifier, so different materials should not share the same name
        colour: np.ndarray,          # RGBA (not used in EnergyPlus simulation, only for visualisation)
        thickness: float,            # meters
        conductivity: float          # W/m*k
    ):
        self.name = name
        self.colour = colour
        self.thickness = thickness
        self.conductivity = conductivity

    def to_openstudio(self, model: openstudio.model.Model) -> openstudio.model.Material:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def __repr__(self):
        return f"Material(Name={self.name})"
    
    #following two methods are needed to do set operations on the material objects, name is the unique identifier
    def __eq__(self, other):
        if not isinstance(other, Material):
            return NotImplemented
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
        

class GlazingMaterial(Material):
    def __init__(
        self, 
        name: str,                   # used as a unique identifier, so different materials should not share the same name
        colour: np.ndarray,          # RGBA (not used in EnergyPlus simulation, only for visualisation)
        thickness: float,            # meters
        conductivity: float,         # W/m*k
        solar_transmittance_norm: float, 
        solar_reflectance_front_norm: float,
        solar_reflectance_back_norm: float,
        visible_transmittance_norm: float,
        visible_reflectance_front_norm: float,
        visible_reflectance_back_norm: float,
        ir_transmittance_norm: float,
        ir_emissivity_front: float,
        ir_emissivity_back: float,
    ):
        super().__init__(name, colour, thickness, conductivity)

        self.solar_transmittance_norm = solar_transmittance_norm
        self.solar_reflectance_front_norm = solar_reflectance_front_norm
        self.solar_reflectance_back_norm = solar_reflectance_back_norm

        self.visible_transmittance_norm = visible_transmittance_norm
        self.visible_reflectance_front_norm = visible_reflectance_front_norm
        self.visible_reflectance_back_norm = visible_reflectance_back_norm

        self.ir_transmittance_norm = ir_transmittance_norm
        self.ir_emissivity_front = ir_emissivity_front
        self.ir_emissivity_back = ir_emissivity_back

    def to_openstudio(self, model: openstudio.model.Model) -> openstudio.model.Material:
        #convert to openstudio glazing material
        os_glazing_material = openstudio.model.StandardGlazing(model)

        #these return false if they fail so you can catch these to check
        os_glazing_material.setName(self.name)
        os_glazing_material.setThickness(self.thickness)
        os_glazing_material.setConductivity(self.conductivity)

        os_glazing_material.setSolarTransmittanceatNormalIncidence(self.solar_transmittance_norm)
        os_glazing_material.setFrontSideSolarReflectanceatNormalIncidence(self.solar_reflectance_front_norm)
        os_glazing_material.setBackSideSolarReflectanceatNormalIncidence(self.solar_reflectance_back_norm)

        os_glazing_material.setVisibleTransmittanceatNormalIncidence(self.visible_transmittance_norm)
        os_glazing_material.setFrontSideVisibleReflectanceatNormalIncidence(self.visible_reflectance_front_norm)
        os_glazing_material.setBackSideVisibleReflectanceatNormalIncidence(self.visible_reflectance_back_norm)

        os_glazing_material.setInfraredTransmittanceatNormalIncidence(self.ir_transmittance_norm)
        os_glazing_material.setFrontSideInfraredHemisphericalEmissivity(self.ir_emissivity_front)
        os_glazing_material.setBackSideInfraredHemisphericalEmissivity(self.ir_emissivity_back)

        return os_glazing_material


class OpaqueMaterial(Material):
    def __init__(
        self, 
        name: str,                   # used as a unique identifier, so different materials should not share the same name
        colour: np.ndarray,          # RGBA (not used in EnergyPlus simulation, only for visualisation)
        thickness: float,            # meters
        conductivity: float,         # W/m*k
        roughness: str,              # “VeryRough”, “Rough”, “MediumRough”, “MediumSmooth”, “Smooth”, and “VerySmooth”
        density: float,              # Kg/M**3
        specific_heat: float,        # J/Kg*K
        thermal_absorptance: float,
        solar_absorptance: float,
        visible_absorptance: float,
    ):
        super().__init__(name, colour, thickness, conductivity)

        self.roughness = roughness
        self.density = density
        self.specific_heat = specific_heat
        self.thermal_absorptance = thermal_absorptance
        self.solar_absorptance = solar_absorptance
        self.visible_absorptance = visible_absorptance

    def to_openstudio(self, model: openstudio.model.Model) -> openstudio.model.Material:
        #convert to openstudio opaque material
        os_opaque_material = openstudio.model.StandardOpaqueMaterial(model)

        #these return false if they fail so you can catch these to check
        os_opaque_material.setName(self.name)
        os_opaque_material.setThickness(self.thickness)
        os_opaque_material.setConductivity(self.conductivity)

        os_opaque_material.setRoughness(self.roughness)
        os_opaque_material.setDensity(self.density)
        os_opaque_material.setSpecificHeat(self.specific_heat)
        os_opaque_material.setThermalAbsorptance(self.thermal_absorptance)
        os_opaque_material.setSolarAbsorptance(self.solar_absorptance)
        os_opaque_material.setVisibleAbsorptance(self.visible_absorptance)

        return os_opaque_material
        

#wall materials
#the name string is used as the unique identifier for the material, so different materials should not share the same name
#this also applies to the materials in the floor material set below
I02_50_mm_insulation_board_wall = OpaqueMaterial(
    name="I02_50_mm_insulation_board",
    colour=np.array([130, 140, 110, 255], dtype=np.uint8), #olive green
    roughness="MediumRough",
    thickness=0.0508,
    conductivity=0.03,  
    density=43, 
    specific_heat=1210,
    thermal_absorptance=0.90, #default
    solar_absorptance=0.70, #default
    visible_absorptance=0.70 #default
)

M08_200mm_lightweight_concrete_block_filled = OpaqueMaterial(
    name="M08_200mm_lightweight_concrete_block_filled",
    colour=np.array([215, 205, 195, 255], dtype=np.uint8), #beige
    roughness="MediumRough",
    thickness=0.2032, 
    conductivity=0.26,
    density=464, 
    specific_heat=880,
    thermal_absorptance=0.90, #default
    solar_absorptance=0.70, #default
    visible_absorptance=0.70 #default
)

M14a_100mm_heavyweight_concrete = OpaqueMaterial(
    name="M14a_100mm_heavyweight_concrete",
    colour=np.array([140, 60, 40, 255], dtype=np.uint8), #dark red
    roughness="MediumRough",
    thickness=0.1016, 
    conductivity=1.95, 
    density=2240,
    specific_heat=900,
    thermal_absorptance=0.90, #default
    solar_absorptance=0.70, #default
    visible_absorptance=0.70 #default
)

low_iron_3mm = GlazingMaterial(
    name="low_iron_3mm",
    colour=np.array([200, 255, 255, 120], dtype=np.uint8), #blueish
    thickness=0.003, 
    conductivity=0.9,
    solar_transmittance_norm=0.899,
    solar_reflectance_front_norm=0.079,
    solar_reflectance_back_norm=0.079,             
    visible_transmittance_norm=0.913,
    visible_reflectance_front_norm=0.082,
    visible_reflectance_back_norm=0.082,
    ir_transmittance_norm=0.0,
    ir_emissivity_front=0.84,
    ir_emissivity_back=0.84
)

LoE_spec_sel_clear_6mm = GlazingMaterial(
    name="LoE_spec_sel_clear_6mm",
    colour=np.array([200, 200, 255, 120], dtype=np.uint8), #greenish
    thickness=0.006,
    conductivity=0.9,
    solar_transmittance_norm=0.430,
    solar_reflectance_front_norm=0.300,
    solar_reflectance_back_norm=0.420,
    visible_transmittance_norm=0.770,
    visible_reflectance_front_norm=0.070,
    visible_reflectance_back_norm=0.060,
    ir_transmittance_norm=0.0,
    ir_emissivity_front=0.84,
    ir_emissivity_back=0.03
)

clear_6mm = GlazingMaterial(
    name="clear_6mm",
    colour=np.array([255, 255, 200, 120], dtype=np.uint8), #yellowish
    thickness=0.006,
    conductivity=0.9,
    solar_transmittance_norm=0.775,
    solar_reflectance_front_norm=0.071,
    solar_reflectance_back_norm=0.071,
    visible_transmittance_norm=0.881,
    visible_reflectance_front_norm=0.080,
    visible_reflectance_back_norm=0.080,
    ir_transmittance_norm=0.0,
    ir_emissivity_front=0.84,
    ir_emissivity_back=0.84
)

#floor materials
M16_300mm_heavyweight_concrete = OpaqueMaterial(
    name="M16_300mm_heavyweight_concrete",
    colour=np.array([40, 40, 40, 255], dtype=np.uint8), #dark grey
    roughness="MediumRough",
    thickness=0.3048,  
    conductivity=1.95,
    density=2240,
    specific_heat=900,
    thermal_absorptance=0.90, #default
    solar_absorptance=0.70, #default
    visible_absorptance=0.70 #default
)

M14_150mm_heavyweight_concrete = OpaqueMaterial(
    name="M14_150mm_heavyweight_concrete",
    colour=np.array([40, 40, 40, 255], dtype=np.uint8), #dark grey
    roughness="MediumRough",
    thickness=0.1524,   
    conductivity=1.95,
    density=2240,
    specific_heat=900,
    thermal_absorptance=0.90, #default
    solar_absorptance=0.70, #default
    visible_absorptance=0.70 #default
)

#window frame material
generic_opaque_windowframe_material = OpaqueMaterial(
    name="generic_window_frame",
    colour=np.array([101, 67, 33, 255], dtype=np.uint8),  #dark brown
    thickness=0.1,
    conductivity=0.17,
    density=700,
    specific_heat=1000,
    roughness="MediumSmooth",
    thermal_absorptance=0.9, #default
    solar_absorptance=0.7, #default
    visible_absorptance=0.7 #default
)

WALL_MATERIAL_SET = {
    I02_50_mm_insulation_board_wall,
    M08_200mm_lightweight_concrete_block_filled,
    M14a_100mm_heavyweight_concrete,
    low_iron_3mm,
    LoE_spec_sel_clear_6mm,
    clear_6mm
}

FLOOR_MATERIAL_SET = {
    M16_300mm_heavyweight_concrete,
    M14_150mm_heavyweight_concrete
}

#all windows are embedded into this material, the window takes up the majority of the surface so these material properties should ideally not make much difference
WINDOW_FRAME_MATERIAL = generic_opaque_windowframe_material
