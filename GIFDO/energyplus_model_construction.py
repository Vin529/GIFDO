import os
import openstudio
import trimesh
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import copy

from candidate import Candidate
from material import Material, GlazingMaterial, WINDOW_FRAME_MATERIAL
from config import GROWING_LEVEL_INITIAL_FLOOR_OFFSET, ILLUMINANCE_GRID_INSET_DISTANCE, ILLUMINANCE_MAP_SAMPLES_PER_AXIS

#unlikely to need to change this value, controls the width of the frame around each window surface
#as to be large enough so that energyplus dosent throw  "area too small" errors, but ideally no larger
WINDOW_INSET_DISTANCE = 0.01 #meters


def build_model(candidate: Candidate, candidate_directory: str, weather_file_path: str) -> tuple[float, float]:
    #move the candidate geometry so it is above z=0, surfaces below this point are treated as "underground" by openstudio
    #take a copy so the original candidate is left untouched
    candidate_above_ground = copy.deepcopy(candidate)
    candidate_above_ground.building_structure.mesh = _get_above_ground_mesh_copy(candidate.building_structure.mesh)

    #create the model
    model = openstudio.model.Model()
    space = _candidate_to_space(model, candidate_above_ground)

    #tweak simulation parameters
    simulation_control = model.getSimulationControl()
    simulation_control.setSolarDistribution("FullExterior") #area is not convex
    simulation_control.setRunSimulationforSizingPeriods(False)
    simulation_control.setRunSimulationforWeatherFileRunPeriods(True) # Ensure annual run is still on

    #create and assign a thermal zone
    thermal_zone_name = "Main Zone"
    thermal_zone = openstudio.model.ThermalZone(model)
    thermal_zone.setName(thermal_zone_name)
    space.setThermalZone(thermal_zone)

    #log ideal heating load
    outputMeter = openstudio.model.OutputMeter(model)
    outputMeter.setSpecificEndUse("DistrictHeatingWater:Facility")
    outputMeter.setReportingFrequency("Hourly")
    outputMeter.setMeterFileOnly(False)
    #log ideal cooling load
    outputMeter = openstudio.model.OutputMeter(model)
    outputMeter.setSpecificEndUse("DistrictCooling:Facility")
    outputMeter.setReportingFrequency("Hourly")
    outputMeter.setMeterFileOnly(False)

    #set the weather file
    if not os.path.exists(weather_file_path):
        raise FileNotFoundError(f"weather file not found: {weather_file_path}")
    
    epw_file = openstudio.EpwFile(openstudio.path(weather_file_path))
    if not openstudio.model.WeatherFile.setWeatherFile(model, epw_file):
        raise RuntimeError(f"failed to set weather file: {weather_file_path}")
    
    #set the ground temperature
    ground_temps = openstudio.model.SiteGroundTemperatureBuildingSurface(model)
    ground_temps.setName("Ground Temperature")

    #add HVAC and thermostat
    ideal_loads = openstudio.model.ZoneHVACIdealLoadsAirSystem(model)
    ideal_loads.addToThermalZone(thermal_zone)

    #minimum and maximum temperature setpoints
    temperature_target_C = candidate.setpoints.temperature
    #use the same target for cooling and heating to effectively maintain a constant temperature
    heating_schedule = openstudio.model.ScheduleRuleset(model)
    heating_schedule.defaultDaySchedule().addValue(openstudio.Time(0, 24, 0), temperature_target_C)
    cooling_schedule = openstudio.model.ScheduleRuleset(model)
    cooling_schedule.defaultDaySchedule().addValue(openstudio.Time(0, 24, 0), temperature_target_C)

    thermostat = openstudio.model.ThermostatSetpointDualSetpoint(model)
    thermostat.setHeatingSetpointTemperatureSchedule(heating_schedule)
    thermostat.setCoolingSetpointTemperatureSchedule(cooling_schedule)
    thermal_zone.setThermostatSetpointDualSetpoint(thermostat)

    #illuminance map
    #height of bottom growing rack
    low_illuminance_map_z = GROWING_LEVEL_INITIAL_FLOOR_OFFSET
    low_grid_origin_x, low_grid_origin_y, low_grid_length_x, low_grid_length_y = _get_illuminance_grid_parameters(candidate_above_ground.building_structure.mesh, low_illuminance_map_z, ILLUMINANCE_GRID_INSET_DISTANCE)

    low_illuminance_map = openstudio.model.IlluminanceMap(model)
    low_illuminance_map.setName(f"Floor Illuminance Map")
    low_illuminance_map.setOriginXCoordinate(low_grid_origin_x)
    low_illuminance_map.setOriginYCoordinate(low_grid_origin_y)
    low_illuminance_map.setOriginZCoordinate(low_illuminance_map_z)
    low_illuminance_map.setXLength(low_grid_length_x)
    low_illuminance_map.setYLength(low_grid_length_y)
    low_illuminance_map.setNumberofXGridPoints(ILLUMINANCE_MAP_SAMPLES_PER_AXIS)
    low_illuminance_map.setNumberofYGridPoints(ILLUMINANCE_MAP_SAMPLES_PER_AXIS)
    low_illuminance_map.setSpace(space)
    thermal_zone.setIlluminanceMap(low_illuminance_map)

    #daylighting control (not used but needs to exist to trigger energyplus to do daylighting calculations for the illuminance map)
    #energyplus is very fussy about where this object gets placed, and will crash if it is within 0.15m of a window
    #hence the complex calculations required
    daylighting_control_x, daylighting_control_y = _get_valid_daylighting_control_position(candidate_above_ground, low_illuminance_map_z)

    daylighting_control = openstudio.model.DaylightingControl(model)
    daylighting_control.setName(f"Daylighting Control")
    daylighting_control.setPositionXCoordinate(daylighting_control_x)
    daylighting_control.setPositionYCoordinate(daylighting_control_y)
    daylighting_control.setPositionZCoordinate(low_illuminance_map_z)
    daylighting_control.setSpace(space)
    thermal_zone.setPrimaryDaylightingControl(daylighting_control)

    #save the openstudio model in a .osm file
    #not directly used in the simultion but useful for debugging
    osm_path = os.path.join(candidate_directory, "model.osm")
    model.save(osm_path, True)

    #convert openstudio model to .idf and save
    ##this is the file that energyplus will actually use for the simulation
    translator = openstudio.energyplus.ForwardTranslator()
    idf_file = translator.translateModel(model)
    idf_path = os.path.join(candidate_directory, "model.idf")
    idf_file.save(idf_path, True)

    #for some reason openstudio wont let me add two illuminance maps, the second one always overwrites the first
    #to overcome this i put the second illuminance map directly into the idf file as a string
    with open(idf_path, "a") as f_idf:
        #place the second illuminance map at the height of the centre of volume of the mesh
        high_illuminance_map_z = candidate_above_ground.building_structure.mesh.centroid[2]
        high_grid_origin_x, high_grid_origin_y, high_grid_length_x, high_grid_length_y = _get_illuminance_grid_parameters(candidate_above_ground.building_structure.mesh, high_illuminance_map_z, ILLUMINANCE_GRID_INSET_DISTANCE)
        high_grid_max_x = high_grid_origin_x + high_grid_length_x
        high_grid_max_y = high_grid_origin_y + high_grid_length_y
        high_illuminance_map_name = "Centroid Illuminance Map"

        new_map_string = f"""
Output:IlluminanceMap,
  {high_illuminance_map_name},                    !- Name
  {thermal_zone_name},                    !- Zone or Space Name
  {high_illuminance_map_z},                    !- Z height {{m}}
  {high_grid_origin_x},                    !- X Minimum Coordinate {{m}}
  {high_grid_max_x},                    !- X Maximum Coordinate {{m}}
  {ILLUMINANCE_MAP_SAMPLES_PER_AXIS},                    !- Number of X Grid Points
  {high_grid_origin_y},                    !- Y Minimum Coordinate {{m}}
  {high_grid_max_y},                    !- Y Maximum Coordinate {{m}}
  {ILLUMINANCE_MAP_SAMPLES_PER_AXIS};                    !- Number of Y Grid Points
"""
        try:
            f_idf.write(new_map_string)
        except IOError as e:
            raise RuntimeError(f"failed to write illuminance map to idf file: {e}")
        
    #avoid memory leak error
    del model

    low_illuminance_map_slice_area = _get_horizontal_mesh_slice_area(mesh=candidate_above_ground.building_structure.mesh, z=low_illuminance_map_z)
    high_illuminance_map_slice_area = _get_horizontal_mesh_slice_area(mesh=candidate_above_ground.building_structure.mesh, z=high_illuminance_map_z)
    return low_illuminance_map_slice_area, high_illuminance_map_slice_area


#translate mesh so that the floor is just above ground level, take a copy so that the original is not modified
def _get_above_ground_mesh_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh_copy = mesh.copy()
    min_z = np.min(mesh_copy.vertices[:, 2])
    if min_z < 0:
        translation_z = -min_z
        mesh_copy.vertices[:, 2] += translation_z    
    return mesh_copy


def _candidate_to_space(model: openstudio.model.Model, candidate: Candidate) -> openstudio.model.Space:
    #build the structure using geometry and materials specified in the candidate
    space = openstudio.model.Space(model)
    space.setName("Main Space")

    face_materials = candidate.building_structure.face_materials
    mesh = candidate.building_structure.mesh
    floor_face_indices = candidate.building_structure.floor_face_indices

    #convert materials to openstudio constructions, these are passed to the openstudio surface constructor along with the geometry
    all_materials_set = set(face_materials)
    all_materials_set.add(WINDOW_FRAME_MATERIAL)
    try:
        construction_dict = _materials_to_constructions(model, all_materials_set)
    except Exception as e:
        raise RuntimeError(f"failed to create constructions from materials: {e}")
    window_frame_construction = construction_dict[WINDOW_FRAME_MATERIAL.name]

    #convert the trimesh faces to openstudio faces, which use point3d objects to define the vertices
    openstudio_faces = _mesh_to_openstudio_faces(mesh)

    #iterate through each face, using its material and geometry to create a surface in openstudio
    for i, openstudio_face in enumerate(openstudio_faces):
        #set the geometry for the base surface
        surface = openstudio.model.Surface(openstudio_face, model)

        face_material = face_materials[i]
        face_material_name = face_material.name

        is_floor = (i in floor_face_indices)
        #set additional parameters for the surface, depending on whether it is a floor or wall
        if is_floor:
            surface.setSurfaceType("Floor")
            surface.setOutsideBoundaryCondition("Ground")
            surface.setSunExposure("NoSun")
            surface.setWindExposure("NoWind")
        else:
            surface.setSurfaceType("Wall")
            surface.setOutsideBoundaryCondition("Outdoors")
            surface.setSunExposure("SunExposed")
            surface.setWindExposure("WindExposed")
        
        #opaque materials are used directly for their surface construction 
        #glazing materials need to be used as a subsurface, so the base construction is set to the window frame material
        if isinstance(face_material, GlazingMaterial):
            #check that winndow is not in the floor
            if is_floor:
                raise ValueError(f"glazing material '{face_material_name}' is assigned to a floor surface at face index {i}, this in not allowed")

            #first build the base surface using the window frame material
            surface.setConstruction(window_frame_construction)
            surface.setName(f"surface_{i}_{WINDOW_FRAME_MATERIAL.name}")

            #then create a subsurface using the glazing material, and inset it slightly so it sits inside the window frame surface
            window_construction = construction_dict[face_material_name]

            #we pull the raw vertices for this face from the trimesh object, and use them to calculate the geometry of the window, which is inset by a small distance
            #the raw face vertices are used over the openstudio faces, as performing the geometric calculations on point3d objects is more difficult
            face = mesh.faces[i]
            vertices_np = mesh.vertices[face]
            inset_window_vertices_np = _get_inset_triangle_polygon_vertices(vertices_np, WINDOW_INSET_DISTANCE)
            openstudio_window_face = [openstudio.Point3d(*vertex_np) for vertex_np in inset_window_vertices_np] #the openstudio docs recommend using a Point3DVector here to contain the Point3d objects, instead of a python list

            subsurface = openstudio.model.SubSurface(openstudio_window_face, model)
            subsurface.setName(f"subsurface_{i}_{face_material_name}")
            subsurface.setSubSurfaceType("FixedWindow")
            subsurface.setSurface(surface)
            subsurface.setConstruction(window_construction)
        else:
            base_construction = construction_dict[face_material_name]
            surface.setConstruction(base_construction)
            surface.setName(f"surface_{i}_{face_material_name}")

        surface.setSpace(space)
    
    return space


def _materials_to_constructions(model: openstudio.model.Model, materials: set["Material"]) -> dict[str, openstudio.model.Construction]:
    construction_dict = {}
    for material in materials:
        construction = openstudio.model.Construction(model)
        construction.setName(material.name)

        os_material = material.to_openstudio(model)
        material_layer_vector = openstudio.model.MaterialVector()
        material_layer_vector.append(os_material)
        construction_success = construction.setLayers(material_layer_vector)

        if construction_success:
            construction_dict[material.name] = construction
        else:
            raise RuntimeError(f"Failed to set material '{material.name}' layer for a construction")
    
    return construction_dict


#check whether  the vertex order needs reversing, depends on how openstudio wants the normals to be oriented (inwards or outwards)
def _mesh_to_openstudio_faces(mesh: trimesh.Trimesh) -> list[list[openstudio.Point3d]]:
    openstudio_faces = []
    for face in mesh.faces:
        vertices_np = mesh.vertices[face]
        openstudio_face = [openstudio.Point3d(*vertex_np) for vertex_np in vertices_np] #the openstudio docs recommend using a Point3DVector here to contain the Point3d objects, instead of a python list
        openstudio_faces.append(openstudio_face)

    return openstudio_faces


def _get_inset_triangle_polygon_vertices(vertices_np: np.ndarray, inset_distance: float) -> np.ndarray:
    if vertices_np.shape != (3, 3):
        raise ValueError(f"Input array shape must be (3, 3) for triangle polygons, got {vertices_np.shape}")
    
    centre_point = np.mean(vertices_np, axis=0)
    inset_vertices = []
    for vertex in vertices_np:
        vector_to_centre = centre_point - vertex
        vector_to_centre_normalised = vector_to_centre / np.linalg.norm(vector_to_centre)
        inset_vertex = vertex + vector_to_centre_normalised * inset_distance
        inset_vertices.append(inset_vertex)

    inset_vertices_np = np.array(inset_vertices)
    return inset_vertices_np


def _get_valid_daylighting_control_position(
    candidate: Candidate,
    daylighting_control_height: float,
    wall_buffer: float = 0.20, #should be set a bit higher than min_clearance_window
    min_clearance_window: float = 0.16, #energyplus requires this to be at least 0.15m, so use 0.16 to avoid floating point errors and be safe
    grid_search_samples: int = 10 #resolution of the fallback grid search
) -> tuple[float, float]:
    #slice our mesh at the given height
    section = candidate.building_structure.mesh.section(plane_origin=[0, 0, daylighting_control_height], plane_normal=[0, 0, 1])
    slice_2D, to_3D = section.to_2D()
    if slice_2D is None:
        raise ValueError(f"failed to create 2D slice in mesh at z={daylighting_control_height}")
    #handles cases where a slice results in multiple polygons or a single polygon with holes
    initial_search_polygon = unary_union([Polygon(p.exterior.coords) for p in slice_2D.polygons_full if len(p.exterior.coords) >= 3])

    #defines the search area as all points more than min_clearance_boundary away from the edge of the slice polygon(s)
    #this interior area is more likely to yield a valid position for our daylighting control point, as it excludes points near the surrounding surfaces
    buffered_search_polygon = initial_search_polygon.buffer(-wall_buffer, join_style=2, mitre_limit=5.0)
    #if there is no such area, then fallback to the original polygon for search
    if not buffered_search_polygon or buffered_search_polygon.is_empty:
        search_polygon = initial_search_polygon
    else:
        search_polygon = buffered_search_polygon

    #get window planes
    #each window surface is treated as an inifinite plane, which is not ideal, as it excluded many points that may not be within 0.15m of the actual window
    #however it hugely simplifies the calclations, and in testing still produces valid points
    window_planes = []
    for face_index, vertex_indices in enumerate(candidate.building_structure.mesh.faces):
        face_material = candidate.building_structure.face_materials[face_index]
        if isinstance(face_material, GlazingMaterial):
            vertices_np = candidate.building_structure.mesh.vertices[vertex_indices]
            centroid, normal = trimesh.points.plane_fit(vertices_np)
            window_planes.append((centroid, normal))

    #representative point is gauranteed to be inslide the search polygon, and makes for a good guess for a potantially valid position
    representative_point_2D = search_polygon.representative_point()
    is_representative_point_inside_polygon = search_polygon.contains(Point(representative_point_2D.x, representative_point_2D.y))
    #in the process of slicing the mesh into a 2D polygon, the x and y coordinates are not preserved
    #so we need to use the to_3D matrix to invert this transformation, and get back the true coordinates of a point in the polygon relative to the mesh
    #this is necessary as the point is being compared to the window planes in the following funciton, so they need to be in the same coordinate space    
    corrected_representative_point_3D = _transform_3D_vector_homogeneous(np.array([representative_point_2D.x, representative_point_2D.y, 0]), to_3D)
    corrected_representative_point_x = corrected_representative_point_3D[0]
    corrected_representative_point_y = corrected_representative_point_3D[1]
    is_representative_point_valid = _is_daylighting_control_point_valid(corrected_representative_point_x, corrected_representative_point_y, daylighting_control_height, min_clearance_window, window_planes)
    
    if is_representative_point_inside_polygon and is_representative_point_valid:
        return corrected_representative_point_x, corrected_representative_point_y

    #if representative point is not valid, then fall back to a grid search over the search polygon
    min_x, min_y, max_x, max_y = search_polygon.bounds
    candidate_x_coords = np.linspace(min_x, max_x, grid_search_samples)
    candidate_y_coords = np.linspace(min_y, max_y, grid_search_samples)
    for candidate_x in candidate_x_coords:
        for candidate_y in candidate_y_coords:
            #check if the candidate point is within the search polygon
            is_candidate_point_inside_polygon = search_polygon.contains(Point(candidate_x, candidate_y))
            #in the process of slicing the mesh into a 2D polygon, the x and y coordinates are not preserved
            #so we need to use the to_3D matrix to invert this transformation, and get back the true coordinates of a point in the polygon relative to the mesh
            #this is necessary as the point is being compared to the window planes in the following funciton, so they need to be in the same coordinate space
            corrected_candidate_point_3D = _transform_3D_vector_homogeneous(np.array([candidate_x, candidate_y, 0]), to_3D)
            corrected_candidate_x = corrected_candidate_point_3D[0]
            corrected_candidate_y = corrected_candidate_point_3D[1]
            is_candidate_point_valid = _is_daylighting_control_point_valid(corrected_candidate_x, corrected_candidate_y, daylighting_control_height, min_clearance_window, window_planes)
            
            if is_candidate_point_inside_polygon and is_candidate_point_valid:
                return corrected_candidate_x, corrected_candidate_y

    raise ValueError(f"could not find valid daylighting point z={daylighting_control_height}")
    

#the point being inside the polygon is verified outside of this function
#valid is defined as not being within min_clearance_window of any window planes
def _is_daylighting_control_point_valid(
    x: float, 
    y: float, 
    z: float, 
    min_clearance_window: float, 
    window_planes: list[np.ndarray, np.ndarray], 
) -> bool:
    #if there are no window planes, then we are clearly not oging to be within 0.15m of one
    if not window_planes: 
        return True
    
    point_3D = np.array([x, y, z])
    for centroid, normal in window_planes:
        distance_to_plane = np.abs(np.dot(point_3D - centroid, normal))
        if distance_to_plane < min_clearance_window:
            return False
        
    return True


def _get_illuminance_grid_parameters(mesh: trimesh.Trimesh, grid_z: float, inset_distance: float) -> tuple[float, float, float, float]:
    #the illuminance map is a 2D horizontal grid, which we position so that it sits just inside the bounding area of the mesh slice at the provided z level
    illuminance_plane_normal = [0, 0, 1] #points upward in the z direction
    section = mesh.section(plane_origin=[0, 0, grid_z], plane_normal=illuminance_plane_normal)
    slice_2D, to_3D = section.to_2D()

    if slice_2D is None:
        raise ValueError(f"failed to create 2D slice in mesh at z={grid_z}")

    x_min, x_max, y_min, y_max = _get_corrected_slice_bounds(slice_2D, to_3D)
    x_min_inset = x_min + inset_distance
    x_max_inset = x_max - inset_distance
    y_min_inset = y_min + inset_distance
    y_max_inset = y_max - inset_distance
    x_length = x_max_inset - x_min_inset
    y_length = y_max_inset - y_min_inset

    return x_min_inset, y_min_inset, x_length, y_length


def _get_corrected_slice_bounds(slice_2D: trimesh.path.Path2D, to_3D: np.ndarray) -> tuple[float, float, float, float]: # type: ignore
    #handles cases where a slice results in multiple polygons or a single polygon with holes
    polygon_geometry = unary_union([Polygon(p.exterior.coords) for p in slice_2D.polygons_full if len(p.exterior.coords) >= 3])
    if polygon_geometry.geom_type == 'Polygon':
        polygons_to_process = [polygon_geometry]
    elif polygon_geometry.geom_type == 'MultiPolygon':
        polygons_to_process = list(polygon_geometry.geoms)

    #iterate through each point that makes up the boundary of the slice polygon(s)
    #apply the inverse transformation matrix to_3D to get the original 3D coordinates of these points before projection to the 2D plane
    #find the bounds of the slice by finding the minimum and maximum x and y coordinates from these points
    all_boundary_points_3D = []
    for polygon in polygons_to_process:
        boundary_2D = list(polygon.exterior.coords)
        #polgyons with 2 or fewer points will not enclos eany area, so do not need to be spanned by the illuminance map
        if len(boundary_2D) >= 3:
            for (x, y) in boundary_2D:
                point_3D = _transform_3D_vector_homogeneous(np.array([x, y, 0]), to_3D)
                all_boundary_points_3D.append(point_3D)

    if not all_boundary_points_3D:
        raise ValueError("the polygons within this slice do not seem to enclose any space")

    boundary_3D_np = np.array(all_boundary_points_3D)
    min_x, min_y, _ = np.min(boundary_3D_np, axis=0)
    max_x, max_y, _ = np.max(boundary_3D_np, axis=0)
    return min_x, max_x, min_y, max_y


def _transform_3D_vector_homogeneous(vector_3D: np.ndarray, matrix_4D: np.ndarray) -> np.ndarray:
    #use homogeneous coordinates
    vector_homogeneous = [vector_3D[0], vector_3D[1], vector_3D[2], 1.0]
    transformed_vector_homogeneous = matrix_4D.dot(vector_homogeneous)
    #convert from homogeneous coordinates (divide by w)
    transformed_vector = transformed_vector_homogeneous[:3] / transformed_vector_homogeneous[3]
    return transformed_vector


def _get_horizontal_mesh_slice_area(mesh: trimesh.Trimesh, z: float) -> float:
    plane_origin = [0, 0, z]
    plane_normal = [0, 0, 1] #normal points upward in the z direction

    #create a section of the mesh at the specified plane
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    #convert the section to 2D and get the area of the slice
    slice_2D, _ = section.to_2D()
    if slice_2D is None:
        raise ValueError(f"failed to create 2D slice in mesh at z={z}")

    area = slice_2D.area
    return area
