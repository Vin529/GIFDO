import numpy as np
import trimesh
import pyvista as pv
import os
import pickle
#from shapely.geometry import Polygon
#from shapely.strtree import STRtree
#from shapely.geometry.base import BaseGeometry

from material import Material, WALL_MATERIAL_SET, FLOOR_MATERIAL_SET
#from material import GlazingMaterial, OpaqueMaterial
from candidate_evaluation import assess_candidate_fitness_single, assess_candidate_fitness_parallel
from candidate import Candidate
from candidate_building_structure import BuildingStructure
from candidate_setpoints import SetPoints
from crop import SIMULATION_CROP


#using pickle is the only way ive found to preserve the colours and the transparency in the mesh
def save_mesh_pickle(save_path: str, mesh: trimesh.Trimesh, face_materials: list[Material]) -> None:
    with open(save_path, 'wb') as file:
        pickle.dump((mesh, face_materials), file)


def show_labelled_mesh_render(candidate: Candidate) -> None:
    trimesh_mesh = candidate.building_structure.mesh
    face_materials = candidate.building_structure.face_materials

    #build pyvista mesh
    num_faces = len(trimesh_mesh.faces)
    faces_pv = np.hstack([
        np.full((num_faces, 1), 3, dtype=trimesh_mesh.faces.dtype),
        trimesh_mesh.faces
    ]).flatten()
    pv_mesh = pv.PolyData(trimesh_mesh.vertices, faces_pv)

    #assign per face rgba colours
    rgba = np.array([m.colour for m in face_materials], dtype=np.uint8)
    pv_mesh.cell_data["RGBA Colors"] = rgba

    #different colouring method
    # rgba_list = []

    # translucent_blue = [0, 100, 255, 60] # RGBA - Light Blue, ~40% transparent
    # opaque_grey = [150, 100, 100, 255]    # RGBA - Opaque light grey
    # default_color = [255, 0, 255, 255]    # Magenta for unexpected types

    # for i, m in enumerate(face_materials):
    #     if isinstance(m, GlazingMaterial):
    #         rgba_list.append(translucent_blue)
    #     elif isinstance(m, OpaqueMaterial):
    #         rgba_list.append(opaque_grey)
    #     else:
    #         print(f"Warning: Face {i} has unexpected material type: {type(m).__name__}. Using default color.")
    #         rgba_list.append(default_color)
    
    # rgba = np.array(rgba_list, dtype=np.uint8)
    # pv_mesh.cell_data["RGBA Colors"] = rgba

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.enable_depth_peeling()

    xmin, xmax, ymin, ymax, zmin, zmax = pv_mesh.bounds
    depth = xmax - xmin   #dimension along Y (north-south)
    width = ymax - ymin   #dimension along X (east-west)
    height = zmax - zmin  #dimension along Z (height)

    #add lines
    p_sw, p_se, p_nw, p_ne = [xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin]
    p_se_top = [xmax, ymin, zmax] #height line currently at south east corner
    line_south, line_east = pv.Line(p_sw, p_se), pv.Line(p_se, p_ne)
    line_north, line_west = pv.Line(p_ne, p_nw), pv.Line(p_nw, p_sw)
    line_height_se = pv.Line(p_se, p_se_top)

    line_color, line_width_val = 'cornflowerblue', 2
    plotter.add_mesh(line_south, color=line_color, line_width=line_width_val)
    plotter.add_mesh(line_east,  color=line_color, line_width=line_width_val)
    plotter.add_mesh(line_north, color=line_color, line_width=line_width_val)
    plotter.add_mesh(line_west,  color=line_color, line_width=line_width_val)
    plotter.add_mesh(line_height_se, color='orange', line_width=line_width_val)

    #label settings
    label_font_size = 20
    label_offset_factor = 0.08
    label_offset = max(width, depth) * label_offset_factor
    vertical_seperation_factor = 0.05
    vertical_label_separation = -height * vertical_seperation_factor

    #North=Green, South=Blue, East=Red, West=Magenta
    #control which labels appear, change depending on which side you wanto to take the render from
    show_north_labels = True
    show_south_labels = False
    show_east_labels  = True
    show_west_labels  = False
    show_height_label = True

    #height label
    if show_height_label:
        plotter.add_point_labels(
            [xmax + label_offset, ymin - label_offset / 2, (zmin + zmax) / 2],
            [f"Height: {height:.2f}"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    #east edge labels
    if show_east_labels:
        east_label_pos_base = [xmax + label_offset, (ymin + ymax) / 2, zmin]
        #east label
        plotter.add_point_labels(
            east_label_pos_base,
            ["East"],
            font_size=label_font_size,
            bold=True,
            text_color='red',
            point_size=0,
            shape=None,
            show_points=False
        )
        #width label
        east_label_pos_dim = [east_label_pos_base[0], east_label_pos_base[1], east_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            east_label_pos_dim,
            [f"Width: {width:.2f}"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    #north edge labels
    if show_north_labels:
        north_label_pos_base = [(xmin + xmax) / 2, ymax + label_offset, zmin]
        #north label
        plotter.add_point_labels(
            north_label_pos_base,
            ["North"],
            font_size=label_font_size,
            bold=True,
            text_color='green',
            point_size=0,
            shape=None,
            show_points=False
        )
        #depth label
        north_label_pos_dim = [north_label_pos_base[0], north_label_pos_base[1], north_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            north_label_pos_dim,
            [f"Depth: {depth:.2f}"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    #south edge labels
    if show_south_labels:
        south_label_pos_base = [(xmin + xmax) / 2, ymin - label_offset, zmin]
        #south label
        plotter.add_point_labels(
            south_label_pos_base,
            ["South"],
            font_size=label_font_size,
            bold=True,
            text_color='blue',
            point_size=0,
            shape=None,
            show_points=False
        )
        #depth label
        south_label_pos_dim = [south_label_pos_base[0], south_label_pos_base[1], south_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            south_label_pos_dim,
            [f"Depth: {depth:.2f}"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    #west edge labels
    if show_west_labels:
        west_label_pos_base = [xmin - label_offset, (ymin + ymax) / 2, zmin]
        #west label
        plotter.add_point_labels(
            west_label_pos_base, ["West"],
            font_size=label_font_size,
            bold=True,
            text_color='magenta',
            point_size=0,
            shape=None,
            show_points=False
        )
        #width label
        west_label_pos_dim = [west_label_pos_base[0], west_label_pos_base[1], west_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            west_label_pos_dim,
            [f"Width: {width:.2f}"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    #add the mesh
    plotter.add_mesh(
        pv_mesh,
        scalars="RGBA Colors",
        rgba=True,
        show_edges=False,
        #edge_color='grey',
        #line_width=1,
        specular=0.0,
        #ambient=0.8,
        #diffuse=0.2,
    )

    plotter.show()


MIN_DIMENSIONS = np.array([5, 5, 5], dtype=float)
MAX_DIMENSIONS = np.array([30, 30, 30], dtype=float)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..")

CANDIDATE_EVALUATION_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "candidate_files")
TOP_CANDIDATE_ARCHIVE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "top_candidate_files")
WEATHER_FILE_PATH = os.path.join(ROOT_DIRECTORY, "data", "weather_files", "SGP_Singapore.486980_IWEC.epw")

candidate_building_structure = BuildingStructure.from_random(
    available_wall_materials=WALL_MATERIAL_SET,
    available_floor_materials=FLOOR_MATERIAL_SET,
    min_dimensions=MIN_DIMENSIONS,
    max_dimensions=MAX_DIMENSIONS,
    subdivision_level=1
)
candidate_setpoints = SetPoints.from_random(
    temperature_range=(18.0, 28.0), #degrees C
    radiation_range=(0.0, 10.0), #MJ/(m**2 * day)
)
candidate = Candidate(0, candidate_building_structure, candidate_setpoints)


def show_mesh_with_colours(mesh: trimesh.Trimesh):
    # Convert Trimesh to PyVista PolyData
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))

    # Generate random colors
    num_faces = len(mesh.faces)
    random_colors = np.random.randint(0, 255, size=(num_faces, 3), dtype=np.uint8)

    # Assign colors to faces
    pv_mesh.cell_data["Face Colors"] = random_colors

    # Plot with PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="Face Colors", rgb=True, show_edges=True)
    plotter.show()


def show_mesh_with_colours(mesh: trimesh.Trimesh, face_materials=None):
    # Convert Trimesh to PyVista PolyData
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))

    # Use materials colors if provided, otherwise generate random colors
    num_faces = len(mesh.faces)
    if face_materials:
        colors = np.array([material.colour for material in face_materials], dtype=np.uint8)
    else:
        colors = np.random.randint(0, 255, size=(num_faces, 3), dtype=np.uint8)

    # Assign colors to faces
    pv_mesh.cell_data["Face Colors"] = colors

    # Plot with PyVista with enhanced visualization
    plotter = pv.Plotter(window_size=[1000, 800])
    
    # Enable depth peeling for better transparency rendering
    plotter.enable_depth_peeling()
    
    # Add the mesh with enhanced visual settings
    plotter.add_mesh(
        pv_mesh,
        scalars="Face Colors", 
        rgb=True,
        show_edges=True,         # Show edges for better definition
        edge_color='black',      # Black edges create a boundary effect
        ambient=0.3,             # Ambient light coefficient
        diffuse=0.7              # Diffuse light coefficient
    )
    
    # Add shadows for better depth perception
    #plotter.enable_shadows()
    
    # Add a directional light for better lighting effects
    #plotter.add_light(pv.Light(position=(1, 2, 3), focal_point=(0, 0, 0)))
    
    # Set a good camera position
    plotter.view_isometric()
    
    # Add floor (optional, uncomment if you want a floor to enhance shadows)
    # bounds = pv_mesh.bounds
    # floor = pv.Plane(
    #     center=(0, 0, bounds[4]), 
    #     direction=(0, 0, 1), 
    #     i_size=1.5*(bounds[1]-bounds[0]), 
    #     j_size=1.5*(bounds[3]-bounds[2])
    # )
    # plotter.add_mesh(floor, color='white', opacity=0.5)
    
    plotter.show()


show_mesh_with_colours(candidate.building_structure.mesh)




#show_labelled_mesh_render(candidate)

#read in delete me from pickle file
# with open("DELETE_ME.pickle", "rb") as file:
#     candidate = pickle.load(file)

#assess_candidate_fitness_single(candidate, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH)
# print(candidate.fitness)
# print(candidate.setpoints)
# print(candidate.annual_cooling_cost_USD)
# print(candidate.annual_heating_cost_USD)
# print(candidate.annual_lighting_cost_USD)
# print(candidate.mean_daily_total_PAR_energy_per_area_MJ)

#candidate.setpoints.temperature += 1.0

# assess_candidate_fitness_single(candidate, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH)
# print(candidate.fitness)
# print(candidate.setpoints)
# print(candidate.annual_cooling_cost_USD)
# print(candidate.annual_heating_cost_USD)
# print(candidate.annual_lighting_cost_USD)
# print(candidate.mean_daily_total_PAR_energy_per_area_MJ)

#candidate.setpoints.radiation += 1.0

# assess_candidate_fitness_single(candidate, SIMULATION_CROP, CANDIDATE_EVALUATION_DIRECTORY, WEATHER_FILE_PATH)
# print(candidate.fitness)
# print(candidate.setpoints)
# print(candidate.annual_cooling_cost_USD)
# print(candidate.annual_heating_cost_USD)
# print(candidate.annual_lighting_cost_USD)
# print(candidate.mean_daily_total_PAR_energy_per_area_MJ)

#candidate.save_pickle("DELETE_ME.pickle")