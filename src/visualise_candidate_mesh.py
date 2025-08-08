import numpy as np
import pyvista as pv
import os
import pickle

from candidate import Candidate

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..")
PICKLE_CANDIDATE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "data", "visualisation_files")
CANDIDATE_PICKLE_FILE_NAME = "oslo_candidate.pickle"


def show_labelled_interactive_mesh_render(candidate: Candidate) -> None:
    trimesh_mesh = candidate.building_structure.mesh
    face_materials = candidate.building_structure.face_materials

    #build pyvista mesh form trimesh mesh, better to work with for visualisation
    num_faces = len(trimesh_mesh.faces)
    faces_pv = np.hstack([np.full((num_faces, 1), 3, dtype=trimesh_mesh.faces.dtype), trimesh_mesh.faces]).flatten()
    pv_mesh = pv.PolyData(trimesh_mesh.vertices, faces_pv)

    #assign per face rgba colours
    rgba = np.array([m.colour for m in face_materials], dtype=np.uint8)
    pv_mesh.cell_data["RGBA Colors"] = rgba

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.enable_depth_peeling()

    xmin, xmax, ymin, ymax, zmin, zmax = pv_mesh.bounds
    width = xmax - xmin   #dimension along Y (north-south)
    depth = ymax - ymin   #dimension along X (east-west)
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
    label_font_size = 30
    label_offset_factor = 0.18
    height_label_offset = 1
    label_offset = max(width, depth) * label_offset_factor
    vertical_seperation_factor = 0.2
    vertical_label_separation = -height * vertical_seperation_factor

    #North=Green, South=Blue, East=Red, West=Magenta
    #control which labels appear, change depending on which side you wanto to take the render from
    show_north_labels = False
    show_south_labels = True
    show_east_labels  = False
    show_west_labels  = True
    show_height_label = True

    #height label
    if show_height_label:
        plotter.add_point_labels(
            [xmax + height_label_offset, ymin - height_label_offset, (zmin + zmax) / 2],
            [f"Height: {height:.2f} m"],
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
        #depth label
        east_label_pos_dim = [east_label_pos_base[0], east_label_pos_base[1], east_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            east_label_pos_dim,
            [f"Depth: {depth:.2f} m"],
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
        #width label
        north_label_pos_dim = [north_label_pos_base[0], north_label_pos_base[1], north_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            north_label_pos_dim,
            [f"Width: {width:.2f} m"],
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
        #width label
        south_label_pos_dim = [south_label_pos_base[0], south_label_pos_base[1], south_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            south_label_pos_dim,
            [f"Width: {width:.2f} m"],
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
        #depth label
        west_label_pos_dim = [west_label_pos_base[0], west_label_pos_base[1], west_label_pos_base[2] + vertical_label_separation]
        plotter.add_point_labels(
            west_label_pos_dim,
            [f"Depth: {depth:.2f} m"],
            font_size=label_font_size,
            bold=False,
            text_color='black',
            point_size=0,
            shape=None,
            show_points=False
        )

    # #add the mesh
    plotter.add_mesh(
        pv_mesh,
        scalars="RGBA Colors",
        rgba=True,
        show_edges=True,
        #edge_color='grey',
        line_width=0.2,
        specular=0.0,
        ambient=0.8,
        diffuse=0.2,
    )

    plotter.show()


def main():
    candidate_pickle_path = os.path.join(PICKLE_CANDIDATE_DIRECTORY, CANDIDATE_PICKLE_FILE_NAME)

    with open(candidate_pickle_path, "rb") as file:
        loaded_candidate = pickle.load(file)

    show_labelled_interactive_mesh_render(loaded_candidate)


if __name__ == "__main__":
    main()
