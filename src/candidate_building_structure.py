#use for optional type hinting
from __future__ import annotations

import numpy as np
import trimesh
import random
from typing import Tuple
import pymeshlab
import tempfile
import os
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.errors import GEOSException

from material import Material


class BuildingStructure:
    def __init__(
        self, 
        mesh: trimesh.Trimesh, 
        face_materials: list[Material], 
        available_wall_materials: set[Material], 
        available_floor_materials: set[Material],
        min_dimensions: np.ndarray[float], 
        max_dimensions: np.ndarray[float], 
        subdivision_level: int = 0,
        floor_vertex_indices: np.ndarray[int] | None = None,
        floor_face_indices: np.ndarray[int] | None = None
    ):
        self.mesh = mesh
        self.face_materials = face_materials
        self.available_wall_materials = available_wall_materials
        self.available_floor_materials = available_floor_materials
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        self.subdivision_level = subdivision_level

        #these checks avoid unnecessary recalculation of the floor vertex and face indices if they were already calculated in the constructor
        if floor_vertex_indices is None:
            self.floor_vertex_indices = BuildingStructure._get_floor_vertex_indices(self.mesh)
        else:
            self.floor_vertex_indices = floor_vertex_indices

        if floor_face_indices is None:
            self.floor_face_indices = BuildingStructure._get_floor_face_indices(self.mesh, self.floor_vertex_indices)
        else:
            self.floor_face_indices = floor_face_indices
        
    @classmethod
    def from_random(
        cls, 
        available_wall_materials: set[Material], 
        available_floor_materials: set[Material],
        min_dimensions: np.ndarray[float], 
        max_dimensions: np.ndarray[float], 
        subdivision_level: int = 0,
        base_vertex_perturbation_count: int = 10,
        max_vertex_displacement_factor: float = 0.2
    ) -> "BuildingStructure":
        if not available_wall_materials:
            raise ValueError("available_wall_materials set passed to from_random() is empty")
        if not available_floor_materials:
            raise ValueError("available_floor_materials set passed to from_random() is empty")

        #initialise a box mesh with randomly distributed dimensions
        base_dimensions = np.random.uniform(low=min_dimensions, high=max_dimensions)
        mesh = trimesh.creation.box(base_dimensions)

        #subdivide to the appropriate level
        for _ in range(subdivision_level):
            mesh = mesh.subdivide()

        floor_vertex_indices = cls._get_floor_vertex_indices(mesh)
        floor_face_indices = cls._get_floor_face_indices(mesh, floor_vertex_indices)

        #check the initial box mesh for validity
        if not cls._check_valid_mesh(mesh, min_dimensions, max_dimensions, floor_vertex_indices, floor_face_indices):
            raise ValueError("initial mesh from from_random() failed validity check")

        #scale the vertex perturbation count and displacement factor with respect to the subdivision level
        vertex_perturbation_count = base_vertex_perturbation_count * (4 ** subdivision_level)
        vertex_displacement_factor = max_vertex_displacement_factor / ((subdivision_level) + 1)
        #perform vertex perturbations, only apply perturbations that lead to a valid mesh
        for _ in range(vertex_perturbation_count):
            new_mesh = mesh.copy()

            displacement_factor_vector = np.random.uniform(-vertex_displacement_factor, vertex_displacement_factor, size=3)
            displacement_vector = displacement_factor_vector * (max_dimensions - min_dimensions)
            random_vertex_index = np.random.randint(0, len(mesh.vertices))

            if random_vertex_index in floor_vertex_indices:
                #vertices that are a part of the floor are limited to lateral movement to keep a flat base
                lateral_displacement_vector = displacement_vector.copy()
                lateral_displacement_vector[2] = 0
                new_mesh.vertices[random_vertex_index] += lateral_displacement_vector
            else:
                new_mesh.vertices[random_vertex_index] += displacement_vector

            if cls._check_valid_mesh(new_mesh, min_dimensions, max_dimensions, floor_vertex_indices, floor_face_indices):
                mesh = new_mesh

        #assign random material to each face, floor and wall faces each have their own seperate pool of materials to choose from
        available_floor_material_list = list(available_floor_materials)
        available_wall_material_list = list(available_wall_materials)
        face_materials = []
        for i in range(len(mesh.faces)):
            if i in floor_face_indices:
                face_materials.append(np.random.choice(available_floor_material_list))
            else:
                face_materials.append(np.random.choice(available_wall_material_list))
        if not cls._check_valid_materials(mesh, face_materials, available_wall_materials, available_floor_materials, floor_face_indices):
            raise ValueError("Invalid face materials assigned during from_random")

        return cls(
            mesh, 
            face_materials, 
            available_wall_materials, 
            available_floor_materials,
            min_dimensions, 
            max_dimensions, 
            subdivision_level, 
            floor_vertex_indices, 
            floor_face_indices
        )
    
    @classmethod
    def crossover(
        cls, 
        parent_1: "BuildingStructure", 
        parent_2: "BuildingStructure", 
        alpha: float | None = None, 
        beta: float | None = None,
        max_random_alpha_attempts: int = 10
    ) -> BuildingStructure | None:
        #perform initial crossover compatibility checks
        if len(parent_1.mesh.vertices) != len(parent_2.mesh.vertices):
            raise ValueError("Both parents must have the same number of vertices")
        if len(parent_1.mesh.faces) != len(parent_2.mesh.faces):
            raise ValueError("Both parents must have the same number of faces")
        if not np.array_equal(parent_1.floor_vertex_indices, parent_2.floor_vertex_indices):
            raise ValueError("Both parents must have the same floor vertex indices")
        if not np.array_equal(parent_1.floor_face_indices, parent_2.floor_face_indices):
            raise ValueError("Both parents must have the same floor face indices")
        
        child_min_dimensions = np.minimum(parent_1.min_dimensions, parent_2.min_dimensions)
        child_max_dimensions = np.maximum(parent_1.max_dimensions, parent_2.max_dimensions)
        child_subdivision_level = parent_1.subdivision_level
        #the crossover operation assumes that these indices are consistent across all meshes with the same subdivision level, in testing this seems to be the case
        child_floor_vertex_indices = parent_1.floor_vertex_indices
        child_floor_face_indices = parent_1.floor_face_indices

        #alpha is the interpolation factor between the two parent meshes
        if alpha is not None:
            #alpha was provided
            if not (0 <= alpha <= 1):
                raise ValueError("Alpha must be between 0 and 1 in crossover")

            #interpolate between the two parent meshes using provided alpha
            #revert to random alpha if the resulting mesh is invalid
            candidate_child_vertices = alpha * parent_1.mesh.vertices + (1 - alpha) * parent_2.mesh.vertices
            candidate_child_faces = parent_1.mesh.faces
            candidate_child_mesh = trimesh.Trimesh(vertices=candidate_child_vertices, faces=candidate_child_faces)
            if cls._check_valid_mesh(candidate_child_mesh, child_min_dimensions, child_max_dimensions, child_floor_vertex_indices, child_floor_face_indices):
                child_mesh = candidate_child_mesh
            else:
                print("crossover gave an invalid child mesh using the provided alpha, reverting to random alpha")
                alpha = None

        if alpha is None:
            #no alpha was provided
            #have multiple attempts at choosing random alpha, break when we find one that results in a valid child mesh
            for _ in range(max_random_alpha_attempts):
                #i found that having alpha be between 0 and 1 allowed a top candidate to effectively "clone" itself occasionally, harming genetic diversity
                #betwen 0.2 and 0.8 still allows variety but prevents cloning
                candidate_alpha = np.random.uniform(0.2, 0.8)
                candidate_child_vertices = candidate_alpha * parent_1.mesh.vertices + (1 - candidate_alpha) * parent_2.mesh.vertices
                candidate_child_faces = parent_1.mesh.faces
                candidate_child_mesh = trimesh.Trimesh(vertices=candidate_child_vertices, faces=candidate_child_faces)
                if cls._check_valid_mesh(candidate_child_mesh, child_min_dimensions, child_max_dimensions, child_floor_vertex_indices, child_floor_face_indices):
                    child_mesh = candidate_child_mesh
                    alpha = candidate_alpha
                    break

            #if no valid alpha was found, return None, the parent function will handle this case
            if alpha is None:
                return None

        #beta is the interpolation factor between the two parent face materials
        if beta is None:
            #prevent cloning all materials from one parent
            beta = np.random.uniform(0.2, 0.8)

        if not (0 <= beta <= 1):
            raise ValueError("Beta must be between 0 and 1 in crossover")

        #choose each face material from parent_1 or parent_2 with probability beta and (1 - beta) respectively
        material_selection_mask = np.random.rand(len(parent_1.face_materials)) < beta
        child_face_materials = list(np.where(material_selection_mask, parent_1.face_materials, parent_2.face_materials))
        child_available_wall_materials = parent_1.available_wall_materials.union(parent_2.available_wall_materials)
        child_available_floor_materials = parent_1.available_floor_materials.union(parent_2.available_floor_materials)

        #create the child, and perform final validity checks
        child = cls(
            child_mesh, 
            child_face_materials, 
            child_available_wall_materials, 
            child_available_floor_materials,
            child_min_dimensions, 
            child_max_dimensions,
            child_subdivision_level,
            child_floor_vertex_indices,
            child_floor_face_indices
        )
        if not cls._check_valid_materials(child_mesh, child_face_materials, child_available_wall_materials, child_available_floor_materials, child_floor_face_indices):
            print("crossover gave an invalid child face materials, child scrapped")
            return None
        if not cls._check_valid_mesh(child_mesh, child_min_dimensions, child_max_dimensions, child_floor_vertex_indices, child_floor_face_indices):
            print("crossover gave an invalid child mesh, child scrapped")
            return None
        
        return child

    #colours each face of the mesh according to its material
    def get_coloured_mesh(self) -> trimesh.Trimesh:
        coloured_mesh = self.mesh.copy()
        face_colours = np.array([material.colour for material in self.face_materials], dtype=np.uint8)
        coloured_mesh.visual.face_colors = face_colours
        return coloured_mesh

    def subdivide(self) -> None:
        subdivided_mesh = self.mesh.subdivide()
        subdivided_mesh_floor_vertex_indices = BuildingStructure._get_floor_vertex_indices(subdivided_mesh)
        subdivided_mesh_floor_face_indices = BuildingStructure._get_floor_face_indices(subdivided_mesh, subdivided_mesh_floor_vertex_indices)
        if not BuildingStructure._check_valid_mesh(subdivided_mesh, self.min_dimensions, self.max_dimensions, subdivided_mesh_floor_vertex_indices, subdivided_mesh_floor_face_indices):
            raise ValueError("subdivision resulted in an invalid mesh")
        
        original_face_count = len(self.face_materials)
        expected_face_count = original_face_count * 4
        new_face_count = len(subdivided_mesh.faces)
        if new_face_count != expected_face_count:
            raise ValueError(f"unexpected number of faces after subdivision: expected {expected_face_count}, got {new_face_count}")
        
        #each face splits into 4 after subdivision, so we need to repeat each material 4 times
        #i checked and this does indeed follow how the faces themselves are reordered after a subdivision, so the materials still apply to the same faces
        new_face_materials = [material for material in self.face_materials for _ in range(4)]
        if not BuildingStructure._check_valid_materials(subdivided_mesh, new_face_materials, self.available_wall_materials, self.available_floor_materials, subdivided_mesh_floor_face_indices):
            raise ValueError("Invalid face materials after subdivision")
        
        self.face_materials = new_face_materials
        self.mesh = subdivided_mesh
        self.subdivision_level += 1
        self.floor_vertex_indices = subdivided_mesh_floor_vertex_indices
        self.floor_face_indices = subdivided_mesh_floor_face_indices

    def mutate(
        self, 
        material_change_probability: float = 0.5, 
        max_material_changes: int = 3,
        scale_change_probability: float = 0.1, 
        scale_factor_range: Tuple[float, float] = (0.8, 1.25),
        max_vertex_movement_attempts: int = 20, 
        max_faces_to_move: int = 3,
        max_vertex_displacement_factor: float = 0.1

    ) -> None:
        #determine the number of faces which should have their material changed, if any
        num_material_changes = 0
        if np.random.rand() < material_change_probability:
            num_material_changes = np.random.randint(1, max_material_changes + 1)
        
        #apply the material changes, revert if resulting amterials are invalid
        if num_material_changes > 0:
            original_face_materials = self.face_materials.copy()
            new_face_materials = self._change_face_material(num_material_changes)

            if BuildingStructure._check_valid_materials(self.mesh, new_face_materials, self.available_wall_materials, self.available_floor_materials, self.floor_face_indices):
                self.face_materials = new_face_materials
            else:
                print("mutation function _change_face_material failed: resulting face materials are invalid")
                self.face_materials = original_face_materials

        #sometimes scale the mesh, revert if resulting mesh is invalid
        if np.random.rand() < scale_change_probability:
            original_mesh = self.mesh.copy()
            scaled_mesh = self._scale_mesh(scale_factor_range)

            if BuildingStructure._check_valid_mesh(scaled_mesh, self.min_dimensions, self.max_dimensions, self.floor_vertex_indices, self.floor_face_indices):
                self.mesh = scaled_mesh
            else:
                #print("mutation function _scale_mesh failed: resulting mesh is invalid")
                self.mesh = original_mesh
            
        #scale the vertex displacement factor with respect to the subdivision level    
        vertex_displacement_factor = max_vertex_displacement_factor / ((self.subdivision_level) + 1)

        faces_to_move = np.random.randint(1, max_faces_to_move + 1)
        for _ in range(faces_to_move):
            #have several attempts at vertex mutation before giving up
            for _ in range(max_vertex_movement_attempts):
                perturbed_mesh = self._move_face_vertices(vertex_displacement_factor)
                if BuildingStructure._check_valid_mesh(perturbed_mesh, self.min_dimensions, self.max_dimensions, self.floor_vertex_indices, self.floor_face_indices):
                    self.mesh = perturbed_mesh
                    break

            #this only triggers if the loop completes without hitting a break
            else:
                print("mutation function _move_face_vertices failed: maximum attempts exceeded")
        
    #randomly changes the material of a number of faces
    def _change_face_material(self, num_material_changes: int) -> list:
        #dont make more changes than there are faces
        num_material_changes = min(np.random.randint(1, num_material_changes + 1), len(self.face_materials))
        new_face_materials = self.face_materials.copy()

        selected_faces = np.random.choice(len(new_face_materials), size=num_material_changes, replace=False)
        for face_index in selected_faces:
            current_material = new_face_materials[face_index]

            #floor and wall faces each have their own seperate pool of materials to choose from
            if face_index in self.floor_face_indices:
                alternative_materials = self.available_floor_materials - {current_material}
                if not alternative_materials:
                    #print(f"no alternative materials to change floor face {face_index} to")
                    continue
            else:
                alternative_materials = self.available_wall_materials - {current_material}
                if not alternative_materials:
                    #print(f"no alternative materials to change wall face {face_index} to")
                    continue
            
            new_material = np.random.choice(list(alternative_materials))
            new_face_materials[face_index] = new_material
        
        return new_face_materials

    #scales the mesh by a random factor in each dimension
    def _scale_mesh(self, scale_factor_range: Tuple[float, float]) -> trimesh.Trimesh:
        new_mesh = self.mesh.copy()

        if scale_factor_range[0] <= 0:
            print("min scale factor must be greater than 0")
            return self.mesh
        
        if scale_factor_range[1] <= scale_factor_range[0]:
            print("max scale factor must be greater than min scale factor")
            return self.mesh
        
        #the scale factor in each dimension is independently sampled
        scale_factors = np.random.uniform(scale_factor_range[0], scale_factor_range[1], size=3)
        scale_matrix = np.eye(4)
        np.fill_diagonal(scale_matrix[:3, :3], scale_factors)
        new_mesh.apply_transform(scale_matrix)
        return new_mesh

    #moves a random selection of vertices on a random face in a random direction
    def _move_face_vertices(self, max_vertex_displacement_factor: float) -> trimesh.Trimesh:
        new_mesh = self.mesh.copy()

        random_face = random.choice(new_mesh.faces) #cant use np.random here as i want to choose a top level element from a 2d array
        vertices_to_move = np.random.choice(random_face, size=np.random.randint(1, 4), replace=False)
        displacement_factor_vector = np.random.uniform(-max_vertex_displacement_factor, max_vertex_displacement_factor, size=3)

        #the maximum movement in each dimension scales with the permitted range of values
        displacement_vector = displacement_factor_vector * (self.max_dimensions - self.min_dimensions)
        for vertex_index in vertices_to_move:
            if vertex_index in self.floor_vertex_indices:
                #vertices that are a part of the floor are limited to lateral movement to keep a flat base
                lateral_displacement_vector = displacement_vector.copy()
                lateral_displacement_vector[2] = 0
                new_mesh.vertices[vertex_index] += lateral_displacement_vector
            else:
                new_mesh.vertices[vertex_index] += displacement_vector

        return new_mesh

    @staticmethod
    def _get_floor_vertex_indices(mesh: trimesh.Trimesh, tolerance: float = 1e-6) -> np.ndarray[int]:
        #this method assumes the mesh is a box, and that the floor is the bottom face of the box
        #therefore it should be called before any vertex pertubations have been done
        vertex_z_coords = mesh.vertices[:, 2]
        floor_z = vertex_z_coords.min()
        floor_vertex_indices = np.where(np.abs(vertex_z_coords - floor_z) < tolerance)[0]
        return floor_vertex_indices
    
    @staticmethod
    def _get_floor_face_indices(mesh: trimesh.Trimesh, floor_vertex_indices: np.ndarray[int]) -> np.ndarray[int]:
        vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
        vertex_mask[floor_vertex_indices] = True
        #mask which identified the faces which contain only floor vertices
        floor_face_mask = np.all(vertex_mask[mesh.faces], axis=1)
        floor_face_indices = np.where(floor_face_mask)[0]
        return floor_face_indices

    @staticmethod
    def _check_valid_mesh(
        mesh: trimesh.Trimesh, 
        min_dimensions: np.ndarray[float], 
        max_dimensions: np.ndarray[float], 
        floor_vertex_indices: np.ndarray[int], 
        floor_face_indices: np.ndarray[int]
    ) -> bool:        
        #check whether the building height falls outside the allowed bounds
        min_building_height = min_dimensions[2]
        max_building_height = max_dimensions[2]
        building_height = mesh.bounding_box.primitive.extents[2]
        if building_height < min_building_height or building_height > max_building_height:
            # print("mesh height is outside the allowed range")
            # print(f"min height: {min_building_height}, max height: {max_building_height}, height: {building_height}")
            return False
        
        #check if the footprint area exceeds the allowed bounds
        min_footprint_area = min_dimensions[0] * min_dimensions[1]
        max_footprint_area = max_dimensions[0] * max_dimensions[1]
        footprint_area = BuildingStructure._get_horizontal_plane_projection_area(mesh)
        if footprint_area < min_footprint_area or footprint_area > max_footprint_area:
            # print(f"mesh footprint area is outside the allowed range")
            # print(f"min footprint area: {min_footprint_area}, max footprint area: {max_footprint_area}, footprint area: {footprint_area}")
            return False

        #check if mesh self intersects
        if BuildingStructure._has_self_intersections(mesh):
            #print("mesh is self intersecting")
            return False
        
        #self intersection check which works for polygons in the 2D floor plane
        if BuildingStructure._has_overlapping_floor_faces(mesh, floor_face_indices):
            #print("mesh has overlapping floor faces")
            return False
        
        #check if any non-floor vertices have z coordinates below the floor
        if BuildingStructure._has_vertices_below_floor(mesh, floor_vertex_indices):
            #print("mesh has non-floor vertices below the floor")
            return False
        
        if BuildingStructure._has_excessively_small_faces(mesh):
            #print("mesh has 1 or more excessively small faces")
            return False
        
        if BuildingStructure._has_proportionally_small_floor_area(mesh, floor_face_indices):
            #print("mesh has a floor with an area that is too small relative to the total mesh area")
            return False

        #check if mesh is fully enclosed
        if not mesh.is_watertight:
            print(f"mesh is not watertight")
            return False
        
        #check if face normals are oriented correctly
        if not mesh.is_volume:
            print("mesh face normals are not oriented correctly")
            return False

        return True
    
    #extra error checking in this funciton as it was causing alot of random crashes
    @staticmethod
    def _get_horizontal_plane_projection_area(mesh: trimesh.Trimesh) -> float:
        polygons = []
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            #project the vertices onto the XY plane
            projected_vertices = vertices[:, :2]
            try:
                polygon = Polygon(projected_vertices)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                    if not polygon.is_valid:
                        print(f"warning: face polygon projected onto the horizontal still invalid after buffer(0)")
                        continue

                polygons.append(polygon)    

            except Exception as e:
                print(f"warning: error creating polygon from face vertices: {e}")
                continue
        
        if not polygons:
            print("warning: no horizontral projected polygons created from mesh faces, returning 0 area")
            return 0.0
        
        try:
            horizontal_plane_polygon = unary_union(polygons)
            horizontal_plane_area = horizontal_plane_polygon.area
        except GEOSException as e:
            #this error caused a crash right before one of my simulations completed, so i catch it explicitly here
            print(f"warning: GEOS TopologyException during unary_union: {e}, returning 0 area")
            return 0.0
        except Exception as e:
            print(f"warning: unexpected error during unary_union: {e}, returning 0 area")
            return 0.0
        
        return horizontal_plane_area

    @staticmethod
    def _has_self_intersections(mesh: trimesh.Trimesh) -> bool:
        #pymeshlab needs to read the mesh from disk
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
            temp_filename = temp_file.name
            mesh.export(temp_filename)

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_filename)
            ms.compute_selection_by_self_intersections_per_face()

            pml_mesh = ms.current_mesh()
            num_selected_faces = pml_mesh.selected_face_number()
            return num_selected_faces > 0
        finally:
            #clean up the temporary file
            os.remove(temp_filename)  

    @staticmethod
    def _has_overlapping_floor_faces(mesh: trimesh.Trimesh, floor_face_indices: np.ndarray[int]) -> bool:
        #convert each floor face into a polygon for use with Shapely
        floor_polygons = []
        for face_index in floor_face_indices:
            face = mesh.faces[face_index]
            #we ignore the z coordinate as Shapely only deals with polygon overlaps in a 2D plane
            #all floor vertices should be at the same z coordinate anyway
            vertex_x_y_coords = mesh.vertices[face, :2].tolist()
            floor_polygon = Polygon(vertex_x_y_coords)
            floor_polygons.append(floor_polygon)
        
        #r-tree allows for fast lookups of polygons with intersecting bounding boxes
        #this reduces the total number of intersection checks needed
        tree = STRtree(floor_polygons)
        for polygon in floor_polygons:
            #iterate through all potentially intersecting polygons, checking for a nonzero intersection area
            intersection_candidate_indices = tree.query(polygon)
            for intersection_candidate_index in intersection_candidate_indices:
                intersection_candidate = floor_polygons[int(intersection_candidate_index)]
                #avoid self comparison
                if intersection_candidate is polygon:
                    continue
                #the first check here is much quicker, but will return True even if the polygons just share a vertex or edge
                #so if the quick check returns True, then we do the more expensive but accurate intersection area check
                if polygon.intersects(intersection_candidate) and polygon.intersection(intersection_candidate).area > 0:
                    return True
        return False
    
    @staticmethod
    def _has_vertices_below_floor(mesh: trimesh.Trimesh, floor_vertex_indices: np.ndarray[int], tolerance: float = 1e-6) -> bool:
        floor_vertex_z_coords = mesh.vertices[floor_vertex_indices, 2]
        floor_z = floor_vertex_z_coords.min()

        floor_vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
        floor_vertex_mask[floor_vertex_indices] = True
        non_floor_vertex_mask = ~floor_vertex_mask
        non_floor_z_coords = mesh.vertices[non_floor_vertex_mask, 2]
        #check if any non-floor vertices are at or below the floor
        return np.any(non_floor_z_coords < floor_z + tolerance)

    @staticmethod
    def _has_excessively_small_faces(mesh: trimesh.Trimesh, relative_min_area_fraction: float = 0.02) -> bool:
        num_faces = len(mesh.faces)
        total_area = mesh.area
        average_face_area = total_area / num_faces

        min_face_area = average_face_area * relative_min_area_fraction
        face_areas = mesh.area_faces
        return np.any(face_areas < min_face_area)
    
    @staticmethod
    def _has_proportionally_small_floor_area(mesh: trimesh.Trimesh, floor_face_indices: np.ndarray[int], min_floor_area_fraction: float = 0.05) -> bool:
        #calculate the area of the floor faces, and total mesh area
        floor_face_areas = mesh.area_faces[floor_face_indices]
        total_floor_area = np.sum(floor_face_areas)
        total_area = mesh.area
        #check if the floor area is less than the minimum required fraction of the total area
        return total_floor_area < (total_area * min_floor_area_fraction)

    @staticmethod
    def _check_valid_materials(
        mesh: trimesh.Trimesh, 
        face_materials: list[Material], 
        available_wall_materials: set[Material], 
        available_floor_materials: set[Material], 
        floor_face_indices: np.ndarray[int]
    ) -> bool:
        #check if all face materials are valid
        for i, material in enumerate(face_materials):
            if i in floor_face_indices:
                if material not in available_floor_materials:
                    print(f"{material.name} material assigned to floor face {i} is not in available_floor_materials")
                    return False     
            else:
                if material not in available_wall_materials:
                    print(f"{material.name} material assigned to wall face {i} is not in available_wall_materials")
                    return False

        #check if there is one material per face
        if len(face_materials) != len(mesh.faces):
            print("number of face materials does not match number of faces")
            return False
        
        return True
