import numpy as np
import itertools
from scipy.spatial import transform
from typing import NamedTuple
from .utils import *


class TriangularMesh(NamedTuple):
    # vertices: [num_vertices, 3_positions]
    # faces: [num_faces, 3_indices]
    vertices: np.ndarray
    faces: np.ndarray

def merge_meshes(mesh_list):
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(vertices=mesh_list[-1].vertices,
                    faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0))


def meshes_list(splits, current_mesh):
    # current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = split_triangle_face(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes

def get_pentagon(scale):
    lat = np.load('/home/wwh/graphcast/location/lats.npy')[253:693,970:1378]
    lon = np.load('/home/wwh/graphcast/location/lons.npy')[253:693,970:1378]
    lat_mean = np.mean(lat)
    lon_mean = np.mean(lon)

    phi = (1 + np.sqrt(5)) / 2

    vertices = np.array([[0, -1, phi],
                         [0,  1, phi],
                         [phi, 0,  1],
                         [1, -phi, 0],
                         [-phi, 0, 1],
                         [-1, -phi, 0]])
    vertices /= np.linalg.norm([1, phi])

    faces = [
        (0, 2, 1),
        (1, 4, 0),
        (3, 2, 0),
        (0, 3, 5),
        (0, 4, 5),
    ]

    grid_center_phi, grid_center_theta = lat_lon_to_spherical(lat_mean, lon_mean)
    mesh_center_phi, mesh_center_theta = cartesian_to_spherical(vertices[0, 0], vertices[0, 1], vertices[0, 2])
    rotate_phi, rotate_theta = mesh_center_phi - grid_center_phi, mesh_center_theta-grid_center_theta
    rotate_matrix =  transform.Rotation.from_euler(
        "zx", [rotate_phi, rotate_theta]).as_matrix().T
    vertices = np.dot(rotate_matrix, vertices.T).T

    for i, vertice in enumerate(vertices):
        vertices[i] = (vertice - vertices[0]) / scale + vertices[0]
        vertices[i] /= np.linalg.norm(vertice)

    return TriangularMesh(vertices=vertices.astype(np.float32), faces=np.array(faces, dtype=np.int32))


def get_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = []
    for c1 in [1., -1.]:
        for c2 in [phi, -phi]:
            vertices.append((c1, c2, 0.))
            vertices.append((0., c1, c2))
            vertices.append((c2, 0., c1))
    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1, phi])
    
    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]
    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform.Rotation.from_euler(seq='y', angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

    return TriangularMesh(vertices=vertices.astype(np.float32), faces=np.array(faces, dtype=np.int32))


def split_triangle_face(triangular_mesh):
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)
    
    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
    
        new_faces.extend([[ind1, ind12, ind31],
                          [ind12, ind2, ind23],
                          [ind31, ind23, ind3],
                          [ind12, ind23, ind31],
                          ])
    return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                          faces=np.array(new_faces, dtype=np.int32))


class _ChildVerticesBuilder:
    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))
    
    def _create_child_vertex(self, parent_vertex_indices):
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(0)
        child_vertex_position /= np.linalg.norm(child_vertex_position)

        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)
    
    def get_new_child_vertex_index(self, parent_vertex_indices):
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]
    
    def get_all_vertices(self):
        return np.array(self._all_vertices_list)

# init_mesh = get_pentagon(8)
# mesh = merge_meshes(meshes_list(1, init_mesh))
# print(mesh.vertices)
# print(mesh.faces)
