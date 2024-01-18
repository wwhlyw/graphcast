from . import icosahedral_mesh
from .utils import *
import numpy as np
import scipy
import trimesh
# import icosahedral_mesh
# from utils import * 


def grid2mesh_edges_indices(
        *,
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        mesh: icosahedral_mesh.TriangularMesh,
        radius: float
    ) -> tuple[np.ndarray, np.ndarray]:

    grid_positions = grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])

    mesh_positions = mesh.vertices
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    query_indices = kd_tree.query_ball_point(x = grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        if len(mesh_neighbors) == 0:
            print(grid_index)
            raise "some grids don't link with mesh"
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)
    
    return grid_edge_indices, mesh_edge_indices


def mesh2grid_edge_indices(
        *,
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        mesh: icosahedral_mesh.TriangularMesh
    ) -> tuple[np.ndarray, np.ndarray]:
    
    grid_positions = grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])

    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions
    )
    
    mesh_edge_indices = mesh.faces[query_face_indices]
    
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices


def mesh2mesh_edge_indices(
    faces,
):
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])  
            
    return senders, receivers



# lat = np.load('../location/lats.npy')[253:693,970:1378]
# lon = np.load('../location/lons.npy')[253:693,970:1378]
# init_mesh = icosahedral_mesh.get_pentagon(5.5)
# mesh = icosahedral_mesh.merge_meshes(icosahedral_mesh.meshes_list(7, init_mesh))
# print(len(mesh.vertices))
# # mesh = icosahedral_mesh.merge_meshes(icosahedral_mesh.meshes_list(6))
# grid_edge_indices, mesh_indices = grid2mesh_edges_indices(grid_latitude=lat, grid_longitude=lon, mesh=mesh, radius=0.002)
# print(grid_edge_indices.shape)
# print(len(set(mesh_indices)))
# import torch
# input = torch.arange(mesh_indices.shape[0]).unsqueeze(0).unsqueeze(-1)
# from torch_scatter import scatter
# output = scatter(input, torch.from_numpy(mesh_indices), dim=1)
# print(output)
def g2m_or_m2g_edges_indices_2d(mesh, radius, type):
    senders = []
    receivers = []
    
    grid_y = np.arange(408).repeat(440) / 440.
    grid_x = np.tile(np.arange(440) / 440., 408)
    grid = np.concatenate([grid_x, grid_y]).reshape([2, -1]).transpose()
    
    kd_tree = scipy.spatial.cKDTree(mesh.vertices[:, :2])

    query_indices = kd_tree.query_ball_point(x=grid, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        if len(mesh_neighbors) == 0:
            raise "some grids don't link with mesh"
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)
 
    return grid_edge_indices, mesh_edge_indices

def mesh2grid_edge_indices_2d(
        mesh: icosahedral_mesh.TriangularMesh
    ) -> tuple[np.ndarray, np.ndarray]:
    
    grid_y = np.arange(408).repeat(440) / 440.
    grid_x = np.tile(np.arange(440) / 440., 408)
    grid_z = np.zeros(440 * 408, )
    grid = np.concatenate([grid_x, grid_y, grid_z]).reshape([3, -1]).transpose()

    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid
    )
    
    mesh_edge_indices = mesh.faces[query_face_indices]
    
    grid_indices = np.arange(grid.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices

def mesh2mesh_edges_indices_2d(faces):
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    arrs = []
    for arr in zip(senders, receivers):
        arr = list(arr)
        if arr[0] > arr[1]:
            arr[0], arr[1] = arr[1], arr[0]
        arrs.append(arr) 

    unique = []
    for arr in arrs:
        if arr not in unique:
            unique.append(arr)
    
    senders = []
    receivers = []

    for arr in unique:
        senders.append(arr[0])
        receivers.append(arr[1])
        senders.append(arr[1])
        receivers.append(arr[0])
    return np.array(senders), np.array(receivers)


# mesh = icosahedral_mesh.get_quadrangle()
# meshes = icosahedral_mesh.meshes_list(7, mesh)
# meshes = icosahedral_mesh.merge_meshes(meshes)
# faces = meshes.faces
# print(len(meshes.vertices))
# senders, receivers = mesh2grid_edge_indices_2d(meshes)
# print(senders[:20])
# print(receivers[:20])
