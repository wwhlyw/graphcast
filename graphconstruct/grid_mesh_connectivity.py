from . import icosahedral_mesh
import numpy as np
import scipy
import trimesh


def grid_lat_lon_to_coordinates(grid_latitude: np.ndarray, grid_longitude: np.ndarray):
    phi_grid, theta_grid = np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    return np.stack([
        np.cos(phi_grid) * np.sin(theta_grid),
        np.sin(phi_grid) * np.sin(theta_grid),
        np.cos(theta_grid)], axis=-1)


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
    g2m_dst_idx=None
):
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])  
            
    return senders, receivers


# lat = np.load('../location/lats.npy')[253:693,970:1378]
# lon = np.load('../location/lons.npy')[253:693,970:1378]
# mesh = icosahedral_mesh.merge_meshes(icosahedral_mesh.meshes_list(6))
# grid_edge_indices, mesh_indices = grid2mesh_edges_indices(grid_latitude=lat, grid_longitude=lon, mesh=mesh, radius=0.03)
# print(grid_edge_indices.shape)
# print(mesh_indices.shape)
# import torch
# input = torch.arange(mesh_indices.shape[0]).unsqueeze(0).unsqueeze(-1)
# from torch_scatter import scatter
# output = scatter(input, torch.from_numpy(mesh_indices), dim=1)
# print(output)


