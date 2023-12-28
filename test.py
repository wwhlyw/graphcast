from scipy.spatial.transform import Rotation
import numpy as np
import torch
import math
import matplotlib.pyplot as plt 
from scipy.spatial import transform



lat = np.load('/home/wwh/graphcast/location/lats.npy')[253:693,970:1378]
lon = np.load('/home/wwh/graphcast/location/lons.npy')[253:693,970:1378]
def grid_lat_lon_to_coordinates(grid_latitude: np.ndarray, grid_longitude: np.ndarray):
    phi_grid, theta_grid = np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    return np.stack([
        np.cos(phi_grid) * np.sin(theta_grid),
        np.sin(phi_grid) * np.sin(theta_grid),
        np.cos(theta_grid)], axis=-1)
grid_latitude =np.array([lat[0,0], lat[0, -1], lat[-1, 0], lat[-1, -1]])
grid_longitude = np.array([lon[0, 0], lon[0, -1], lon[-1, 0], lon[-1, -1]])
print(grid_latitude)
print(grid_longitude)
print(grid_lat_lon_to_coordinates(grid_latitude, grid_longitude))
pos = grid_lat_lon_to_coordinates()
vertices = []
phi = (1 + np.sqrt(5)) / 2
for c1 in [1., -1.]:
    for c2 in [phi, -phi]:
        vertices.append((c1, c2, 0.))
        vertices.append((0., c1, c2))
        vertices.append((c2, 0., c1))
diss = []
for v in vertices:
    dis = np.sum(np.power(pos - v, 2))
    diss.append(dis)

pos1 = [0, -1, phi]
pos1 = pos1 / np.linalg.norm(pos1)

def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    phi = np.arctan2(y, x)
    with np.errstate(invalid='ignore'):
        theta = np.arccos(z)
    return phi, theta

def lat_lon_to_spherical(lat, lon):
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90 - lat)
    return phi, theta 

# pos1 是最近点的位置， centers是要找的目标点的位置

centers_phi, centers_theta = lat_lon_to_spherical(35, -87)
receiver_phi, receiver_theta = cartesian_to_spherical(pos1[0], pos1[1], pos1[2])
print(centers_phi, centers_theta)
print(receiver_phi, receiver_theta)

rotate_phi = centers_phi - receiver_phi # +
rotate_theta = centers_theta - receiver_theta # +

rotate = transform.Rotation.from_euler(
        "zx", [-rotate_phi, -rotate_theta]).as_matrix().T

print(pos)
new = np.dot(rotate, pos1)
print(new)

distances = []
for v in vertices:
    distances.append(np.sum(np.power(pos1 - v, 2)))

pos11 = [0, 1, phi]
pos12 = [phi, 0, 1]
pos13 = [1, -phi, 0]
pos14 = [-phi, 0, 1]
pos15 = [-1, -phi, 0]

pos1 = [0, -1, phi]
vertices = [pos1, pos11, pos12, pos13, pos14, pos15]
vertices = np.array(vertices, dtype=np.float32)
vertices /= np.linalg.norm([1, phi])
faces = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 5),
    (0, 4, 1),
    (0, 5, 4),
]
vertices = np.dot(rotate, vertices.T).T

print(vertices)
# print(np.sum(np.power(vertices[2] - vertices[3], 2)))
# [1, 2, 3, 5, 9]
# 1-2 1-4  2-3 3-5 4-5

