import numpy as np

def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    phi = np.arctan2(y, x)
    with np.errstate(invalid='ignore'):
        theta = np.arccos(z)
    return phi, theta

def spherical_to_cartesian(phi: np.ndarray, theta: np.ndarray):
    return (np.cos(phi) * np.sin(theta), 
            np.sin(phi) * np.sin(theta),
            np.cos(theta))

def spherical_to_lat_lon(phi: np.ndarray, theta: np.ndarray):
    lon = np.mod(np.rad2deg(phi), 360)
    lat = 90 - np.rad2deg(theta)
    return lat, lon

def lat_lon_to_spherical(lat, lon):
    phi = np.deg2rad(lon)
    theta = np.deg2rad(90 - lat)
    return phi, theta 

def grid_lat_lon_to_coordinates(grid_latitude: np.ndarray, grid_longitude: np.ndarray):
    phi_grid, theta_grid = np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    return np.stack([
        np.cos(phi_grid) * np.sin(theta_grid),
        np.sin(phi_grid) * np.sin(theta_grid),
        np.cos(theta_grid)], axis=-1)