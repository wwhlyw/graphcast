import numpy as np
from scipy.spatial import transform


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

def get_bipartite_graph_spatial_features(
    *,
    senders_lat: np.ndarray,
    senders_lon: np.ndarray,
    senders: np.ndarray,
    receivers_lat: np.ndarray,
    receivers_lon: np.ndarray,
    receivers: np.ndarray,    
):
    senders_phi, senders_theta = lat_lon_to_spherical(
        senders_lat, senders_lon
    )
    receivers_phi, receivers_theta = lat_lon_to_spherical(
        receivers_lat, receivers_lon
    )

    senders_features = []
    receivers_features = []

    # add cos of node latitude
    senders_features.append(np.cos(senders_theta))
    receivers_features.append(np.cos(receivers_theta))

    # add cos and sin of node longitude
    senders_features.append(np.cos(senders_phi))
    senders_features.append(np.sin(senders_phi))
    receivers_features.append(np.cos(receivers_phi))
    receivers_features.append(np.sin(receivers_phi))

    senders_features = np.stack(senders_features, axis=-1)
    receivers_features = np.stack(receivers_features, axis=-1)

    # add relative positions
    # compute some edge features
    edge_features = []
    relative_position = get_bipartite_relative_position_in_receiver_local_coordinates(
        senders_phi=senders_phi,
        senders_theta=senders_theta,
        receivers_phi=receivers_phi,
        receivers_theta=receivers_theta,
        senders=senders,
        receivers=receivers,
    )

    relative_edge_distances = np.linalg.norm(relative_position, axis=-1, keepdims=True)

    edge_normlization_factor = relative_edge_distances.max()
    edge_features.append(relative_edge_distances / edge_normlization_factor)
    edge_features.append(relative_position / edge_normlization_factor)
    edge_features = np.concatenate(edge_features, axis=-1)

    return senders_features, receivers_features, edge_features

def get_bipartite_relative_position_in_receiver_local_coordinates(
    senders_phi: np.ndarray,
    senders_theta: np.ndarray,
    senders: np.ndarray,
    receivers_phi: np.ndarray,
    receivers_theta: np.ndarray,
    receivers: np.ndarray,
):
    senders_pos = np.stack(spherical_to_cartesian(senders_phi, senders_theta), axis=-1)
    receivers_pos = np.stack(spherical_to_cartesian(receivers_phi, receivers_theta), axis=-1)

    # get rotation matrices for the local space for every receiver node
    receiver_rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receivers_phi,
        reference_theta=receivers_theta,
    )
    edge_rotation_matrices = receiver_rotation_matrices[receivers]

    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, receivers_pos[receivers]
    )
    sender_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, senders_pos[senders]
    )
  
    return sender_pos_in_rotated_space - receiver_pos_in_rotated_space

def get_rotation_matrices_to_local_coordinates(reference_phi, reference_theta):
    azimuthal_rotation = -reference_phi
    polar_rotation = -reference_theta + np.pi / 2

    return transform.Rotation.from_euler(
        "zy", np.stack([azimuthal_rotation, polar_rotation], axis=1)
    ).as_matrix()

def rotate_with_matrices(
    rotation_matrices, positions
):
    return np.einsum('bji,bi->bj', rotation_matrices, positions)


def get_homogeneous_graph_spatial_features(
    *,
    node_lat,
    node_lon,
    senders,
    receivers,
):
    node_phi, node_theta = lat_lon_to_spherical(node_lat, node_lon)

    edge_features = []
    relative_position = get_homogeneous_relative_position_in_receiver_local_coordinates(
        node_phi=node_phi,
        node_theta=node_theta,
        senders=senders,
        receivers=receivers,
    )
    relative_edge_distances = np.linalg.norm(
        relative_position, axis=-1, keepdims=True
    )

    max_edge_distance = relative_edge_distances.max()
    edge_features.append(relative_edge_distances / max_edge_distance)
    edge_features.append(relative_position / max_edge_distance)

    edge_features = np.concatenate(edge_features, axis=-1)
    return edge_features


def get_homogeneous_relative_position_in_receiver_local_coordinates(
    node_phi, 
    node_theta, 
    senders,
    receivers,
):
    node_pos = np.stack(spherical_to_cartesian(node_phi, node_theta), axis=-1)
    rotation_matrix = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi,
        reference_theta=node_theta,
    )
    edge_rotation_matrices = rotation_matrix[receivers]
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[receivers]
    )
    sender_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[senders]
    )

    return sender_pos_in_rotated_space - receiver_pos_in_rotated_space