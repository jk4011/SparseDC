
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay


def point_line_distance_vectorized(points, line_start, line_end):
    # This function is intended to handle multiple points at once for a single line segment
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    points_vec = points - line_start
    points_proj_length = np.dot(points_vec, line_unitvec)
    points_proj_length_clipped = np.clip(points_proj_length, 0, line_len)
    nearest_points = line_start + np.outer(points_proj_length_clipped, line_unitvec)
    distances = np.linalg.norm(points - nearest_points, axis=1)
    return distances


def get_hull_values(sparse_map, lambda_param=30):
    # Get the coordinates of the non-zero points in the sparse map
    
    h, w = sparse_map.shape
    y, x = np.where(sparse_map)
    points = np.vstack([x, y]).T
    
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    hull = ConvexHull(points)
    delaunay = Delaunay(points[hull.vertices])
    is_inside = delaunay.find_simplex(grid_points) >= 0
    values = np.zeros(grid_points.shape[0])

    # For each edge in the hull, compute the distance from all grid points to the edge
    for edge in hull.simplices:
        line_start = points[edge[0]]
        line_end = points[edge[1]]
        distances = point_line_distance_vectorized(grid_points, line_start, line_end)
        if 'min_distances' not in locals():
            min_distances = distances
        else:
            min_distances = np.minimum(min_distances, distances)

    # Compute values for each point outside the hull
    values[is_inside] = 1
    values[~is_inside] = np.maximum(1 - min_distances[~is_inside] / lambda_param, 0)
    mask = torch.tensor(values).reshape(sparse_map.shape)

    return mask


def get_min_distances(A, B):
    # Calculate the squared differences in each dimension
    diff_squared = (A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2
    # Sum the squared differences across the dimensions to get squared distances
    distances_squared = np.sum(diff_squared, axis=2)
    # Take the square root to get Euclidean distances
    distances = np.sqrt(distances_squared)
    # Find the minimum distance for each point in A
    min_distances = np.min(distances, axis=1)
    return min_distances


def get_circle_value(sparse_map, lambda_param=30, constant=True):

    h, w = sparse_map.shape
    y, x = np.where(sparse_map)
    # TODO: check y, x order
    points = np.vstack([y, x]).T
    
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    values = np.zeros(grid_points.shape[0])

    min_distances = get_min_distances(grid_points, points)
    if constant:
        values[(1 - min_distances / lambda_param) > 0] = True
    else:
        values = np.maximum(1 - min_distances / lambda_param, 0)
    return values
