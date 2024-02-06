import torch
import numpy as np
from sklearn.cluster import DBSCAN


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim == 2:
        target = target.unsqueeze(0)
    if prediction.ndim == 2:
        mask = mask.unsqueeze(0)
    
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def cluster_depth_map(depth_map, eps=1.5, min_samples=5):
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)
    
    if depth_map.ndim != 2:
        raise ValueError('Depth map must be 2D')
    
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    
    
    # Get the height and width of the depth map
    H, W = depth_map.shape

    # Create an array of [x, y] coordinates for each pixel
    y, x = np.mgrid[0:H, 0:W]

    # Stack the coordinates with the depth map to get a [x, y, depth] feature for each pixel
    features = np.stack((x.ravel(), y.ravel(), depth_map.ravel() * 100), axis=1)

    # Apply DBSCAN clustering
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    clustering = DBSCAN(eps=eps, min_samples=min_samples, p=1).fit(features)

    # The labels_ attribute will have the cluster labels for each point
    labels = clustering.labels_

    # Reshape the labels back to the original depth map shape
    labels_map = labels.reshape(H, W)
    
    labels_map = torch.tensor(labels_map)

    return labels_map
