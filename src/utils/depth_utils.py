import torch
import numpy as np
from sklearn.cluster import DBSCAN
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    if prediction.ndim == 2:
        prediction = prediction[None, :]
    if prediction.ndim == 2:
        target = target[None, :]
    if prediction.ndim == 2:
        mask = mask[None, :]

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


sam = None


def get_depth_sam(rgb, depth_pred, depth_gt_sparse):
    global sam
    if sam is None:
        sam_checkpoint = "/data2/wlsgur4011/SparseDC/pretrain/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        model.to(device="cuda")
        sam = SamAutomaticMaskGenerator(model)
    clusters = sam.generate(rgb)

    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)

    new_depth = torch.zeros_like(depth_pred)

    for cluster in clusters:
        cluster = torch.tensor(cluster['segmentation']).cuda()
        cluster_depth = cluster * depth_gt_sparse
        if cluster_depth.sum() < 2:
            continue

        mask = cluster_depth != 0
        scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
        new_depth[cluster] = scale * depth_pred[cluster] + shift

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth = scale * depth_pred + shift

    mask = new_depth == 0
    new_depth[mask] = scaled_depth[mask]

    dbscan_mask = ~mask

    return new_depth, dbscan_mask


def get_diff_depth(depth_pred, depth_gt_sparse, depth_gt):
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)
    if depth_gt.ndim == 3:
        depth_gt = depth_gt.squeeze(0)

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth = scale * depth_pred + shift

    diff_depth = scaled_depth - depth_gt

    return diff_depth


def get_depth_affine(depth_pred, depth_gt_sparse):
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth = scale * depth_pred + shift

    return scaled_depth


def get_scaled_depth_sparse(depth_pred, depth_gt_sparse):
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth_sparse = (depth_gt_sparse - shift) / scale
    scaled_depth_sparse[~mask] = 0

    return scaled_depth_sparse

def get_depth_dbscan(depth_pred, depth_gt_sparse):
    # TODO: 위 함수랑 변수명 통일
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)

    cluster_map = cluster_depth_map(depth_pred, eps=1.4, min_samples=4).cuda()

    new_depth = torch.zeros_like(depth_pred)

    for i in range(cluster_map.max() + 1):
        cluster = (cluster_map == i)
        cluster_depth = cluster * depth_gt_sparse
        if cluster_depth.sum() < 2:
            continue

        mask = cluster_depth != 0
        scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
        new_depth[cluster] = scale * depth_pred[cluster] + shift

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth = scale * depth_pred + shift

    mask = new_depth == 0
    new_depth[mask] = scaled_depth[mask]

    dbscan_mask = ~mask

    return new_depth, dbscan_mask


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
