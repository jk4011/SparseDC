import torch
import numpy as np
from sklearn.cluster import DBSCAN
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt

import PIL
from scipy.spatial import QhullError
from torchvision import transforms
from src.utils.mask_utils import (
    get_hull_values,
    get_circle_value,
)


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    if prediction.ndim == 2:
        prediction = prediction[None, :]
    if target.ndim == 2:
        target = target[None, :]
    if mask.ndim == 2:
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
clusters = None


def get_depth_sam_hull(rgb, depth_pred, depth_gt_sparse, scale_margin=[0.8, 1.25], shift_margin=[0.7, 1.4], err_margin=0.1, lambda_param=30):
    device = depth_gt_sparse.device
    
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)
    if rgb.ndim == 4:
        rgb = rgb.squeeze(0)
    
    if isinstance(rgb, torch.Tensor):
        rgb = transforms.ToPILImage()(rgb.cpu())
    if isinstance(rgb, PIL.Image.Image):
        rgb = np.array(rgb)
        
    global sam, clusters
    if sam is None:
        sam_checkpoint = "/data2/wlsgur4011/SparseDC/pretrain/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        model.to(device=device)
        sam = SamAutomaticMaskGenerator(model)
    clusters = sam.generate(rgb)
    global_scale, global_shift = compute_scale_and_shift(depth_pred, depth_gt_sparse, mask=(depth_gt_sparse != 0))

    depth_sum = torch.zeros_like(depth_pred)
    hull_sum = torch.zeros_like(depth_pred)
    hull_max = torch.zeros_like(depth_pred)

    for i, cluster in enumerate(clusters):
        cluster = torch.tensor(cluster['segmentation'], device=device)
        cluster_depth = cluster * depth_gt_sparse

        mask = cluster_depth != 0

        err = torch.ones(100)

        while True:
            if mask.sum() <= 3:
                break

            scale, shift = compute_scale_and_shift(depth_pred, depth_gt_sparse, mask)

            # TODO: reject
            err = ((scale * depth_pred + shift) - depth_gt_sparse).abs()
            err[~mask] = 0

            if (err < err_margin).all():
                break
            else:
                # Replace the maximum value with 0
                mask[err == err.max()] = 0

        # n sample must be larger than 3
        if mask.sum() <= 3:
            # TODO: use shifted depth
            continue

        if not (scale_margin[0] < abs(scale / global_scale) < scale_margin[1] or
                shift_margin[0] < abs(shift / global_shift) < shift_margin[1]):
            continue

        depth_sparse_local = (depth_gt_sparse * mask).cpu().numpy().astype(bool)
        try:
            hull_values = get_hull_values(depth_sparse_local, lambda_param=lambda_param).to(depth_pred.device)
        except QhullError:
            # pixels are collinear
            continue

        # TODO: use hull_values linear interpolation
        depth_scaled = scale * depth_pred + shift
        hull_values = cluster * hull_values
        depth_sum += hull_values * depth_scaled
        hull_sum += hull_values
        hull_max = torch.max(hull_max, hull_values)

        # hull_mask = (hull_values != 0).to(depth_pred.device)
        # cluster = cluster * hull_mask
        # depth_sum[cluster] += scale * depth_pred[cluster] + shift
        # depth_n += cluster

    depth_hull = torch.zeros_like(depth_pred)
    sam_mask = hull_max > 0
    depth_hull[sam_mask] = depth_sum[sam_mask] / hull_sum[sam_mask]
    
    depth_scaled = global_scale * depth_pred + global_shift
        
    new_depth = hull_max * depth_hull + (1 - hull_max) * depth_scaled

    # sam_mask = depth_n != 0
    # new_depth[sam_mask] = depth_sum[sam_mask] / depth_n[sam_mask]
    # new_depth[~sam_mask] = (global_scale * depth_pred + global_shift)[~sam_mask]

    return new_depth, hull_max


def get_depth_sam_shifted(rgb, depth_pred, depth_gt_sparse):
    device = depth_gt_sparse.device
    
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.squeeze(0)
    if depth_gt_sparse.ndim == 3:
        depth_gt_sparse = depth_gt_sparse.squeeze(0)
    if rgb.ndim == 4:
        rgb = rgb.squeeze(0)
    
    if isinstance(rgb, torch.Tensor):
        rgb = transforms.ToPILImage()(rgb.cpu())
    if isinstance(rgb, PIL.Image.Image):
        rgb = np.array(rgb)

    global sam, clusters
    if sam is None:
        sam_checkpoint = "/data2/wlsgur4011/SparseDC/pretrain/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        model.to(device=device)
        sam = SamAutomaticMaskGenerator(model)
    clusters = sam.generate(rgb)
    global_scale, global_shift = compute_scale_and_shift(depth_pred, depth_gt_sparse, mask=(depth_gt_sparse != 0))
    depth_affine = global_scale * depth_pred + global_shift
    
    depth_shifted = depth_affine.clone()

    for i, cluster in enumerate(clusters):
        cluster = torch.tensor(cluster['segmentation'], device=device)
        cluster_depth = cluster * depth_gt_sparse

        mask = cluster_depth != 0
        points_pred = depth_affine[mask]
        points_gt = cluster_depth[mask]
        
        local_shift = (points_gt - points_pred).mean()
        
        depth_shifted[cluster] = depth_affine[cluster] + local_shift
    
    return depth_shifted


#TODO: get_depth_sam_CIRCLE


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


def depth_map_to_twilight(depth_map, norm=True):
    if norm:
        # Normalize the depth map for the color map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) / 2 + 0.5
    else:
        depth_map = depth_map / 2 + 0.5

    # Apply the twilight color map
    twilight_color_map = plt.get_cmap('twilight')
    depth_map_colored = twilight_color_map(depth_map.squeeze())  # Remove channel dim if it exists

    # Convert to tensor and keep only RGB channels (discard alpha)
    depth_map_colored = torch.tensor(depth_map_colored[..., :3], dtype=torch.float32).permute(2, 0, 1)

    return depth_map_colored


def combine_depth_results(image, depth_gt, depth_gt_sparse, depth_pred):
    depth_gt = depth_gt.squeeze().cpu()
    depth_gt_sparse = depth_gt_sparse.squeeze().cpu()
    depth_pred = depth_pred.squeeze().cpu()
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu()

    # 1. depth_gt and depth_pred

    depth_combined = torch.cat([depth_gt, depth_pred], dim=1)
    depth_combined = depth_map_to_twilight(depth_combined)

    # 2. depth_gt_sparse
    depth_mask = torch.cat([depth_gt_sparse != 0, depth_gt_sparse != 0], dim=1)
    depth_combined[:, depth_mask] = 1

    # 3. depth_diff
    depth_diff = (depth_gt - depth_pred)
    depth_diff_rgb = depth_map_to_twilight(depth_diff, norm=False)
    depth_combined = torch.cat([depth_combined, depth_diff_rgb], dim=2)

    # 4. image
    image = torch.from_numpy(np.array(image)).float().squeeze()
    if image.shape[0] != 3:
        image = image.permute(2, 0, 1)
    if image.max() > 1:
        image /= 255

    depth_combined = torch.cat([image, depth_combined], dim=2)

    return depth_combined
