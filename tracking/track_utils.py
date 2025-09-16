#!/usr/bin/env python3
"""
track_utils.py
Reusable helper functions for the 3D fruit-tracking pipeline.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as SciRot
import cv2
from pathlib import Path
FX = 1272.44
FY = 1272.67
CX = 920.062
CY = 618.949


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Uniformly downsample 3D points by voxel grid average.
    """
    if points.size == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int32)
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    down = np.zeros((len(unique), 3), dtype=np.float32)
    for i in range(len(unique)):
        down[i] = points[inverse == i].mean(axis=0)
    return down

def world_to_camera(points_w: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """
    Transform world points into camera frame.
    """
    T_cw = np.linalg.inv(T_wc)
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    return (R_cw @ points_w.T).T + t_cw

def rotation_matrix_to_angle_axis(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to an angle-axis (rotvec) vector.
    """
    return SciRot.from_matrix(R).as_rotvec()



def load_poses(path):
    """
    Read a text file of 4×4 camera-to-world poses (row-major, angle-axis->R + t)
    and return a list of 4×4 numpy arrays.
    """
    Ts = []
    with open(path, 'r') as f:
        for line in f:
            v = list(map(float, line.split()))
            R = np.array([v[0:3], v[4:7], v[8:11]], dtype=np.float64)
            t = np.array([v[3], v[7], v[11]], dtype=np.float64)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            Ts.append(T)
    return Ts


def invert_T(T):
    """
    Invert a 4×4 rigid-body transform.
    """
    R, t = T[:3, :3], T[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    Tinv = np.eye(4)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3] = tinv
    return Tinv


def transform_cam_to_world(points_cam, T_wc):
    """
    Transform an (N×3) array of points from camera frame to world frame.

    Args:
        points_cam: numpy array of shape (N,3)
        T_wc: 4×4 numpy array (camera-to-world)

    Returns:
        numpy array of shape (N,3)
    """
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
    return (R_wc @ points_cam.T).T + t_wc


def bbox_to_mask(bbox, H, W):
    """
    Rasterize an axis-aligned bbox into a binary mask.

    Args:
        bbox: [x1,y1,x2,y2] (ints or floats)
        H, W: mask height and width

    Returns:
        (H,W) uint8 mask with 1 inside the bbox, else 0
    """
    m = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1c, x2c = max(0, min(int(x1), W-1)), max(0, min(int(x2), W-1))
    y1c, y2c = max(0, min(int(y1), H-1)), max(0, min(int(y2), H-1))
    if x2c >= x1c and y2c >= y1c:
        m[y1c:y2c+1, x1c:x2c+1] = 1
    return m


def compute_iou(a, b):
    """
    Compute the Intersection-over-Union of two binary masks.
    """
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / union if union > 0 else 0.0


def max_consecutive(frames):
    """
    Given a list of integer frame indices, return the length of the longest run
    of consecutive numbers.
    """
    if not frames:
        return 0
    frames = sorted(frames)
    longest = curr = 1
    for prev, nxt in zip(frames, frames[1:]):
        if nxt == prev + 1:
            curr += 1
            longest = max(longest, curr)
        else:
            curr = 1
    return longest


def backproject_bbox_dense(bbox, depth):
    H, W = depth.shape
    x1, y1, x2, y2 = map(int, bbox)
    x1c, x2c = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
    y1c, y2c = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((0, 3))
    xs = np.arange(x1c, x2c + 1)
    ys = np.arange(y1c, y2c + 1)
    gx, gy = np.meshgrid(xs, ys)
    zs = depth[gy, gx].astype(np.float32)
    mask = (zs > 0) & np.isfinite(zs)
    xg, yg, zg = gx[mask], gy[mask], zs[mask]
    X = (xg - CX) * zg / FX
    Y = (yg - CY) * zg / FY
    return np.stack([X, Y, zg], axis=1)

def filter_highest_density_cluster(points, eps=2, min_samples=10):
    if len(points) < min_samples:
        return points
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    if np.max(labels) < 0:
        return points
    best_label = max(set(labels) - {-1}, key=lambda lbl: np.sum(labels == lbl))
    return points[labels == best_label]

def reject_outliers_by_mode_z(points, delta=10):
    """
    Keep only points within ±delta mm of the most common Z value.
    Assumes Z is the 3rd column (depth in mm).
    """
    if len(points) == 0:
        return points

    z_vals = points[:, 2]
    z_vals_rounded = np.round(z_vals / 5) * 5  # bin to nearest 5mm
    mode_z = float(np.bincount(z_vals_rounded.astype(int)).argmax())

    keep_mask = (z_vals >= mode_z - delta) & (z_vals <= mode_z + delta)
    return points[keep_mask]

def project_cloud_to_mask(cloud, H, W):
    if cloud.size == 0:
        return np.zeros((H,W), dtype=np.uint8)

    X, Y, Z = cloud[:,0], cloud[:,1], cloud[:,2]
    valid = (Z > 0) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = (FX * X / Z + CX).round().astype(int)
    v = (FY * Y / Z + CY).round().astype(int)

    inside = (u>=0)&(u<W)&(v>=0)&(v<H)
    m = np.zeros((H,W), dtype=np.uint8)
    m[v[inside], u[inside]] = 1
    return m

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / union if union > 0 else 0.0

def draw_sphere_mask(u: float, v: float, z: float,
                     radius_mm: float,
                     H: int, W: int,
                     fx: float, fy: float,
                     cx: float, cy: float) -> np.ndarray:
    """
    Draw a filled circle of physical radius `radius_mm` at image
    position (u,v) given depth z, returns a bool mask of size H×W.
    """
    from cv2 import circle
    mask = np.zeros((H, W), np.uint8)
    r_px = int(round(fx * radius_mm / z))
    circle(mask, (int(round(u)), int(round(v))), r_px, 255, -1)
    return mask > 0

def convert_ceres_poses(
    refined_cameras_path: Path,
    output_poses_path: Path
) -> int:
    """
    Read a Ceres `refined_cameras.txt` (angle-axis rx ry rz, tx ty tz per line)
    and write out a camera-to-world 3×4 row-major file at `output_poses_path`.
    Returns the number of poses written.

    Each input line has 6 floats:
      [rx, ry, rz, tx, ty, tz]
    where (rx,ry,rz) is the angle-axis rotation of world→camera,
    and (tx,ty,tz) is the translation in the camera frame.

    The output row is:
      [R_cw | t_cw] = [R_wcᵀ | –R_wcᵀ·tvec]
    laid out as
      r11 r12 r13  tx   r21 r22 r23  ty   r31 r32 r33  tz
    """
    rows = []
    with open(refined_cameras_path, 'r') as f:
        for ln, line in enumerate(f, 1):
            vals = list(map(float, line.split()))
            if len(vals) != 6:
                raise ValueError(f"Line {ln}: expected 6 values, got {len(vals)}")

            # split into angle-axis and translation
            rvec = np.array(vals[:3], dtype=float)
            tvec = np.array(vals[3:], dtype=float).reshape(3, 1)

            # world→camera rotation matrix
            R_wc, _ = cv2.Rodrigues(rvec)
            # invert to get camera→world
            R_cw = R_wc.T
            t_cw = -R_cw @ tvec

            # flatten into one row
            row = np.hstack([
                R_cw[0], [t_cw[0,0]],
                R_cw[1], [t_cw[1,0]],
                R_cw[2], [t_cw[2,0]],
            ])
            rows.append(row)

    # write out as 12 numbers per line
    output_poses_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_poses_path, 'w') as f:
        for r in rows:
            f.write(" ".join(f"{x:.6f}" for x in r) + "\n")

    return len(rows)
