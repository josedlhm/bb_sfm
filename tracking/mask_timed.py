from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.optimize import linear_sum_assignment
import time  # ⏱ added
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

from track_utils import (
    load_poses, invert_T,
    filter_highest_density_cluster,
    reject_outliers_by_mode_z, bbox_to_mask,  # kept for compatibility
    compute_iou,                               # kept for compatibility
    transform_cam_to_world, project_cloud_to_mask
)

# ---------- XY coefficient cache (exact math, fewer ops) ----------
_XCOEF = None
_YCOEF = None
_XY_SHAPE = None
_XY_PARAMS = None  # (fx, fy, cx, cy)

def _ensure_xy(H: int, W: int, fx: float, fy: float, cx: float, cy: float):
    global _XCOEF, _YCOEF, _XY_SHAPE, _XY_PARAMS
    if _XY_SHAPE == (H, W) and _XY_PARAMS == (fx, fy, cx, cy) and _XCOEF is not None:
        return
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    Xs, Ys = np.meshgrid(xs, ys)
    _XCOEF = (Xs - cx) / np.float32(fx)
    _YCOEF = (Ys - cy) / np.float32(fy)
    _XY_SHAPE = (H, W)
    _XY_PARAMS = (fx, fy, cx, cy)

@dataclass
class Detection:
    frame: int
    det_id: int
    bbox: List[int]
    mask_coords: List[List[int]]      = field(default_factory=list)
    mask_points_3d: List[List[float]] = field(default_factory=list)
    centroid_3d: Optional[List[float]] = None
    # --- per-detection classification fields ---
    class_id: Optional[int] = None
    score: Optional[float] = None


def backproject_mask_dense(mask: np.ndarray,
                           depth: np.ndarray,
                           fx: float, fy: float,
                           cx: float, cy: float) -> np.ndarray:
    # (kept for compatibility; not used by _get_cloud anymore)
    m = mask > 0
    ys, xs = np.nonzero(m)
    zs = depth[ys, xs].astype(np.float32)
    valid = (zs > 0) & np.isfinite(zs)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    return np.stack([X, Y, Z], axis=1)


def _get_cloud(frame_idx: int,
               bbox: List[int],
               mask_coords: List[List[int]],
               depth_dir: Path,
               outlier_delta: float,
               fx: float, fy: float,
               cx: float, cy: float) -> np.ndarray:
    """
    Accuracy-preserving, faster version:
    - Loads depth once (mmap).
    - Computes on bbox crop only (exact same points as full-image path).
    """
    x1, y1, x2, y2 = map(int, bbox)
    mask_local = np.array(mask_coords, dtype=np.uint8)

    depth_path = depth_dir / f"depth_{frame_idx:06d}.npy"
    depth = np.load(str(depth_path), mmap_mode='r')  # exact values, faster IO
    H, W = depth.shape

    # Validate mask shape vs bbox (same behavior as original)
    mask_h, mask_w = mask_local.shape
    if mask_h != (y2 - y1 + 1) or mask_w != (x2 - x1 + 1):
        raise ValueError(f"Mask shape {mask_local.shape} does not match bbox dimensions {(y2-y1+1, x2-x1+1)}")

    # Crop-first backprojection (exact)
    m = mask_local > 0
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    depth_crop = depth[y1:y2+1, x1:x2+1]
    zs = depth_crop[ys, xs].astype(np.float32)
    valid = (zs > 0) & np.isfinite(zs)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs_g = xs[valid].astype(np.int64) + x1  # global coords
    ys_g = ys[valid].astype(np.int64) + y1
    z    = zs[valid]

    # Precompute per-pixel coefficients once per resolution/intrinsics
    _ensure_xy(H, W, fx, fy, cx, cy)
    X = _XCOEF[ys_g, xs_g] * z
    Y = _YCOEF[ys_g, xs_g] * z
    Z = z
    dense = np.stack([X, Y, Z], axis=1)

    # Cluster and reject outliers (unchanged)
    cluster = filter_highest_density_cluster(dense)
    filtered = reject_outliers_by_mode_z(cluster, delta=outlier_delta)
    return filtered


# ---------- Exact IoU via integral images (no approximation) ----------
def _integral_image_bool(mask_bool: np.ndarray) -> np.ndarray:
    """Integral image with 1-pixel padding: S[y+1,x+1] = sum(mask_bool[0:y+1, 0:x+1])."""
    H, W = mask_bool.shape
    S = np.zeros((H + 1, W + 1), dtype=np.int32)
    S[1:, 1:] = mask_bool.astype(np.uint8, copy=False).cumsum(axis=0).cumsum(axis=1)
    return S

def _compute_row_costs_exact(m0: np.ndarray,
                             x1: np.ndarray, y1: np.ndarray,
                             x2: np.ndarray, y2: np.ndarray,
                             area1: np.ndarray,
                             H: int, W: int) -> np.ndarray:
    """
    Exact 1 - IoU between mask m0 (boolean) and each bbox (given by x1,y1,x2,y2,area1).
    Vectorized intersection from integral image.
    """
    m0b = m0.astype(bool, copy=False)
    S = _integral_image_bool(m0b)
    ones0 = int(S[-1, -1])  # total 1s in m0

    n1 = x1.shape[0]
    costs = np.empty(n1, dtype=np.float32)
    if n1 == 0:
        return costs

    valid = (x2 >= x1) & (y2 >= y1)
    inter = np.zeros(n1, dtype=np.int32)
    if np.any(valid):
        xv1 = x1[valid]; yv1 = y1[valid]
        xv2 = x2[valid]; yv2 = y2[valid]
        # Integral image indexing (+1 offset)
        inter_v = (S[yv2 + 1, xv2 + 1] - S[yv1, xv2 + 1] -
                   S[yv2 + 1, xv1] + S[yv1, xv1]).astype(np.int32)
        inter[valid] = inter_v

    union = ones0 + area1 - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return 1.0 - iou

def _build_cost_matrix(masks0: List[np.ndarray],
                       bbs1: List[List[float]],
                       H: int,
                       W: int,
                       iou_thresh: float) -> np.ndarray:
    """
    Exact-cost version (no approximations):
    - Computes 1 - IoU(m0, bbox_mask(bb)) using integral images (vectorized per row).
    - Preserves dummy columns for 'unmatched' like your original.
    """
    n0 = len(masks0)
    n1 = len(bbs1)
    if n0 == 0 and n1 == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n0 == 0:
        return np.zeros((0, n1), dtype=np.float32)

    # Precompute bbox ints and areas once per frame
    if n1 > 0:
        B = np.asarray(bbs1, dtype=np.float32)
        x1 = np.clip(B[:, 0].astype(np.int32), 0, W - 1)
        y1 = np.clip(B[:, 1].astype(np.int32), 0, H - 1)
        x2 = np.clip(B[:, 2].astype(np.int32), 0, W - 1)
        y2 = np.clip(B[:, 3].astype(np.int32), 0, H - 1)
        area1 = ((x2 - x1 + 1) * (y2 - y1 + 1)).astype(np.int32)

        # Row-wise parallel, order preserved
        workers = max(2, min(8, cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows_list = list(ex.map(lambda i: _compute_row_costs_exact(
                                        masks0[i], x1, y1, x2, y2, area1, H, W),
                                    range(n0)))
        C = np.vstack(rows_list).astype(np.float32, copy=False)
    else:
        C = np.zeros((n0, 0), dtype=np.float32)

    # Pad with dummy columns (unmatched option), identical semantics
    pad = np.full((n0, n0), 1.0 - iou_thresh, dtype=np.float32)
    return np.hstack([C, pad])


def run(
    detections_json: Path,
    depth_dir:        Path,
    poses_file:       Path,
    output_json:      Path,
    iou_thresh:       float,
    outlier_delta:    float,
    min_track_len:    int,
    min_consecutive:  int,
    fx:               float,
    fy:               float,
    cx:               float,
    cy:               float,
    max_cam_distance: float = 1150.0,  # Z-only distance gate in camera space (units = depth). ≤0 disables.
) -> Path:
    import track_utils
    for name, val in (("FX", fx), ("FY", fy), ("CX", cx), ("CY", cy)):
        setattr(track_utils, name, val)



    dets = json.load(open(detections_json))

    poses = load_poses(poses_file)
  

    # iterate over actual detected frame indices (e.g., 400 → 401 → 405 …)
    frames = sorted(int(k[4:10]) for k in dets.keys() if k.startswith("img_") and k.endswith(".png"))

    print(frames)
    
    N = len(frames)


    tracks = {}
    next_id = 0
    prev_map = {}

    loop_start_all = time.perf_counter()  # ⏱ total over all frames

    for idx in range(N - 1):
        frame = frames[idx]
        next_frame = frames[idx + 1]
        print(f"\n▶️ Processing frame {idx+1}/{N}  (t={frame:06d} → t'={next_frame:06d})")

        iter_start = time.perf_counter()  # ⏱ per-iteration

        key0 = f"img_{frame:06d}.png"
        key1 = f"img_{next_frame:06d}.png"

        objs0, objs1 = dets[key0], dets[key1]

        # Build clouds in parallel (order preserved by map)
        t0 = time.perf_counter()
        def _gc(args):
            return _get_cloud(*args)
        args0 = [(frame, o["bbox"], o["mask"], depth_dir, outlier_delta, fx, fy, cx, cy) for o in objs0]
        args1 = [(next_frame, o["bbox"], o["mask"], depth_dir, outlier_delta, fx, fy, cx, cy) for o in objs1]
        workers = max(2, min(8, cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            clouds0 = list(ex.map(_gc, args0))
            clouds1 = list(ex.map(_gc, args1))
        print(f" ⏱ clouds built: {time.perf_counter() - t0:.3f}s")

        # ---- Z-only distance filter in camera space (using median Z) ----
        if max_cam_distance and max_cam_distance > 0:
            def _apply_z_filter_inplace(clouds: List[np.ndarray], max_dist: float):
                for k, c in enumerate(clouds):
                    if c.size:
                        z_med = float(np.median(c[:, 2]))
                        if abs(z_med) > max_dist:
                            clouds[k] = np.empty((0, 3), dtype=np.float32)
            _apply_z_filter_inplace(clouds0, max_cam_distance)
            _apply_z_filter_inplace(clouds1, max_cam_distance)
            culled0 = sum(1 for c in clouds0 if c.size == 0)
            culled1 = sum(1 for c in clouds1 if c.size == 0)
            print(f" ⏱ Z-filter (>{max_cam_distance:.1f}): culled {culled0} in t, {culled1} in t'")

        # Relative pose (very fast)
        t_rel = time.perf_counter()
        Trel = invert_T(poses[idx + 1]) @ poses[idx]
        Rrel, trel = Trel[:3, :3], Trel[:3, 3]
        P0 = poses[idx]
        P1 = poses[idx + 1]
        print(f" ⏱ relative pose: {time.perf_counter() - t_rel:.6f}s")

        # Project clouds from t into t' masks
        t1 = time.perf_counter()
        H1, W1 = np.load(str(depth_dir / f"depth_{next_frame:06d}.npy"), mmap_mode='r').shape
        masks0 = [project_cloud_to_mask((Rrel @ c.T).T + trel, H1, W1) for c in clouds0]
        print(f" ⏱ projection to masks: {time.perf_counter() - t1:.3f}s")

        # Exact cost matrix + Hungarian
        t2 = time.perf_counter()
        Cfull = _build_cost_matrix(masks0, [o["bbox"] for o in objs1], H1, W1, iou_thresh)
        rows, cols = linear_sum_assignment(Cfull) if Cfull.size else ([], [])
        print(f" ⏱ cost matrix + assignment: {time.perf_counter() - t2:.3f}s")

        # ---------- Fused & parallel precompute of next-frame attrs ----------
        t_prep = time.perf_counter()

        R1 = P1[:3, :3]
        t_vec = P1[:3, 3]

        def _prep_one(arg):
            j, c1 = arg
            if c1.size:
                t_w0 = time.perf_counter()
                # exact world transform (same as transform_cam_to_world(c1, P1))
                pts = (c1 @ R1.T) + t_vec
                cent = pts.mean(0).tolist()
                t_w = time.perf_counter() - t_w0

                t_m0 = time.perf_counter()
                m = project_cloud_to_mask(c1, H1, W1)
                t_m = time.perf_counter() - t_m0

                t_c0 = time.perf_counter()
                ys, xs = np.where(m)
                coords = np.column_stack((xs, ys)).tolist()  # same order as before
                t_c = time.perf_counter() - t_c0
            else:
                pts = np.empty((0, 3))
                cent = None
                m = np.zeros((H1, W1), dtype=bool)
                coords = []
                t_w = t_m = t_c = 0.0
            # micro-opt: ensure contiguous float32 before tolist() later
            pts = np.ascontiguousarray(pts, dtype=np.float32)
            return j, pts, cent, m, coords, t_w, t_m, t_c

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_prep_one, enumerate(clouds1)))

        pts1_world    = [None] * len(clouds1)
        cent1_all     = [None] * len(clouds1)
        m1_cache      = [None] * len(clouds1)
        coords1_cache = [None] * len(clouds1)

        tw = tm = tc = 0.0
        for j, pts, cent, m, coords, t_w, t_m, t_c in results:
            pts1_world[j]    = pts
            cent1_all[j]     = cent
            m1_cache[j]      = m
            coords1_cache[j] = coords
            tw += t_w; tm += t_m; tc += t_c

        print(f" ⏱ precompute next-frame attrs: {time.perf_counter() - t_prep:.3f}s "
              f"(world={tw:.3f}s, mask={tm:.3f}s, coords={tc:.3f}s)")

        # ---------- Lazy caches for i-side (only when new track starts) ----------
        pts0_world_cache = [None] * len(clouds0)
        cent0_cache      = [None] * len(clouds0)
        m0_cache         = [None] * len(clouds0)
        coords0_cache    = [None] * len(clouds0)

        def ensure_i_cached(i_idx: int):
            if pts0_world_cache[i_idx] is not None:
                return
            p0 = transform_cam_to_world(clouds0[i_idx], P0) if clouds0[i_idx].size else np.empty((0, 3))
            # micro-opt: contiguous float32 before tolist()
            p0 = np.ascontiguousarray(p0, dtype=np.float32)
            pts0_world_cache[i_idx] = p0
            cent0_cache[i_idx] = p0.mean(0).tolist() if p0.size else None
            m0 = project_cloud_to_mask(clouds0[i_idx], H1, W1)
            m0_cache[i_idx] = m0
            ys, xs = np.where(m0)
            coords0_cache[i_idx] = np.column_stack((xs, ys)).tolist()  # same order as before

        # Track updates (logic unchanged; now uses caches)
        t3 = time.perf_counter()
        new_map, matched = {}, set()
        for i, j in zip(rows, cols):
            if j < len(objs1) and Cfull[i, j] <= (1.0 - iou_thresh):
                tid = prev_map.get(i, next_id)
                if i not in prev_map:
                    tracks[tid] = []
                    next_id += 1
                    ensure_i_cached(i)
                    tracks[tid].append(Detection(
                        frame, i, objs0[i]["bbox"],
                        coords0_cache[i],
                        pts0_world_cache[i].tolist() if pts0_world_cache[i].size else [],
                        cent0_cache[i],
                        class_id=objs0[i].get("class_id", None),
                        score=objs0[i].get("score", None),
                    ))
                tracks[tid].append(Detection(
                    frame + 1, j, objs1[j]["bbox"],
                    coords1_cache[j],
                    pts1_world[j].tolist() if pts1_world[j].size else [],
                    cent1_all[j],
                    class_id=objs1[j].get("class_id", None),
                    score=objs1[j].get("score", None),
                ))
                new_map[j] = tid
                matched.add(j)

        for j in set(range(len(objs1))) - matched:
            tid = next_id
            next_id += 1
            tracks[tid] = [Detection(
                frame + 1, j, objs1[j]["bbox"],
                coords1_cache[j],
                pts1_world[j].tolist() if pts1_world[j].size else [],
                cent1_all[j],
                class_id=objs1[j].get("class_id", None),
                score=objs1[j].get("score", None),
            )]
            new_map[j] = tid

        prev_map = new_map
        print(f" ⏱ track updates: {time.perf_counter() - t3:.3f}s")

        print(f" ⏱ total this frame: {time.perf_counter() - iter_start:.3f}s")

    # Post-filter + JSON write timing
    t_post = time.perf_counter()

    final = {}
    new_id = 0
    for tid, dets_list in tracks.items():
        frames_tmp = [d.frame for d in dets_list]
        if len(frames_tmp) < min_track_len:
            continue
        if max(np.diff(sorted(frames_tmp))) > (len(frames_tmp) - min_consecutive + 1):
            continue
        for d in dets_list:
            d.det_id = new_id
        final[new_id] = [{
            "frame": d.frame,
            "det":   d.det_id,
            "bbox":  d.bbox,
            "mask_coords": d.mask_coords,
            "mask_points_3d": d.mask_points_3d,
            "centroid_3d":   d.centroid_3d,
            "class_id":      d.class_id,   # per-detection class
            "score":         d.score,      # per-detection score
        } for d in dets_list]
        new_id += 1

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n⏱ post-filter + write: {time.perf_counter() - t_post:.3f}s")
    print(f"⏱ total runtime: {time.perf_counter() - loop_start_all:.3f}s")
    print(f"Total final tracks saved: {len(final)}")

    return output_json
