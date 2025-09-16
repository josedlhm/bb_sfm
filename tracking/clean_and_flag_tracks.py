#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import ijson
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict, Any, Optional
from track_utils import load_poses, world_to_camera, voxel_downsample


def _extract_frame_num(d: Dict[str, Any]) -> Optional[int]:
    """Return an integer frame number from a detection dict.
    Tries 'frame' (int/float/str with digits) then common filename keys.
    """
    v = d.get("frame", None)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        digits = "".join(ch for ch in v if ch.isdigit())
        if digits:
            return int(digits)

    for k in ("image", "img", "filename", "file"):
        s = d.get(k)
        if isinstance(s, str):
            digits = "".join(ch for ch in s if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _detect_frame_start(tracks_json: Path) -> int:
    """Stream the JSON once to find the minimum absolute frame number (e.g., 400)."""
    frame_start: Optional[int] = None
    with open(tracks_json, "rb") as f:
        for _, dets in ijson.kvitems(f, "", use_float=True):
            for d in dets:
                fn = _extract_frame_num(d)
                if fn is not None:
                    frame_start = fn if frame_start is None else min(frame_start, fn)
    return 0 if frame_start is None else int(frame_start)


def run(
    tracks_json: Path,
    poses_file:  Path,
    output_json: Path,
    voxel_size:  float = 5.0,
    eps:         float = 20.0,
    min_samples: int   = 10,
    min_cluster_detections: int = 4,
) -> Path:
    """
    Clean and flag 3D tracks via DBSCAN, streaming one track at a time.

    - Streams input with ijson to avoid loading the entire JSON.
    - Applies DBSCAN-based cleaning and cluster-selection logic.
    - Reassigns contiguous IDs.
    - Maps absolute frame numbers (e.g., 400, 401, ...) to pose indices [0..N)
      using the detected frame_start offset.
    """
    poses = load_poses(poses_file)
    frame_start = _detect_frame_start(tracks_json)  # <<< key fix for frame offset

    flagged: List[Tuple[str, int, int]] = []
    new_id = 0

    with open(tracks_json, 'rb') as infile, open(output_json, 'w') as outfile:
        outfile.write('{\n')
        first_out = True

        # iterate over each top-level (track_id, detections)
        for tid, detections in ijson.kvitems(infile, '', use_float=True):
            # 1) collect per-detection 3D points
            all_pts_world: List[np.ndarray] = []
            detection_meta: List[Tuple[Dict[str, Any], np.ndarray]] = []
            for d in detections:
                pts = np.array(d.get("mask_points_3d", []), dtype=float)
                if pts.size > 0:
                    all_pts_world.append(pts)
                    detection_meta.append((d, pts))
                else:
                    print(f"Skipped detection for track {tid}, frame {d.get('frame')}: no mask points")

            if not all_pts_world:
                continue  # nothing to keep

            # 2) small-track shortcut before big alloc
            total_pts = sum(pts.shape[0] for pts in all_pts_world)
            if total_pts < min_samples:
                cleaned = [d for d, _ in detection_meta]
            else:
                # 3) voxel-downsample + DBSCAN on full cloud
                all_stack = np.vstack(all_pts_world)
                ds_pts = voxel_downsample(all_stack, voxel_size=voxel_size)
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(ds_pts)

                # 4) global noise removal
                keep_mask = labels != -1
                ds_pts, labels = ds_pts[keep_mask], labels[keep_mask]
                if ds_pts.size == 0:
                    continue  # drop track

                # 5) flag multi-cluster
                unique_labels = set(labels)
                if len(unique_labels) > 1:
                    flagged.append((tid, len(unique_labels), ds_pts.shape[0]))

                # 6) compute centroids
                centroids = {
                    lbl: ds_pts[labels == lbl].mean(axis=0)
                    for lbl in unique_labels
                }

                # 7) strip stray points per detection
                filtered_meta: List[Tuple[Dict[str, Any], np.ndarray]] = []
                cent_stack = np.stack(list(centroids.values()))
                for d, pts in detection_meta:
                    dists = np.linalg.norm(pts[:, None, :] - cent_stack[None, :, :], axis=2)
                    keep = pts[np.min(dists, axis=1) <= eps]
                    if keep.size:
                        d["mask_points_3d"] = keep.tolist()
                        filtered_meta.append((d, keep))
                    else:
                        print(f"Dropped detection in track {tid}, frame {d.get('frame')}: all points noise")
                if not filtered_meta:
                    continue

                # 8) if only one cluster, keep all
                if len(unique_labels) <= 1:
                    cleaned = [d for d, _ in filtered_meta]
                else:
                    # assign detections to nearest centroid
                    cluster_to_dets: Dict[int, List[Dict[str, Any]]] = {}
                    for d, pts in filtered_meta:
                        ctr = pts.mean(axis=0)
                        best_lbl = min(
                            centroids.keys(),
                            key=lambda l: np.linalg.norm(ctr - centroids[l])
                        )
                        cluster_to_dets.setdefault(best_lbl, []).append(d)

                    # 9) select best cluster by (a) count threshold, then
                    #    (b) lowest mean camera-space Z when pose available;
                    #    fallback to max count if no valid pose index.
                    eligible = {lbl: dets for lbl, dets in cluster_to_dets.items()
                                if len(dets) >= min_cluster_detections}
                    if not eligible:
                        continue

                    # Prefer clusters with valid pose indices for their first frame
                    best_lbl = None
                    best_z = float('inf')
                    best_count = -1
                    in_range_found = False

                    for lbl, dets in eligible.items():
                        frame0 = _extract_frame_num(dets[0])
                        if frame0 is None:
                            continue
                        idx = frame0 - frame_start
                        if 0 <= idx < len(poses):
                            in_range_found = True
                            pts3d = np.vstack([d.get("centroid_3d", [0, 0, 0]) for d in dets])
                            cam = world_to_camera(pts3d, poses[idx])
                            zmean = cam[:, 2].mean()
                            if (zmean < best_z) or (zmean == best_z and len(dets) > best_count):
                                best_z = zmean
                                best_lbl = lbl
                                best_count = len(dets)

                    # If none had a valid pose index, fall back to max count
                    if not in_range_found:
                        best_lbl = max(eligible.items(), key=lambda kv: len(kv[1]))[0]

                    if best_lbl is None:
                        continue
                    cleaned = eligible[best_lbl]

            # write out if non-empty
            if cleaned:
                for d in cleaned:
                    d["det"] = new_id
                if not first_out:
                    outfile.write(',\n')
                outfile.write(f'"{new_id}": ')
                json.dump(cleaned, outfile, indent=2)
                first_out = False
                new_id += 1

        outfile.write('\n}\n')

    # summary
    print(f"\nSaved cleaned tracks to {output_json}")
    print(f"Final track count: {new_id}")
    print(f"Found {len(flagged)} tracks with multiple clusters:")
    for tid, n_clust, pts in flagged:
        print(f" • Track {tid}: {n_clust} clusters, {pts} downsampled points")

    return output_json
