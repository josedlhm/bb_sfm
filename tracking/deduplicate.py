#!/usr/bin/env python3
"""
deduplicate.py

Pipeline:
1) Summarize each track (geometry + class stats)
2) Deduplicate by geometry/time only (no class checks)
3) Restream detections with merged IDs and a voted `track_class`
4) Write a minimal summary JSON containing ONLY:
   - total_merged_groups
   - tracks_per_class          (TRACK COUNTS per voted class, numeric 1..K + "None")
   - tracks_per_class_named    (same counts keyed by class name order)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import defaultdict

import ijson
import numpy as np
import yaml
from tqdm import tqdm

from track_utils import (
    convert_ceres_poses,
    load_poses,
    mask_iou,
    project_cloud_to_mask,
    world_to_camera,
)

# --------- class order (1-based IDs) ----------
CLASS_NAMES = ['cuajo', 'flor', 'maduro', 'pinton', 'verde']
CLASS_ID_TO_NAME = {i + 1: name for i, name in enumerate(CLASS_NAMES)}  # 1→'cuajo', …, 5→'verde'

# --------- intrinsics + image size ----------
CONFIG = yaml.safe_load(open("config.yaml", "r"))
FX, FY = CONFIG["intrinsics"]["fx"], CONFIG["intrinsics"]["fy"]
CX, CY = CONFIG["intrinsics"]["cx"], CONFIG["intrinsics"]["cy"]
IMG_W, IMG_H = CONFIG["image"]["width"], CONFIG["image"]["height"]


def project_point(point_w: np.ndarray, T_wc: np.ndarray) -> Tuple[float, float, float]:
    x_cam, y_cam, z_cam = world_to_camera(point_w[None, :], T_wc)[0]
    u = FX * (x_cam / z_cam) + CX
    v = FY * (y_cam / z_cam) + CY
    return u, v, z_cam


def _vote_class(class_weights: Dict[int, float], class_counts: Dict[int, int]) -> Optional[int]:
    """
    Score-weighted majority -> counts -> smallest class_id.
    Returns None if no class info is present.
    """
    if not class_weights:
        return None
    max_w = max(class_weights.values())
    cands = [cid for cid, w in class_weights.items() if w == max_w]
    if len(cands) == 1:
        return cands[0]
    max_c = max(class_counts.get(c, 0) for c in cands)
    cands = [c for c in cands if class_counts.get(c, 0) == max_c]
    return min(cands) if cands else None


def summarize_tracks(tracks_json: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    with open(tracks_json, "rb") as f:
        for tid_str, dets in ijson.kvitems(f, "", use_float=True):
            tid = int(tid_str)

            cents = [np.array(d["centroid_3d"], float) for d in dets if "centroid_3d" in d]
            N = len(cents)
            centroid_w = np.mean(cents, axis=0) if N > 0 else None

            frames = sorted({d["frame"] for d in dets})
            frames_set: Set[int] = set(frames)

            mask0 = np.zeros((IMG_H, IMG_W), dtype=bool)
            if frames:
                first_frame = frames[0]
                for d in dets:
                    if d.get("frame") == first_frame and "mask_coords" in d:
                        for x, y in d["mask_coords"]:
                            if 0 <= x < IMG_W and 0 <= y < IMG_H:
                                mask0[y, x] = True

            mask_pts: List[List[float]] = []
            for d in dets:
                pts = d.get("mask_points_3d")
                if pts:
                    mask_pts.extend(pts)

            class_weights: Dict[int, float] = {}
            class_counts: Dict[int, int] = {}
            for d in dets:
                cid = d.get("class_id", None)
                if cid is None:
                    continue
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                score = d.get("score", None)
                w = float(score) if (score is not None and np.isfinite(score)) else 1.0
                class_weights[cid_int] = class_weights.get(cid_int, 0.0) + w
                class_counts[cid_int] = class_counts.get(cid_int, 0) + 1

            summaries.append({
                "id": tid,
                "centroid": centroid_w,
                "N": N,
                "frames": frames_set,
                "mask0": mask0,
                "mask_pts": mask_pts,
                "class_weights": class_weights,
                "class_counts": class_counts,
            })
    return summaries


def dedupe_summaries(
    summaries: List[Dict[str, Any]],
    Ts: List[np.ndarray],
    centroid_distance: float,
    iou_min: float,
    frame0_offset: int = 0,
) -> Dict[int, int]:
    used: Set[int] = set()
    merged_map: Dict[int, int] = {}
    merged_id = 0

    for i, si in enumerate(tqdm(summaries, desc="Deduplicating")):
        ti = si["id"]
        if ti in used:
            continue

        merged_map[ti] = merged_id

        for sj in summaries[i+1:]:
            tj = sj["id"]
            if tj in used:
                continue

            ci, cj = si["centroid"], sj["centroid"]
            if ci is None or cj is None:
                continue
            if np.linalg.norm(ci - cj) > centroid_distance:
                continue

            if si["frames"] & sj["frames"]:
                continue

            if not si["frames"]:
                continue
            f0 = min(si["frames"])
            idx = f0 - frame0_offset
            if idx < 0 or idx >= len(Ts):
                continue
            T_wc = Ts[idx]

            u_i, v_i, _ = project_point(ci, T_wc)
            u_j, v_j, _ = project_point(cj, T_wc)
            if not (0 <= u_i < IMG_W and 0 <= v_i < IMG_H and
                    0 <= u_j < IMG_W and 0 <= v_j < IMG_H):
                continue

            pts_w_j = np.array(sj["mask_pts"], float)
            if pts_w_j.size == 0:
                continue
            pts_cam_j = world_to_camera(pts_w_j, T_wc)
            mask_j = project_cloud_to_mask(pts_cam_j, IMG_H, IMG_W).astype(bool)
            if mask_iou(si["mask0"], mask_j) < iou_min:
                continue

            merged_map[tj] = merged_id
            used.add(tj)

            si["frames"].update(sj["frames"])
            Ni, Nj = si["N"], sj["N"]
            si["centroid"] = (ci * Ni + cj * Nj) / (Ni + Nj)
            si["N"] = Ni + Nj

        merged_id += 1

    return merged_map


def compute_group_votes(
    summaries: List[Dict[str, Any]],
    merged_map: Dict[int, int],
) -> Dict[int, Optional[int]]:
    group_weights: Dict[int, Dict[int, float]] = {}
    group_counts: Dict[int, Dict[int, int]] = {}

    for s in summaries:
        orig_id = s["id"]
        if orig_id not in merged_map:
            continue
        gid = merged_map[orig_id]
        gw = group_weights.setdefault(gid, {})
        gc = group_counts.setdefault(gid, {})
        for cid, w in s.get("class_weights", {}).items():
            gw[cid] = gw.get(cid, 0.0) + float(w)
        for cid, c in s.get("class_counts", {}).items():
            gc[cid] = gc.get(cid, 0) + int(c)

    group_vote: Dict[int, Optional[int]] = {}
    for gid in set(merged_map.values()):
        cw = group_weights.get(gid, {})
        cc = group_counts.get(gid, {})
        group_vote[gid] = _vote_class(cw, cc)
    return group_vote


def restream_tracks(
    tracks_json: Path,
    merged_map: Dict[int, int],
    group_vote: Dict[int, Optional[int]],
    output_json: Path,
) -> Tuple[Path, int]:
    groups_written: Set[int] = set()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(tracks_json, "rb") as infile, open(output_json, "w") as outfile:
        outfile.write("{\n")
        first = True
        for tid_str, dets in ijson.kvitems(infile, "", use_float=True):
            orig_id = int(tid_str)
            if orig_id not in merged_map:
                continue
            mid = merged_map[orig_id]
            groups_written.add(mid)
            voted = group_vote.get(mid, None)

            for d in dets:
                d["det"] = mid
                d["track_class"] = voted

            if not first:
                outfile.write(",\n")
            outfile.write(f'"{mid}": ')
            json.dump(dets, outfile, indent=2)
            first = False
        outfile.write("\n}\n")

    return output_json, len(groups_written)


def run(
    tracks_json:            Path,
    ceres_poses:            Path,
    converted_ceres_poses:  Path,
    output_json:            Path,
    centroid_distance:      float,
    iou_min:                float,
) -> Path:
    # 1) camera→world poses
    convert_ceres_poses(ceres_poses, converted_ceres_poses)
    Ts = load_poses(converted_ceres_poses)

    # 2) summarize (geometry + class stats)
    summaries = summarize_tracks(tracks_json)

    # 3) dedupe (geometry-only)
    frame0_offset = min(min(s["frames"]) for s in summaries if s["frames"])
    merged_map = dedupe_summaries(summaries, Ts, centroid_distance, iou_min, frame0_offset)

    # 4) vote per merged group (after merges fixed)
    group_vote = compute_group_votes(summaries, merged_map)

    # 5) restream merged tracks & stamp votes
    result_path, total_merged_groups = restream_tracks(
        tracks_json, merged_map, group_vote, output_json
    )

    # 6) count TRACKS per class from votes (NOT detections)
    # numeric-keyed (1..K) + "None"
    track_counts_per_class: Dict[str, int] = defaultdict(int)
    named_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    none_count = 0

    for gid, cls in group_vote.items():
        if cls is None:
            none_count += 1
            track_counts_per_class["None"] += 1
            continue
        key = str(cls)  # preserve 1-based numeric label as string
        track_counts_per_class[key] += 1
        # also into name-keyed, only if within expected range
        name = CLASS_ID_TO_NAME.get(int(cls))
        if name is not None:
            named_counts[name] += 1
        else:
            # unseen/overflow class id: still counted numerically; ignore in named block
            pass

    # ensure numeric keys appear in order 1..K if missing
    for cid in range(1, len(CLASS_NAMES) + 1):
        track_counts_per_class.setdefault(str(cid), 0)
    if none_count and "None" not in track_counts_per_class:
        track_counts_per_class["None"] = none_count

    # 7) minimal summary (two requested fields + name-keyed convenience block)
    summary = {
        "total_merged_groups": total_merged_groups,
        "tracks_per_class": {**{str(i): track_counts_per_class[str(i)] for i in range(1, len(CLASS_NAMES)+1)},
                             **({"None": track_counts_per_class["None"]} if "None" in track_counts_per_class else {})},
        "tracks_per_class_named": {name: named_counts[name] for name in CLASS_NAMES}
                                   | ({"None": none_count} if none_count else {}),
    }
    summary_path = output_json.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Merged {len(summaries)} → {total_merged_groups} tracks")
    print(f"✅ deduplicate wrote {result_path}")
    print(f"🧾 summary wrote {summary_path}")
    return result_path
