from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any
from track_utils import load_poses, rotation_matrix_to_angle_axis

def is_valid_centroid(c: Any, length: int = 3) -> bool:
    """
    Return True if c is a list/tuple/ndarray of exactly `length`
    numeric (int/float) elements.
    """
    if not isinstance(c, (list, tuple, np.ndarray)):
        return False
    if len(c) != length:
        return False
    return all(isinstance(x, (int, float, np.floating, np.integer)) for x in c)


def run(
    tracks_json: Path,
    poses_txt:    Path,
    out_dir:      Path     = Path("output/ceres_input"),
) -> Path:
    """
    Prepare Ceres solver input files (cameras.txt, points.txt, observations.txt)
    from cleaned tracks and camera poses, skipping any detection
    where `centroid_3d` is missing or malformed.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    cameras_txt = out_dir / "cameras.txt"
    points_txt = out_dir / "points.txt"
    observations_txt = out_dir / "observations.txt"

    # load tracks
    with open(tracks_json, 'r') as f:
        tracks: Dict[str, List[Dict[str, Any]]] = json.load(f)

    # load extrinsics (camera-to-world)
    poses_c2w = load_poses(poses_txt)
    # convert to world-to-camera
    poses_w2c = [np.linalg.inv(T) for T in poses_c2w]

    bundle = []  # point observations
    for track_id, dets in tracks.items():
        centroids: List[List[float]] = []
        observations: List[Dict[str, Any]] = []

        for det in dets:
            cent = det.get("centroid_3d")
            if not is_valid_centroid(cent):
                # skip missing or malformed centroid
                continue

            # collect valid centroid
            centroids.append([float(x) for x in cent])

            # compute UV from bbox center
            x1, y1, x2, y2 = det["bbox"]
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            observations.append({
                "frame": det["frame"],
                "uv": [u, v]
            })

        # skip tracks with no valid centroids / observations
        if not centroids or not observations:
            continue

        # compute mean 3D point
        mean_xyz = np.mean(np.array(centroids, dtype=float), axis=0).tolist()
        bundle.append({
            "point_id": int(track_id),
            "xyz": mean_xyz,
            "observations": observations
        })

    # write cameras.txt
    with open(cameras_txt, 'w') as f_cam:
        for T in poses_w2c:
            R = T[:3, :3]
            t = T[:3, 3]
            aa = rotation_matrix_to_angle_axis(R)
            vec = list(aa) + list(t)
            f_cam.write(" ".join(map(str, vec)) + "\n")

    # write points.txt
    with open(points_txt, 'w') as f_pts:
        for item in bundle:
            f_pts.write(" ".join(map(str, item["xyz"])) + "\n")

    # write observations.txt
    with open(observations_txt, 'w') as f_obs:
        for pid, item in enumerate(bundle):
            for obs in item["observations"]:
                frame = obs["frame"]
                u, v = obs["uv"]
                f_obs.write(f"{frame} {pid} {u} {v}\n")

    print(f"✅ prepare_ceres_input wrote files to {out_dir}")
    return out_dir