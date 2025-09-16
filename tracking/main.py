# main.py
import yaml
import subprocess
import time
from pathlib import Path

from mask_timed           import run as build_tracks
from clean_and_flag_tracks import run as clean_tracks
from prepare_ceres_input  import run as prepare_ceres
from deduplicate          import run as deduplicate
from deltas               import run as apply_deltas
from measure              import run as measure

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config= yaml.safe_load(f)

paths = config["paths"]

# map to local vars for clarity
DEPTH_FOLDER       = Path(paths["depth_folder"])
DETECTIONS_JSON    = Path(paths["detections_json"])
POSES_FILE         = Path(paths["poses_file"])
IMAGES_DIR         = paths["images_dir"]

CERES_SOLVER_BIN   = Path(paths["ceres_solver_bin"])
REFINED_POSES_FILE = Path(paths["refined_poses_file"])

OUTPUT_DIR         = Path(paths["output_dir"])
TRACKS_JSON        = OUTPUT_DIR / paths["tracks_json"]
CLEANED_JSON       = OUTPUT_DIR / paths["cleaned_tracks_json"]
CERES_DIR          = OUTPUT_DIR / paths["ceres_dir"]
POST_DELTA_JSON    = OUTPUT_DIR / paths["post_delta_json"]
MERGED_JSON        = OUTPUT_DIR / paths["merged_tracks_json"]
# ─────────────────────────────────────────────────────────────────────────

def pipeline() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    total_start = time.perf_counter()

    # 1) build tracks
    t0 = time.perf_counter()
    build_tracks(
        detections_json=DETECTIONS_JSON,
        depth_dir=DEPTH_FOLDER,
        poses_file=POSES_FILE,
        output_json=TRACKS_JSON,
        iou_thresh=config["thresholds"]["iou_thresh"],
        outlier_delta=config["thresholds"]["outlier_delta"],
        min_track_len=config["thresholds"]["min_track_len"],
        min_consecutive=config["thresholds"]["min_consecutive"],
        **config["intrinsics"]
    )
    print(f"Step 1 completed in {time.perf_counter() - t0:.2f}s")

    # 2) clean tracks
    t0 = time.perf_counter()
    clean_tracks(
        tracks_json=TRACKS_JSON,
        poses_file=POSES_FILE,
        output_json=CLEANED_JSON,
        voxel_size=config["clustering"]["voxel_size"],
        eps=config["clustering"]["dbscan_eps"],
        min_samples=config["clustering"]["dbscan_samples"],
        min_cluster_detections=config["clustering"]["min_cluster_detections"]
    )
    print(f"Step 2 completed in {time.perf_counter() - t0:.2f}s")

    # 3) prepare Ceres
    t0 = time.perf_counter()
    prepare_ceres(
        tracks_json=CLEANED_JSON,
        poses_txt=POSES_FILE,
        out_dir=CERES_DIR
    )
    print(f"Step 3 completed in {time.perf_counter() - t0:.2f}s")

    # 4) bundle-adjustment
    t0 = time.perf_counter()
    subprocess.run(
        [str(CERES_SOLVER_BIN), "cameras.txt", "points.txt", "observations.txt"],
        cwd=CERES_DIR, check=True
    )
    print(f"Step 4 completed in {time.perf_counter() - t0:.2f}s")

    # 5) apply deltas
    t0 = time.perf_counter()
    apply_deltas(
        tracks_json=CLEANED_JSON,
        centroid_deltas=CERES_DIR / "centroid_deltas.txt",
        output_json=POST_DELTA_JSON
    )
    print(f"Step 5 completed in {time.perf_counter() - t0:.2f}s")

    # 6) deduplicate
    t0 = time.perf_counter()
    deduplicate(
        tracks_json=POST_DELTA_JSON,
        ceres_poses=CERES_DIR / "refined_cameras.txt",
        converted_ceres_poses=OUTPUT_DIR/ 'post_ceres_poses.txt',
        output_json=MERGED_JSON, 
        centroid_distance=config["deduplication"]["max_centroid_distance_mm"],
        iou_min=config["deduplication"]["min_mask_iou"],
          
    )
    print(f"Step 6 completed in {time.perf_counter() - t0:.2f}s")

    # 7) measure
    t0 = time.perf_counter()
    measure(
        input_file = MERGED_JSON,
        images_dir = IMAGES_DIR,

    )
    print(f"Step 7 completed in {time.perf_counter() - t0:.2f}s")

    print(f"\nPipeline complete in {time.perf_counter() - total_start:.2f}s")
    return MERGED_JSON

if __name__ == "__main__":
    merged = pipeline()
    print(f"Merged tracks JSON at {merged}")
