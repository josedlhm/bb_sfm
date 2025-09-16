#!/usr/bin/env python3
# run_tracking.py — call tracking/main.py in-process (no subprocess/env hops)

import sys, time, json, yaml, importlib.util
from pathlib import Path

def load_module_from(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_tracking.py <CAPTURE_ROOT>")
        sys.exit(1)

    t0 = time.time()
    capture_root = Path(sys.argv[1]).expanduser().resolve()
    out_root     = capture_root.parent / f"{capture_root.name}_output"
    track_out    = out_root / "03_tracking"
    track_out.mkdir(parents=True, exist_ok=True)

    # Inputs produced by earlier stages
    sfm_dir   = out_root / "02_sfm"
    images    = sfm_dir / "images_pruned"
    depths    = sfm_dir / "depth_pruned"
    poses_txt = sfm_dir / "poses_mm_yup.txt"
    det_json  = out_root / "detections" / "detections.json"

    missing = [p for p in [images, depths, poses_txt, det_json] if not p.exists()]
    if missing:
        print("[TRK] Missing required inputs:")
        for m in missing: print("  -", m)
        sys.exit(1)

    # Skip if already completed
    success_f = track_out / "SUCCESS"
    if success_f.exists():
        print(f"[TRK] SKIP (SUCCESS exists): {track_out}")
        return

    # Load intrinsics from the capture folder
    intr_path = capture_root / "intrinsics.json"
    try:
        fx, fy, cx, cy = map(float, json.loads(intr_path.read_text()))
    except Exception:
        print(f"[TRK] intrinsics.json must look like [fx, fy, cx, cy]: {intr_path}")
        sys.exit(1)

    # Build config.yaml exactly where main.py expects it (same folder)
    trk_dir = Path(__file__).parent
    cfg = {
        "paths": {
            "depth_folder":        str(depths),
            "images_dir":          str(images),
            "detections_json":     str(det_json),
            "poses_file":          str(poses_txt),
            "ceres_solver_bin":    "/Users/josedlh/new_ba/build/bundle_adjustment",
            "refined_poses_file":  "refined_poses.txt",
            "output_dir":          str(track_out),
            "tracks_json":         "tracks.json",
            "cleaned_tracks_json": "tracks_cleaned.json",
            "ceres_dir":           "ceres",
            "post_delta_json":     "post_delta.json",
            "merged_tracks_json":  "merged_tracks.json",
        },
        "thresholds":   {"iou_thresh": 0.15, "outlier_delta": 5, "min_track_len": 4, "min_consecutive": 4},
        "clustering":   {"voxel_size": 5, "dbscan_eps": 15, "dbscan_samples": 10, "min_cluster_detections": 4},
        "deduplication":{"max_centroid_distance_mm": 50.0, "min_mask_iou": 0.1},
        "image":        {"width": 1920, "height": 1200},
        "intrinsics":   {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }
    cfg_path = trk_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[TRK] Config written: {cfg_path}")

    # Import and run your tracker **in this same interpreter/env**
    main_py = trk_dir / "main.py"   # your orchestrator file
    mod = load_module_from(main_py)
    merged_path = mod.pipeline()    # calls your existing pipeline()

    # Mark success + summary
    success_f.write_text("")
    (track_out / "tracking_done.json").write_text(json.dumps({
        "tracking_done": True,
        "runtime_s": round(time.time() - t0, 3),
        "capture_root": str(capture_root),
        "merged_tracks_json": str(merged_path),
    }, indent=2))
    print(f"TRACKING_DONE {track_out}")

if __name__ == "__main__":
    main()
