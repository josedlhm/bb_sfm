#!/usr/bin/env python3
# cvrecon/cli.py – DEMO: one-arg in, fixed outputs, always prune + scale

import sys, json, time
from pathlib import Path

from glomap import run_glomap
from scale import rgbd_scale_and_export
from prune import prune_sequence   # NOTE: your file is prune.py

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m cvrecon.cli <CAPTURE_ROOT>", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    capture_root = Path(sys.argv[1]).expanduser().resolve()
    images_dir   = capture_root / "images"
    depth_dir    = capture_root / "depth"
    out_root     = capture_root.parent / f"{capture_root.name}_output" / "02_sfm"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── sanity checks ───────────────────────────────────────────────
    if not images_dir.is_dir() or not any(images_dir.glob("img_*.png")):
        print(f"[SFM] Missing or empty images dir: {images_dir}", file=sys.stderr); sys.exit(1)
    if not depth_dir.is_dir() or not any(depth_dir.glob("depth_*.npy")):
        print(f"[SFM] Missing or empty depth dir: {depth_dir}", file=sys.stderr); sys.exit(1)

    intr_path = capture_root / "intrinsics.json"
    if not intr_path.exists():
        print(f"[SFM] Missing intrinsics.json in: {capture_root}", file=sys.stderr); sys.exit(1)
    arr = json.loads(intr_path.read_text())
    if not (isinstance(arr, list) and len(arr) == 4):
        print(f"[SFM] intrinsics.json must be like [fx, fy, cx, cy]", file=sys.stderr); sys.exit(1)
    intrinsics = tuple(map(float, arr))

    # ── prune ──────────────────────────────────────────────────────
    print("[SFM] Pruning redundant frames …")
    pruned_img_dir = out_root / "images_pruned"
    pruned_dep_dir = out_root / "depth_pruned"
    _, _, kept, total = prune_sequence(
        images_dir=images_dir,
        depth_dir=depth_dir,
        out_images_dir=pruned_img_dir,
        out_depth_dir=pruned_dep_dir,
    )
    print(f"[SFM] Pruning complete: kept {kept}/{total} frames.")

    # ── SfM (COLMAP + GLOMAP) ───────────────────────────────────────
    print("[SFM] Running COLMAP/GLOMAP …")
    run_glomap(pruned_img_dir, out_root, intrinsics)

    # ── scaling (always) ────────────────────────────────────────────
    print("[SFM] Scaling with RGB-D …")
    poses_path = rgbd_scale_and_export(
        model_dir  = out_root,
        depth_dir  = pruned_dep_dir,
        intrinsics = intrinsics,
        out_path   = out_root / "poses_mm_yup.txt",
    )

    # ── success marker & summary ────────────────────────────────────
    (out_root / "SUCCESS").touch()
    summary = {
        "num_frames_input": total,
        "num_frames_kept": kept,
        "used_depth": True,
        "poses_path": str(poses_path),
        "runtime_s": round(time.time() - t0, 3),
    }
    (out_root / "sfm_done.json").write_text(json.dumps(summary, indent=2))
    print(f"SFM_DONE {out_root}")

if __name__ == "__main__":
    main()
