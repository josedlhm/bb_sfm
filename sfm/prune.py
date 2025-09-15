#!/usr/bin/env python3
# cvrecon/prune_rgbd.py
# ORB + RANSAC pruning for img_******.png + depth_******.npy sequences.

from pathlib import Path
import cv2
import numpy as np
import shutil
import re
from typing import Optional, Tuple, List

# --- Defaults (edit if you want; can also pass as kwargs to prune_sequence) ---
MAX_SIDE      = 1920   # resize longest side for feature extraction
NFEATURES     = 3000   # ORB features per image
MIN_KEYPOINTS = 300    # discard if fewer keypoints than this
RATIO_THRESH  = 0.75   # Lowe ratio-test
RANSAC_THRESH = 3.0    # px
SIM_THRESH    = 0.40   # inlier-ratio >= threshold => drop frame
COMPARE_K     = 8      # compare against last K kept frames
PNG_COMPR     = 3      # output PNG compression (0-9)

_ID_RE = re.compile(r"img_(\d{6})")

def _imread_color(p: Path) -> Optional[np.ndarray]:
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def _to_gray_resized(bgr: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if max_side and max(bgr.shape[:2]) > max_side:
        h, w = bgr.shape[:2]
        s = float(max_side) / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def _extract_orb(gray: np.ndarray, nfeatures: int):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kpts, desc = orb.detectAndCompute(gray, None)
    return (kpts or []), desc

def _inlier_ratio(descA, kptsA, descB, kptsB,
                  ratio: float, ransac_thresh: float, min_matches: int = 12) -> float:
    if descA is None or descB is None or len(kptsA) < min_matches or len(kptsB) < min_matches:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(descA, descB, k=2)
    good = [m for m, n in knn if (n is not None and m.distance < ratio * n.distance)]
    if len(good) < min_matches:
        return 0.0
    ptsA = np.float32([kptsA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kptsB[m.trainIdx].pt for m in good])
    _, inliers = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_thresh)
    if inliers is None:
        return 0.0
    inl = int(inliers.sum())
    return float(inl) / float(max(1, min(len(kptsA), len(kptsB))))

def prune_sequence(
    images_dir: Path,
    depth_dir: Path,
    out_images_dir: Path,
    out_depth_dir: Path,
    *,
    max_side: int = MAX_SIDE,
    nfeatures: int = NFEATURES,
    min_keypoints: int = MIN_KEYPOINTS,
    ratio_thresh: float = RATIO_THRESH,
    ransac_thresh: float = RANSAC_THRESH,
    sim_thresh: float = SIM_THRESH,
    compare_k: int = COMPARE_K,
    png_compr: int = PNG_COMPR,
) -> Tuple[Path, Path, int, int]:
    """
    Prunes an RGB(+depth) sequence with ORB+RANSAC similarity.
    Returns: (out_images_dir, out_depth_dir, kept_count, total_count).
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    # Natural sort by the numeric part in img_******.png
    imgs: List[Path] = sorted([p for p in images_dir.glob("img_*.png")],
                              key=lambda p: int(_ID_RE.search(p.stem).group(1)) if _ID_RE.search(p.stem) else 10**12)
    total = len(imgs)
    if total == 0:
        print(f"[prune] No img_*.png in {images_dir}.")
        return images_dir, depth_dir, 0, 0

    print(f"[prune] Extracting ORB on {total} images …")
    feats = []  # list of (path, kpts, desc)
    for p in imgs:
        bgr = _imread_color(p)
        if bgr is None:
            print(f"[prune] ⚠️  Could not read {p}, skipping.")
            feats.append((p, [], None))
            continue
        gray = _to_gray_resized(bgr, max_side)
        kpts, desc = _extract_orb(gray, nfeatures)
        feats.append((p, kpts, desc))

    kept_feats: List[Tuple[Path, list, Optional[np.ndarray]]] = []
    kept_count = 0

    for i, (img_path, kpts, desc) in enumerate(feats, 1):
        if desc is None or len(kpts) < min_keypoints:
            continue

        if not kept_feats:
            keep = True
        else:
            refs = kept_feats[-compare_k:] if compare_k > 0 else kept_feats
            sims = [_inlier_ratio(desc, kpts, d2, k2, ratio_thresh, ransac_thresh)
                    for (_, k2, d2) in refs]
            max_sim = max(sims) if sims else 0.0
            keep = (max_sim < sim_thresh)

        if keep:
            new_id = f"{kept_count:06d}"

            # Save full-res RGB
            bgr_full = _imread_color(img_path)
            if bgr_full is None:
                print(f"[prune] ⚠️  Re-read failed for {img_path}, skipping.")
                continue
            cv2.imwrite(str(out_images_dir / f"img_{new_id}.png"),
                        bgr_full, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compr)])

            # Copy matching depth if available
            m = _ID_RE.match(img_path.stem)
            if m:
                old_id = m.group(1)
                src_depth = depth_dir / f"depth_{old_id}.npy"
                dst_depth = out_depth_dir / f"depth_{new_id}.npy"
                if src_depth.exists():
                    shutil.copyfile(src_depth, dst_depth)
                else:
                    print(f"[prune] ⚠️  Missing depth_{old_id}.npy")
            else:
                print(f"[prune] ⚠️  Could not parse ID for {img_path.name}")

            kept_feats.append((img_path, kpts, desc))
            kept_count += 1

        if i % 500 == 0:
            print(f"[prune] Scanned {i}/{total} – kept {kept_count}")

    print(f"[prune] Finished. Kept {kept_count} / {total}")
    return out_images_dir, out_depth_dir, kept_count, total

# Optional: allow running directly (no argparse; just edit paths here if you want)
if __name__ == "__main__":
    # Example quick test (edit these or call from cli.py)
    IMAGES_DIR = Path("data/images")
    DEPTH_DIR  = Path("data/depth")
    OUT_IMG    = Path("data/pruned_images")
    OUT_DEP    = Path("data/depth_pruned")
    prune_sequence(IMAGES_DIR, DEPTH_DIR, OUT_IMG, OUT_DEP)
