# reconlib/depth_scale.py
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Tuple, List, Dict
from utils import load_intrinsics        # reuse if desired

S_FLIP = np.diag([1., 1., 1.])         # placeholder for now. original code did yflip

def load_points3d_txt(path_txt: Path) -> Dict[int, np.ndarray]:
    """Return dict {point3D_id: XYZ[np.float64]}  – only first 4 columns used."""
    pts = {}
    with path_txt.open() as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            toks = ln.split()
            pid  = int(toks[0])
            xyz  = np.array(list(map(float, toks[1:4])), dtype=np.float64)
            pts[pid] = xyz
    return pts


class ImageRec:
    __slots__ = ("name", "qvec", "tvec", "points_xy", "points3d_ids")
    def __init__(self, name: str, qvec: np.ndarray, tvec: np.ndarray,
                 pts_xy: np.ndarray, pids: np.ndarray):
        self.name, self.qvec, self.tvec = name, qvec, tvec
        self.points_xy, self.points3d_ids = pts_xy, pids



def load_images_txt(path_txt: Path) -> List["ImageRec"]:
    """Parse COLMAP images.txt (TXT model) into ImageRec objects."""
    images: List[ImageRec] = []
    with path_txt.open() as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        ln1 = lines[i].strip()
        if ln1.startswith("#") or not ln1:
            i += 1
            continue
        ln2 = lines[i + 1].strip()
        i += 2

        toks    = ln1.split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = toks[:9]
        name    = toks[9]
        qvec    = np.array([float(qx), float(qy), float(qz), float(qw)], np.float64)
        tvec    = np.array([float(tx), float(ty), float(tz)],            np.float64)

        # second line tokens: x y point3D_id  (repeated)
        pts     = ln2.split()
        n_trip  = len(pts) // 3                    # complete triples only
        xy_flat = [float(v) for k in range(n_trip) for v in pts[3*k : 3*k+2]]
        ids     = [int(pts[3*k+2]) for k in range(n_trip)]

        xy   = np.asarray(xy_flat, dtype=np.float64).reshape(-1, 2)
        pids = np.asarray(ids,      dtype=np.int64)

        images.append(ImageRec(name, qvec, tvec, xy, pids))

    return images


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """qx,qy,qz,qw  →  3×3"""
    return R.from_quat(q).as_matrix()


def bilinear(depth: np.ndarray, x: float, y: float) -> float | None:
    h, w = depth.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None
    xf, yf = int(math.floor(x)), int(math.floor(y))
    xi, yi = x - xf, y - yf
    ztl, ztr = depth[yf, xf], depth[yf, xf + 1]
    zbl, zbr = depth[yf + 1, xf], depth[yf + 1, xf + 1]
    if min(ztl, ztr, zbl, zbr) <= 0.0:
        return None
    zl = ztl * (1 - yi) + zbl * yi
    zr = ztr * (1 - yi) + zbr * yi
    return zl * (1 - xi) + zr * xi


# main function. everything is in mm
def rgbd_scale_and_export(model_dir: Path,
                          depth_dir: Path,
                          intrinsics: Tuple[float, float, float, float],
                          out_path: Path | None = None) -> Path:
    """
    Port of Alex Yu’s C++ scale-estimation, but:
      • reads depth as <image_stem>.npy  (float, already in mm)
      • writes poses in millimetres (no extra ×1000)
    """
    model_dir, depth_dir = Path(model_dir), Path(depth_dir)
    fx, fy, cx, cy = intrinsics

    images = load_images_txt(model_dir / "images.txt")
    pts3d  = load_points3d_txt(model_dir / "points3D.txt")

    ratios: List[float] = []

    for im in images:
        stem = Path(im.name).stem          # img_00000
        # strip the "img_" prefix and add "depth_"
        depth_path = depth_dir / f"depth_{stem[4:]}.npy"
        if not depth_path.exists():
            continue
        depth = np.load(depth_path).astype(np.float32)    # depth in **mm**
        if depth.ndim != 2:
            raise ValueError(f"{depth_path} is not a 2-D array")

        idxs = [j for j, pid in enumerate(im.points3d_ids) if pid >= 0]

        Rwc = qvec_to_rotmat(im.qvec).T      # world→cam → cam→world
        tcw = -Rwc @ im.tvec

        for j in idxs:
            pid = im.points3d_ids[j]
            P_w = pts3d[pid]
            P_c = Rwc.T @ (P_w - tcw)
            if P_c[2] <= 0.0:
                continue
            x = fx * P_c[0] / P_c[2] + cx
            y = fy * P_c[1] / P_c[2] + cy
            z_d = bilinear(depth, x, y)


            if z_d is None:
                continue

            if (not np.isfinite(z_d)) or z_d <= 0.0 \
            or (not np.isfinite(P_c[2])) or P_c[2] <= 1e-6:
                continue
     # returns mm
            if z_d is None or z_d <= 0.0:
                continue
            ratios.append(z_d / P_c[2])      # mm / (COLMAP-units)

    if not ratios:
        print("[WARN] No valid RGB-D correspondences; scale=1")
        scale = 1.0
    else:
        ratios = np.asarray(ratios, np.float64)
        scale  = ratios.mean()
        print(f"[INFO] global scale = {scale:.6f} mm/colmap "
              f"(±{ratios.std():.6f}, {len(ratios)} samples)")

    # ─── write 3×4 poses (mm, Y-up) ──────────────────────────────────────────
    lines = []
    for im in sorted(images, key=lambda x: x.name):
        Rwc = qvec_to_rotmat(im.qvec).T
        tcw = -Rwc @ im.tvec * scale                 # now in mm
        Rfl = S_FLIP @ Rwc @ S_FLIP
        tfl = S_FLIP @ tcw
        T   = np.hstack([Rfl, tfl[:, None]]).reshape(-1)  # 12 numbers
        lines.append(" ".join(f"{v:.6f}" for v in T))

    out_path = (out_path or model_dir / "poses_mm_yup.txt")
    Path(out_path).write_text("\n".join(lines))
    print(f"[INFO] wrote {len(lines)} poses → {out_path}")
    return out_path