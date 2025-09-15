from __future__ import annotations
import logging, subprocess
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("cvrecon")

def run(cmd: List[str | Path]) -> None:
    """Thin wrapper around subprocess.run with nice logging."""
    cmd = list(map(str, cmd))
    log.info("▶ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """qx,qy,qz,qw → 3×3."""
    return R.from_quat(q).as_matrix()

def bilinear(depth: np.ndarray, x: float, y: float) -> float | None:
    h, w = depth.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None
    xf, yf = int(x), int(y)
    xi, yi = x - xf, y - yf
    z = depth
    vals = [z[yf, xf], z[yf, xf+1], z[yf+1, xf], z[yf+1, xf+1]]
    if any(v <= 0 for v in vals):                       # invalid pixels
        return None
    zl = vals[0]*(1-yi) + vals[2]*yi
    zr = vals[1]*(1-yi) + vals[3]*yi
    return zl*(1-xi) + zr*xi


def load_intrinsics(src: str | Tuple[float, float, float, float]
                     ) -> Tuple[float, float, float, float]:
    """Accept (fx,fy,cx,cy) tuple, CSV string, or path to 1-line txt."""
    if isinstance(src, (tuple, list)) and len(src) == 4:
        return tuple(map(float, src))
    if isinstance(src, str):
        if "," in src:
            return tuple(map(float, src.split(",")))
        return tuple(map(float, Path(src).read_text().split()))
    raise ValueError("intrinsics must be (fx,fy,cx,cy), CSV, or path")