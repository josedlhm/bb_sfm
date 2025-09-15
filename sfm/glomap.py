from pathlib import Path
from utils import load_intrinsics, run
import numpy as np

def run_glomap(
    images_dir: Path,
    out_dir: Path,
    intrinsics: tuple[float, float, float, float],
    *,
    quality: str = "high",        # Not used by GLOMAP but kept for API consistency
    gpu: bool = True,
):
    """
    Launch COLMAP feature extractor, sequential matcher,
    run GLOMAP mapper, convert to TXT, and dump poses_raw.txt.
    Returns path to that poses file.
    """
    images_dir = images_dir.expanduser()
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "database.db"

    fx, fy, cx, cy = intrinsics
    intrin_txt = f"{fx},{fy},{cx},{cy}"

    # 1. Feature extraction (replicates COLMAP default from automatic_reconstructor)
    run([
        "colmap", "feature_extractor",
        "--image_path", str(images_dir),
        "--database_path", str(db_path),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", intrin_txt,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1" if gpu else "0",
    ])

    # 2. Sequential feature matching
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1" if gpu else "0",
    ])

    # 3. Run GLOMAP global SfM
    sparse_dir = out_dir / "sparse"
    glomap_cmd = [
        "glomap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        # Prevent intrinsics refinement by default:
        "--BundleAdjustment.optimize_intrinsics", "0",
    ]
    run(glomap_cmd)

    # Select the first sparse model
    models = sorted(
        (d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name)
    )
    if not models:
        raise RuntimeError("GLOMAP produced no sparse model")
    final = models[0]

    # Convert to TXT format
    run([
        "colmap", "model_converter",
        "--input_path", str(final),
        "--output_path", str(out_dir),
        "--output_type", "TXT",
    ])

    # Parse `images.txt` to extract camera poses
    poses = []
    with open(out_dir / "images.txt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, ln.split()[1:8])
            poses.append((tx, ty, tz, qx, qy, qz, qw))
    poses_path = out_dir / "poses_raw.txt"
    np.savetxt(poses_path, poses)

    (out_dir / "SUCCESS").touch()
    return poses_path
