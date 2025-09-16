#!/usr/bin/env python3
"""
Module: extract_svo2.py

Provides `extract_svo2` to extract RGB, depth, and poses from an SVO2 using ZED SDK 5.0.
"""
import os
import cv2
import numpy as np
import pyzed.sl as sl

def extract_svo2(
    svo_path: str,
    output_dir: str,
    max_frames: int = None
) -> None:
    """
    Extracts up to `max_frames` frames from an SVO2 file, saving:
      - Left RGB images (lossless PNG)
      - Depth maps (full-precision .npy)
      - Camera poses (CSV: frame, tx, ty, tz, qx, qy, qz, qw)

    Parameters:
        svo_path:    Path to input .svo or .svo2 file
        output_dir:  Directory where `images/`, `depth/`, and `poses.csv` will be created
        max_frames:  Maximum number of frames to extract (None = all frames)
    """
    # Prepare output dirs
    img_dir   = os.path.join(output_dir, "images")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(img_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # 1. Initialize ZED in playback mode
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode   = False
    init_params.depth_mode           = sl.DEPTH_MODE.NEURAL_PLUS       
    init_params.coordinate_units     = sl.UNIT.MILLIMETER
    init_params.camera_resolution    = sl.RESOLUTION.HD1200
    init_params.coordinate_system    = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps           = 30

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open SVO2: {svo_path}")

    # 2. Enable positional tracking with IMU fusion, GEN_3 mode
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    tracking_params.mode             = sl.POSITIONAL_TRACKING_MODE.GEN_3
    if zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError("Could not enable positional tracking (GEN_3 + IMU fusion)")

    # 3. Prepare Mats and pose container
    left_mat  = sl.Mat()
    depth_mat = sl.Mat()
    runtime   = sl.RuntimeParameters()
    pose      = sl.Pose()

    # 4. Open CSV for poses
    poses_csv = os.path.join(output_dir, "poses.csv")
    with open(poses_csv, "w", newline="") as f:
        writer = __import__('csv').writer(f)
        writer.writerow(["frame","tx","ty","tz","qx","qy","qz","qw"])

        # 5. Frame loop
        total = zed.get_svo_number_of_frames()
        limit = total if max_frames is None else min(total, max_frames)
        frame_idx = 0
        while frame_idx < limit:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                # Retrieve image & depth
                zed.retrieve_image(left_mat, sl.VIEW.LEFT)
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

                # Retrieve pose (WORLD frame)
                zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                # Translation via Translation.get()
                py_trans = sl.Translation()
                pose.get_translation(py_trans)
                trans = py_trans.get()  # [x, y, z]
                tx, ty, tz = trans[0], trans[1], trans[2]
                # Orientation via Orientation.get()
                py_orient = sl.Orientation()
                pose.get_orientation(py_orient)
                orient = py_orient.get()  # [x, y, z, w]
                ox, oy, oz, ow = orient[0], orient[1], orient[2], orient[3]

                # Save RGB
                img_rgba = left_mat.get_data()
                img_bgr  = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(
                    os.path.join(img_dir, f"img_{frame_idx:06d}.png"),
                    img_bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0]
                )

                # Save depth (float32 mm)
                depth_data = depth_mat.get_data()
                np.save(os.path.join(depth_dir, f"depth_{frame_idx:06d}.npy"), depth_data)

                # Log pose
                writer.writerow([
                    frame_idx,
                    tx, ty, tz,
                    ox, oy, oz, ow
                ])

                frame_idx += 1
            else:
                break

    # Cleanup
    zed.close()
    print(f"✅ Extracted {frame_idx} frames (limit={limit}) to '{output_dir}'")

# Example usage:
# extract_svo2("path/to/file.svo2", "output_dir", max_frames=100)
