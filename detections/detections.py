# make_sahi_detections_json.py
#!/usr/bin/env python3
import sys, json
from pathlib import Path
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─── CONFIG (kept from your original; minimal edits) ───────────────────────────
CONF_THRESH = 0.5
DEVICE      = "cpu"  # "cuda" if available

BASE_DIR    = Path(__file__).parent.resolve()


WEIGHTS_PATH = str(BASE_DIR / "model_weights" / "arandanos_18.pth")
CONFIG_PATH  = str(BASE_DIR / "model_weights" / "arandanos_18.yaml")

SLICE_HEIGHT = 640
SLICE_WIDTH  = 640
OVERLAP_H    = 0.2
OVERLAP_W    = 0.2
POSTPROC     = "GREEDYNMM"

DETS_JSON = "detections.json"

WRITE_ANNOTATED = True
ANNOTATED_SUBDIR = "annotated"




FIXED_CLASS_COLORS = {
    1: (0,   0, 255),   # cuajo  -> red
    2: (255, 255, 255), # flor   -> white
    3: (255, 0,   0),   # maduro -> blue
    4: (0,   165, 255), # pinto  -> orange
    5: (255, 0, 255),   # rosado -> pink
    6: (0, 255, 0),     # verde  -> green
}

def color_for_class_id(cid: int) -> tuple[int, int, int]:
    if cid in FIXED_CLASS_COLORS:
        return FIXED_CLASS_COLORS[cid]
    # fallback: deterministic random vivid color
    rng = np.random.default_rng(int(cid) & 0xFFFFFFFF)
    return tuple(int(x) for x in rng.integers(50, 255, size=3))
# ───────────────────────────────────────────────────────────────────────────────

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_captures(root: Path) -> list[Path]:
    """
    Accept either:
      - a single CAPTURE_ROOT (expects CAPTURE_ROOT_output/02_sfm/images_pruned/)
      - a parent folder containing many CAPTURE_ROOTs
    Returns the list of CAPTURE_ROOT paths to process.
    """
    root = root.expanduser().resolve()
    captures: list[Path] = []

    # Case 1: single capture root
    if (root.parent / f"{root.name}_output" / "02_sfm" / "images_pruned").is_dir():
        captures.append(root)
        return captures

    # Case 2: parent folder → scan its immediate subfolders as possible capture roots
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            imgp = sub.parent / f"{sub.name}_output" / "02_sfm" / "images_pruned"
            if imgp.is_dir():
                captures.append(sub)

    return captures

def detect_for_capture(model: AutoDetectionModel, capture_root: Path) -> None:
    """
    For one capture root:
      - Input images: <capture>_output/02_sfm/images_pruned/
      - Output detections: <capture>_output/detections/detections.json
      - Skip if <capture>_output/detections/SUCCESS exists
    """
    out_base   = capture_root.parent / f"{capture_root.name}_output"
    input_dir  = out_base / "02_sfm" / "images_pruned"
    output_dir = out_base / "detections"
    success_f  = output_dir / "SUCCESS"

    # Skip if already done
    if success_f.exists():
        print(f"[detect] SKIP (already done): {capture_root.name}")
        return

    if not input_dir.is_dir():
        print(f"[detect] No images_pruned found for: {capture_root}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if WRITE_ANNOTATED:
        (output_dir / ANNOTATED_SUBDIR).mkdir(exist_ok=True)

    all_detections: dict[str, list[dict]] = {}

    for img_path in sorted(input_dir.iterdir()):
        if not (img_path.is_file() and img_path.suffix.lower() in EXTS):
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        vis = img_bgr.copy()

        # SAHI needs RGB
        res = get_sliced_prediction(
            image=img_bgr[..., ::-1],  # BGR -> RGB
            detection_model=model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_H,
            overlap_width_ratio=OVERLAP_W,
            postprocess_type=POSTPROC,
            postprocess_class_agnostic=True,
        )

        dets_for_image: list[dict] = []
        for p in res.object_prediction_list:
            # bbox
            x1 = int(round(getattr(p.bbox, "minx", 0)))
            y1 = int(round(getattr(p.bbox, "miny", 0)))
            x2 = int(round(getattr(p.bbox, "maxx", 0)))
            y2 = int(round(getattr(p.bbox, "maxy", 0)))

            # clamp & order
            x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1

            # mask
            if getattr(p, "mask", None) is not None and getattr(p.mask, "bool_mask", None) is not None:
                full_mask = (p.mask.bool_mask.astype(np.uint8) * 255)
            else:
                full_mask = np.zeros((H, W), dtype=np.uint8)
            cropped = full_mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)

            # class & score
            try:
                cls_id = int(p.category.id)
            except Exception:
                cls_id = 0
            score = float(p.score.value) if hasattr(p.score, "value") else float(p.score)

            dets_for_image.append({
                "bbox":     [x1, y1, x2, y2],  # inclusive
                "score":    score,
                "class_id": cls_id,
                "mask":     cropped.tolist(),
            })

            # optional viz

            if WRITE_ANNOTATED and full_mask.any():
                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                col_bgr = color_for_class_id(cls_id)   # ← pick class color
                cv2.drawContours(vis, contours, -1, col_bgr, 1)

        if WRITE_ANNOTATED:
            cv2.imwrite(str(output_dir / ANNOTATED_SUBDIR / img_path.name), vis)

        all_detections[img_path.name] = dets_for_image

    with open(output_dir / DETS_JSON, "w") as f:
        json.dump(all_detections, f, indent=2)

    success_f.write_text("")
    print(f"[detect] Wrote: {output_dir / DETS_JSON}")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python make_sahi_detections_json.py <PATH_TO_CAPTURE_OR_PARENT>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    captures = find_captures(root)
    if not captures:
        print(f"[detect] No captures found under: {root}")
        sys.exit(1)

    # Load model once
    print("[detect] Loading model …")
    model = AutoDetectionModel.from_pretrained(
        model_type="detectron2",
        model_path=WEIGHTS_PATH,
        config_path=CONFIG_PATH,
        confidence_threshold=CONF_THRESH,
        device=DEVICE,
    )

    for cap in captures:
        detect_for_capture(model, cap)

if __name__ == "__main__":
    main()
