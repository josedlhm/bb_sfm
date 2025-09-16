#!/usr/bin/env python3
import sys, json
from pathlib import Path
from extract import extract_svo2  # <-- your function

# fixed intrinsics for your ZED X Mini
INTRINSICS = [1272.44, 1272.67, 920.062, 618.949]

def process_one(svo_path: Path, out_root: Path, max_frames: int | None = None):
    """Extract one .svo2 into its own capture folder."""
    capture_name = svo_path.stem
    capture_dir  = out_root / capture_name
    capture_dir.mkdir(parents=True, exist_ok=True)

    # run the extractor
    extract_svo2(
        svo_path   = str(svo_path),
        output_dir = str(capture_dir),
        max_frames = max_frames
    )

    # write intrinsics.json
    intr_path = capture_dir / "intrinsics.json"
    intr_path.write_text(json.dumps(INTRINSICS))
    print(f"[extraction] wrote {intr_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: run_extraction.py <SVO2_FILE_OR_FOLDER> [OUT_DIR]")
        sys.exit(1)

    src = Path(sys.argv[1]).expanduser().resolve()
    out_root = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else src.parent

    if src.is_file() and src.suffix.lower() in {".svo", ".svo2"}:
        process_one(src, out_root)
    elif src.is_dir():
        for svo in sorted(src.glob("*.svo2")):
            process_one(svo, out_root)
    else:
        print(f"[extraction] no .svo2 files found in {src}")

if __name__ == "__main__":
    main()
