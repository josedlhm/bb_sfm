#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, List
import numpy as np
import ijson
import sys

def load_deltas(path: Path) -> np.ndarray:
    d = np.loadtxt(str(path), dtype=float)
    if d.ndim == 1:
        d = d.reshape(1,3)
    return d

def shift_point(p: List[float], delta: np.ndarray) -> List[float]:
    return [p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]]

def shift_mask_points(pts: Any, delta: np.ndarray) -> Any:
    if not isinstance(pts, list) or len(pts) == 0:
        return pts
    first = pts[0]
    if isinstance(first, (list, tuple)) and len(first) == 3:
        return [shift_point(p, delta) for p in pts]
    if len(pts) == 3 and all(isinstance(x, (int, float)) for x in pts):
        return shift_point(pts, delta)
    return pts

def run(
    tracks_json:      Path,
    centroid_deltas:  Path,
    output_json:      Path
) -> Path:
    # 1) load the (small) deltas
    deltas = load_deltas(centroid_deltas)

    # 2) stream‐in the cleaned tracks one track at a time
    with open(tracks_json, 'rb') as infile, open(output_json, 'w') as outfile:
        outfile.write('{\n')
        first = True

        for tid_str, dets in ijson.kvitems(infile, '', use_float=True):
            tid = int(tid_str)
            if not (0 <= tid < deltas.shape[0]):
                sys.exit(f"Error: track ID {tid} out of range [0,{deltas.shape[0]-1}]")
            delta = deltas[tid]

            # shift each detection in this one track
            for d in dets:
                if 'centroid_3d' in d:
                    d['centroid_3d'] = shift_point(d['centroid_3d'], delta)
                if 'mask_points_3d' in d:
                    d['mask_points_3d'] = shift_mask_points(d['mask_points_3d'], delta)

            # write it out immediately
            if not first:
                outfile.write(',\n')
            outfile.write(json.dumps(str(tid)) + ': ')
            json.dump(dets, outfile, indent=2)
            first = False

        outfile.write('\n}\n')

    print(f"Wrote refined tracks to {output_json}")
    return output_json
