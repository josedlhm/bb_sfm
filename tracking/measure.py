#!/usr/bin/env python3
"""
enrich_tracks.py

Stream‑based enrichment of fruit tracks with centered flag and ellipsoid measurements:

1) Stream in each track (one JSON property at a time) instead of json.load()
2) For each track: set d['centered'], compute measurement if possible
3) Stream out to a temporary JSON and atomically replace the input file
"""

import json
import os
import numpy as np
import numpy.linalg as la
from PIL import Image
from pathlib import Path
import ijson

# === Ellipsoid fitting & measurement ===

def outer_ellipsoid_fit(points, tol=0.009, reg=1e-8):
    pts = np.asmatrix(points)
    N, d = pts.shape
    Q = np.vstack((pts.T, np.ones((1, N))))
    u = np.ones(N) / N
    err = tol + 1
    I = np.eye(d + 1)
    while err > tol:
        X = Q @ np.diag(u) @ Q.T
        try:
            invX = la.inv(X)
        except la.LinAlgError:
            invX = la.inv(X + reg * I)
        M = np.diag(Q.T @ invX @ Q)
        j = np.argmax(M)
        step = (M[j] - d - 1.0) / ((d + 1) * (M[j] - 1.0))
        u = (1 - step) * u
        u[j] += step
        err = la.norm(u - (1 - step) * u - step * u)
    c = u @ pts
    P = pts.T @ np.diag(u) @ pts - c.T @ c
    try:
        invP = la.inv(P)
    except la.LinAlgError:
        invP = la.inv(P + reg * np.eye(d))
    return invP / d

def calculate_axes(A):
    _, D, _ = la.svd(A)
    axes = 2.0 / np.sqrt(D)
    return np.sort(axes)[::-1].tolist()

def compute_volume(axes):
    a, b, c = (ax / 2.0 for ax in axes)
    return (4.0 / 3.0) * np.pi * a * b * c

def compute_diameter(volume):
    return ((6.0 * volume) / np.pi) ** (1.0 / 3.0)

# === Detection filtering & per‑track measurement ===

def is_centered(det, images_dir, margin=0.2):
    frame = det.get('frame')
    bbox  = det.get('bbox')
    if frame is None or not bbox:
        return False
    img_path = Path(images_dir) / f"img_{frame:06d}.png"
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except FileNotFoundError:
        return False
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (margin * w < cx < (1 - margin) * w) and (margin * h < cy < (1 - margin) * h)

def measure_axes(track):
    centered = [d for d in track if d.get('centered')]
    sample   = centered[:5] if len(centered) >= 5 else track[:5]
    fits = []
    for det in sample:
        pts = np.array(det['mask_points_3d'])
        try:
            A = outer_ellipsoid_fit(pts)
            fits.append(calculate_axes(A))
        except Exception:
            pass
    if not fits:
        raise ValueError('No valid fits')
    return np.median(np.array(fits), axis=0).tolist()

# === Run enrichment ===

def run(input_file: Path, images_dir: Path):
    tmp_file = input_file.with_suffix('.tmp.json')
    with open(input_file, 'rb') as infile, open(tmp_file, 'w') as outfile:
        outfile.write('{\n')
        first = True
        for tid_str, entry in ijson.kvitems(infile, '', use_float=True):
            # handle either dict-with-detections or bare list
            if isinstance(entry, dict) and 'detections' in entry:
                dets = entry['detections']
            else:
                dets = entry

            # set centered flag
            for d in dets:
                d['centered'] = is_centered(d, images_dir)

            # compute measurement if possible
            try:
                axes = measure_axes(dets)
                vol  = compute_volume(axes)
                diam = compute_diameter(vol)
                out_val = {'detections': dets, 'measurement': {'axes': axes, 'volume': vol, 'diameter': diam}}
            except ValueError:
                out_val = dets

            if not first:
                outfile.write(',\n')
            outfile.write(f'"{tid_str}": ')
            json.dump(out_val, outfile, indent=2)
            first = False

        outfile.write('\n}\n')

    os.replace(str(tmp_file), str(input_file))
