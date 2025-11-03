import os
import cv2
import numpy as np

def read_index_lines(index_path="index.txt"):
    """Yield tuples: (frame1, ann1, frame2, ann2) from index.txt"""
    with open(index_path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                yield tuple(parts)

def load_image(path):
    """Read image as BGR uint8 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def read_annotations(txt_path):
    """
    Parse VIRAT-style annotation file:
    class_id track_id frame x y w h confidence
    """
    boxes, labels = [], []
    with open(txt_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 7:
                continue
            try:
                cls = int(vals[0])
                x = float(vals[3])
                y = float(vals[4])
                w = float(vals[5])
                h = float(vals[6])
            except ValueError:
                continue  # skip malformed rows (e.g., if last field has %)

            x1, y1 = x, y
            x2, y2 = x + w, y + h

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

    return np.array(boxes, dtype=float), np.array(labels, dtype=int)

def filter_small(boxes, labels, img_shape, min_frac=2e-4):
    """Remove boxes smaller than min_frac × (H×W)."""
    H, W = img_shape[:2]
    min_area = min_frac * H * W
    keep = []
    for i, b in enumerate(boxes):
        area = (b[2]-b[0])*(b[3]-b[1])
        if area >= min_area:
            keep.append(i)
    if not keep:
        return np.empty((0,4)), np.empty((0,), dtype=int), []
    return boxes[keep], labels[keep], keep
