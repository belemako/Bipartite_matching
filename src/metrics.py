import numpy as np
import cv2

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def centroid_distance(p, q):
    return float(np.hypot(p[0]-q[0], p[1]-q[1]))

def centroids_areas(boxes):
    if len(boxes) == 0:
        return np.empty((0,2)), np.empty((0,))
    cx = (boxes[:,0] + boxes[:,2]) * 0.5
    cy = (boxes[:,1] + boxes[:,3]) * 0.5
    area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return np.stack([cx, cy], axis=1), area

def crop_hist(img, box, bins=16):
    """HSV histogram for appearance similarity."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2,img.shape[1]-1), min(y2,img.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = []
    for ch in range(3):
        h = cv2.calcHist([hsv],[ch],None,[bins],[0,256]).flatten()
        hist.append(h)
    h = np.concatenate(hist)
    return h / (np.linalg.norm(h) + 1e-8)

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))
