import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import iou, centroid_distance, centroids_areas, crop_hist, cosine_sim

def build_cost_matrix(img1, img2, boxes1, boxes2, labels1, labels2,
                      w_iou=0.6, w_ctr=0.4, w_app=0.0, class_penalty=1e3):
    """Return cost matrix (|S1|×|S2|)."""
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    D = 0.5 * (np.hypot(W1,H1) + np.hypot(W2,H2))
    ctr1,_ = centroids_areas(boxes1)
    ctr2,_ = centroids_areas(boxes2)
    hists1 = [crop_hist(img1,b) if w_app>0 else None for b in boxes1]
    hists2 = [crop_hist(img2,b) if w_app>0 else None for b in boxes2]
    C = np.zeros((len(boxes1), len(boxes2)), dtype=float)
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            cost  = w_iou*(1 - iou(boxes1[i], boxes2[j]))
            cost += w_ctr*(centroid_distance(ctr1[i], ctr2[j]) / (D+1e-8))
            if w_app > 0:
                sim = cosine_sim(hists1[i], hists2[j])
                cost += w_app*(1 - sim)
            if labels1[i] != labels2[j]:
                cost += class_penalty
            C[i,j] = cost
    return C

def hungarian_with_threshold(C, boxes1, boxes2, min_iou_thresh=0.1):
    """Run Hungarian and keep only pairs with IoU≥threshold."""
    rows, cols = linear_sum_assignment(C)
    matches, unmatched1, unmatched2 = [], set(range(len(boxes1))), set(range(len(boxes2)))
    for r, c in zip(rows, cols):
        if iou(boxes1[r], boxes2[c]) >= min_iou_thresh:
            matches.append((r,c))
            unmatched1.discard(r)
            unmatched2.discard(c)
    return matches, sorted(list(unmatched1)), sorted(list(unmatched2))
