import os
from .io_utils import read_index_lines, load_image, read_annotations, filter_small
from .matcher import build_cost_matrix, hungarian_with_threshold
from .viz import draw_matches_and_save

def process_pair(pair_id, f1, a1, f2, a2, cfg):
    img1, img2 = load_image(f1), load_image(f2)
    boxes1, labels1 = read_annotations(a1)
    boxes2, labels2 = read_annotations(a2)

    # skip empty
    if len(boxes1)==0 or len(boxes2)==0:
        print(f"{pair_id}: empty annotations")
        return

    # remove tiny boxes
    boxes1, labels1, _ = filter_small(boxes1, labels1, img1.shape, cfg["min_area_frac"])
    boxes2, labels2, _ = filter_small(boxes2, labels2, img2.shape, cfg["min_area_frac"])

    # build cost + Hungarian
    C = build_cost_matrix(img1, img2, boxes1, boxes2, labels1, labels2,
                          w_iou=cfg["w_iou"], w_ctr=cfg["w_ctr"],
                          w_app=cfg["w_app"], class_penalty=cfg["class_penalty"])
    matches, un1, un2 = hungarian_with_threshold(C, boxes1, boxes2, cfg["min_iou_gate"])

    # visualize and save
    draw_matches_and_save(pair_id, img1, img2, boxes1, boxes2, labels1, labels2, matches, cfg["out_dir"])

    print(f"{pair_id}: {len(matches)} matches, {len(un1)} unmatched in frame1, {len(un2)} unmatched in frame2")

def main():
    cfg = dict(
        w_iou=0.6,
        w_ctr=0.4,
        w_app=0.0,          # set >0 to use appearance
        class_penalty=1000,
        min_iou_gate=0.10,
        min_area_frac=2e-4,
        out_dir="outputs"
    )
    os.makedirs(cfg["out_dir"], exist_ok=True)

    for idx, (f1, a1, f2, a2) in enumerate(read_index_lines("index.txt"), start=1):
        pair_id = f"pair_{idx:03d}"
        process_pair(pair_id, f1, a1, f2, a2, cfg)

if __name__ == "__main__":
    main()
