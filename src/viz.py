import cv2, os, random

def draw_matches_and_save(pair_id, img1, img2, boxes1, boxes2, labels1, labels2, matches,
                          out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    img1, img2 = img1.copy(), img2.copy()
    rng = random.Random(1234)
    for (i, j) in matches:
        color = (rng.randint(0,255), rng.randint(0,255), rng.randint(0,255))
        # frame 1
        x1,y1,x2,y2 = [int(round(v)) for v in boxes1[i]]
        cv2.rectangle(img1, (x1,y1), (x2,y2), color, 3)
        cv2.putText(img1, f"id{i}_cls{labels1[i]}", (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # frame 2
        x1,y1,x2,y2 = [int(round(v)) for v in boxes2[j]]
        cv2.rectangle(img2, (x1,y1), (x2,y2), color, 3)
        cv2.putText(img2, f"id{j}_cls{labels2[j]}", (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(f"{out_dir}/{pair_id}_matched_frame1.png", img1)
    cv2.imwrite(f"{out_dir}/{pair_id}_matched_frame2.png", img2)
