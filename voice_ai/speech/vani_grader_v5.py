"""
vani_detect_grade.py
====================
YOLO-based vegetable detector + quality grader.

Usage
-----
    python vani_detect_grade.py --model best.pt --source image.jpg
    python vani_detect_grade.py --model best.pt --source images_folder/
    python vani_detect_grade.py --model best.pt --source 0          # webcam

Outputs
-------
    • Annotated image(s) saved alongside source (or to --output dir)
    • grade_log.json updated with every result
    • Console table with detection + grade per vegetable
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── ultralytics / torch are required at runtime ──────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "❌  ultralytics is not installed.\n"
        "    Run:  pip install ultralytics"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  COMMODITY PROFILES  (tune HSV ranges per vegetable)
# ─────────────────────────────────────────────────────────────────────────────

COMMODITY_PROFILES = {
    "potato": {
        "healthy_hsv_range": [(10, 20, 80), (30, 120, 220)],
        "rot_dark_threshold": 50,
        "rot_hue_range": [(0, 40), (160, 180)],
        "rot_saturation_min": 40,
        "blemish_min_area": 80,
        "rot_min_area": 60,
        "rot_weight": 22,
        "blemish_weight": 4,
        "tolerance_pct": 8,
    },
    "onion": {
        "healthy_hsv_range": [(10, 10, 100), (40, 80, 255)],
        "rot_dark_threshold": 45,
        "rot_hue_range": [(0, 20), (160, 180)],
        "rot_saturation_min": 50,
        "blemish_min_area": 100,
        "rot_min_area": 50,
        "rot_weight": 25,
        "blemish_weight": 3,
        "tolerance_pct": 12,
    },
    "tomato": {
        "healthy_hsv_range": [(0, 120, 100), (15, 255, 255)],
        "rot_dark_threshold": 60,
        "rot_hue_range": [(15, 50), (130, 180)],
        "rot_saturation_min": 30,
        "blemish_min_area": 50,
        "rot_min_area": 40,
        "rot_weight": 30,
        "blemish_weight": 8,
        "tolerance_pct": 5,
    },
    "default": {
        "healthy_hsv_range": None,
        "rot_dark_threshold": 55,
        "rot_hue_range": [(0, 30), (140, 180)],
        "rot_saturation_min": 40,
        "blemish_min_area": 80,
        "rot_min_area": 60,
        "rot_weight": 20,
        "blemish_weight": 5,
        "tolerance_pct": 8,
    },
}

# Grade colour palette  (BGR for OpenCV)
GRADE_COLORS = {
    "A": (0, 200, 0),       # green
    "B": (0, 165, 255),     # orange
    "C": (0, 0, 220),       # red
    "Reject": (80, 80, 80), # grey
}

GRADE_LOG_PATH = Path("grade_log.json")


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _filter_blobs_by_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lid in range(1, n):
        if stats[lid, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lid] = 255
    return out


def segment_body(hsv: np.ndarray) -> np.ndarray:
    raw = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([180, 255, 255]))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(raw,  cv2.MORPH_CLOSE, kc)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def build_healthy_mask(hsv, body_mask, profile):
    body_px = hsv[body_mask > 0]
    val = body_px[:, 2]
    thresh = np.percentile(val, 60)
    bright = body_px[val >= thresh]
    mean, std = np.mean(bright, 0), np.std(bright, 0)
    lo = np.clip(mean - 1.5 * std, 0, 255).astype(np.uint8)
    hi = np.clip(mean + 1.5 * std, 0, 255).astype(np.uint8)
    adapt = cv2.inRange(hsv, lo, hi)

    if profile["healthy_hsv_range"] is not None:
        lp = np.array(profile["healthy_hsv_range"][0], np.uint8)
        hp = np.array(profile["healthy_hsv_range"][1], np.uint8)
        combined = cv2.bitwise_or(adapt, cv2.inRange(hsv, lp, hp))
    else:
        combined = adapt

    healthy = cv2.bitwise_and(combined, body_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(healthy, cv2.MORPH_CLOSE, k)


def detect_blemishes(hsv, body_mask, healthy_mask, profile):
    not_healthy = cv2.bitwise_and(cv2.bitwise_not(healthy_mask), body_mask)
    not_dark = (hsv[:, :, 2] > profile["rot_dark_threshold"]).astype(np.uint8) * 255
    raw = cv2.bitwise_and(not_healthy, not_dark)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    blemish = _filter_blobs_by_area(raw, profile["blemish_min_area"])
    area = max(cv2.countNonZero(body_mask), 1)
    return blemish, cv2.countNonZero(blemish) / area * 100


def detect_rot(hsv, body_mask, blemish_mask, profile):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark = cv2.bitwise_and(cv2.inRange(v, 0, int(profile["rot_dark_threshold"])), body_mask)
    rot_hue = np.zeros_like(h, dtype=np.uint8)
    for lo, hi in profile["rot_hue_range"]:
        rot_hue = cv2.bitwise_or(rot_hue, cv2.inRange(h, int(lo), int(hi)))
    high_sat = cv2.inRange(s, int(profile["rot_saturation_min"]), 255)
    primary = cv2.bitwise_and(dark, rot_hue)
    surface = cv2.bitwise_and(cv2.bitwise_and(high_sat, rot_hue), blemish_mask)
    raw = cv2.bitwise_and(cv2.bitwise_or(primary, surface), body_mask)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k7)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k3)
    rot = _filter_blobs_by_area(raw, profile["rot_min_area"])
    area = max(cv2.countNonZero(body_mask), 1)
    return rot, cv2.countNonZero(rot) / area * 100


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING  →  GRADE  (A / B / C / Reject)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(blemish_ratio: float, rot_ratio: float, profile: dict):
    adj_blemish = max(0.0, blemish_ratio - profile["tolerance_pct"])
    score = 100.0 - rot_ratio * profile["rot_weight"] - adj_blemish * profile["blemish_weight"]
    score = round(max(0.0, min(100.0, score)), 2)

    if score >= 85:
        grade, label = "A", "Premium"
    elif score >= 65:
        grade, label = "B", "Standard"
    elif score >= 45:
        grade, label = "C", "Discount"
    else:
        grade, label = "Reject", "Reject"

    return score, grade, label


# ─────────────────────────────────────────────────────────────────────────────
#  CORE GRADER  (works on a cropped BGR patch)
# ─────────────────────────────────────────────────────────────────────────────

def grade_crop(crop_bgr: np.ndarray, commodity: str) -> dict:
    """
    Run full quality analysis on a single vegetable crop.
    Returns dict with score, grade, label, blemish_ratio, rot_ratio.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"score": 0, "grade": "Reject", "label": "Reject",
                "blemish_ratio": 0.0, "rot_ratio": 0.0}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    profile = COMMODITY_PROFILES.get(commodity.lower(), COMMODITY_PROFILES["default"])

    body_mask = segment_body(hsv)
    if cv2.countNonZero(body_mask) == 0:
        return {"score": 0, "grade": "Reject", "label": "Reject",
                "blemish_ratio": 0.0, "rot_ratio": 0.0}

    healthy_mask = build_healthy_mask(hsv, body_mask, profile)
    blemish_mask, blemish_ratio = detect_blemishes(hsv, body_mask, healthy_mask, profile)
    rot_mask, rot_ratio = detect_rot(hsv, body_mask, blemish_mask, profile)
    score, grade, label = compute_score(blemish_ratio, rot_ratio, profile)

    return {
        "score": score,
        "grade": grade,
        "label": label,
        "blemish_ratio": round(blemish_ratio, 2),
        "rot_ratio": round(rot_ratio, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

def _load_log() -> list:
    if GRADE_LOG_PATH.exists():
        try:
            with open(GRADE_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(log: list) -> None:
    with open(GRADE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def log_result(img_path: str, commodity: str, box_idx: int, result: dict) -> None:
    log = _load_log()
    log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": img_path,
        "commodity": commodity,
        "detection_index": box_idx,
        **result,
    })
    _save_log(log)


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION  (draw box + grade label on the full image)
# ─────────────────────────────────────────────────────────────────────────────

def annotate(
    img: np.ndarray,
    box: tuple,          # (x1, y1, x2, y2)  integers
    commodity: str,
    conf: float,
    result: dict,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    grade = result["grade"]
    color = GRADE_COLORS.get(grade, (200, 200, 200))

    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # top label  →  "Tomato  Grade A  93.4"
    top_text = f"{commodity.capitalize()}  Grade {grade}  {result['score']}"
    (tw, th), _ = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, top_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # bottom label  →  "B:3.2%  R:0.0%  conf:0.91"
    bot_text = (
        f"B:{result['blemish_ratio']}%  "
        f"R:{result['rot_ratio']}%  "
        f"conf:{conf:.2f}"
    )
    cv2.putText(img, bot_text, (x1 + 2, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    model: "YOLO",
    img_path: str,
    output_dir: Path,
    conf_thresh: float = 0.40,
    verbose: bool = True,
) -> list[dict]:
    """
    Run YOLO detection on one image, grade every detected vegetable,
    save annotated output, log results.

    Returns list of result dicts (one per detection).
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Cannot read image: {img_path}")
        return []

    # ── YOLO inference ──────────────────────────────────────────────────────
    results_yolo = model(img, conf=conf_thresh, verbose=False)
    detections = results_yolo[0]

    if detections.boxes is None or len(detections.boxes) == 0:
        print(f"⚠️  No vegetables detected in {img_path}")
        return []

    names   = model.names          # {0: 'potato', 1: 'tomato', ...}
    boxes   = detections.boxes.xyxy.cpu().numpy().astype(int)
    confs   = detections.boxes.conf.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy().astype(int)

    annotated = img.copy()
    all_results = []

    print(f"\n{'─'*60}")
    print(f"  Image : {img_path}")
    print(f"  Found : {len(boxes)} detection(s)")
    print(f"{'─'*60}")
    print(f"  {'#':<4} {'Commodity':<12} {'Conf':>6}  {'Score':>6}  {'Grade':>6}  {'Blemish':>8}  {'Rot':>6}")
    print(f"{'─'*60}")

    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
        commodity = names.get(cls_id, "default").lower()
        x1, y1, x2, y2 = box

        # pad crop slightly (5 px) so edges of the vegetable are included
        H, W = img.shape[:2]
        cx1 = max(0, x1 - 5)
        cy1 = max(0, y1 - 5)
        cx2 = min(W, x2 + 5)
        cy2 = min(H, y2 + 5)
        crop = img[cy1:cy2, cx1:cx2]

        result = grade_crop(crop, commodity)

        annotate(annotated, (x1, y1, x2, y2), commodity, conf, result)
        log_result(img_path, commodity, i, result)
        all_results.append({"commodity": commodity, "conf": float(conf), **result})

        print(
            f"  {i:<4} {commodity.capitalize():<12} {conf:>6.2f}  "
            f"{result['score']:>6}  {result['grade']:>6}  "
            f"{result['blemish_ratio']:>7}%  {result['rot_ratio']:>5}%"
        )

    print(f"{'─'*60}")

    # ── save annotated image ────────────────────────────────────────────────
    stem = Path(img_path).stem
    out_path = output_dir / f"{stem}_graded.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  💾 Saved → {out_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE WEBCAM MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(model: "YOLO", conf_thresh: float = 0.40) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌  Cannot open webcam.")

    print("🎥  Webcam mode — press Q to quit, S to save a frame")
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_yolo = model(frame, conf=conf_thresh, verbose=False)
        detections = results_yolo[0]
        annotated = frame.copy()

        if detections.boxes is not None and len(detections.boxes):
            names   = model.names
            boxes   = detections.boxes.xyxy.cpu().numpy().astype(int)
            confs   = detections.boxes.conf.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                commodity = names.get(cls_id, "default").lower()
                x1, y1, x2, y2 = box
                H, W = frame.shape[:2]
                crop = frame[max(0, y1-5):min(H, y2+5), max(0, x1-5):min(W, x2+5)]
                result = grade_crop(crop, commodity)
                annotate(annotated, (x1, y1, x2, y2), commodity, conf, result)

        cv2.imshow("Vani Grader — live", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"webcam_frame_{frame_id:04d}_graded.jpg"
            cv2.imwrite(fname, annotated)
            print(f"💾 Frame saved → {fname}")
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO vegetable detector + quality grader (Vani)"
    )
    p.add_argument("--model",  default="best.pt",  help="Path to best.pt")
    p.add_argument("--source", default=".",         help="Image file, folder, or webcam index (0)")
    p.add_argument("--output", default="graded",    help="Output directory for annotated images")
    p.add_argument("--conf",   default=0.40, type=float, help="Detection confidence threshold")
    p.add_argument("--show",   action="store_true", help="Display each result image on screen")
    return p.parse_args()


def main():
    args = parse_args()

    # load model
    print(f"📦  Loading model: {args.model}")
    model = YOLO(args.model)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.source

    # ── webcam ──────────────────────────────────────────────────────────────
    if source.isdigit():
        run_webcam(model, conf_thresh=args.conf)
        return

    # ── single image ────────────────────────────────────────────────────────
    source_path = Path(source)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        images = [p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        images.sort()
    else:
        sys.exit(f"❌  Source not found: {source}")

    if not images:
        sys.exit(f"❌  No images found in {source}")

    print(f"🔍  Processing {len(images)} image(s) …\n")
    t0 = time.time()
    total_detections = 0

    for img_path in images:
        res = process_image(model, str(img_path), out_dir, conf_thresh=args.conf)
        total_detections += len(res)

        if args.show and res:
            ann = cv2.imread(str(out_dir / (img_path.stem + "_graded.jpg")))
            if ann is not None:
                cv2.imshow(img_path.name, ann)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n✅  Done — {total_detections} vegetable(s) graded in {elapsed:.1f}s")
    print(f"📂  Annotated images → {out_dir.resolve()}")
    print(f"📋  Grade log        → {GRADE_LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()