import cv2
import numpy as np
import torch

INPUT_W, INPUT_H = 112, 112
W, H = 640, 480
F_X, F_Y = 686, 686
C_X, C_Y = W // 2, H // 2

MAX_IRIS_DIAMETER = 98.2

# ── Canonical preprocessing geometry ─────────────────────────────────────────
# original (640×480)
#   → center_crop(CROP_SIZE × CROP_SIZE)   offset: (CC_X, CC_Y) into original
#   → random_crop / center_crop (RANDOM_CROP × RANDOM_CROP)  offset: crop_xy into above
#   → optional HFlip
#   → ÷2 → heatmap (INPUT_W × INPUT_H)
CROP_SIZE    = 320
RANDOM_CROP  = 224
CC_X         = (W - CROP_SIZE)    // 2   # 160 — x offset of center crop in original
CC_Y         = (H - CROP_SIZE)    // 2   # 80  — y offset of center crop in original
EVAL_CROP_XY = (                          # deterministic crop for eval/inference:
    (CROP_SIZE - RANDOM_CROP) // 2,       # 48 — center RANDOM_CROP within CROP_SIZE
    (CROP_SIZE - RANDOM_CROP) // 2,       # 48
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def render_gaussian(height, width, hm_x, hm_y, sigma):
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width,  dtype=torch.float32),
        indexing='ij',
    )
    return torch.exp(-((xs - hm_x)**2 + (ys - hm_y)**2) / (2 * sigma**2))


def _to_hm(px, py, crop_xy, flipped):
    """
    Map a raw 640×480 landmark pixel to 112×112 heatmap coordinates.

    center_crop is a pixel-space translation, not a scale:
      hm_x = (px - CC_X - crop_xy[0]) / 2

    crop_xy: (left, top) of the RANDOM_CROP within the center-crop region.
             Pass EVAL_CROP_XY for deterministic (non-augmented) runs.
    """
    hm_x = (px - CC_X - crop_xy[0]) / 2
    hm_y = (py - CC_Y - crop_xy[1]) / 2
    if flipped:
        hm_x = INPUT_W - hm_x
    return hm_x, hm_y


def project_2d(x, y, z):
    u = (F_X * (x / z) + C_X) / W
    v = (F_Y * (y / z) + C_Y) / H
    return (u, v)


def derive_pupil_stats(iris, pupil_size):
    iris_pts = np.array([(p[0], p[1]) for p in iris], dtype=np.float32)
    (cx, cy), (axis_w, axis_h), _ = cv2.fitEllipse(iris_pts)
    iris_diameter = float(np.sqrt(axis_w * axis_h))
    return cx, cy, (pupil_size * iris_diameter) / MAX_IRIS_DIAMETER


def convert(obj, crop_xy=EVAL_CROP_XY, flipped=False):
    """
    Build ground-truth tensors from a parsed UnityEyes JSON object.

    crop_xy : (left, top) of the RANDOM_CROP within the CROP_SIZE center crop.
              Defaults to EVAL_CROP_XY — the canonical deterministic center crop.
    flipped : whether RandomHorizontalFlip was applied.
    """
    gt  = {}
    cam = obj['cameras']['cam0']

    gx, gy, gz, _ = obj['eye_details']['look_vec']
    gt['gaze_vector'] = torch.tensor([-gx if flipped else gx, gy, gz], dtype=torch.float32)

    pupil_x, pupil_y, gt['pupil_diameter'] = derive_pupil_stats(
        cam['iris_2d'], obj['eye_details']['pupil_size']
    )

    eye_heatmaps = [render_gaussian(INPUT_H, INPUT_W,
                                    *_to_hm(pupil_x, pupil_y, crop_xy, flipped), 1.5)]

    upper = list(cam['upper_interior_margin_2d'])
    lower = list(cam['lower_interior_margin_2d'])
    if flipped:
        upper = upper[::-1]
        lower = lower[::-1]

    for side in (upper, lower):
        for pt in side:
            pos = pt['pos']
            eye_heatmaps.append(render_gaussian(INPUT_H, INPUT_W,
                                                *_to_hm(pos[0], pos[1], crop_xy, flipped), 2))

    gt['eye_heatmaps'] = torch.stack(eye_heatmaps)
    return gt
