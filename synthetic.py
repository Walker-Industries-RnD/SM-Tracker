import cv2
import numpy as np
import json
import torch

INPUT_W, INPUT_H = 112, 112
W, H = 640, 480
F_X, F_Y = 686, 686
C_X, C_Y = W // 2, H // 2

MAX_IRIS_DIAMETER = 98.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def render_gaussian(height, width, hm_x, hm_y, sigma):
    """Render a 2-D Gaussian at heatmap-space pixel (hm_x, hm_y)."""
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    return torch.exp(-((xs - hm_x)**2 + (ys - hm_y)**2) / (2 * sigma**2))

def _to_hm(px, py, crop_xy, flipped, resize):
    """
    Map a raw 640×480 landmark pixel to 112×112 heatmap coordinates.

    Pipeline:
      original (W×H=640×480) → Resize(resize) → crop(crop_xy) → optional HFlip → /2 → heatmap
    """
    rw, rh = resize
    hm_x = (px * rw / W - crop_xy[0]) / 2
    hm_y = (py * rh / H - crop_xy[1]) / 2
    if flipped:
        hm_x = INPUT_W - hm_x
    return hm_x, hm_y

def project_2d(x, y, z):
    u = (F_X * (x / z) + C_X) / W
    v = (F_Y * (y / z) + C_Y) / H
    return (u, v)

def derive_pupil_stats(iris, pupil_size):
    iris_pts = np.array([(p[0], p[1]) for p in iris], dtype=np.float32)
    (cx, cy), (axis_w, axis_h), angle = cv2.fitEllipse(iris_pts)
    iris_diameter = float(np.sqrt(axis_w * axis_h))
    return cx, cy, (pupil_size * iris_diameter) / MAX_IRIS_DIAMETER

def convert(obj, crop_xy=(0, 0), flipped=False, resize=(224, 224)):
    """
    Build ground-truth tensors from a parsed UnityEyes JSON object.

    crop_xy : (x, y) pixel offset of RandomCrop in the resized image space.
    flipped : whether RandomHorizontalFlip was applied.
    resize  : (w, h) the image was resized to before cropping.
              Default (224, 224) is correct for the no-augmentation inference
              path (direct Resize to 224, no crop).
    """
    gt = {}
    cam = obj['cameras']['cam0']

    gx, gy, gz, _ = obj['eye_details']['look_vec']
    gt['gaze_vector'] = torch.tensor([-gx if flipped else gx, gy, gz], dtype=torch.float32)

    pupil_x, pupil_y, gt['pupil_diameter'] = derive_pupil_stats(cam['iris_2d'], obj['eye_details']['pupil_size'])

    eye_heatmaps = []
    hm_x, hm_y = _to_hm(pupil_x, pupil_y, crop_xy, flipped, resize)
    eye_heatmaps.append(render_gaussian(INPUT_H, INPUT_W, hm_x, hm_y, 1.5))

    upper = list(cam['upper_interior_margin_2d'])
    lower = list(cam['lower_interior_margin_2d'])
    if flipped:
        upper = upper[::-1]
        lower = lower[::-1]

    for side in (upper, lower):
        for pt in side:
            pos = pt['pos']
            hm_x, hm_y = _to_hm(pos[0], pos[1], crop_xy, flipped, resize)
            eye_heatmaps.append(render_gaussian(INPUT_H, INPUT_W, hm_x, hm_y, 2))

    gt['eye_heatmaps'] = torch.stack(eye_heatmaps)

    return gt
