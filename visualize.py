import argparse
import os, glob, random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from model import ShinraCNN
from dataset import SyntheticDS, BORDER_PAD
from synthetic import RANDOM_CROP

device = torch.device('cpu')

GAZE_ARROW_LEN = 90       # display length of the gaze arrow in pixels
GAZE_FLIP_Y    = True     # UnityEyes look_vec has +y up, image +y is down
WEBCAM_CROP    = 256      # side length of the center crop fed to the model in webcam mode

MAX_IRIS_DIAMETER = 98.2  # pixels in 640×480 space; preserved through center_crop (no scaling)

# Display size for the canonical preprocessing output (RANDOM_CROP + 2*BORDER_PAD).
DISP_SIZE = RANDOM_CROP + 2 * BORDER_PAD   # 240

# For raw uint8 images (e.g. webcam center-crops) that haven't been through the
# canonical pipeline, apply a simple resize + normalize before inference.
gray_inference_transforms = v2.Compose([
    v2.Resize((RANDOM_CROP, RANDOM_CROP)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

# Legacy: Resize-based transform for raw 640×480 synthetic images.
# Not used in the canonical pipeline (SyntheticDS now returns preprocessed tensors).
synth_inference_transforms = v2.Compose([
    v2.Resize((RANDOM_CROP, RANDOM_CROP)),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])


def decode_heatmaps(heatmaps, out_w, out_h, border_pad=0):
    """
    Soft argmax: (17, H, W) → (17, 2) coords in the target display space.

    border_pad: offset added after scaling, equal to BORDER_PAD when decoding
                into padded-image display space (target = RANDOM_CROP + 2*BORDER_PAD).
    """
    C, H, W = heatmaps.shape
    weights = torch.softmax(heatmaps.reshape(C, -1), dim=-1).reshape(C, H, W)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    cx = (weights * grid_x).sum(dim=(-2, -1)) * out_w + border_pad
    cy = (weights * grid_y).sum(dim=(-2, -1)) * out_h + border_pad
    return torch.stack([cx, cy], dim=-1)  # (17, 2)


def hard_argmax_2d(hm, out_w, out_h, border_pad=0):
    """
    Hard argmax: (17, H, W) array or tensor → (17, 2) numpy coords.

    border_pad: same semantics as decode_heatmaps — pass BORDER_PAD when
                decoding into padded-image display space.
    """
    if isinstance(hm, torch.Tensor):
        hm = hm.numpy()
    C, H, W = hm.shape
    flat_idx = hm.reshape(C, -1).argmax(axis=-1)
    ys = flat_idx // W
    xs = flat_idx % W
    return np.stack([
        xs * out_w / W + border_pad,
        ys * out_h / H + border_pad,
    ], axis=-1).astype(np.float32)


def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]), reverse=True,
    )
    for phase_dir in phase_dirs:
        ckpts = sorted(
            glob.glob(os.path.join(phase_dir, 'shinra_checkpoint_*.pth')),
            key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]),
        )
        if ckpts:
            return ckpts[-1]
    return None


def smooth_curve(points, degree=2):
    pts = np.asarray(points, dtype=np.float64)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs, ys = pts[:, 0], pts[:, 1]
    if len(xs) < 2:
        return pts
    deg = min(degree, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, deg)
    xx = np.linspace(xs.min(), xs.max(), 120)
    return np.column_stack([xx, np.polyval(coeffs, xx)])


def load_model():
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError('No phase_*/shinra_checkpoint_*.pth checkpoint found.')
    print(f'Loading checkpoint: {ckpt_path}')
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    shinra = ShinraCNN(backbone, out_channels=17).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    shinra.eval()
    return shinra


def infer(shinra, img_tensor, already_gray=False, already_preprocessed=False):
    """
    img_tensor: uint8 or float32 [C, H, W].

    already_preprocessed=True: tensor is already in the canonical pipeline format
        (center_crop → center_crop → normalize → pad → DISP_SIZE×DISP_SIZE float32).
        Passed directly to the model without any further transform.

    already_gray=True (and not already_preprocessed): applies gray_inference_transforms
        (resize + normalize), used for webcam crops.
    """
    if already_preprocessed:
        x = img_tensor.unsqueeze(0).to(device)
    else:
        transform = gray_inference_transforms if already_gray else synth_inference_transforms
        x = transform(img_tensor).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        (diameter_pred, _), (heatmap_logits, _), (gaze_vec, _) = shinra(x, decode=False)
    diameter = diameter_pred[0].squeeze().item()
    heatmaps = heatmap_logits[0]   # (17, 112, 112)
    gaze     = gaze_vec[0].numpy() # (3,)
    return diameter, heatmaps, gaze


def synthetic_mode(shinra):
    ds = SyntheticDS(transforms=None)
    print(f'Synthetic mode. Dataset size: {len(ds)}. Close window for next sample.')

    while True:
        idx = random.randrange(len(ds))
        # SyntheticDS(transforms=None) returns a canonically preprocessed DISP_SIZE×DISP_SIZE
        # float32 tensor — pass it directly to the model.
        raw_img, gt = ds[idx]
        img_np = raw_img.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            img_np = img_np[..., 0]

        diameter, heatmaps, gaze = infer(shinra, raw_img, already_preprocessed=True)

        # Decode heatmap peaks into padded-image display space (DISP_SIZE × DISP_SIZE).
        # out_w = RANDOM_CROP gives the 2× scale (heatmap is half-res of the crop),
        # border_pad = BORDER_PAD shifts into the padded coordinate frame.
        lm_px    = hard_argmax_2d(heatmaps,              RANDOM_CROP, RANDOM_CROP, border_pad=BORDER_PAD)
        gt_lm_px = hard_argmax_2d(gt['eye_heatmaps'],    RANDOM_CROP, RANDOM_CROP, border_pad=BORDER_PAD)

        pupil_px = lm_px[0]
        upper_px = lm_px[1:9]
        lower_px = lm_px[9:17]

        diam_px    = diameter          * MAX_IRIS_DIAMETER
        gt_diam_px = float(gt['pupil_diameter']) * MAX_IRIS_DIAMETER
        gt_pupil_px = gt_lm_px[0]

        mse_heatmaps = float(np.mean((lm_px - gt_lm_px) ** 2))
        mse_gaze     = float(np.mean((gaze - gt['gaze_vector'].numpy()) ** 2))

        _, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)

        ax.scatter(lm_px[1:, 0], lm_px[1:, 1],
                   c='black', s=35, zorder=4, edgecolors='white', linewidths=0.5)
        for lid_px in (upper_px, lower_px):
            curve = smooth_curve(lid_px)
            ax.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.6, zorder=3)

        ax.plot([gt_pupil_px[0] - gt_diam_px / 2, gt_pupil_px[0] + gt_diam_px / 2],
                [gt_pupil_px[1] - 6, gt_pupil_px[1] - 6],
                color='limegreen', linewidth=2.0, zorder=5)
        ax.plot([pupil_px[0] - diam_px / 2, pupil_px[0] + diam_px / 2],
                [pupil_px[1], pupil_px[1]],
                color='white', linewidth=2.5, zorder=5)

        ax.scatter([pupil_px[0]], [pupil_px[1]], c='blue', s=45, zorder=6,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(f'({pupil_px[0]:.0f}, {pupil_px[1]:.0f})',
                    (pupil_px[0], pupil_px[1]), xytext=(10, -10), textcoords='offset points',
                    color='blue', fontsize=10, fontweight='bold')

        gx, gy = gaze[0], (-gaze[1] if GAZE_FLIP_Y else gaze[1])
        ax.annotate('', xy=(pupil_px[0] + gx * GAZE_ARROW_LEN, pupil_px[1] + gy * GAZE_ARROW_LEN),
                    xytext=(pupil_px[0], pupil_px[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.2, mutation_scale=22))

        ax.set_xlim(0, DISP_SIZE)
        ax.set_ylim(DISP_SIZE, 0)
        ax.set_title(f'idx={idx}   (close window for next sample)')
        ax.axis('off')

        mse_text = (
            'Per-head MSE (pred vs ground truth)\n'
            f'  heatmaps  {mse_heatmaps:.4e} px²\n'
            f'  gaze      {mse_gaze:.4e}\n'
            f'  ── pred diameter  ── gt diameter'
        )
        ax.text(0.012, 0.985, mse_text,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, family='monospace', color='white',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=6))
        plt.tight_layout()
        plt.show()


def draw_overlays_cv2(frame, diameter, lm_px, gaze, origin_x, origin_y, crop_size):
    """lm_px: (17, 2) pixel coords in crop-local space."""
    lm_abs    = lm_px + np.array([origin_x, origin_y])
    pupil_abs = lm_abs[0]
    upper_abs = lm_abs[1:9]
    lower_abs = lm_abs[9:17]

    px, py  = pupil_abs
    diam_px = diameter * MAX_IRIS_DIAMETER * (crop_size / (640))

    for lid_pts in (upper_abs, lower_abs):
        curve = smooth_curve(lid_pts).astype(np.int32)
        cv2.polylines(frame, [curve], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    for (x, y) in lm_abs[1:]:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0),       -1, cv2.LINE_AA)

    cv2.line(frame,
             (int(px - diam_px / 2), int(py)),
             (int(px + diam_px / 2), int(py)),
             (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(frame, (int(px), int(py)), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(px), int(py)), 4, (255, 0, 0),     -1, cv2.LINE_AA)
    cv2.putText(frame, f'({int(px)}, {int(py)})',
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    gx, gy = gaze[0], (-gaze[1] if GAZE_FLIP_Y else gaze[1])
    cv2.arrowedLine(frame, (int(px), int(py)),
                    (int(px + gx * GAZE_ARROW_LEN), int(py + gy * GAZE_ARROW_LEN)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)


def webcam_mode(shinra, camera_idx):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {camera_idx}')
    print(f'Webcam mode (camera {camera_idx}). Press q to quit.')

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame from camera.')
            break
        H, W = frame.shape[:2]
        if W < WEBCAM_CROP or H < WEBCAM_CROP:
            print(f'Camera resolution {W}×{H} smaller than required {WEBCAM_CROP}×{WEBCAM_CROP} crop.')
            break

        x0 = (W - WEBCAM_CROP) // 2
        y0 = (H - WEBCAM_CROP) // 2

        crop_bgr  = frame[y0:y0 + WEBCAM_CROP, x0:x0 + WEBCAM_CROP]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        gray_tensor          = torch.from_numpy(crop_gray).unsqueeze(0)  # [1, H, W] uint8
        diameter, heatmaps, gaze = infer(shinra, gray_tensor, already_gray=True)
        lm_px = hard_argmax_2d(heatmaps, WEBCAM_CROP, WEBCAM_CROP)  # (17, 2) crop-local px

        max_hm = torch.sigmoid(heatmaps).max(dim=0).values.numpy()  # (112, 112) float32
        hm_u8  = (max_hm * 255).clip(0, 255).astype(np.uint8)
        hm_crop = cv2.resize(hm_u8, (WEBCAM_CROP, WEBCAM_CROP), interpolation=cv2.INTER_NEAREST)
        frame[y0:y0 + WEBCAM_CROP, x0:x0 + WEBCAM_CROP] = cv2.applyColorMap(hm_crop, cv2.COLORMAP_HOT)

        draw_overlays_cv2(frame, diameter, lm_px, gaze,
                          origin_x=x0, origin_y=y0, crop_size=WEBCAM_CROP)

        cv2.rectangle(frame, (x0, y0), (x0 + WEBCAM_CROP, y0 + WEBCAM_CROP),
                      (0, 255, 255), 1)
        cv2.putText(frame, 'q: quit', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Shinra-Meisin live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize Shinra-Meisin eye-tracking predictions.')
    parser.add_argument('--webcam', action='store_true',
                        help='Run on a live webcam feed (256×256 center crop) instead of synthetic samples.')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index for --webcam mode (default 0).')
    args = parser.parse_args()

    shinra = load_model()
    if args.webcam:
        webcam_mode(shinra, args.camera)
    else:
        synthetic_mode(shinra)


if __name__ == '__main__':
    main()
