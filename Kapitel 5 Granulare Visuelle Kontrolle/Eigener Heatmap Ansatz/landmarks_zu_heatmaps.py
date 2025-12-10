
import os
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
import cv2
from matplotlib import pyplot as plt



def landmarks_to_heatmaps(points, H=64, W=64, sigma=2.5, device='cuda'):
    """
    Convert given normalized landmark points to heatmaps.

    Args:
        points (np.ndarray or torch.Tensor): shape (N, 2), normalized (x,y) in [0,1].
        H (int): height of heatmaps.
        W (int): width of heatmaps.
        sigma (float): Gaussian radius.
        device (str): torch device.

    Returns:
        torch.Tensor: heatmaps (N, H, W)
    """
    if isinstance(points, np.ndarray):
        pts = torch.tensor(points, device=device, dtype=torch.float32)
    else:
        pts = points.to(device).float()

    # Normalize points to bounding box [0,1] internally (optional, depending on use case)
    min_xy, _ = pts.min(dim=0)
    max_xy, _ = pts.max(dim=0)
    size = torch.clamp(max_xy - min_xy, min=1e-6)
    pts = (pts - min_xy) / size

    # Scale to heatmap coordinates
    pts[:, 0] *= (W - 1)
    pts[:, 1] *= (H - 1)

    xx = torch.arange(W, device=device).view(1, 1, W).expand(len(pts), H, W)
    yy = torch.arange(H, device=device).view(1, H, 1).expand(len(pts), H, W)

    heatmaps = torch.exp(-((xx - pts[:, 0].view(-1, 1, 1)) ** 2 + (yy - pts[:, 1].view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))

    return heatmaps

def process_landmark_file(npz_path, output_dir, H=64, W=64, sigma=2.5):
    """Load saved expr/pose/mouth landmarks and generate separate heatmaps."""
    data = np.load(npz_path)

    expr = data["expr"]   # (N_expr, 2)
    pose = data["pose"]   # (N_pose, 2)
    mouth = data["mouth"] # (N_mouth, 2)

    #Combine Pose and Expression
    pose_and_expr = np.concatenate([pose, expr], axis=0)

    # Generate heatmaps
    mouth_hm = landmarks_to_heatmaps_refined(torch.tensor(mouth), mode='mouth', H=64, W=64, sigma=sigma)
    pose_and_expr_hm = landmarks_to_heatmaps_refined(torch.tensor(pose_and_expr), mode='absolute', H=64, W=64, sigma=sigma)

    # Save compressed npz
    base = os.path.splitext(os.path.basename(npz_path))[0].replace("landmarks_", "")
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(os.path.join(output_dir, f"{base}_pose_and_expr.npz"),
                        heatmaps=(pose_and_expr_hm.cpu().numpy() * 255).astype(np.uint8))
    np.savez_compressed(os.path.join(output_dir, f"{base}_mouth.npz"),
                        heatmaps=(mouth_hm.cpu().numpy() * 255).astype(np.uint8))

def visualize_saved_heatmap(npz_path, image=None, alpha=0.5, figsize=(6, 6)):
    """
    Load a saved .pt heatmap tensor and visualize it.

    Args:
        pt_path: str, path to the .pt file
        image: np.ndarray, optional background image (H,W,3) BGR
        alpha: float, blending factor if image is provided
        figsize: tuple, matplotlib figure size
    """
    # Load tensor
    # Load heatmaps
    data = np.load(npz_path)
    heatmaps = data['heatmaps']  # shape (N, H, W), dtype=uint8
    heatmaps = heatmaps.astype(np.float32) / 255.0  # scale back to [0,1]
    print(heatmaps.shape)


    # Collapse across channels for visualization
    combined_heatmap = heatmaps.sum(axis=0)
    combined_heatmap = combined_heatmap / (combined_heatmap.max() + 1e-8)

    H, W = combined_heatmap.shape
    heatmap_img = (combined_heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    if image is not None:
        # Resize heatmap to match image
        img_h, img_w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap_color, (img_w, img_h))
        overlay = cv2.addWeighted(image, 1.0, heatmap_resized, alpha, 0)
    else:
        overlay = heatmap_color

    # Display
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def process_all_landmarks(dataset_root, num_workers=None):
    """Process all landmark files in all clips in-place."""
    landmark_files = []

    # Iterate over every clip folder
    for clip_name in tqdm(os.listdir(dataset_root), desc="Processing clips"):
        clip_path = os.path.join(dataset_root, clip_name)
        if not os.path.isdir(clip_path):
            continue

        for subfolder_name in os.listdir(clip_path):
            subfolder_path = os.path.join(clip_path, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue

            landmarks_dir = os.path.join(subfolder_path, "landmarks")
            if not os.path.exists(landmarks_dir):
                continue

            heatmaps_dir = os.path.join(subfolder_path, "heatmaps")  # same level as landmarks
            for f in os.listdir(landmarks_dir):
                if f.endswith(".npz") and "landmarks" in f:
                    landmark_files.append((os.path.join(landmarks_dir, f), heatmaps_dir))

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(landmark_files))

    print(f"Starting with {num_workers} workers...")
    with mp.Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_landmark_file_wrapper, landmark_files),
                      total=len(landmark_files), desc="Generating heatmaps"):
            pass

def process_landmark_file_wrapper(args):
    return process_landmark_file(*args)

_grid_cache = {}


def landmarks_to_heatmaps_refined(points, H=64, W=64, sigma=2.5, mode='absolute'):
    """
    Convert landmarks to heatmaps with different normalization strategies.

    Args:
        points: (N, 2) normalized coordinates in [0,1]
        mode: 'absolute' (pose_and_expr), 'mouth' (mouth shape)
    """
    pts = points.clone()

    if mode == 'mouth':
        center = pts.mean(dim=0, keepdim=True)
        pts = pts - center
        max_dist = pts.abs().max()
        if max_dist > 1e-6:
            pts = pts * (0.5 / max_dist)
        pts = pts + 0.5

    pts = pts * torch.tensor([W - 1, H - 1], device=pts.device, dtype=pts.dtype)

    inv_2sigma2 = -0.5 / (sigma * sigma)

    # Cache coordinate grids
    cache_key = (H, W, pts.device, pts.dtype)
    if cache_key not in _grid_cache:
        xx = torch.arange(W, device=pts.device, dtype=pts.dtype).view(1, 1, W)
        yy = torch.arange(H, device=pts.device, dtype=pts.dtype).view(1, H, 1)
        _grid_cache[cache_key] = (xx, yy)

    xx, yy = _grid_cache[cache_key]

    dx = xx - pts[:, 0].view(-1, 1, 1)
    dy = yy - pts[:, 1].view(-1, 1, 1)

    heatmaps = torch.exp((dx.square() + dy.square()) * inv_2sigma2)

    return heatmaps

def landmarks_to_heatmaps_refined_old(points, H=128, W=128, sigma=2.5, mode='absolute'):
    """
    Convert landmarks to heatmaps with different normalization strategies.

    Args:
        points: (N, 2) normalized coordinates in [0,1]
        mode: 'absolute' (pose), 'centered' (expression), 'mouth' (mouth shape)
    """
    pts = points.clone()

    if mode == 'absolute':
        # No normalization - keep absolute positions for pose
        pass

    elif mode == 'centered':
        # Center around mean, gentle scaling for expression
        center = pts.mean(dim=0)
        pts = pts - center
        std = pts.std(dim=0).mean()
        if std > 1e-6:
            pts = pts / (std * 3)
        pts = pts + 0.5
        pts = torch.clamp(pts, 0, 1)

    elif mode == 'mouth':
        # Center and normalize for mouth shape
        center = pts.mean(dim=0)
        pts = pts - center
        max_dist = torch.abs(pts).max()
        if max_dist > 1e-6:
            pts = pts / (max_dist * 2)
        pts = pts + 0.5

    # Scale to heatmap coordinates
    pts[:, 0] *= (W - 1)
    pts[:, 1] *= (H - 1)

    xx = torch.arange(W, device=pts.device).view(1, 1, W).expand(len(pts), H, W)
    yy = torch.arange(H, device=pts.device).view(1, H, 1).expand(len(pts), H, W)

    heatmaps = torch.exp(-((xx - pts[:, 0].view(-1, 1, 1)) ** 2 +
                           (yy - pts[:, 1].view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))

    return heatmaps

def visualize_saved_landmarks(npz_path):
    data = np.load(npz_path)

    expr = data["expr"]  # (N_expr, 2)
    #expr = np.concatenate([expr[37:53], expr[71:86]], axis=0)
    print(expr.shape)
    pose = data["pose"]  # (N_pose, 2)
    mouth = data["mouth"]  # (N_mouth, 2)

    plt.figure(figsize=(6, 6))
    if expr.shape[0] > 0:
        plt.scatter(expr[:, 0], expr[:, 1], c='r', label='Expression', alpha=0.6)
    if pose.shape[0] > 0:
        plt.scatter(pose[:, 0], pose[:, 1], c='g', label='Pose', alpha=0.6)
    if mouth.shape[0] > 0:
        plt.scatter(mouth[:, 0], mouth[:, 1], c='b', label='Mouth', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.legend()
    plt.title('Landmark Visualization')
    plt.show()


def visualize_single_heatmap(heatmap, title="Heatmap", colormap='jet',
                             figsize=(6, 6), show_colorbar=True):
    """
    Visualize a single heatmap or combined heatmaps.

    Args:
        heatmap (torch.Tensor or np.ndarray):
            - Shape (H, W) for single heatmap
            - Shape (N, H, W) for multiple heatmaps (will be combined)
        title (str): Title for the plot
        colormap (str): 'jet', 'hot', 'viridis', etc.
        figsize (tuple): Figure size
        show_colorbar (bool): Whether to show colorbar
    """
    # Convert to numpy if needed
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    # If multiple heatmaps, combine them
    if heatmap.ndim == 3:
        combined = heatmap.sum(axis=0)
        # Normalize
        combined = combined / (combined.max() + 1e-8)
    else:
        combined = heatmap

    plt.figure(figsize=figsize)
    plt.imshow(combined, cmap=colormap, interpolation='bilinear')
    if show_colorbar:
        plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def rotate_landmarks(pts, angle_deg):
    """
    Rotate landmarks around their center point.

    Args:
        pts: (N, 2) numpy array of landmarks
        angle_deg: rotation angle in degrees

    Returns:
        Rotated landmarks, clipped to [0, 1]
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Center landmarks
    center = pts.mean(axis=0)
    pts_centered = pts - center

    # Apply rotation
    x_rot = pts_centered[:, 0] * cos_a - pts_centered[:, 1] * sin_a
    y_rot = pts_centered[:, 0] * sin_a + pts_centered[:, 1] * cos_a

    # Recenter and clip
    pts_rotated = np.stack([x_rot, y_rot], axis=1) + center
    return np.clip(pts_rotated, 0, 1)


def visualize_landmarks_on_image(npz_path, image_path):
    # Bild laden
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

    # Landmarks laden
    data = np.load(npz_path)
    expr = data["expr"]
    pose = data["pose"]
    mouth = data["mouth"]

    h, w, _ = image.shape

    # Landmarks auf Bildmaßstab skalieren (0-1 → Pixel)
    expr_img = expr * [w, h]
    pose_img = pose * [w, h]
    mouth_img = mouth * [w, h]

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    if expr_img.shape[0] > 0:
        plt.scatter(expr_img[:, 0], expr_img[:, 1], c='r', label='Expression', alpha=0.6)
    if pose_img.shape[0] > 0:
        plt.scatter(pose_img[:, 0], pose_img[:, 1], c='g', label='Pose', alpha=0.6)
    if mouth_img.shape[0] > 0:
        plt.scatter(mouth_img[:, 0], mouth_img[:, 1], c='b', label='Mouth', alpha=0.6)

    plt.legend()
    plt.title('Landmarks on Image')
    plt.axis('off')
    plt.show()


