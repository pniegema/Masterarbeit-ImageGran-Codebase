import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time
import subprocess
import tempfile
import shutil
import torch

# MediaPipe solutions
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Landmark indices from your existing code
POSE_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94
]

EXPR_NO_MOUTH_LANDMARKS = [
    285, 295, 282, 283, 276, 300, 293, 334, 296, 336,  # right brow
    55, 65, 52, 53, 46, 70, 63, 103, 66, 105,  # left brow
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  # right inner eye
    464, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 263,  # right outer eye
    33, 7, 163, 144, 145, 153, 154, 155, 246, 161, 160, 159, 158, 157, 173, 133, 155,  # left inner eye
    130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25,  # left outer eye
    104, 69, 108, 151, 337, 299, 333, 9,  # Forehead
    117, 118, 101, 36, 203, 206, 205, 50, 123, 147, 187, 207, 216, 212, 214, 192, 213,  # Left Cheek
    423, 266, 330, 347, 346, 352, 280, 425, 426, 436, 427, 411, 376, 432, 434, 416, 433  # Right Cheek
]

MOUTH_LANDMARKS = [
    0, 11, 12, 13, 14, 15, 16, 17, 37, 72, 38, 82, 87, 86, 85, 84, 39, 73, 41, 81, 178, 179, 180, 181,
    40, 74, 42, 80, 88, 89, 90, 91, 185, 62, 61, 146, 267, 302, 268, 312, 317, 316, 315, 314, 269, 303,
    271, 311, 402, 403, 404, 405, 270, 304, 272, 310, 318, 319, 320, 321, 409, 306, 375, 57, 287, 291
]

# Face regions for masking (from your existing code)
FACE_OVAL_IDX = list(set(range(468)))
CUSTOM_MOUTH_AND_JAW_IDX = [150, 169, 210, 202, 57, 186, 165, 97, 0, 326, 391, 410, 287, 422, 430, 394, 379, 378, 400,
                            377, 152, 148, 176, 149]


# Global instances for multiprocessing
_selfie_segmentation_instance = None
_face_mesh_instance = None


def get_landmark_points(landmarks, indices, w, h):
    return np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])


def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        idx = []
        for pt in pts:
            for i, p in enumerate(points):
                if np.linalg.norm(p - pt) < 1.0:
                    idx.append(i)
                    break
        if len(idx) == 3:
            delaunay_tri.append(tuple(idx))
    return delaunay_tri

def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in t_src]
    t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in t_dst]

    src_rect = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1, 1, 1))

    mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(src_rect, mat, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    dst_patch = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_patch * (1 - mask) + warped * mask


def init_mediapipe_instances(model_selection):
    global _selfie_segmentation_instance, _face_mesh_instance
    _selfie_segmentation_instance = mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)
    _face_mesh_instance = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )


def split_video_to_clips(video_path, temp_dir, max_duration=3.0, target_sample_rate=16000, target_fps=25):
    """
    Split a video into clips and extract audio at 16kHz, resample all clips to target FPS.
    Produces clean folder names like: onacY99veEQ_1_clip_0000
    """
    clips = []

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    frames_per_clip = int(target_fps * max_duration)
    num_clips = int(np.ceil(duration / max_duration))

    # Base name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(num_clips):
        start_time = i * max_duration
        clip_duration = min(max_duration, duration - start_time)

        # Clean clip folder
        clip_dir = os.path.join(temp_dir, f"{video_name}_clip_{i:04d}")
        os.makedirs(clip_dir, exist_ok=True)

        # Extract video clip at target FPS
        video_clip_path = os.path.join(clip_dir, "clip.mp4")
        ffmpeg_video_cmd = [
            'ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(clip_duration),
            '-r', str(target_fps),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-an',  # no audio
            '-y', video_clip_path
        ]
        subprocess.run(ffmpeg_video_cmd, capture_output=True, check=True)

        # Extract audio for this clip
        audio_path = os.path.join(clip_dir, "audio.wav")
        ffmpeg_audio_cmd = [
            'ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(clip_duration),
            '-ar', str(target_sample_rate), '-ac', '1', '-vn', '-y', audio_path
        ]
        subprocess.run(ffmpeg_audio_cmd, capture_output=True, check=True)

        clips.append({
            'clip_dir': clip_dir,
            'video_path': video_clip_path,
            'start_time': start_time,
            'clip_duration': clip_duration,
            'audio_path': audio_path
        })

    return clips



def create_mask(image_shape, landmarks, indices):
    """Create mask for specific landmark indices"""
    h, w = image_shape[:2]
    points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])

    if points.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)

    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def apply_mask(image, mask):
    """Apply mask to image"""
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, 3, axis=-1)
    return cv2.bitwise_and(image, mask)


def apply_inverse_mask(image, mask):
    """Apply inverse mask to image"""
    inverse_mask = cv2.bitwise_not(mask)
    return apply_mask(image, inverse_mask)


def extract_landmarks(image_rgb, face_mesh):
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def create_heatmap_vectorized(image_shape, landmarks, indices, sigma=5):
    """Vectorized Gaussian heatmap for landmarks"""
    h, w = image_shape[:2]
    Y, X = np.ogrid[:h, :w]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for idx in indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        heatmap = np.maximum(heatmap, np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2)))

    return (heatmap * 255).astype(np.uint8)



def save_landmarks_npz(out_path, expr_landmarks, pose_landmarks, mouth_landmarks):
    """Save landmarks as compressed numpy arrays"""
    expr_xy = np.array([[lm.x, lm.y] for lm in expr_landmarks], dtype=np.float32)
    pose_xy = np.array([[lm.x, lm.y] for lm in pose_landmarks], dtype=np.float32)
    mouth_xy = np.array([[lm.x, lm.y] for lm in mouth_landmarks], dtype=np.float32)
    np.savez_compressed(out_path, expr=expr_xy, pose=pose_xy, mouth=mouth_xy)





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

def process_clip(clip_info, output_root, mouth_offset=5):
    """Faster version of process_clip with streaming, vectorized heatmaps, and single RGB conversion"""
    global _selfie_segmentation_instance, _face_mesh_instance

    clip_dir = clip_info['clip_dir']
    video_path = clip_info['video_path']

    # Extract base clip name (e.g., "onacY99veEQ_1") from original video path
    video_name = os.path.splitext(os.path.basename(clip_info['clip_dir'].rsplit('_clip_', 1)[0]))[0]

    # Extract clip index like "0000" from clip_dir path suffix
    clip_index = clip_info['clip_dir'].rsplit('_clip_', 1)[-1]

    # Construct output folder as desired:
    # output_root/clip_name/clip_name_clip_0000/
    output_clip_dir = os.path.join(output_root, video_name, f"{video_name}_clip_{clip_index}")

    # Create subdirectories inside this folder
    dirs = {
        'pose': os.path.join(output_clip_dir, 'pose'),
        'expression': os.path.join(output_clip_dir, 'expression'),
        'mouth': os.path.join(output_clip_dir, 'mouth'),
        'identity': os.path.join(output_clip_dir, 'identity'),
        'landmarks': os.path.join(output_clip_dir, 'landmarks'),
        'heatmaps': os.path.join(output_clip_dir, 'heatmaps')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy audio
    if os.path.exists(clip_info['audio_path']):
        shutil.copy2(clip_info['audio_path'], os.path.join(output_clip_dir, 'audio.wav'))

    try:
        cap = cv2.VideoCapture(video_path)
        future_frames_buffer = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            results_landmarks = _face_mesh_instance.process(frame_rgb)
            if not results_landmarks.multi_face_landmarks:
                frame_idx += 1
                continue
            landmarks = results_landmarks.multi_face_landmarks[0].landmark

            # Selfie segmentation
            seg_results = _selfie_segmentation_instance.process(frame_rgb)
            if seg_results.segmentation_mask is None:
                frame_idx += 1
                continue
            binary_mask = (seg_results.segmentation_mask > 0.5).astype(np.uint8) * 255
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            segmented_frame = apply_mask(frame, binary_mask)

            # Masks for regions
            full_face_mask = create_mask(frame.shape, landmarks, FACE_OVAL_IDX)
            mouth_mask = create_mask(frame.shape, landmarks, CUSTOM_MOUTH_AND_JAW_IDX)

            pose_frame = apply_inverse_mask(segmented_frame.copy(), full_face_mask)
            pose_frame = cv2.resize(pose_frame, (128, 128))
            expression_frame = apply_mask(frame, full_face_mask)
            mouth_frame = apply_mask(frame, mouth_mask)
            identity_frame = segmented_frame.copy()
            stitched = expression_frame.copy()

            # Handle mouth offset (future frame)
            if len(future_frames_buffer) >= mouth_offset:
                future_frame, future_landmarks = future_frames_buffer.pop(0)

                points1 = get_landmark_points(future_landmarks, CUSTOM_MOUTH_AND_JAW_IDX, w, h)
                points1_clamped = points1.copy()
                points1_clamped[:, 0] = np.clip(points1_clamped[:, 0], 0, w - 1)
                points1_clamped[:, 1] = np.clip(points1_clamped[:, 1], 0, h - 1)
                points1 = points1_clamped

                points2 = get_landmark_points(landmarks, CUSTOM_MOUTH_AND_JAW_IDX, w, h)
                points2_clamped = points2.copy()
                points2_clamped[:, 0] = np.clip(points2_clamped[:, 0], 0, w - 1)
                points2_clamped[:, 1] = np.clip(points2_clamped[:, 1], 0, h - 1)
                points2 = points2_clamped

                # Warp mouth triangles if valid
                if points1.shape[0] >= 3 and points2.shape[0] >= 3:
                    rect = (0, 0, w, h)
                    triangles = get_delaunay_triangles(rect, points2)
                    for tri in triangles:
                        t1 = [points1[tri[0]], points1[tri[1]], points1[tri[2]]]
                        t2 = [points2[tri[0]], points2[tri[1]], points2[tri[2]]]
                        warp_triangle(future_frame, stitched, t1, t2)

                    future_mouth_mask = create_mask(future_frame.shape, future_landmarks, CUSTOM_MOUTH_AND_JAW_IDX)
                    future_mouth_region = apply_mask(future_frame, future_mouth_mask)
                    mouth_area = mouth_mask > 0
                    expression_frame[mouth_area] = future_mouth_region[mouth_area]

            # Save current frame to future buffer
            future_frames_buffer.append((frame.copy(), landmarks))

            # Extract and save landmarks
            expr_landmarks = [landmarks[j] for j in EXPR_NO_MOUTH_LANDMARKS]
            mouth_landmarks = [landmarks[j] for j in MOUTH_LANDMARKS]
            pose_landmarks = [landmarks[j] for j in POSE_LANDMARKS]

            expr_xy = np.array([[lm.x, lm.y] for lm in expr_landmarks], dtype=np.float32)
            pose_xy = np.array([[lm.x, lm.y] for lm in pose_landmarks], dtype=np.float32)
            mouth_xy = np.array([[lm.x, lm.y] for lm in mouth_landmarks], dtype=np.float32)
            landmarks_path = os.path.join(dirs['landmarks'], f"landmarks_{frame_idx:04d}.npz")
            np.savez_compressed(landmarks_path, expr=expr_xy, pose=pose_xy, mouth=mouth_xy)

            # Generate heatmaps
            pose_heatmap = landmarks_to_heatmaps(pose_xy, H=64, W=64, sigma=2.5, device=device)
            mouth_heatmap = landmarks_to_heatmaps(mouth_xy, H=64, W=64, sigma=2.5, device=device)

            base = f"{frame_idx:04d}"
            # Save outputs
            cv2.imwrite(os.path.join(dirs['pose'], f"{base}.jpg"), pose_frame)
            cv2.imwrite(os.path.join(dirs['expression'], f"{base}.jpg"), stitched)
            cv2.imwrite(os.path.join(dirs['mouth'], f"{base}.jpg"), mouth_frame)
            cv2.imwrite(os.path.join(dirs['identity'], f"{base}.jpg"), identity_frame)

            np.savez_compressed(os.path.join(dirs['heatmaps'], f"{base}_pose.npz"),
                                heatmaps=(pose_heatmap.cpu().numpy() * 255).astype(np.uint8))
            np.savez_compressed(os.path.join(dirs['heatmaps'], f"{base}_mouth.npz"),
                                heatmaps=(mouth_heatmap.cpu().numpy() * 255).astype(np.uint8))

            frame_idx += 1

        cap.release()
        return f"[OK] Processed {frame_idx} frames from {clip_dir}"

    except Exception as e:
        return f"[Error] {clip_dir}: {str(e)}"


def process_video_worker(args):
    """Worker function for processing a single video"""
    video_path, output_root, model_selection = args

    try:
        # Create temporary directory for this video
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split video into clips
            clips = split_video_to_clips(video_path, temp_dir)

            results = []
            for clip_info in clips:
                result = process_clip(clip_info, output_root)
                results.append(result)

            return results

    except Exception as e:
        return [f"[Error] Video {video_path}: {str(e)}"]


def process_videos_parallel(input_dir, output_dir, workers=4, model_selection=0):
    """Process all videos in parallel"""

    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} video files to process.")

    if not video_files:
        print("No video files found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_mediapipe_instances,
            initargs=(model_selection,)
    ) as executor:

        tasks = [(video_path, output_dir, model_selection) for video_path in video_files]

        results = list(tqdm(
            executor.map(process_video_worker, tasks),
            total=len(video_files),
            desc="Processing videos"
        ))

        # Print results
        for video_results in results:
            for result in video_results:
                if result.startswith("[Error]"):
                    print(result)

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    input_videos_dir = r"C:\Users\pnieg\Documents\HDTF\HDTF"
    output_dir = r"C:\Users\pnieg\Documents\HDTF\HDTF_datatset"

    # Adjust worker count based on your system
    # Note: Each worker will use significant RAM due to MediaPipe models + FFmpeg processing
    num_workers = 8
    target_fps = 25

    process_videos_parallel(input_videos_dir, output_dir, workers=num_workers, model_selection=0)