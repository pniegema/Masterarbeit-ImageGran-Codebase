import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import pickle
from landmarks_zu_Heamaps import landmarks_to_heatmaps_refined, visualize_single_heatmap
import cv2
from scipy.ndimage import rotate


class Hybrid_Datensatz(Dataset):
    def __init__(self, root_dir, transform=None, min_frames=10, cache_file="dataset_cache.pkl"):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.cache_file = cache_file
        self.scale_tensor = torch.tensor([127.0, 127.0], dtype=torch.float32)

        if os.path.exists(cache_file):
            print(f"Loading dataset cache from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.clips = cache["clips"]
            self.valid_frame_ids = cache["valid_frame_ids"]
            print(f"Loaded {len(self.clips)} clips from cache.")
        else:
            self.clips = []
            self.valid_frame_ids = {}
            skip_count = 0
            count = 0

            print("Initializing dataset...")
            for video_folder in sorted(os.listdir(root_dir)):
                video_path = os.path.join(root_dir, video_folder)
                if not os.path.isdir(video_path):
                    continue

                for clip in sorted(os.listdir(video_path)):
                    clip_path = os.path.join(video_path, clip)
                    if not os.path.isdir(clip_path):
                        continue

                    identity_dir = os.path.join(clip_path, "identity")
                    landmarks_dir = os.path.join(clip_path, "landmarks")
                    heatmaps_dir = os.path.join(clip_path, "heatmaps")

                    # Skip if required subdirs missing
                    if not all(os.path.isdir(p) for p in [identity_dir, landmarks_dir, heatmaps_dir]):
                        skip_count += 1
                        continue

                    identity_emb_path = os.path.join(clip_path, "identity_emb.npy")
                    if not os.path.isfile(identity_emb_path):
                        skip_count += 1
                        continue

                    identity_files = [f for f in os.listdir(identity_dir) if f.endswith(".jpg")]
                    landmarks_files = [f for f in os.listdir(landmarks_dir) if f.endswith(".npz")]

                    identity_ids = set(os.path.splitext(f)[0] for f in identity_files)
                    landmarks_ids = set(os.path.splitext(f)[0].replace("landmarks_", "") for f in landmarks_files)
                    common_ids = sorted(identity_ids & landmarks_ids)

                    # Remove frames without combined heatmaps
                    valid_ids = []
                    for fid in common_ids:
                        combined_hm_path = os.path.join(heatmaps_dir, f"{fid}_combined.npz")
                        if os.path.isfile(combined_hm_path):
                            valid_ids.append(fid)

                    if len(valid_ids) < min_frames:
                        skip_count += 1
                        continue

                    self.clips.append(clip_path)
                    self.valid_frame_ids[clip_path] = valid_ids

                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} clips...")
                        print(f"  Skipped clips so far: {skip_count}")

            print(f"Saving dataset cache to {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {"clips": self.clips, "valid_frame_ids": self.valid_frame_ids},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        print(f"Dataset initialized with {len(self.clips)} clips having valid frames.")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        valid_frames = self.valid_frame_ids[clip_path]

        # Select driving frame
        driving_frame_id = random.choice(valid_frames)
        frame_id_str = f"{int(driving_frame_id):04d}"

        # --- Identity frames ---
        n_id = min(1, len(valid_frames))
        id_sample_ids = random.sample(valid_frames, n_id)
        identity_frames = []
        for frame_id in id_sample_ids:
            img_path = self.identity_image_paths[clip_path][valid_frames.index(frame_id)]
            with Image.open(img_path) as img:
                identity_frames.append(self.transform(img.convert("RGB")))
        identity_frames = torch.stack(identity_frames)
        del img

        # --- Load driving frame image ---
        driving_img_path = os.path.join(clip_path, "identity", f"{frame_id_str}.jpg")
        with Image.open(driving_img_path) as img:
            driving_frame = self.transform(img.convert("RGB"))

        # --- Load pre-computed heatmaps (combined file) ---
        heatmaps_dir = os.path.join(clip_path, "heatmaps")
        combined_hm_path = os.path.join(heatmaps_dir, f"{frame_id_str}_combined.npz")

        # Load both heatmaps from single file
        heatmap_data = np.load(combined_hm_path, allow_pickle=False)
        mouth_hm = heatmap_data["mouth"].astype(np.float32) / 255.0
        pose_expr_hm = heatmap_data["pose_and_expr"]
        pose_hm = pose_expr_hm[:47]
        heatmap_data.close()

        # --- Load landmarks ---
        landmarks_path = os.path.join(clip_path, "landmarks", f"landmarks_{frame_id_str}.npz")
        landmarks_data = np.load(landmarks_path, allow_pickle=False)
        mouth = landmarks_data["mouth"]
        expr = landmarks_data["expr"]
        landmarks_data.close()

        expr_tensor = torch.from_numpy(expr).float()

        # Generate expression heatmap with mode='mouth' (centering/normalization happens inside)
        expression_hm = landmarks_to_heatmaps_refined(expr_tensor, H=64, W=64, sigma=2.5, mode='mouth')
        expression_hm = expression_hm.numpy()

        # Now rotate the expression heatmap
        angle = np.random.uniform(-30, 30)
        rotated_expr = np.zeros_like(expression_hm)
        for i in range(expression_hm.shape[0]):
            rotated_expr[i] = rotate(expression_hm[i], angle, reshape=False, order=1, mode='constant', cval=0)
        expression_hm = rotated_expr

        # Rotate mouth heatmap (keep as before)
        angle = np.random.uniform(-30, 30)
        rotated = np.zeros_like(mouth_hm)
        for i in range(mouth_hm.shape[0]):
            rotated[i] = rotate(mouth_hm[i], angle, reshape=False, order=1, mode='constant', cval=0)
        mouth_hm = rotated

        # Force contiguous arrays
        if not mouth_hm.flags['C_CONTIGUOUS']:
            mouth_hm = np.ascontiguousarray(mouth_hm)
        if not expression_hm.flags['C_CONTIGUOUS']:
            expression_hm = np.ascontiguousarray(expression_hm)

        # Pad/truncate mouth_hm
        expected_mouth_channels = 66
        if mouth_hm.shape[0] != expected_mouth_channels:
            temp = np.zeros((expected_mouth_channels, 64, 64), dtype=np.float32)
            copy_size = min(mouth_hm.shape[0], expected_mouth_channels)
            temp[:copy_size] = mouth_hm[:copy_size]
            mouth_hm = temp

        expected_pose_channels = 47
        if pose_hm.shape[0] != expected_pose_channels:
            temp = np.zeros((expected_pose_channels, 64, 64), dtype=np.float32)
            copy_size = min(pose_hm.shape[0], expected_pose_channels)
            temp[:copy_size] = pose_hm[:copy_size]
            pose_hm = temp

        expected_expression_channels = 95
        if expression_hm.shape[0] != expected_expression_channels:
            temp = np.zeros((expected_expression_channels, 64, 64), dtype=np.float32)
            copy_size = min(expression_hm.shape[0], expected_expression_channels)
            temp[:copy_size] = expression_hm[:copy_size]
            expression_hm = temp

        # Convert to tensors
        driving_mouth_hm = torch.from_numpy(mouth_hm).float()
        driving_pose_hm = torch.from_numpy(pose_hm).float()
        driving_expression_hm = torch.from_numpy(expression_hm).float()

        driving_landmarks = {"mouth": mouth, "expr": expr}

        real_frame_id = random.choice(valid_frames)
        real_frame_id_str = f"{int(real_frame_id):04d}"
        real_img_path = os.path.join(clip_path, "identity", f"{real_frame_id_str}.jpg")
        with Image.open(real_img_path) as img:
            real_image = self.transform(img.convert("RGB"))

        return {
            "identity_frames": None,
            "identity_embedding": identity_frames,
            "driving_mouth_hm": driving_mouth_hm,
            "driving_frame": driving_frame,
            "driving_landmarks": driving_landmarks,
            "real_image": real_image,
            "driving_pose_hm": driving_pose_hm,
            "driving_expression_hm": driving_expression_hm,
        }



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

def average_brightness(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return 0  # Treat missing or unreadable image as very dark
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to model input size
        transforms.ToTensor(),  # scale to [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])  # → [-1,1]
    ])


    start_time = time.time()
    dataset = Hybrid_Datensatz(
        root_dir=r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset_test",
        transform=transform,
        min_frames=10,
        cache_file="dataset_cache_HDTF_with_id_embeddings_test.pkl"

    )
    end_time = time.time()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    #mismatch_count = 0


    for i, sample in enumerate(dataloader):
        pose_hm = sample['driving_pose_hm']
        expression_hm = sample['driving_expression_hm']
        mouth_heatmaps = sample['driving_mouth_hm']
        print(f"Batch {i} mouth heatmap shape: {mouth_heatmaps.shape}")
        print(f"Batch {i} pose heatmap shape: {pose_hm.shape}")
        print(f"Batch {i} expression heatmap shape: {expression_hm.shape}")
        visualize_single_heatmap(mouth_heatmaps[0], title="Sample Mouth Heatmaps")
        visualize_single_heatmap(expression_hm[0], title="Sample Expression Heatmaps")
        visualize_single_heatmap(pose_hm[0], title="Sample Pose Heatmaps")




        if i >= 3:
            break








