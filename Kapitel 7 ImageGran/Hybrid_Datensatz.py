import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import pickle
from utils import landmarks_to_heatmaps_refined, visualize_single_heatmap
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

        # --- Load identity embedding (mmap, no copy) ---
        identity_emb_path = os.path.join(clip_path, "identity_emb.npy")
        identity_embedding = torch.from_numpy(
            np.load(identity_emb_path, mmap_mode="r")
        ).float()

        # --- Load driving frame image ---
        driving_img_path = os.path.join(clip_path, "identity", f"{frame_id_str}.jpg")
        with Image.open(driving_img_path) as img:
            driving_frame = self.transform(img.convert("RGB"))

        # --- Load pre-computed heatmaps (combined file) ---
        heatmaps_dir = os.path.join(clip_path, "heatmaps")
        combined_hm_path = os.path.join(heatmaps_dir, f"{frame_id_str}_combined.npz")

        # Load both heatmaps from single file
        heatmap_data = np.load(combined_hm_path, allow_pickle=False)
        mouth_hm = heatmap_data["mouth"].astype(np.float32) / 255.0  # Convert to float BEFORE rotation
        pose_expr_hm = heatmap_data["pose_and_expr"]
        heatmap_data.close()

        # Kleine zufällige Rotation ±30° (now working with float values)
        angle = np.random.uniform(-30, 30)
        rotated = np.zeros_like(mouth_hm)
        for i in range(mouth_hm.shape[0]):
            rotated[i] = rotate(mouth_hm[i], angle, reshape=False, order=1, mode='constant', cval=0)
        mouth_hm = rotated

        # Force contiguous arrays for faster GPU transfer
        if not mouth_hm.flags['C_CONTIGUOUS']:
            mouth_hm = np.ascontiguousarray(mouth_hm)
        if not pose_expr_hm.flags['C_CONTIGUOUS']:
            pose_expr_hm = np.ascontiguousarray(pose_expr_hm)

        # Pad/truncate mouth_hm (already float)
        expected_mouth_channels = 66
        if mouth_hm.shape[0] != expected_mouth_channels:
            temp = np.zeros((expected_mouth_channels, 64, 64), dtype=np.float32)
            copy_size = min(mouth_hm.shape[0], expected_mouth_channels)
            temp[:copy_size] = mouth_hm[:copy_size]
            mouth_hm = temp

        # Pad/truncate pose_expr_hm (still uint8)
        expected_pose_expr_channels = 175
        if pose_expr_hm.shape[0] != expected_pose_expr_channels:
            temp = np.zeros((expected_pose_expr_channels, 64, 64), dtype=np.uint8)
            copy_size = min(pose_expr_hm.shape[0], expected_pose_expr_channels)
            temp[:copy_size] = pose_expr_hm[:copy_size]
            pose_expr_hm = temp

        # Convert to tensors
        driving_mouth_hm = torch.from_numpy(mouth_hm).float()  # Already in [0,1]
        driving_pose_and_expr_hm = torch.from_numpy(pose_expr_hm).float().div_(255.0)

        # --- Load only mouth landmarks ---
        landmarks_path = os.path.join(clip_path, "landmarks", f"landmarks_{frame_id_str}.npz")
        landmarks_data = np.load(landmarks_path, allow_pickle=False)
        mouth = landmarks_data["mouth"]
        expr = landmarks_data["expr"]
        driving_landmarks = {"mouth": mouth, "expr": expr}
        landmarks_data.close()

        real_frame_id = random.choice(valid_frames)
        real_frame_id_str = f"{int(real_frame_id):04d}"
        real_img_path = os.path.join(clip_path, "identity", f"{real_frame_id_str}.jpg")
        with Image.open(real_img_path) as img:
            real_image = self.transform(img.convert("RGB"))

        return {
            "identity_embedding": identity_embedding,
            "driving_pose_and_expr": driving_pose_and_expr_hm,
            "driving_mouth_hm": driving_mouth_hm,
            "driving_frame": driving_frame,
            "driving_landmarks": driving_landmarks,
            "real_image": real_image,
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
        #Examine first batch identity embeddings
        identity_emb = sample['identity_embedding']
        print(f"Batch {i} identity embedding shape: {identity_emb.shape}")
        pose_expr_heatmaps = sample['driving_pose_and_expr']
        mouth_heatmaps = sample['driving_mouth_hm']
        print(f"Batch {i} pose heatmap shape: {pose_expr_heatmaps.shape}")
        print(f"Batch {i} mouth heatmap shape: {mouth_heatmaps.shape}")
        visualize_single_heatmap(pose_expr_heatmaps[0], title="Sample Pose Heatmaps")
        visualize_single_heatmap(mouth_heatmaps[0], title="Sample Mouth Heatmaps")


        if i >= 3:
            break





