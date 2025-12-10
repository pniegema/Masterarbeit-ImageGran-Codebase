import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import show_first_batch_sample_heatmaps
import time
import shutil
import pickle




class TalkingHeadDataset(Dataset):
    def __init__(self, root_dir, landmarks_dir, transform=None, min_frames=10, cache_file="dataset_cache.pkl"):
        """

        Args:
            root_dir (str): Root directory containing video folders -> clip folders structured dataset.
            landmarks_dir (str): Root directory where pose/expression .npz landmarks are stored
            transform (callable, optional): Transform applied to images (PIL Images).
        """
        self.root_dir = root_dir
        self.landmarks_dir = landmarks_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.cache_file = cache_file

        if os.path.exists(cache_file):
            print(f"Loading dataset cache from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.clips = cache["clips"]
            self.valid_frame_ids = cache["valid_frame_ids"]
            self.identity_image_paths = cache["identity_image_paths"]
            self.mouth_image_paths = cache["mouth_image_paths"]
            self.landmark_paths = cache["landmark_paths"]
            self.mel_path = cache["mel_path"]
            print(f"Loaded {len(self.clips)} clips from cache.")
        else:
            self.clips = []
            self.valid_frame_ids = {}
            self.identity_image_paths = {}
            self.mouth_image_paths = {}
            self.landmark_paths = {}
            self.mel_path = {}

            print("Initializing dataset...")
            for video_folder in sorted(os.listdir(root_dir)):
                video_path = os.path.join(root_dir, video_folder)
                if not os.path.isdir(video_path):
                    continue
                for clip in sorted(os.listdir(video_path)):
                    clip_path = os.path.join(video_path, clip)
                    if not os.path.isdir(clip_path):
                        continue

                    # Ensure masked folder exists (driving frames reside here)
                    masked_dir = os.path.join(clip_path, "masked")
                    if not os.path.isdir(masked_dir):
                        print(f"Warning: masked dir not found, skipping clip: {masked_dir}")
                        continue

                    # Collect driving frame IDs from mouth images
                    mouth_files = [f for f in os.listdir(masked_dir) if f.lower().endswith("_mouth.jpg")]
                    driving_frame_ids = sorted(set(f.split("_")[0] for f in mouth_files))

                    if len(driving_frame_ids) < min_frames:
                        print(f"Deleting clip {clip_path} with only {len(driving_frame_ids)} driving frames")
                        try:
                            shutil.rmtree(clip_path)
                        except Exception as e:
                            print(f"Failed to delete {clip_path}: {e}")
                        continue

                    # Check identity folder exists
                    identity_dir = os.path.join(clip_path, "identity")
                    if not os.path.isdir(identity_dir):
                        print(f"Warning: identity dir not found, skipping clip: {identity_dir}")
                        continue

                    # Only keep frames that have identity images
                    existing_identity_ids = set(f[:-4] for f in os.listdir(identity_dir) if f.lower().endswith(".jpg"))
                    common_frame_ids = [fid for fid in driving_frame_ids if fid in existing_identity_ids]

                    if len(common_frame_ids) < min_frames:
                        print(f"Deleting clip {clip_path} with only {len(common_frame_ids)} valid frames")
                        try:
                            shutil.rmtree(clip_path)
                        except Exception as e:
                            print(f"Failed to delete {clip_path}: {e}")
                        continue

                    # Identity image paths
                    self.identity_image_paths[clip_path] = [
                        os.path.join(identity_dir, f"{fid}.jpg") for fid in common_frame_ids
                    ]

                    # Mouth image paths
                    self.mouth_image_paths[clip_path] = [
                        os.path.join(masked_dir, f"{fid}_mouth.jpg") for fid in common_frame_ids
                    ]

                    # Landmark paths (pose & expression)
                    self.landmark_paths[clip_path] = [
                        {
                            "pose": os.path.join(landmarks_dir, video_folder, clip, f"{int(fid):04d}_pose.npz"),
                            "expression": os.path.join(landmarks_dir, video_folder, clip,
                                                       f"{int(fid):04d}_expression.npz")
                        }
                        for fid in common_frame_ids
                    ]

                    # Mel features path
                    self.mel_path[clip_path] = os.path.join(clip_path, "mel.npy")

                    self.clips.append(clip_path)
                    self.valid_frame_ids[clip_path] = common_frame_ids

            # Save cache
            print(f"Saving dataset cache to {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "clips": self.clips,
                    "valid_frame_ids": self.valid_frame_ids,
                    "identity_image_paths": self.identity_image_paths,
                    "mouth_image_paths": self.mouth_image_paths,
                    "landmark_paths": self.landmark_paths,
                    "mel_path": self.mel_path,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Dataset initialized with {len(self.clips)} clips having valid frames.")


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        valid_frames = self.valid_frame_ids[clip_path]

        # --- Identity frames ---
        n_id = min(8, len(valid_frames))
        id_sample_ids = random.sample(valid_frames, n_id)
        identity_frames = []
        for frame_id in id_sample_ids:
            img_path = os.path.join(self.identity_image_paths[clip_path][valid_frames.index(frame_id)])
            if not os.path.isfile(img_path):
                raise RuntimeError(f"Identity image missing: {img_path}")
            with Image.open(img_path) as img:
                identity_frames.append(self.transform(img.convert("RGB")))
        identity_frames = torch.stack(identity_frames)  # (N, C, H, W)

        # --- Driving frame ---
        driving_candidates = [fid for fid in valid_frames if fid not in id_sample_ids]
        if len(driving_candidates) == 0:
            driving_frame_id = random.choice(valid_frames)
        else:
            driving_frame_id = random.choice(driving_candidates)
        frame_idx = valid_frames.index(driving_frame_id)

        # Load mouth image
        mouth_path = self.mouth_image_paths[clip_path][frame_idx]
        if not os.path.isfile(mouth_path):
            raise RuntimeError(f"Mouth image missing: {mouth_path}")
        with Image.open(mouth_path) as img:
            driving_mouth = self.transform(img.convert("RGB"))

        # Load pose & expression heatmaps from .npz
        landmark_info = self.landmark_paths[clip_path][frame_idx]

        def load_heatmap(npz_path):
            if not os.path.isfile(npz_path):
                raise RuntimeError(f"Landmark npz missing: {npz_path}")
            data = np.load(npz_path)
            heatmaps = data["heatmaps"].astype(np.float32) / 255.0  # normalize to [0,1]
            return torch.tensor(heatmaps)  # (N_landmarks, H, W)

        driving_pose = load_heatmap(landmark_info["pose"])
        driving_expression = load_heatmap(landmark_info["expression"])

        #If driving pose shpae isnt (18,H,W) or driving expression shape isnt (140,H,W) pad with zeros
        if driving_pose.shape[0] != 18:
            padded_pose = torch.zeros((18, driving_pose.shape[1], driving_pose.shape[2]))
            padded_pose[:driving_pose.shape[0], :, :] = driving_pose
            driving_pose = padded_pose
        if driving_expression.shape[0] != 140:
            padded_expression = torch.zeros((140, driving_expression.shape[1], driving_expression.shape[2]))
            padded_expression[:driving_expression.shape[0], :, :] = driving_expression
            driving_expression = padded_expression

        # --- Driving frame (original RGB image) for reconstruction/loss ---
        # Use any identity image for the driving frame itself
        driving_img_path = self.identity_image_paths[clip_path][frame_idx]
        with Image.open(driving_img_path) as img:
            driving_frame = self.transform(img.convert("RGB"))

        # --- Mel features ---
        mel_path = self.mel_path[clip_path]
        if not os.path.isfile(mel_path):
            raise RuntimeError(f"Mel feature file missing: {mel_path}")
        mel = np.load(mel_path, mmap_mode="r", allow_pickle=False)
        mel_feature = torch.tensor(mel[frame_idx], dtype=torch.float32)  # (freq, time)

        # --- Real image for contrastive/real loss ---
        real_frame_id = random.choice(valid_frames)
        real_img_path = self.identity_image_paths[clip_path][valid_frames.index(real_frame_id)]
        with Image.open(real_img_path) as img:
            real_image = self.transform(img.convert("RGB"))

        return {
            "identity_frames": identity_frames,  # (N=8, C, H, W)
            "driving_pose": driving_pose,  # (N_landmarks, H, W)
            "driving_expression": driving_expression,  # (N_landmarks, H, W)
            "driving_mouth": driving_mouth,  # (C, H, W)
            "driving_frame": driving_frame,  # (C, H, W)
            "mel_feature": mel_feature,  # (freq, time)
            "real_image": real_image,  # (C, H, W)
            "clip_path": clip_path,
            "driving_frame_idx": int(frame_idx),
        }



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to model input size
        transforms.ToTensor(),  # scale to [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])  # → [-1,1]
    ])


    start_time = time.time()
    dataset = TalkingHeadDataset(
        root_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\dataset",
        landmarks_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\landmarks",
        transform=transform,
        cache_file="dataset_cache_heatmap.pkl"
    )
    end_time = time.time()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    mismatch_count = 0

    dataset_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\dataset"
    landmarks_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\landmarks"


    for i, sample in enumerate(dataloader):
        show_first_batch_sample_heatmaps(sample)
        break





