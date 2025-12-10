import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import display_mel_feature, show_sample, show_first_batch_sample
import time
import shutil
import pickle


class TalkingHeadDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_frames=10, cache_file="dataset_cache.pkl"):
        """
        PyTorch Dataset for talking head data with:
        - Sampling identity frames only where driving frames exist (frames with pose/expression/mouth images)
        - Sampling one driving frame distinct from identity frames
        - Loading corresponding mel features per frame

        Args:
            root_dir (str): Root directory containing video folders -> clip folders structured dataset.
            transform (callable, optional): Transform applied to images (PIL Images).
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.cache_file = cache_file

        if os.path.exists(cache_file):
            print(f"Loading dataset cache from {cache_file}...")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.clips = cache["clips"]
            self.valid_frame_ids = cache["valid_frame_ids"]
            self.identity_image_paths = cache["identity_image_paths"]
            self.masked_image_paths = cache["masked_image_paths"]
            self.mel_path = cache["mel_path"]
            print(f"Loaded {len(self.clips)} clips from cache.")
        else:
            self.clips = []
            self.valid_frame_ids = {}
            self.identity_image_paths = {}
            self.masked_image_paths = {}
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

                    # Collect existing driving frame IDs from masked folder filenames
                    masked_files = [f for f in os.listdir(masked_dir) if f.lower().endswith(".jpg") and "_" in f]
                    driving_frame_ids = sorted(set(f.split("_")[0] for f in masked_files))

                    if len(driving_frame_ids) < min_frames:
                        print(f"Deleting clip {clip_path} with only {len(driving_frame_ids)} driving frames")
                        try:
                            shutil.rmtree(clip_path)
                        except Exception as e:
                            print(f"Failed to delete {clip_path}: {e}")
                        continue  # skip appending this clip since it's deleted

                    # Check identity folder exists
                    identity_dir = os.path.join(clip_path, "identity")
                    if not os.path.isdir(identity_dir):
                        print(f"Warning: identity dir not found, skipping clip: {identity_dir}")
                        continue


                    # Optional: check identity frames exist for the driving frame IDs
                    existing_identity_ids = set(f[:-4] for f in os.listdir(identity_dir) if f.lower().endswith(".jpg"))
                    # Intersection - only keep frames that have identity frames as well
                    common_frame_ids = [fid for fid in driving_frame_ids if fid in existing_identity_ids]

                    if len(common_frame_ids) < min_frames:
                        print(f"Deleting clip {clip_path} with only {len(common_frame_ids)} valid frames")
                        try:
                            shutil.rmtree(clip_path)
                        except Exception as e:
                            print(f"Failed to delete {clip_path}: {e}")
                        continue
                    # idetity paths
                    self.identity_image_paths[clip_path] = [
                        os.path.join(identity_dir, f"{fid}.jpg")
                        for fid in common_frame_ids
                    ]

                    self.clips.append(clip_path)
                    self.valid_frame_ids[clip_path] = common_frame_ids

                    # masked paths
                    masked_dir = os.path.join(clip_path, "masked")
                    self.masked_image_paths[clip_path] = {
                        key: [
                            os.path.join(masked_dir, f"{fid}_{key}.jpg")
                            for fid in common_frame_ids
                        ]
                        for key in ("pose", "expression", "mouth")
                    }

                    self.mel_path[clip_path] = os.path.join(clip_path, "mel.npy")
            print(f"Saving dataset cache to {cache_file}...")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "clips": self.clips,
                    "valid_frame_ids": self.valid_frame_ids,
                    "identity_image_paths": self.identity_image_paths,
                    "masked_image_paths": self.masked_image_paths,
                    "mel_path": self.mel_path,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Dataset initialized with {len(self.clips)} clips having valid frames.")
        # After building, save cache


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        valid_frames = self.valid_frame_ids[clip_path]

        # Sample 8 identity frames from valid frames
        n_id = min(8, len(valid_frames))
        id_sample_ids = random.sample(valid_frames, n_id)
        identity_dir = os.path.join(clip_path, "identity")

        identity_frames = []
        for frame_id in id_sample_ids:
            img_path = os.path.join(identity_dir, f"{frame_id}.jpg")
            if not os.path.isfile(img_path):
                raise RuntimeError(f"Identity image missing: {img_path}")
            with Image.open(img_path) as img:
                identity_frames.append(self.transform(img.convert("RGB")))


        # From valid_frames excluding the sampled identity frames, sample driving frame id
        driving_candidates = [fid for fid in valid_frames if fid not in id_sample_ids]
        if len(driving_candidates) == 0:
            # In rare case no driving frame is exclusive, sample any
            driving_frame_id = random.choice(valid_frames)
        else:
            driving_frame_id = random.choice(driving_candidates)

        #Extract full driving frame for later loss computation
        driving_path = os.path.join(identity_dir, f"{driving_frame_id}.jpg")
        if not os.path.isfile(driving_path):
            raise RuntimeError(f"Identity image missing: {driving_path}")
        with Image.open(driving_path) as img:
            driving_img = self.transform(img.convert("RGB"))

        masked_dir = os.path.join(clip_path, "masked")


        def load_masked_image(frame_id, key):
            img_path = os.path.join(masked_dir, f"{frame_id}_{key}.jpg")
            if not os.path.isfile(img_path):
                raise RuntimeError(f"Masked {key} image missing: {img_path}")
            with Image.open(img_path) as img:
                return self.transform(img.convert("RGB"))

        driving_pose = load_masked_image(driving_frame_id, "pose")
        driving_expression = load_masked_image(driving_frame_id, "expression")
        driving_mouth = load_masked_image(driving_frame_id, "mouth")

        mel_path = self.mel_path[clip_path]
        if not os.path.isfile(mel_path):
            raise RuntimeError(f"Mel feature file missing: {mel_path}")
        mel = np.load(mel_path, mmap_mode="r", allow_pickle=False)


        driving_frame_idx = int(driving_frame_id)  # Convert to int for indexing

        # Select just the frame you need
        frame_idx = int(driving_frame_id)
        mel_frame = mel[frame_idx]  # this is a 2D numpy view

        # Convert to Torch tensor
        mel_feature = torch.tensor(mel_frame, dtype=torch.float32)

        # Sample a single random real image frame from valid frames (independent of identity/driving)
        real_image_frame_id = random.choice(valid_frames)
        real_image_path = os.path.join(identity_dir, f"{real_image_frame_id}.jpg")
        if not os.path.isfile(real_image_path):
            raise RuntimeError(f"Real image missing: {real_image_path}")
        with Image.open(real_image_path) as real_img:
            real_image = self.transform(real_img.convert("RGB"))

        return {
            "identity_frames": torch.stack(identity_frames),  # Tensor (N=8, C, H, W)
            "driving_pose": driving_pose,  # Tensor (C, H, W)
            "driving_expression": driving_expression,  # Tensor (C, H, W)
            "driving_mouth": driving_mouth,  # Tensor (C, H, W)
            "mel_feature": mel_feature,  # Tensor (freq, time)
            "clip_path": clip_path,
            "driving_frame_idx": driving_frame_idx,
            "real_image": real_image,  # new added real image tensor (C, H, W)
            "driving_frame": driving_img,
        }


def benchmark_num_workers(dataset, batch_size=1, num_trials=3, max_workers=None):
    """Benchmark DataLoader performance with varying num_workers."""

    if max_workers is None:
        max_workers = min(16, os.cpu_count())  # limit max workers for testing

    results = {}
    for workers in range(2, max_workers + 1, max(1, max_workers // 8)):
        print(f"\nTesting num_workers = {workers}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=workers, pin_memory=True)
        elapsed = 0
        for trial in range(num_trials):
            start = time.time()
            for batch in dataloader:
                # Simulate a minimal training step or just iterate
                # Optionally do something light with batch for realism
                # For speed test, just pass
                pass
            end = time.time()
            trial_time = end - start
            print(f" Trial {trial + 1}: {trial_time:.3f} sec")
            elapsed += trial_time
        avg_time = elapsed / num_trials
        results[workers] = avg_time
        print(f"Average time with {workers} workers: {avg_time:.3f} sec")

    # Find best num_workers
    best_workers = min(results, key=results.get)
    print(f"\nBest num_workers: {best_workers} with avg time {results[best_workers]:.3f} sec")

    return results


if __name__ == "__main__":
    #Sanity Test of Dataset

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model input size
        transforms.ToTensor(),  # scale to [0,1]
    ])


    start_time = time.time()
    dataset = TalkingHeadDataset(
        root_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\dataset",
        transform=transform
    )
    end_time = time.time()
    print(f"Dataset initialized in {end_time - start_time:.2f} seconds")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i, sample in enumerate(dataloader):
        show_first_batch_sample(sample)
        print(f"Batch {i + 1}:")
        print(f" Identity frames shape: {sample['identity_frames'].shape}")
        print(f" Driving pose shape: {sample['driving_pose'].shape}")
        print(f" Driving expression shape: {sample['driving_expression'].shape}")
        print(f" Driving mouth shape: {sample['driving_mouth'].shape}")
        print(f" Mel feature shape: {sample['mel_feature'].shape}")
        print(f" Clip path: {sample['clip_path']}")
        print(f" Driving frame index: {sample['driving_frame_idx']}\n")
        print(f" driving frames shape: {sample['driving_frame'].shape}")
        break

        #Display mel feature
        #Optionally display mel spectrogram (if you want)
        # if len(sample['mel_feature'].shape) == 3:
        #     mel_to_show = sample['mel_feature'][0]  # first sample in batch, shape (80, T)
        # else:
        #     mel_to_show = sample['mel_feature']
        #
        # display_mel_feature(mel_to_show)

    #benchmark_num_workers(dataset, batch_size=16, num_trials=1, max_workers=16)





