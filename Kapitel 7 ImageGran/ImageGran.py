"""
Complete Inference Pipeline for Face Animation
Inputs:
    - Identity: Single image frame
    - Pose: Video (head pose and expression)
    - Visual: Video (lip sync)
Output:
    - Animated video
"""


import torch
from pathlib import Path
from insightface.app import FaceAnalysis
import mediapipe as mp
from scipy.ndimage import rotate
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2
import moviepy.editor as mpy
import tempfile
import os
from tqdm import tqdm


from models import HeatmapEncoder64, StyleGAN1Generator_256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Landmark indices (from your preprocessing)
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

# Global cache for grid coordinates
_grid_cache = {}


class ImageGran:
    def __init__(self,
                 generator,
                 visual_encoder,
                 pose_expr_encoder,
                 device='cuda',
                 det_size=(256, 256)):
        """
        ImageGran face animation pipeline.

        Args:
            generator: Trained generator model
            visual_encoder: Trained visual encoder
            pose_expr_encoder: Trained pose+expr encoder
            device: 'cuda' or 'cpu'
            det_size: Face detection size for InsightFace
        """
        self.device = device
        self.G = generator.to(device).eval()
        self.visual_enc = visual_encoder.to(device).eval()
        self.pose_expr_enc = pose_expr_encoder.to(device).eval()

        # Initialize InsightFace for identity embedding
        self.face_app = FaceAnalysis(name="buffalo_sc")
        ctx_id = 0 if device == 'cuda' else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Selfie Segmentation
        self.selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0: general, 1: landscape (better quality)
        )

    def _apply_mask(self, image, mask):
        """
        Apply binary mask to image (set non-mask areas to black).

        Args:
            image: (H, W, 3) BGR image
            mask: (H, W) binary mask (0 or 255)

        Returns:
            masked_image: (H, W, 3) BGR image with black background
        """
        # Ensure mask has same dimensions as image
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)

        # Apply mask
        masked_image = np.where(mask > 0, image, 0).astype(np.uint8)

        return masked_image

    def segment_identity_image(self, image_path, output_path=None, threshold=0.5):
        """
        Segment identity image with black background using MediaPipe.

        Args:
            image_path: Path to identity image
            output_path: Path to save segmented image (optional)
            threshold: Segmentation threshold (default: 0.5)

        Returns:
            segmented_image: (H, W, 3) numpy array with black background
            output_path: Path where image was saved (if output_path provided)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run segmentation
        seg_results = self.selfie_segmentation.process(image_rgb)

        if seg_results.segmentation_mask is None:
            raise ValueError("Segmentation failed - no mask generated")

        # Create binary mask
        binary_mask = (seg_results.segmentation_mask > threshold).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Apply mask (set background to black)
        segmented_image = self._apply_mask(image, binary_mask)

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, segmented_image)
            print(f"Segmented image saved to: {output_path}")

        return segmented_image, output_path


    def extract_identity_embedding(self, image_path):
        """
        Extract identity embedding from a single image.

        Args:
            image_path: Path to identity image

        Returns:
            identity_embedding: (512,) tensor
        """
        img, _ = self.segment_identity_image(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")



        # Extract face
        faces = self.face_app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in identity image")

        # Get embedding from first face
        face = faces[0]
        embedding = face.embedding  # (512,)
        #emb = np.mean(embedding, axis=0)
        embedding /= np.linalg.norm(embedding)  # L2-normalize

        return torch.from_numpy(embedding).float().to(self.device)

    def extract_landmarks_from_video(self, video_path):
        """
        Extract MediaPipe landmarks from all frames of a video.

        Args:
            video_path: Path to video file

        Returns:
            landmarks_list: List of dicts with keys 'pose', 'expr', 'mouth'
            fps: Video FPS
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        landmarks_list = []

        with tqdm(desc=f"Extracting landmarks from {Path(video_path).name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)

                if not results.multi_face_landmarks:
                    pbar.update(1)
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                # Extract landmark subsets
                pose_lm = np.array([[landmarks[j].x, landmarks[j].y] for j in POSE_LANDMARKS])
                expr_lm = np.array([[landmarks[j].x, landmarks[j].y] for j in EXPR_NO_MOUTH_LANDMARKS])
                mouth_lm = np.array([[landmarks[j].x, landmarks[j].y] for j in MOUTH_LANDMARKS])

                landmarks_list.append({
                    'pose': pose_lm,
                    'expr': expr_lm,
                    'mouth': mouth_lm
                })

                pbar.update(1)

        cap.release()

        if len(landmarks_list) == 0:
            raise ValueError(f"No faces detected in video: {video_path}")

        return landmarks_list, fps

    def landmarks_to_heatmaps(self, points, H=64, W=64, sigma=2.5, mode='absolute'):
        """
        Convert landmarks to heatmaps.

        Args:
            points: (N, 2) numpy array with normalized coordinates in [0,1]
            mode: 'absolute' (pose_and_expr) or 'mouth' (mouth shape)

        Returns:
            heatmaps: (N, H, W) numpy array
        """
        pts = torch.from_numpy(points).float()

        if mode == 'mouth':
            center = pts.mean(dim=0, keepdim=True)
            pts = pts - center
            max_dist = pts.abs().max()
            if max_dist > 1e-6:
                pts = pts * (0.5 / max_dist)
            pts = pts + 0.5

        pts = pts * torch.tensor([W - 1, H - 1], dtype=pts.dtype)

        inv_2sigma2 = -0.5 / (sigma * sigma)

        # Cache coordinate grids
        cache_key = (H, W, 'cpu', pts.dtype)
        if cache_key not in _grid_cache:
            xx = torch.arange(W, dtype=pts.dtype).view(1, 1, W)
            yy = torch.arange(H, dtype=pts.dtype).view(1, H, 1)
            _grid_cache[cache_key] = (xx, yy)

        xx, yy = _grid_cache[cache_key]

        dx = xx - pts[:, 0].view(-1, 1, 1)
        dy = yy - pts[:, 1].view(-1, 1, 1)

        heatmaps = torch.exp((dx.square() + dy.square()) * inv_2sigma2)

        return heatmaps.numpy()

    def create_heatmaps_from_landmarks(self, landmarks_dict, mode='pose_expr'):
        """
        Create combined pose+expr and mouth heatmaps from landmarks.

        Args:
            landmarks_dict: Dict with keys 'pose', 'expr', 'mouth'

        Returns:
            pose_expr_hm: (175, 64, 64) numpy array
            mouth_hm: (66, 64, 64) numpy array
        """
        if mode == 'pose_expr':
            # Create pose heatmaps
            pose_points = landmarks_dict['pose']

            # Create expr heatmaps
            expr_points = landmarks_dict['expr']


            pose_expr_points = np.concatenate([pose_points, expr_points], axis=0)
            #print("Pose and Expr Points", pose_expr_points.shape)

            # Combine pose and expr
            pose_expr_hm = self.landmarks_to_heatmaps(pose_expr_points, mode='absolute')

            # Ensure correct number of channels
            if pose_expr_hm.shape[0] != 175:
                temp = np.zeros((175, 64, 64), dtype=np.float32)
                copy_size = min(pose_expr_hm.shape[0], 175)
                temp[:copy_size] = pose_expr_hm[:copy_size]
                pose_expr_hm = temp
            return pose_expr_hm

        if mode == 'mouth':
            # Create mouth heatmaps
            mouth_points = landmarks_dict['mouth']
            mouth_hm = self.landmarks_to_heatmaps(mouth_points, mode='mouth')



            if mouth_hm.shape[0] != 66:
                temp = np.zeros((66, 64, 64), dtype=np.float32)
                copy_size = min(mouth_hm.shape[0], 66)
                temp[:copy_size] = mouth_hm[:copy_size]
                mouth_hm = temp
            return mouth_hm


    def rotate_mouth_heatmap(self, mouth_hm, max_angle=30):
        """
        Rotate mouth heatmap to remove position information.

        Args:
            mouth_hm: (C, H, W) numpy array
            max_angle: Maximum rotation angle in degrees

        Returns:
            Rotated heatmap
        """
        angle = np.random.uniform(-max_angle, max_angle)
        rotated = np.zeros_like(mouth_hm)
        for i in range(mouth_hm.shape[0]):
            rotated[i] = rotate(mouth_hm[i], angle, reshape=False,
                                order=1, mode='constant', cval=0)
        return rotated

    def generate_video(self,
                       identity_image_path,
                       pose_expr_video_path,
                       visual_video_path,
                       output_path,
                       apply_mouth_rotation=False,
                       output_resolution=(256, 256)):
        """
        Generate animated video from identity image and two driving videos only outputs the generated video.

        Args:
            identity_image_path: Path to identity image
            pose_expr_video_path: Path to video for pose/expression
            visual_video_path: Path to video for lip sync
            output_path: Path to save output video
            apply_mouth_rotation: Whether to rotate mouth heatmaps
            output_resolution: (width, height) for output

        Returns:
            output_path: Path to generated video
        """
        print("=== Starting Face Animation Pipeline ===")

        # Step 1: Extract identity embedding
        print("\n[1/4] Extracting identity embedding...")
        identity_embedding = self.extract_identity_embedding(identity_image_path)
        identity_embedding = identity_embedding.unsqueeze(0)  # Add batch dim
        print("Max value in identity embedding:", identity_embedding.max().item())
        print("Min value in identity embedding:", identity_embedding.min().item())
        print("Identity embedding shape:", identity_embedding.shape)


        # Step 2: Extract landmarks from pose video
        print("\n[2/4] Extracting landmarks from pose video...")
        pose_expr_landmarks, pose_fps = self.extract_landmarks_from_video(pose_expr_video_path)
        print(f"Extracted {len(pose_expr_landmarks)} frames from pose video.")

        # Step 3: Extract landmarks from visual video
        print("\n[3/4] Extracting landmarks from visual video...")
        visual_landmarks, visual_fps = self.extract_landmarks_from_video(visual_video_path)
        print(f"Extracted {len(visual_landmarks)} frames from visual video.")

        # Determine output FPS (use pose video FPS)
        output_fps = pose_fps
        print(f"\nOutput video FPS set to: {output_fps:.2f}")

        # Step 4: Generate frames
        print(f"\n[4/4] Generating {min(len(pose_expr_landmarks), len(visual_landmarks))} frames...")

        # Resample visual landmarks to match pose video fps
        if visual_fps != pose_fps:
            scale = pose_fps / visual_fps
            indices = np.linspace(0, len(visual_landmarks) - 1, int(len(visual_landmarks) * scale), dtype=int)
            visual_landmarks = [visual_landmarks[i] for i in indices]

        # Match lengths
        min_len = min(len(pose_expr_landmarks), len(visual_landmarks))
        pose_expr_landmarks = pose_expr_landmarks[:min_len]
        visual_landmarks = visual_landmarks[:min_len]

        max_frames = min(len(pose_expr_landmarks), len(visual_landmarks))
        generated_frames = []

        with torch.no_grad():
            for idx in tqdm(range(max_frames), desc="Generating frames"):
                # Get landmarks (reuse last if index exceeds list)
                pose_idx = min(idx, len(pose_expr_landmarks) - 1)
                visual_idx = min(idx, len(visual_landmarks) - 1)

                pose_expr_lm = pose_expr_landmarks[pose_idx]
                visual_lm = visual_landmarks[visual_idx]


                # Create heatmaps from pose_expr video
                pose_expr_hm = self.create_heatmaps_from_landmarks(pose_expr_lm, mode='pose_expr')

                # Create heatmaps from visual video (only mouth)
                mouth_hm = self.create_heatmaps_from_landmarks(visual_lm, mode='mouth')


                # Apply rotation to mouth heatmap
                if apply_mouth_rotation:
                    mouth_hm = self.rotate_mouth_heatmap(mouth_hm)

                # Convert to tensors
                pose_expr_tensor = torch.from_numpy(pose_expr_hm).float().unsqueeze(0).to(self.device)
                mouth_tensor = torch.from_numpy(mouth_hm).float().unsqueeze(0).to(self.device)

                # Encode latents
                pose_expr_lat = self.pose_expr_enc(pose_expr_tensor)
                visual_lat = self.visual_enc(mouth_tensor)

                # Concatenate motion latents
                motion_lat = torch.cat([pose_expr_lat, visual_lat], dim=1)

                # Generate frame
                generated = self.G(identity_embedding, motion_lat)

                # Convert to numpy image [0, 255]
                img_np = ((generated + 1.0) / 2.0).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_uint8 = (img_np * 255.0).astype(np.uint8)

                # Resize to output resolution
                img_resized = cv2.resize(img_uint8, output_resolution, interpolation=cv2.INTER_LINEAR)
                generated_frames.append(img_resized)

        # Step 5: Save video
        print("\n[5/5] Saving video...")

        # Try to extract audio from visual video
        audio_path = None
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio_path = temp_audio.name
            temp_audio.close()

            video_clip = mpy.VideoFileClip(visual_video_path)
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video_clip.close()
        except:
            audio_path = None

        # Save video with or without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = generated_frames[0].shape[:2]
        writer = cv2.VideoWriter(temp_video, fourcc, output_fps, (width, height))

        for frame in generated_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        if audio_path and os.path.exists(audio_path):
            # Add audio
            video_clip = mpy.VideoFileClip(temp_video)
            audio_clip = mpy.AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(output_path, codec="libx264",
                                       fps=output_fps, audio_codec="aac",
                                       verbose=False, logger=None)
            video_clip.close()
            audio_clip.close()

            # Clean up
            os.remove(audio_path)
            os.remove(temp_video)
        else:
            # No audio, just move temp video
            import shutil
            shutil.move(temp_video, output_path)

        print(f"\n✓ Video saved to: {output_path}")
        return output_path

    def generate_video_4way(self,
                            identity_image_path,
                            pose_expr_video_path,
                            visual_video_path,
                            output_path,
                            apply_mouth_rotation=False,
                            output_fps=25,
                            output_resolution=(256, 256),
                            background_path=None):
        """
        Generate animated video from identity image and two driving videos,
        with time-aligned frames and optional background image. Output video also contains all driving sources.

        Args:
            identity_image_path: Path to identity image
            pose_expr_video_path: Path to video for pose/expression
            visual_video_path: Path to video for lip sync
            output_path: Path to save output video
            apply_mouth_rotation: Whether to rotate mouth heatmaps
            output_fps: Output FPS of generated video
            output_resolution: (width, height) for output
            background_path: Optional path to background image to composite over

        Returns:
            output_path: Path to generated video
        """


        print("=== Starting Face Animation Pipeline ===")

        # ---------------- Step 1: Identity embedding ----------------
        print("\n[1/4] Extracting identity embedding...")
        identity_embedding = self.extract_identity_embedding(identity_image_path)
        identity_embedding = identity_embedding.unsqueeze(0).to(self.device)
        print("Identity embedding shape:", identity_embedding.shape,
              "Max:", identity_embedding.max().item(),
              "Min:", identity_embedding.min().item())

        # ---------------- Step 2: Extract landmarks ----------------
        print("\n[2/4] Extracting landmarks from pose video...")
        pose_expr_landmarks, pose_fps = self.extract_landmarks_from_video(pose_expr_video_path)
        print(f"Pose video: {len(pose_expr_landmarks)} frames at {pose_fps} FPS")

        print("\n[3/4] Extracting landmarks from visual video...")
        visual_landmarks, visual_fps = self.extract_landmarks_from_video(visual_video_path)
        print(f"Visual video: {len(visual_landmarks)} frames at {visual_fps} FPS")

        # ---------------- Step 3: Load video frames ----------------
        def load_video_frames(video_path, resolution):
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resolution)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            return frames

        print("Loading pose frames...")
        pose_frames = load_video_frames(pose_expr_video_path, output_resolution)
        print("Loading visual frames...")
        visual_frames = load_video_frames(visual_video_path, output_resolution)

        # ---------------- Step 4: Load background if provided ----------------
        bg = None
        if background_path is not None:
            bg = cv2.imread(background_path)
            bg = cv2.resize(bg, output_resolution)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        # ---------------- Step 5: Determine output duration ----------------
        duration_pose = len(pose_expr_landmarks) / pose_fps
        duration_visual = len(visual_landmarks) / visual_fps
        output_duration = min(duration_pose, duration_visual)
        num_frames = int(output_fps * output_duration)
        print(f"Output video duration: {output_duration:.2f}s, {num_frames} frames at {output_fps} FPS")

        # Load identity frame
        identity_frame = cv2.imread(identity_image_path)
        identity_frame = cv2.resize(identity_frame, output_resolution)
        identity_frame = cv2.cvtColor(identity_frame, cv2.COLOR_BGR2RGB)

        generated_frames = []
        viz_frames = []

        # Helper for labeling frames
        def add_label(frame, text):
            labeled = frame.copy()
            cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 1, cv2.LINE_AA)
            return labeled

        # Helper: composite frame over background
        def composite_frame(frame, bg):
            if bg is None:
                return frame
            frame_f = frame.astype(np.float32)
            bg_f = bg.astype(np.float32)
            mask = (frame_f.sum(axis=2) > 10).astype(np.float32)[..., None]  # non-black pixels
            composite = (frame_f * mask + bg_f * (1 - mask)).astype(np.uint8)
            return composite

        print("\n[4/4] Generating frames...")
        with torch.no_grad():
            for i in tqdm(range(num_frames), desc="Generating frames"):
                # Time-aligned indices
                pose_idx = min(int(i * pose_fps / output_fps), len(pose_expr_landmarks) - 1)
                visual_idx = min(int(i * visual_fps / output_fps), len(visual_landmarks) - 1)

                # Get landmarks
                pose_lm = pose_expr_landmarks[pose_idx]
                visual_lm = visual_landmarks[visual_idx]

                # Heatmaps
                pose_expr_hm = self.create_heatmaps_from_landmarks(pose_lm, mode='pose_expr')
                mouth_hm = self.create_heatmaps_from_landmarks(visual_lm, mode='mouth')
                if apply_mouth_rotation:
                    mouth_hm = self.rotate_mouth_heatmap(mouth_hm)

                # Encode
                pose_tensor = torch.from_numpy(pose_expr_hm).float().unsqueeze(0).to(self.device)
                mouth_tensor = torch.from_numpy(mouth_hm).float().unsqueeze(0).to(self.device)
                pose_lat = self.pose_expr_enc(pose_tensor)
                visual_lat = self.visual_enc(mouth_tensor)
                motion_lat = torch.cat([pose_lat, visual_lat], dim=1)

                # Generate frame
                generated = self.G(identity_embedding, motion_lat)
                img_np = ((generated + 1.0) / 2.0).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                gen_frame = (img_np * 255).astype(np.uint8)

                # Composite over background if provided
                gen_frame = composite_frame(gen_frame, bg)
                generated_frames.append(gen_frame)

                # 4-way visualization
                identity_labeled = add_label(identity_frame, "Identity")
                pose_labeled = add_label(pose_frames[pose_idx], "Pose+Expr")
                visual_labeled = add_label(visual_frames[visual_idx], "Visual")
                gen_labeled = add_label(gen_frame, "Generated")

                top_row = np.hstack([identity_labeled, pose_labeled])
                bottom_row = np.hstack([visual_labeled, gen_labeled])
                combined = np.vstack([top_row, bottom_row])
                viz_frames.append(combined)

        # ---------------- Step 6: Save videos ----------------
        print("Saving videos...")
        import tempfile
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Save generated video
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        h, w = generated_frames[0].shape[:2]
        writer = cv2.VideoWriter(temp_video, fourcc, output_fps, (w, h))
        for frame in generated_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Save visualization video
        viz_path = output_path.replace(".mp4", "_viz.mp4")
        vh, vw = viz_frames[0].shape[:2]
        writer_viz = cv2.VideoWriter(viz_path, fourcc, output_fps, (vw, vh))
        for frame in viz_frames:
            writer_viz.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer_viz.release()

        # Merge audio into generated video
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            video_clip = mpy.VideoFileClip(visual_video_path)
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            video_clip.close()

            video_clip = mpy.VideoFileClip(temp_video).set_fps(output_fps)
            audio_clip = mpy.AudioFileClip(temp_audio).set_duration(video_clip.duration)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(output_path, codec="libx264", fps=output_fps,
                                       audio_codec="aac", verbose=False, logger=None)
            video_clip.close()
            audio_clip.close()
            os.remove(temp_video)
            os.remove(temp_audio)
        except Exception as e:
            print(f"Warning: audio merge failed ({e})")
            import shutil
            shutil.move(temp_video, output_path)

        # Merge audio into visualization video
        try:
            video_clip = mpy.VideoFileClip(viz_path)
            audio_clip = mpy.AudioFileClip(visual_video_path)
            video_clip = video_clip.set_audio(audio_clip.set_duration(video_clip.duration))
            viz_with_audio = viz_path.replace(".mp4", "_audio.mp4")
            video_clip.write_videofile(viz_with_audio, codec="libx264", fps=output_fps,
                                       audio_codec="aac", verbose=False, logger=None)
            video_clip.close()
            audio_clip.close()
            os.remove(viz_path)
            viz_path = viz_with_audio
        except Exception as e:
            print(f"Warning: could not add audio to visualization ({e})")

        print(f"\n✓ Generated video saved: {output_path}")
        print(f"✓ Visualization saved:   {viz_path}")
        return output_path



# Example usage
if __name__ == "__main__":

    # Load your models
    pose_and_expr_enc = HeatmapEncoder64(in_channels=175, latent_dim=256, first_layer_channels=128)
    visual_enc = HeatmapEncoder64(in_channels=66, latent_dim=128, first_layer_channels=64)
    generator = StyleGAN1Generator_256(
        identity_dim=512,  # buffalo_sc embedding size
        motion_dim=384,  # pose(128) + mouth(128) + expression(128)
        style_dim=512
    )

    checkpoint_path = 'last_epoch_Stylegan1_attention_pose_and_expr_enc_256_full_GAN_v2.pth'
    #checkpoint_path = 'last_epoch_Stylegan1_attention_pose_and_expr_enc_256_v2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    pose_and_expr_enc.load_state_dict(checkpoint['pose_and_expr_enc'])
    visual_enc.load_state_dict(checkpoint['visual_enc'])


    # Initialize pipeline
    pipeline = ImageGran(
        generator=generator,
        visual_encoder=visual_enc,
        pose_expr_encoder=pose_and_expr_enc,
        device='cuda'
    )

    #Example Inputs used for testing, plotting and video generation
    #identity_img_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WRA_DaveCamp_000\WRA_DaveCamp_000_clip_0000\identity\0037.jpg"
    #identity_img_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WDA_BarackObama_000\WDA_BarackObama_000_clip_0000\identity\0055.jpg"
    #identity_img_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WDA_XavierBecerra_001\WDA_XavierBecerra_001_clip_0001\identity\0021.jpg"
    #identity_img_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WRA_JohnKasich1_001\WRA_JohnKasich1_001_clip_0003\identity\0068.jpg"
    identity_img_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WDA_TammyBaldwin1_000\WDA_TammyBaldwin1_000_clip_0001\identity\0021.jpg"
    #train_emb = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset\WRA_DaveCamp_000\WRA_DaveCamp_000_clip_0000\identity_emb.npy"
    #identity_img_path = r"C:\Users\pnieg\Documents\Masterarbeit\identity_images\man3.jpg"
    #identity_img_path = r"D:\videos_HDTF\_2g7CTRkaxc_0.mp4"
    #identity_img_path = r"D:\HDTF_raw\WRA_DaveCamp_000.mp4"
    #identity_img_path = r"D:\HDTF_raw\WRA_JohnKasich1_000.mp4"
    #pose_expr_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\WRA_AdamKinzinger0_000.mp4"
    #pose_expr_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\WRA_AustinScott0_000.mp4"
    #pose_expr_video_path = r"D:\HDTF_raw\WRA_JohnHoeven_000.mp4"
    pose_expr_video_path = r"D:\HDTF_raw\_5ukjsqqLg4_13.mp4"
    #pose_expr_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\_NKXqc5vAN8_3.mp4"
    #pose_expr_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\_Yu-EWF7Fxc_5.mp4"
    #pose_expr_video_path = r"D:\HDTF_raw\WDA_BarackObama_000.mp4"
    #visual_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\WRA_AustinScott0_000.mp4"
    #visual_video_path = r"D:\HDTF_raw\_2g7CTRkaxc_0.mp4"
    visual_video_path = r"D:\HDTF_raw\_bkBt4Z6NQ8_3.mp4"
    #visual_video_path = r"D:\HDTF_raw\WDA_BarackObama_000.mp4"
    #visual_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\_MuxVqB3I7E_1.mp4"
    #visual_video_path = r"C:\Users\pnieg\Documents\HDTF\HDTF_raw\1AhrD2-cvrw_1.mp4"

    output_video_path = r"C:\Users\pnieg\PycharmProjects\ImageGran\videos\adversarial_2.mp4"


    background_path = r"C:\Users\pnieg\Documents\Masterarbeit\identity_images\hintergrund4.png"
    # Generate video
    pipeline.generate_video_4way(
        identity_image_path=identity_img_path,
        pose_expr_video_path= pose_expr_video_path,
        visual_video_path=visual_video_path,
        output_path=output_video_path,
        apply_mouth_rotation=False,
        background_path=None,
    )



