import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Global instance for multiprocessing
_selfie_segmentation_instance = None

def init_mediapipe_instance(model_selection):
    global _selfie_segmentation_instance
    _selfie_segmentation_instance = mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

def apply_mask(image, binary_mask):
    if binary_mask.ndim == 2:
        binary_mask = np.expand_dims(binary_mask, axis=-1)
    if binary_mask.shape[-1] == 1:
        binary_mask = np.repeat(binary_mask, 3, axis=-1)
    return cv2.bitwise_and(image, binary_mask)

def process_image(args):
    img_path, root_dir, segmented_root, masks_root = args
    global _selfie_segmentation_instance

    try:
        image = cv2.imread(img_path)
        if image is None:
            return f"[Skipped] Could not read: {img_path}"

        rel_dir = os.path.relpath(os.path.dirname(img_path), root_dir)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        segmented_dir = os.path.join(segmented_root, rel_dir)
        masks_dir = os.path.join(masks_root, rel_dir)
        os.makedirs(segmented_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        mask_path = os.path.join(masks_dir, base_name + ".png")
        seg_path = os.path.join(segmented_dir, base_name + ".jpg")

        if os.path.exists(mask_path) and os.path.exists(seg_path):
            return f"[Exists] Skipping: {img_path}"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _selfie_segmentation_instance.process(image_rgb)
        if results.segmentation_mask is None:
            return f"[Skipped] No mask: {img_path}"

        binary_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented_frame = apply_mask(image, binary_mask)

        cv2.imwrite(mask_path, binary_mask)
        cv2.imwrite(seg_path, segmented_frame)
        return f"[OK] {img_path}"

    except Exception as e:
        return f"[Error] {img_path}: {str(e)}"


def process_dataset(root_dir, segmented_root, masks_root, workers=1, model_selection=0):
    all_image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        images = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        full_paths = [os.path.join(dirpath, f) for f in images]
        all_image_paths.extend(full_paths)

    print(f"Found {len(all_image_paths)} images to process.")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers, initializer=init_mediapipe_instance,
                             initargs=(model_selection,)) as executor:
        tasks = ((img_path, root_dir, segmented_root, masks_root) for img_path in all_image_paths)

        # Wrap executor.map with tqdm explicitly
        results = executor.map(process_image, tasks)
        for result in tqdm(results, total=len(all_image_paths), desc="Segmenting", smoothing=0.05):
            if result.startswith("[Error]"):
                print(result)

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds ({len(all_image_paths) / elapsed:.2f} FPS).")

if __name__ == "__main__":
    input_frames_dir = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\cropped_data"
    output_segmented_dir = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\segmented_data"
    output_masks_dir = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\segmented_mask"

    # Adjust worker count to match your CPU
    process_dataset(input_frames_dir, output_segmented_dir, output_masks_dir, workers=1, model_selection=0)
