import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import time

ROOT_DIR = r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset"
MAX_IMAGES = 8
DET_SIZE = (256, 256)

MODEL_NAME = "buffalo_sc"  # SC = Small & Compact, statt buffalo_l


def select_evenly_spaced_images(img_paths, max_images):
    n = len(img_paths)
    if n <= max_images:
        return img_paths
    indices = np.linspace(0, n - 1, max_images, dtype=int)
    return [img_paths[i] for i in indices]


def process_identity_folder(identity_dir, app):
    """Optimierte Version mit minimalem Overhead"""
    clip_dir = os.path.dirname(identity_dir)
    save_path = os.path.join(clip_dir, "identity_emb.npy")

    # Early return für bereits verarbeitete
    if os.path.exists(save_path):
        return None

    # Glob nur einmal
    img_paths = sorted(glob.glob(os.path.join(identity_dir, "*.jpg")))
    img_paths = select_evenly_spaced_images(img_paths, MAX_IMAGES)

    if not img_paths:
        return None

    # Direktes Laden ohne Thread-Overhead (für kleine Anzahl Bilder schneller)
    embeddings = []
    for path in img_paths:
        # Kombiniertes Laden + Verarbeiten (reduziert Memory-Kopien)
        img = cv2.imread(path)
        if img is None:
            continue

        # Nur konvertieren wenn nötig
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = app.get(img)
        if faces and len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if embeddings:
        emb = np.mean(embeddings, axis=0)
        emb /= np.linalg.norm(emb)
        np.save(save_path, emb)
        return True

    return False


def batch_process_folders(identity_dirs, app, batch_size=100):
    """Process in batches with progress tracking"""
    results = {"done": 0, "skip": 0, "fail": 0}

    for i in tqdm(range(0, len(identity_dirs), batch_size), desc="Batches"):
        batch = identity_dirs[i:i + batch_size]

        for identity_dir in batch:
            result = process_identity_folder(identity_dir, app)
            if result is None:
                results["skip"] += 1
            elif result:
                results["done"] += 1
            else:
                results["fail"] += 1

    return results


if __name__ == "__main__":
    start_time = time.time()

    print(f"Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Det size: {DET_SIZE}")
    print(f"  Max images: {MAX_IMAGES}")
    print()

    # GPU check
    try:
        import torch

        use_gpu = torch.cuda.is_available()
        ctx_id = 0 if use_gpu else -1
    except:
        ctx_id = -1
        use_gpu = False

    print(f"Using {'GPU' if use_gpu else 'CPU'}")

    # Initialize model einmal
    print("Loading model...")
    t_model = time.time()
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    print(f"Model loaded in {time.time() - t_model:.2f}s\n")

    # Find directories
    print("Scanning directories...")
    identity_dirs = []
    for root, dirs, files in os.walk(ROOT_DIR):
        if os.path.basename(root).lower() == "identity":
            identity_dirs.append(root)

    print(f"Found {len(identity_dirs)} identity folders.")

    # Pre-filter already processed (schneller Check)
    t_filter = time.time()
    to_process = []
    for d in identity_dirs:
        if not os.path.exists(os.path.join(os.path.dirname(d), "identity_emb.npy")):
            to_process.append(d)

    print(f"Filtered in {time.time() - t_filter:.2f}s")
    print(f"Processing {len(to_process)} remaining folders.\n")

    if len(to_process) == 0:
        print("All done!")
        exit(0)

    # Process all
    results = batch_process_folders(to_process, app)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.2f}s ({elapsed / 60:.1f} min)")
    print(f"Average: {elapsed / len(to_process):.3f}s per identity")
    print(f"Throughput: {len(to_process) / elapsed:.2f} identities/sec")
    print(f"{'=' * 60}")
    print(f"\nResults:")
    print(f"  Processed: {results['done']}")
    print(f"  Skipped: {results['skip']}")
    print(f"  Failed: {results['fail']}")