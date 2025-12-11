import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp_pool  # alias to avoid conflict with mediapipe.mp
import matplotlib.pyplot as plt


mp_face_mesh = mp.solutions.face_mesh



POSE_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
    136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175

]


EXPR_NO_MOUTH_LANDMARKS = [285, 295, 282, 283, 276, 300, 293, 334, 296, 336, #right brow
                            55, 65, 52, 53, 46, 70, 63, 103, 66, 105, #left brow
                            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,    #right inner eye
                            464, 414, 286 ,258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 263, #right outer eye 37-53
                            33, 7, 163, 144, 145, 153, 154, 155, 246, 161, 160, 159, 158, 157, 173, 133, 155, #left inner eye # STARTS 54
                            130, 247, 30, 29 ,27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25, #left outer eye Start 71 -86
                            104, 69, 108, 151, 337, 299, 333, 9, # Forehead
                            117, 118, 101, 36, 203, 206, 205, 50, 123, 147, 187, 207, 216, 212,214, 192, 213, #Left Cheek
                            423, 266, 330, 347, 346, 352, 280, 425, 426, 436, 427, 411, 376, 432, 434, 416, 433  #Right Cheek

                            ]

LANDMARKS = [0, 11, 12,13, 14, 15, 16, 17, 37, 72, 38, 82, 87, 86, 85, 84, 39, 73, 41 ,81, 178, 179, 180, 181,
                40, 74, 42, 80, 88, 89, 90, 91, 185, 62, 61, 146, 267, 302, 268, 312, 317, 316, 315, 314, 269, 303,
                271, 311, 402, 403, 404, 405, 270, 304, 272, 310, 318, 319, 320, 321, 409, 306, 375, 57, 287, 291
                   ]

def landmarks_to_xy_array(landmarks):
    return np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)


def extract_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def process_clip_worker(args):
    clip_dir, output_dir = args

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)
    try:
        process_clip(clip_dir, output_dir, face_mesh, mouth_offset=5, debug=True)
    finally:
        face_mesh.close()

def save_landmarks_npz(out_path, expr_landmarks, pose_landmarks, mouth_landmarks):
    # Extract (x, y) for each set
    expr_xy = np.array([[lm.x, lm.y] for lm in expr_landmarks], dtype=np.float32)
    pose_xy = np.array([[lm.x, lm.y] for lm in pose_landmarks], dtype=np.float32)
    mouth_xy = np.array([[lm.x, lm.y] for lm in mouth_landmarks], dtype=np.float32)
    np.savez_compressed(out_path, expr=expr_xy, pose=pose_xy, mouth=mouth_xy)

def visualize_landmarks(image, expr_xy, pose_xy, mouth_xy, colors=None, radius=2):
    """
    Draw landmarks on image for each group.
    - image: np.ndarray, BGR (H,W,3)
    - expr_xy, pose_xy, mouth_xy: np.ndarray, shape (N,2), normalized [0,1]
    - colors: dict, e.g. {'expr': (0,255,0), 'pose': (255,0,0), 'mouth': (0,0,255)}
    """
    h, w = image.shape[:2]
    out_img = image.copy()
    if colors is None:
        colors = {'expr': (0,255,0), 'pose': (255,0,0), 'mouth': (0,0,255)}
    # Indices to highlight in red
    #highlight_idx = [44,70,93,103,123]
    #nose_surrounding= [98, 119, 78, 20, 93]
    blur_index = [104, 122, 123, 103, 93, 89, 100, 118]

    for i, pt in enumerate(expr_xy):
        x, y = int(pt[0] * w), int(pt[1] * h)
        if i in blur_index:
            color = (0, 0, 255)  # Red in BGR
        else:
            color = colors['expr']
        cv2.circle(out_img, (x, y), radius, color, -1)
    for pt in pose_xy:
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(out_img, (x, y), radius, colors['pose'], -1)
    for pt in mouth_xy:
        x, y = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(out_img, (x, y), radius, colors['mouth'], -1)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def visualize_saved_landmarks(npz_path, image_path):
    data = np.load(npz_path)
    expr_xy = data['expr']
    pose_xy = data['pose']
    mouth_xy = data['mouth']
    image = cv2.imread(image_path)
    visualize_landmarks(image, expr_xy, pose_xy, mouth_xy)

def process_clip(clip_dir, output_dir, face_mesh, mouth_offset=5, debug=False):
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith('.jpg')])
    os.makedirs(output_dir, exist_ok=True)

    for i, frame_name in enumerate(frames):
        path = os.path.join(clip_dir, frame_name)
        image = cv2.imread(path)
        if image is None:
            continue

        landmarks = extract_landmarks(image, face_mesh)
        if landmarks is None:
            continue




        expr_landmarks = [landmarks[j] for j in EXPR_NO_MOUTH_LANDMARKS]
        mouth_landmarks = [landmarks[j] for j in MOUTH_LANDMARKS]
        pose_landmarks = [landmarks[j] for j in POSE_LANDMARKS]




        #Save landmarks
        #Save (x, y) arrays for each frame
        out_name = f"landmarks_{i:04d}.npz"
        out_path = os.path.join(output_dir, out_name)
        save_landmarks_npz(out_path, expr_landmarks, pose_landmarks, mouth_landmarks)


def process_all_clips_parallel(root_dir, output_root, num_workers=None):
    """
    Collect all clip folders under root_dir containing .jpg images and process them in parallel.
    Each clip is processed in a separate process spawning its own MediaPipe FaceMesh instance.
    """
    clips = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.lower().endswith('.jpg') for f in filenames):
            rel_path = os.path.relpath(dirpath, root_dir)
            out_dir = os.path.join(output_root, rel_path)
            clips.append((dirpath, out_dir))

    if num_workers is None:
        num_workers = min(mp_pool.cpu_count(), len(clips))

    print(f"Starting multiprocessing with {num_workers} workers.")
    with mp_pool.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_clip_worker, clips), total=len(clips), desc="Processing all clips"))





if __name__ == "__main__":


    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)

    start = time.time()
    input_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\segmented_data"
    output_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\landmarks"
    Number_of_workers = 8
    print("Workers:", Number_of_workers)
    process_all_clips_parallel(input_root, output_root, num_workers=Number_of_workers)
    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds.")

