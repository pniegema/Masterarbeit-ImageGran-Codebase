import cv2
import mediapipe as mp
import os
import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp_pool  # alias to avoid conflict with mediapipe.mp
import time


mp_face_mesh = mp.solutions.face_mesh

# Face oval (whole face): more reliable than FULL_FACE_IDX
FACE_OVAL_IDX = list(set(range(468)))

# Upper face (exclude mouth & jaw)
UPPER_FACE_IDX = list(range(70, 103))  # rough upper face approximation


CUSTOM_MOUTH_AND_JAW_IDX = [150, 169, 210, 202, 57, 186, 165, 97, 0, 326, 391, 410, 287, 422, 430, 394 ,379, 378, 400, 377, 152, 148, 176, 149]

reference_image_path = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\Referenz_Bild2.jpg"
ref_means = (32.358811310640576, 131.58898418654516, 133.04683060841597)
ref_stds = (58.579945180119964, 5.36601516974633, 9.093122016322967)
reference_image_mouth_path = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\0007_mouth.jpg"
ref_means_mouth = (8.685531616210938, 128.6955108642578, 128.14324951171875)
ref_stds_mouth = (38.52105060532664, 3.1739883836853693, 0.665567843239770)

def create_mask(image_shape, landmarks, indices):
    h, w = image_shape[:2]
    points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])

    if points.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)

    hull = cv2.convexHull(points)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def extract_landmarks(image, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def apply_inverse_mask(image, mask):
    inverse_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image, image, mask=inverse_mask)

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


def process_clip_worker(args):
    clip_dir, output_dir = args

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)
    try:
        process_clip(clip_dir, output_dir, face_mesh)
    finally:
        face_mesh.close()


def process_clip(clip_dir, output_dir, face_mesh, mouth_offset=5):
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

        h, w = image.shape[:2]
        full_face_mask = create_mask(image.shape, landmarks, FACE_OVAL_IDX)
        mouth_mask = create_mask(image.shape, landmarks, CUSTOM_MOUTH_AND_JAW_IDX)

        upper_face = apply_inverse_mask(image, full_face_mask)
        expression_face = apply_mask(image, full_face_mask)
        original_mouth = apply_mask(image, mouth_mask)

        stitched = expression_face.copy()
        if i + mouth_offset < len(frames):
            future_path = os.path.join(clip_dir, frames[i + mouth_offset])
            future_img = cv2.imread(future_path)
            future_landmarks = extract_landmarks(future_img, face_mesh)
            if future_landmarks:
                points1 = get_landmark_points(future_landmarks, CUSTOM_MOUTH_AND_JAW_IDX, w, h)
                points2 = get_landmark_points(landmarks, CUSTOM_MOUTH_AND_JAW_IDX, w, h)

                # Safety checks
                if points1.shape[0] < 3 or points2.shape[0] < 3:
                    continue
                if np.any(points1 < 0) or np.any(points1[:, 0] >= w) or np.any(points1[:, 1] >= h):
                    continue
                if np.any(points2 < 0) or np.any(points2[:, 0] >= w) or np.any(points2[:, 1] >= h):
                    continue

                rect = (0, 0, w, h)
                triangles = get_delaunay_triangles(rect, points2)

                for tri in triangles:
                    t1 = [points1[tri[0]], points1[tri[1]], points1[tri[2]]]
                    t2 = [points2[tri[0]], points2[tri[1]], points2[tri[2]]]
                    warp_triangle(future_img, stitched, t1, t2)

        # Pixelwise augmentations and rotations
        stitched = pixelwise_augmentation(stitched, ref_means, ref_stds)
        stitched = random_rotate_image(stitched, max_angle=15)
        stitched = cv2.resize(stitched, (w, h), interpolation=cv2.INTER_LINEAR)
        upper_face = pixelwise_augmentation(upper_face, ref_means, ref_stds)
        original_mouth = pixelwise_augmentation(original_mouth, ref_means_mouth, ref_stds_mouth)
        original_mouth = random_rotate_image(original_mouth, max_angle=15)
        original_mouth = cv2.resize(original_mouth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Reapply masks to keep background black
        upper_face = apply_inverse_mask(upper_face, full_face_mask)
        stitched = apply_mask(stitched, full_face_mask)
        original_mouth = apply_mask(original_mouth, mouth_mask)

        base = os.path.splitext(frame_name)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_pose.jpg"), upper_face)
        # cv2.imwrite(os.path.join(output_dir, f"{base}_expression.png"), expression_face) # optional raw expression
        cv2.imwrite(os.path.join(output_dir, f"{base}_mouth.jpg"), original_mouth)
        cv2.imwrite(os.path.join(output_dir, f"{base}_expression.jpg"), stitched)

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



def color_normalization(target, ref_means, ref_stds):
    """
    Normalize target image colors to reference using LAB color space statistics.
    target: BGR image np.array
    ref_means, ref_stds: tuples/lists of means and stddevs for (L, A, B) channels
    """
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    l, a, b = cv2.split(target_lab)

    mean_t = [l.mean(), a.mean(), b.mean()]
    std_t = [l.std(), a.std(), b.std()]

    # Normalize each channel
    l_norm = ((l - mean_t[0]) / (std_t[0] + 1e-8)) * ref_stds[0] + ref_means[0]
    a_norm = ((a - mean_t[1]) / (std_t[1] + 1e-8)) * ref_stds[1] + ref_means[1]
    b_norm = ((b - mean_t[2]) / (std_t[2] + 1e-8)) * ref_stds[2] + ref_means[2]

    norm_lab = cv2.merge([np.clip(l_norm, 0, 255),
                          np.clip(a_norm, 0, 255),
                          np.clip(b_norm, 0, 255)]).astype(np.uint8)

    return cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)


def pixelwise_augmentation(img, ref_means, ref_stds, blur_ksize=3, sharpen_amount=1):
    # 1. Color normalization
    img_norm = color_normalization(img, ref_means, ref_stds)

    # 2. Blur
    img_blur = cv2.GaussianBlur(img_norm, (blur_ksize, blur_ksize), 0)

    # 3. Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.addWeighted(img_blur, sharpen_amount,
                                cv2.filter2D(img_blur, -1, kernel), 1 - sharpen_amount, 0)
    return img_sharp



def compute_reference_stats(reference_image_path):
    """ Compute mean and stddev for each LAB channel of the reference image for color normalization."""
    ref_img = cv2.imread(reference_image_path)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(ref_lab)
    ref_means = (l.mean(), a.mean(), b.mean())
    ref_stds = (l.std(), a.std(), b.std())
    return ref_means, ref_stds

def random_rotate_image(image, max_angle=15):
    """
    Rotate the image by a random angle between -max_angle and +max_angle degrees.
    Keeps the entire rotated image inside the frame with black background.

    Args:
        image (np.array): Input image (BGR).
        max_angle (float): Maximum absolute angle for random rotation (degrees).

    Returns:
        np.array: Rotated image.
    """
    angle = random.uniform(-max_angle, max_angle)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the bounding dimensions of the new image
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to consider translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(0,0,0))
    return rotated




if __name__ == "__main__":
    start = time.time()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                       max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5)

    #ref_means, ref_stds = compute_reference_stats(reference_image_path)
    #print("Reference Means:", ref_means)
    #print("Reference Stds:", ref_stds)



    input_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\segmented_data"
    output_root = r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\masked_data"
    Number_of_workers = 6
    print("Workers:", Number_of_workers)
    process_all_clips_parallel(input_root, output_root, num_workers=Number_of_workers)
    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds.")

