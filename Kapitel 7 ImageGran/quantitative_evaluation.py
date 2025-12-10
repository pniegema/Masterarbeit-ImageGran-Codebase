import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import csv
import numpy as np
from torchvision import transforms
from models import HeatmapEncoder64,  StyleGAN1Generator_256
from Hybrid_Datensatz import Hybrid_Datensatz
from torch.utils.data import DataLoader, random_split
from PIL import Image
import cv2

# Metrik-Libs
from piq import ssim, psnr
import lpips
from torch_fidelity import calculate_metrics

# === DLIB für M-LMD ===
import dlib

# Dlib Landmark-Predictor (muss heruntergeladen werden)
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


# Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


def get_dlib_mouth_landmarks(image_np, detector, predictor):
    """
    Extrahiert die 20 Mund-Landmarks (Indizes 48-67) mit Dlib.

    Args:
        image_np: numpy array (H, W, 3), RGB, uint8 [0-255]
        detector: dlib.get_frontal_face_detector()
        predictor: dlib.shape_predictor(path)

    Returns:
        numpy array (20, 2) mit (x, y) Koordinaten oder None bei Fehler
    """
    # Dlib erwartet BGR
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Gesicht detektieren
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None  # Kein Gesicht gefunden

    # Nehme erstes Gesicht (bei Talking-Head meist nur eins)
    face = faces[0]

    # Landmarks extrahieren
    shape = predictor(gray, face)

    # Mund-Landmarks (48-67) als numpy array
    mouth_landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                                for i in range(48, 68)])

    return mouth_landmarks


def calculate_mlmd_batch(fake_images, real_images, detector, predictor):
    """
    Berechnet M-LMD für einen Batch.

    Args:
        fake_images: torch.Tensor (B, C, H, W) in [0, 1]
        real_images: torch.Tensor (B, C, H, W) in [0, 1]
        detector: dlib face detector
        predictor: dlib landmark predictor

    Returns:
        list of float: M-LMD pro Sample (None wenn Landmark-Detektion fehlschlägt)
    """
    mlmd_values = []

    for i in range(fake_images.shape[0]):
        # Konvertiere zu numpy [0, 255] RGB
        fake_np = (fake_images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        real_np = (real_images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # Extrahiere Landmarks
        fake_mouth = get_dlib_mouth_landmarks(fake_np, detector, predictor)
        real_mouth = get_dlib_mouth_landmarks(real_np, detector, predictor)

        # Berechne Distanz falls beide erfolgreich
        if fake_mouth is not None and real_mouth is not None:
            # Euklidische Distanz pro Landmark, dann Mittelwert
            distances = np.linalg.norm(fake_mouth - real_mouth, axis=1)
            mlmd = float(np.mean(distances))
            mlmd_values.append(mlmd)
        else:
            mlmd_values.append(None)  # Fehler bei Detektion

    return mlmd_values


def evaluate_model(
        G,
        pose_enc, visual_enc,
        dataloader,
        device='cuda',
        csv_out="eval_results.csv",
        max_batches=None,
        compute_fid=True,
        compute_mlmd=True
):
    G.to(device).eval()
    pose_enc.to(device).eval()
    visual_enc.to(device).eval()

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # === DLIB initialisieren ===
    dlib_detector = None
    dlib_predictor = None

    if compute_mlmd:
        try:
            dlib_detector = dlib.get_frontal_face_detector()
            dlib_predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
            print(f"✓ Dlib Landmark-Predictor geladen von {DLIB_PREDICTOR_PATH}")
        except Exception as e:
            print(f"⚠ Warnung: Dlib-Predictor konnte nicht geladen werden: {e}")
            print(
                f"   M-LMD wird übersprungen. Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            compute_mlmd = False

    # Sammel-Arrays
    all_ssim = []
    all_psnr = []
    all_lpips = []
    all_mlmd = []

    # Temporäre Verzeichnisse für FID
    if compute_fid:
        temp_real_dir = Path("temp_eval_real")
        temp_fake_dir = Path("temp_eval_fake")
        temp_real_dir.mkdir(parents=True, exist_ok=True)
        temp_fake_dir.mkdir(parents=True, exist_ok=True)

        for f in temp_real_dir.glob("*.png"):
            f.unlink()
        for f in temp_fake_dir.glob("*.png"):
            f.unlink()

        img_counter = 0

    # CSV
    csv_file = Path(csv_out)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(csv_file, 'w', newline='')
    csv_writer = csv.writer(csv_f)

    # Erweiterte Header
    header = ['sample_id', 'ssim', 'psnr', 'lpips']
    if compute_mlmd:
        header.append('mlmd')
    csv_writer.writerow(header)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
            if (max_batches is not None) and (batch_idx >= max_batches):
                break

            # Daten auf Device
            id_embedding = batch['identity_embedding'].to(device, non_blocking=True)
            pose_and_expr_hm = batch['driving_pose_and_expr'].to(device, non_blocking=True)
            visual_mouth_i = batch['driving_mouth_hm'].to(device, non_blocking=True)
            gt_imgs = batch['driving_frame'].to(device, non_blocking=True)  # in [-1,1]

            # Motion-latents erzeugen
            pose_lat = pose_enc(pose_and_expr_hm)
            vis_lat = visual_enc(visual_mouth_i)
            motion_lat = torch.cat([pose_lat, vis_lat], dim=1)

            # Generiere Bilder
            fake_imgs = G(id_embedding, motion_lat)  # [-1,1]

            # Für SSIM/PSNR/LPIPS -> [0,1]
            fake_01 = (fake_imgs + 1.0) / 2.0
            gt_01 = (gt_imgs + 1.0) / 2.0

            # === M-LMD für gesamten Batch ===
            mlmd_batch = []
            if compute_mlmd:
                mlmd_batch = calculate_mlmd_batch(
                    fake_01,
                    gt_01,
                    dlib_detector,
                    dlib_predictor
                )

            # Speichere Bilder für FID
            if compute_fid:
                for i in range(gt_imgs.shape[0]):
                    img_real = (gt_01[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(img_real).save(temp_real_dir / f"{img_counter:06d}.png")

                    img_fake = (fake_01[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(img_fake).save(temp_fake_dir / f"{img_counter:06d}.png")

                    img_counter += 1

            # Per-Sample Metriken
            for i in range(gt_imgs.shape[0]):
                f = fake_imgs[i:i + 1]  # [-1,1]
                g = gt_imgs[i:i + 1]
                f01 = fake_01[i:i + 1]  # [0,1]
                g01 = gt_01[i:i + 1]

                s = float(ssim(f01, g01, data_range=1.0).item())
                p = float(psnr(f01, g01, data_range=1.0).item())
                lp = float(lpips_fn(f, g).mean().item())

                all_ssim.append(s)
                all_psnr.append(p)
                all_lpips.append(lp)

                # CSV-Zeile
                row = [f"b{batch_idx}_i{i}", s, p, lp]

                # M-LMD hinzufügen
                if compute_mlmd:
                    mlmd_val = mlmd_batch[i]
                    if mlmd_val is not None:
                        all_mlmd.append(mlmd_val)
                        row.append(mlmd_val)
                    else:
                        row.append('N/A')  # Landmark-Detektion fehlgeschlagen

                csv_writer.writerow(row)

    csv_f.close()

    # FID berechnen
    fid_score = None
    if compute_fid:
        print("\nBerechne FID mit torch-fidelity...")
        metrics = calculate_metrics(
            input1=str(temp_fake_dir),
            input2=str(temp_real_dir),
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
            batch_size=32
        )
        fid_score = metrics['frechet_inception_distance']

        # Cleanup
        print("Räume temporäre Dateien auf...")
        import shutil
        shutil.rmtree(temp_real_dir)
        shutil.rmtree(temp_fake_dir)

    # Zusammenfassung
    results = {
        'SSIM_mean': float(np.mean(all_ssim)),
        'PSNR_mean': float(np.mean(all_psnr)),
        'LPIPS_mean': float(np.mean(all_lpips)),
        'N_samples': len(all_ssim)
    }

    if fid_score is not None:
        results['FID'] = float(fid_score)

    if compute_mlmd and len(all_mlmd) > 0:
        results['M-LMD_mean'] = float(np.mean(all_mlmd))
        results['M-LMD_samples'] = len(all_mlmd)
        success_rate = len(all_mlmd) / len(all_ssim) * 100
        results['M-LMD_success_rate'] = f"{success_rate:.1f}%"

    return results


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = Hybrid_Datensatz(
        root_dir=r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset",
        transform=transform,
        min_frames=10,
        cache_file="dataset_cache_HDTF_with_id_embeddings_HDTF_only.pkl"
    )

    test_ratio = 0.1
    test_size = int(len(full_dataset) * test_ratio)
    train_size = len(full_dataset) - test_size

    generator_split = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator_split
    )

    print(f"Train: {len(train_dataset)},  Test: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    pose_and_expr_enc = HeatmapEncoder64(in_channels=175, latent_dim=256, first_layer_channels=128)
    visual_enc = HeatmapEncoder64(in_channels=66, latent_dim=128, first_layer_channels=64)
    generator = StyleGAN1Generator_256(
        identity_dim=512,
        motion_dim=384,
        style_dim=512
    )

    print("Pose+Expr Encoder Params:", sum(p.numel() for p in pose_and_expr_enc.parameters() if p.requires_grad))
    print("Visual Encoder Params:", sum(p.numel() for p in visual_enc.parameters() if p.requires_grad))
    print("Generator Params:", sum(p.numel() for p in generator.parameters() if p.requires_grad))

    # Lade Checkpoint
    checkpoint_path = 'weights_heatmap/last_epoch_Stylegan1_attention__GAN_v2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    pose_and_expr_enc.load_state_dict(checkpoint['pose_and_expr_enc'])
    visual_enc.load_state_dict(checkpoint['visual_enc'])

    # Evaluation
    results = evaluate_model(
        G=generator,
        pose_enc=pose_and_expr_enc,
        visual_enc=visual_enc,
        dataloader=test_loader,
        device='cuda',
        csv_out="eval_results_with_mlmd.csv",
        max_batches=None,  # Setze z.B. auf 10 für schnellen Test
        compute_fid=True,
        compute_mlmd=True
    )

    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")