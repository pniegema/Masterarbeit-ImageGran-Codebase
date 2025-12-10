import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm

import numpy as np
from models import HeatmapEncoder64, IdentityEncoder, StyleGenerator
from utils import VGGPerceptualLoss, normalize_vgg
from matplotlib import pyplot as plt
from Hybrid_Datensatz import HybridDataset
from torchvision import transforms
from utils import show_debug_images

from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_first_crop(img, landmarks):
    """
    Visualize the first mouth crop for debugging (no resizing, normalized [-1,1] image).

    img: [B, C, H, W] tensor normalized [-1,1]
    landmarks: [B, N, 2] tensor normalized [0,1] (x,y)
    """
    # Take first sample
    img_0 = img[0:1]  # keep batch dim
    lm_0 = landmarks[0].detach().cpu().numpy()

    _, _, H, W = img_0.shape

    # Convert normalized coordinates to pixels
    lm_0[:, 0] = np.clip(lm_0[:, 0] * W, 0, W - 1)
    lm_0[:, 1] = np.clip(lm_0[:, 1] * H, 0, H - 1)

    x_min, y_min = int(lm_0[:, 0].min()) - 1, int(lm_0[:, 1].min()) - 1
    x_max, y_max = int(lm_0[:, 0].max()) + 1, int(lm_0[:, 1].max()) + 1

    # Crop
    crop = img_0[:, :, y_min:y_max, x_min:x_max]
    crop_np = crop.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    # Denormalize [-1,1] -> [0,1] for matplotlib
    crop_np = (crop_np + 1) / 2.0
    crop_np = crop_np.clip(0, 1)

    plt.imshow(crop_np)
    plt.title(f"First mouth crop: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
    plt.axis("off")
    plt.show()

    plt.imshow((img_0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) + 1) / 2.0)
    plt.title("First full image")
    plt.axis("off")
    plt.show()


class CustomLoss(torch.nn.Module):
    def __init__(self, identity_enc, l1_weight=8.0, perceptual_weight=0.2, identity_weight=1.0,
                 mouth_l1_weight=4.0, mouth_perceptual_weight=0.1):
        super(CustomLoss, self).__init__()
        self.identity_enc = identity_enc
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
        self.mouth_l1_weight = mouth_l1_weight
        self.mouth_perceptual_weight = mouth_perceptual_weight
        self.perceptual_loss_fn = VGGPerceptualLoss().to(device)

    def crop_mouth(self, img, landmarks, padding=10):
        """
        img: [B, C, H, W] tensor
        landmarks: [B, N, 2] tensor normalized [0,1] (x,y)
        padding: extra pixels around mouth region
        """
        B, C, H, W = img.shape
        crops = []

        for b in range(B):
            lm = landmarks[b].detach().cpu().numpy()
            # Convert normalized [0,1] -> pixel coordinates
            lm[:, 0] = np.clip(lm[:, 0] * W, 0, W - 1)
            lm[:, 1] = np.clip(lm[:, 1] * H, 0, H - 1)

            # Add padding around mouth region
            x_min = max(0, int(lm[:, 0].min()) - padding)
            y_min = max(0, int(lm[:, 1].min()) - padding)
            x_max = min(W, int(lm[:, 0].max()) + padding)
            y_max = min(H, int(lm[:, 1].max()) + padding)

            if x_max <= x_min or y_max <= y_min:
                # invalid crop → fallback
                crop = torch.zeros((1, C, 64, 64), device=img.device, dtype=img.dtype)
            else:
                crop = img[b:b + 1, :, y_min:y_max, x_min:x_max]
                crop = F.interpolate(crop, size=(64, 64), mode='bilinear', align_corners=False)

            crops.append(crop)

        return torch.cat(crops, dim=0) if crops else None

    def forward(self, fake, real, landmarks=None):
        # Resize real from 256x256 to 128x128
        real = F.interpolate(real, size=(128, 128), mode='bilinear', align_corners=False)
        fake = F.interpolate(fake, size=(128, 128), mode='bilinear', align_corners=False)

        # --- Global losses ---
        l1_loss = F.l1_loss(fake, real)

        fake_norm = normalize_vgg(fake)
        real_norm = normalize_vgg(real)
        percep_loss = self.perceptual_loss_fn(fake_norm, real_norm)

        # --- Mouth region losses ---
        mouth_l1 = mouth_percep = torch.tensor(0.0, device=fake.device)
        if landmarks is not None:
            fake_mouth = self.crop_mouth(fake, landmarks, padding=10)
            real_mouth = self.crop_mouth(real, landmarks, padding=10)

            if fake_mouth is not None and real_mouth is not None:
                mouth_l1 = F.l1_loss(fake_mouth, real_mouth)

        # --- Total loss ---
        total_loss = (self.l1_weight * l1_loss +
                      self.perceptual_weight * percep_loss +
                      self.mouth_l1_weight * mouth_l1)

        return total_loss, l1_loss, percep_loss, mouth_l1


def train_visual(
        identity_enc, pose_enc, visual_enc, expr_enc,
        G,
        opt_G, opt_Enc,
        loader,
        epochs=20,
        device='cuda'
):
    # ---- Move to device ----
    modules = [identity_enc, G, pose_enc, visual_enc, expr_enc]
    for m in modules: m.to(device)

    # Set models to train mode
    for m in modules: m.train()

    # ---- AMP ----
    scaler_G = GradScaler()

    # ---- Logging buffers ----
    l1_losses, vgg_losses, adv_losses, identity_losses, l1_mouth_losses = [], [], [], [], []
    steps = []
    global_step = 0

    # loss weights
    l1_weight, adv_weight = 10.0, 0.3
    l1_weight_mouth = 1.5 * l1_weight
    perceptual_weight = 0.20
    identity_weight = 1

    loss_fn = CustomLoss(
        identity_enc,
        l1_weight=l1_weight,
        perceptual_weight=perceptual_weight,
        identity_weight=identity_weight,
        mouth_l1_weight=l1_weight_mouth,
        mouth_perceptual_weight=perceptual_weight
    )

    for epoch in range(epochs):

        epoch_start = time.time()
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for i, batch in enumerate(loader_iter):
            global_step += 1
            iter_start = time.time()

            # ---- Data ----
            t0 = time.time()
            id_i = batch['identity_frames'].to(device)
            pose_i = batch['driving_pose_hm'].to(device)
            expr_i = batch['driving_expression_hm'].to(device)
            visual_mouth_i = batch['driving_mouth_hm'].to(device)
            driving_img = batch['driving_frame'].to(device)

            landmarks_mouth = batch['driving_landmarks']['mouth'].to(device)
            data_time = time.time() - t0

            # ---- Encode ----
            t0 = time.time()
            id_lat = identity_enc(id_i)
            pose_lat = pose_enc(pose_i)
            visual_lat = visual_enc(visual_mouth_i)
            expr_lat = expr_enc(expr_i)
            encoding_time = time.time() - t0

            z = torch.cat([id_lat, pose_lat, visual_lat, expr_lat], dim=1)

            t_gen_start = time.time()
            opt_G.zero_grad(set_to_none=True)
            opt_Enc.zero_grad(set_to_none=True)

            with autocast():
                fake = G(z)

                # Reconstruction losses
                total_G_loss, l1_loss, percep_loss, mouth_l1 = loss_fn(
                    fake, driving_img, landmarks=landmarks_mouth
                )


            scaler_G.scale(total_G_loss).backward()

            # Unscale and step both optimizers
            scaler_G.unscale_(opt_G)
            scaler_G.unscale_(opt_Enc)

            # Optional: gradient clipping
            # torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(list(identity_enc.parameters()) +
            #                                list(pose_enc.parameters()) +
            #                                list(visual_enc.parameters()), max_norm=1.0)

            scaler_G.step(opt_G)
            scaler_G.step(opt_Enc)
            scaler_G.update()

            t_gen_total = time.time() - t_gen_start
            total_iter_time = time.time() - iter_start

            # ---- tqdm ----
            loader_iter.set_postfix({
                "G": f"{total_G_loss.item():.3f}",
                "L1": f"{(l1_loss.item() * l1_weight):.3f}",
                "Perc": f"{(percep_loss.item() * perceptual_weight):.3f}",
                "MouthL1": f"{(mouth_l1.item() * l1_weight_mouth):.3f}",
                "T": f"{total_iter_time:.2f}s",

            })

            if i < 3:
                print(f"[Step {i}] DL={data_time:.2f}s, Enc={encoding_time:.2f}s, "
                      f"Gen={t_gen_total:.2f}s, Total={total_iter_time:.2f}s, "
                      f"LR_G={opt_G.param_groups[0]['lr']:.2e}, LR_Enc={opt_Enc.param_groups[0]['lr']:.2e}")

            # Logging values
            if (i % 10) == 0 and ((epoch == 0 and i > 200) or (epoch > 0)):
                steps.append(global_step)
                l1_losses.append(l1_loss.item() * l1_weight)
                vgg_losses.append(percep_loss.item() * perceptual_weight)
                l1_mouth_losses.append(mouth_l1.item() * l1_weight_mouth)

            # Debug / plotting
            if i % 1000 == 0:
                print(f"[Step {i}] G: {total_G_loss.item():.4f} | "
                      f"L1: {l1_loss.item() * l1_weight:.4f} | "

                      f"Perc: {percep_loss.item() * perceptual_weight:.4f} | "
                      f"Mouth L1: {mouth_l1.item() * l1_weight:.4f}")

                with torch.no_grad():
                    show_debug_images(fake, driving_img, title=f"Epoch {epoch + 1} Step {i} (Training)")

                # Latent visualization
                id_lat_np = to_numpy_float32(id_lat[0])
                pose_lat_np = to_numpy_float32(pose_lat[0])
                visual_lat_np = to_numpy_float32(visual_lat[0])
                expr_lat_np = to_numpy_float32(expr_lat[0])
                z_latent_np = to_numpy_float32(z[0])
                w_latent_np = to_numpy_float32(G.MappingNetwork(z[0].unsqueeze(0)).squeeze(0))

                plt.figure(figsize=(12, 8))
                plots = [
                    (id_lat_np, "Identity"),
                    (pose_lat_np, "Pose"),
                    (visual_lat_np, "Visual Mouth"),
                    (expr_lat_np, "Expression"),
                    (z_latent_np, "Concatenated Z"),
                    (w_latent_np, "Mapped W"),
                ]

                for idx, (arr, title) in enumerate(plots, start=1):
                    plt.subplot(3, 2, idx)
                    plt.title(title)
                    plt.plot(arr)
                    plt.xlabel("Index")
                    plt.ylabel("Value")

                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

                # Loss plots
                plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1)
                plt.plot(steps, l1_losses, label='L1 * w')
                plt.plot(steps, vgg_losses, label='Perceptual * w')
                plt.plot(steps, l1_mouth_losses, label='Mouth L1 * w')
                plt.title(f"Generator Losses - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

        print(f"[Epoch {epoch + 1}/{epochs}] completed in {time.time() - epoch_start:.1f}s")

        # ---- Save checkpoint ----
        checkpoint = {
            'epoch': epoch + 1,
            'identity_enc': identity_enc.state_dict(),
            'pose_enc': pose_enc.state_dict(),
            'visual_enc': visual_enc.state_dict(),
            'expr_enc': expr_enc.state_dict(),
            'generator': G.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_Enc': opt_Enc.state_dict(),
        }

        torch.save(checkpoint, f'weights_heatmap/last_epoch_Stylegan1.pth')

        if epoch % 5 == 0:
            torch.save(checkpoint, f'weights_heatmap/epoch_{epoch + 1}_Stylegan1.pth')


def to_numpy_float32(tensor):
    return tensor.detach().cpu().float().numpy()




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HybridDataset(
        root_dir=r"C:\Users\pnieg\Documents\HDTF\HDTF_dataset",
        transform=transform,
        min_frames=10,
        cache_file="dataset_cache_HDTF.pkl"
    )

    dataloader = DataLoader(dataset, batch_size=64, num_workers=4,
                            shuffle=True, pin_memory=True if device == 'cuda' else False,
                            persistent_workers=True)

    pose_enc = HeatmapEncoder64(in_channels=47, latent_dim=128)
    identity_enc = IdentityEncoder(in_channels=3, latent_dim=512)
    expr_enc = HeatmapEncoder64(in_channels=128, latent_dim=256, first_layer_channels=128)
    visual_enc = HeatmapEncoder64(in_channels=66, latent_dim=128, first_layer_channels=128)
    generator = StyleGenerator(latent_dim=1024, style_dim=512)

    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.99))
    opt_encoders = torch.optim.Adam(
        list(identity_enc.parameters()) +
        list(pose_enc.parameters()) +
        list(expr_enc.parameters()) +
        list(visual_enc.parameters()),
        lr=0.0001, betas=(0.5, 0.99)
    )


    train_visual(
        identity_enc, pose_enc, visual_enc, expr_enc,
        generator,
        opt_G=opt_G,
        opt_Enc=opt_encoders,
        loader=dataloader,
        epochs=120,
        device=device
    )