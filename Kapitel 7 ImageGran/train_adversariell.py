import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm

import numpy as np
from models import HeatmapEncoder64,  StyleGAN1Generator_256
from utils import VGGPerceptualLoss
from matplotlib import pyplot as plt

from Hybrid_Datensatz import Hybrid_Datensatz
from torchvision import transforms
from utils import show_debug_images
from critic import R3GANCritic, r3gan_d_loss, r3gan_g_loss, r3gan_d_loss_fast, r3gan_d_loss_alternating

from torch.utils.data import DataLoader, random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def visualize_first_eye_crop(img, landmarks, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=2):
    """
    Visualize the first eye crops for debugging (no resizing, normalized [-1,1] image).

    img: [B, C, H, W] tensor normalized [-1,1]
    landmarks: [B, N, 2] tensor normalized [0,1] (x,y)
    left_idxs/right_idxs: landmark index ranges for left/right eyes
    """

    img_0 = img[0:1]  # keep batch dimension
    lm_0 = landmarks[0].detach().cpu().numpy()
    _, _, H, W = img_0.shape

    # convert normalized to pixel coordinates
    lm_0[:, 0] = np.clip(lm_0[:, 0] * W, 0, W - 1)
    lm_0[:, 1] = np.clip(lm_0[:, 1] * H, 0, H - 1)

    def crop_region(lm_points, label):
        x_min = int(lm_points[:, 0].min()) - padding
        y_min = int(lm_points[:, 1].min()) - padding
        x_max = int(lm_points[:, 0].max()) + padding
        y_max = int(lm_points[:, 1].max()) + padding
        crop = img_0[:, :, y_min:y_max, x_min:x_max]
        crop_np = crop.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        crop_np = (crop_np + 1) / 2.0
        crop_np = crop_np.clip(0, 1)
        plt.imshow(crop_np)
        plt.title(f"{label} eye crop: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
        plt.axis("off")
        plt.show()

    # extract regions
    left_eye = lm_0[list(left_idxs)]
    right_eye = lm_0[list(right_idxs)]

    # plot both crops
    crop_region(left_eye, "Left")
    crop_region(right_eye, "Right")

    # also show full image for reference
    full_img = (img_0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) + 1) / 2.0
    plt.imshow(full_img)
    plt.title("Full image with both eyes visible")
    plt.axis("off")
    plt.show()

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


def crop_region(img, lm, idxs, padding=1, out_size=32):
    """
    Allgemeine Region-Crop-Funktion.
    img: [1, C, H, W]
    lm: [N, 2] normalized [0,1]
    idxs: Indices der Region (z.B. Augen oder Mund)
    """
    _, C, H, W = img.shape
    coords = lm[idxs, :].copy()
    coords[:, 0] = np.clip(coords[:, 0] * W, 0, W - 1)
    coords[:, 1] = np.clip(coords[:, 1] * H, 0, H - 1)

    x_min = max(0, int(coords[:, 0].min()) - padding)
    y_min = max(0, int(coords[:, 1].min()) - padding)
    x_max = min(W, int(coords[:, 0].max()) + padding)
    y_max = min(H, int(coords[:, 1].max()) + padding)

    crop = img[:, :, y_min:y_max, x_min:x_max]
    if crop.shape[-1] < 2 or crop.shape[-2] < 2:
        return torch.zeros((1, C, out_size, out_size), device=img.device, dtype=img.dtype)
    crop = F.interpolate(crop, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return crop


def crop_eyes(img, landmarks, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=2):
    """
    Cropt linke + rechte Augenregion, kombiniert sie als Batch.
    """
    B, C, H, W = img.shape
    crops = []

    for b in range(B):
        lm = landmarks[b].detach().cpu().numpy()
        left_crop = crop_region(img[b:b + 1], lm, list(left_idxs), padding)
        right_crop = crop_region(img[b:b + 1], lm, list(right_idxs), padding)
        crops.append(torch.cat([left_crop, right_crop], dim=0))

    return torch.cat(crops, dim=0)


def crop_mouth(img, landmarks, padding=1):
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
            crop = torch.zeros((1, C, 32, 32), device=img.device, dtype=img.dtype)
        else:
            crop = img[b:b + 1, :, y_min:y_max, x_min:x_max]
            crop = F.interpolate(crop, size=(32, 32), mode='bilinear', align_corners=False)


        crops.append(crop)

    return torch.cat(crops, dim=0) if crops else None


class CustomLoss(torch.nn.Module):
    def __init__(self, l1_weight=8.0, perceptual_weight=0.2,
                 mouth_l1_weight=4.0, lpips_weight=0.2, eye_l1_weight=2.0,):
        super(CustomLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.mouth_l1_weight = mouth_l1_weight
        self.lpips_weight = lpips_weight
        #self.mouth_perceptual_weight = mouth_perceptual_weight
        self.perceptual_loss_fn = VGGPerceptualLoss().to(device)
        #self.lpips_loss_fn = lpips.LPIPS(net='squeeze').to(device)
        self.eye_l1_weight = eye_l1_weight



    def forward(self, fake, real, landmarks_mouth=None, landmarks_expr=None):
        fake_128 = F.interpolate(fake, size=(128, 128), mode='bilinear', align_corners=False)
        real_128 = F.interpolate(real, size=(128, 128), mode='bilinear', align_corners=False)

        # --- Global losses ---
        l1_loss = F.l1_loss(fake, real)
        percep_loss = self.perceptual_loss_fn(fake_128, real_128)

        # --- Mouth region losses ---
        mouth_l1 = torch.tensor(0.0, device=fake.device)
        eyes_l1 = torch.tensor(0.0, device=fake.device)
        if landmarks_mouth is not None and landmarks_expr is not None:
            fake_mouth = crop_mouth(fake, landmarks_mouth, padding=2)
            real_mouth = crop_mouth(real, landmarks_mouth, padding=2)
            fake_eyes = crop_eyes(fake, landmarks_expr, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=2)
            real_eyes = crop_eyes(real, landmarks_expr, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=2)



            if fake_mouth is not None and real_mouth is not None:
                mouth_l1 = F.l1_loss(fake_mouth, real_mouth)

            if fake_eyes is not None and real_eyes is not None:
                eyes_l1 = F.l1_loss(fake_eyes, real_eyes)




        # --- Total loss ---
        total_loss = (self.l1_weight * l1_loss +
                      #self.lpips_weight * lpips_loss +
                      self.mouth_l1_weight * mouth_l1 +
                      self.perceptual_weight * percep_loss +
                      self.eye_l1_weight * eyes_l1)

        return total_loss, l1_loss, mouth_l1, percep_loss, eyes_l1#, lpips_loss

# Freeze first blocks helper
def freeze_first_blocks_generator(model, freeze=True):
    # Beispiel für Generator: gefrostete Blocks sind self.progression[0], identity_mapping, motion_mapping bis Layer 1
    # Passe an deine Architektur an
    if freeze:
        for param in model.identity_mapping.parameters():
            param.requires_grad = False
        for param in model.motion_mapping.parameters():
            param.requires_grad = False
        for param in model.progression[0].parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def freeze_first_blocks_encoder(encoder, freeze=True):
    """
    Freeze or unfreeze die ersten Encoder-Blöcke im HeatmapEncoder64.
    Hier werden die ersten drei Conv2d-Blöcke (6 Layers mit Norm+ReLU zusammen) eingefroren,
    also bis (inkl.) Layer 8 in `self.encoder` (Sequenz von Modulen).
    """
    # Die self.encoder ist ein nn.Sequential von Schichten:
    # Index      Layer-Typ
    # 0          Conv2d (in_channels -> first_layer_channels)
    # 1          InstanceNorm2d
    # 2          ReLU
    # 3          Conv2d (first_layer_channels->128)
    # 4          InstanceNorm2d
    # 5          ReLU
    # 6          Conv2d (128->256)
    # 7          InstanceNorm2d
    # 8          ReLU
    # 9, 10, 11 ... weitere Schichten

    # Wir frieren die Layer 0 bis 8 ein (erster bis dritter Block inkl. Norm + Activation)
    freeze_layers = list(range(9))  # 0 bis 8

    for idx, layer in enumerate(encoder.encoder):
        if idx in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = not freeze  # False wenn freeze=True, sonst True

    # Optional: auch fc und scaling einfrieren wenn gewünscht
    if freeze:
        for param in encoder.fc.parameters():
            param.requires_grad = False
        for param in encoder.scaling.parameters():
            param.requires_grad = False
    else:
        for param in encoder.fc.parameters():
            param.requires_grad = True
        for param in encoder.scaling.parameters():
            param.requires_grad = True


def train_visual(
        pose_and_expr_enc, visual_enc,
        G, D,
        opt_G, opt_Enc, opt_D,
        loader,
        epochs=20,
        device='cuda',
        freeze_epochs=5
):
    # ---- Move to device ----
    modules = [G, pose_and_expr_enc, visual_enc, D]
    for m in modules: m.to(device)

    # Set models to train mode
    for m in modules: m.train()

    # ---- AMP ----
    scaler_G = GradScaler()
    scaler_D = GradScaler()

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs, eta_min=1e-6)
    scheduler_Enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_Enc, T_max=epochs, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs, eta_min=1e-6)

    # ---- Logging buffers ----
    l1_losses, vgg_losses, adv_losses, identity_losses, l1_mouth_losses, lpips_losses, eyes_losses = [], [], [], [], [], [], []
    d_losses, r1_losses, r2_losses, base_losses, diff_scores = [], [], [], [], []
    steps = []
    global_step = 0

    # loss weights
    l1_weight, adv_weight = 5.0, 0.025
    l1_weight_mouth = 2 * l1_weight
    perceptual_weight = 0.375
    lpips_weight = 0.25
    eye_l1_weight = 1.5 * l1_weight

    reg_interval = 16  # Paper nutzt 4-16

    loss_fn = CustomLoss(
        l1_weight=l1_weight,
        perceptual_weight=perceptual_weight,
        mouth_l1_weight=l1_weight_mouth,
        lpips_weight=lpips_weight,
        eye_l1_weight=eye_l1_weight
    )

    for epoch in range(epochs):

        epoch_start = time.time()
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        if epoch < freeze_epochs:
            freeze_first_blocks_generator(G, freeze=True)
            freeze_first_blocks_encoder(pose_and_expr_enc, freeze=True)
            freeze_first_blocks_encoder(visual_enc, freeze=True)
        else:
            freeze_first_blocks_generator(G, freeze=False)
            freeze_first_blocks_encoder(pose_and_expr_enc, freeze=False)
            freeze_first_blocks_encoder(visual_enc, freeze=False)

        for i, batch in enumerate(loader_iter):
            global_step += 1
            iter_start = time.perf_counter()

            # ---- Data Transfer ----
            t_data = time.perf_counter()
            id_embedding = batch['identity_embedding'].to(device, non_blocking=True)
            visual_mouth_i = batch['driving_mouth_hm'].to(device, non_blocking=True)
            driving_img = batch['driving_frame'].to(device, non_blocking=True)
            landmarks_mouth = batch['driving_landmarks']['mouth'].to(device, non_blocking=True)
            landmarks_expr = batch['driving_landmarks']['expr'].to(device, non_blocking=True)
            pose_and_expr_hm = batch['driving_pose_and_expr'].to(device, non_blocking=True)
            real_img = batch['real_image'].to(device, non_blocking=True)
            data_time = time.perf_counter() - t_data

            # ---- Encoding ----
            t_enc = time.perf_counter()
            with torch.cuda.amp.autocast():
                pose_and_expr_lat = pose_and_expr_enc(pose_and_expr_hm)
                visual_lat = visual_enc(visual_mouth_i)
                motion_lat = torch.cat([pose_and_expr_lat, visual_lat], dim=1)
            encoding_time = time.perf_counter() - t_enc

            # ---- Generator Forward ----
            t_gen_fwd = time.perf_counter()
            with torch.cuda.amp.autocast():
                fake_img = G(id_embedding, motion_lat)
            gen_forward_time = time.perf_counter() - t_gen_fwd

            # ----- Critic Update (nur alle 2 Steps) -----
            t_critic = time.perf_counter()


            opt_D.zero_grad(set_to_none=True)

            do_r1 = (i % (reg_interval * 2) == 0)
            do_r2 = (i % (reg_interval * 2) == reg_interval)

            with torch.cuda.amp.autocast():
                d_loss, d_loss_terms = r3gan_d_loss_alternating(
                    D, real_img, fake_img.detach(),
                    lambda_r1=10.0,
                    lambda_r2=10.0,
                    do_r1=do_r1,
                    do_r2=do_r2
                )

                D_real = D(real_img)
                D_fake = D(fake_img.detach())
                diff_score = (D_real - D_fake).mean().item()
                if global_step % 2000 == 0:
                    print(D_real)
                    print(D_fake)

                scaler_D.scale(d_loss).backward()
                scaler_D.step(opt_D)
                scaler_D.update()


            critic_time = time.perf_counter() - t_critic

            # ----- Generator Loss Computation -----
            t_loss = time.perf_counter()
            with torch.cuda.amp.autocast():
                fake_img = G(id_embedding, motion_lat)
                total_G_loss, l1_loss, mouth_l1, vgg_loss, eyes_l1 = loss_fn(fake_img, driving_img,
                                                                             landmarks_mouth=landmarks_mouth,
                                                                             landmarks_expr=landmarks_expr)
                g_adv_loss = r3gan_g_loss(D, real_img, fake_img)
                gen_loss = total_G_loss + adv_weight * g_adv_loss
            loss_compute_time = time.perf_counter() - t_loss

            # ----- Generator Backward + Step -----
            t_gen_bwd = time.perf_counter()
            opt_G.zero_grad(set_to_none=True)
            opt_Enc.zero_grad(set_to_none=True)

            scaler_G.scale(gen_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.step(opt_Enc)
            scaler_G.update()
            gen_backward_time = time.perf_counter() - t_gen_bwd

            total_iter_time = time.perf_counter() - iter_start

            # ---- tqdm ----
            loader_iter.set_postfix({
                "G": f"{gen_loss.item():.3f}",
                "D": f"{d_loss.item():.3f}",
                "L1": f"{(l1_loss.item() * l1_weight):.3f}",
                "VGG": f"{(vgg_loss.item() * perceptual_weight):.3f}",
                "MouthL1": f"{(mouth_l1.item() * l1_weight_mouth):.3f}",
                "EyesL1": f"{(eyes_l1.item() * eye_l1_weight):.3f}",
                "Adv": f"{(g_adv_loss.item() * adv_weight):.3f}",
                "T": f"{total_iter_time:.2f}s",
                "Time": f"D={data_time * 1000:.0f}ms,E={encoding_time * 1000:.0f}ms,GF={gen_forward_time * 1000:.0f}ms,C={critic_time * 1000:.0f}ms,L={loss_compute_time * 1000:.0f}ms,GB={gen_backward_time * 1000:.0f}ms"
            })

            # Logging values
            if (i % 100) == 0 and ((epoch == 0 and i > 0) or (epoch > 0)):
                steps.append(global_step)
                l1_losses.append(l1_loss.item() * l1_weight)
                vgg_losses.append(vgg_loss.item() * perceptual_weight)
                l1_mouth_losses.append(mouth_l1.item() * l1_weight_mouth)
                eyes_losses.append(eyes_l1.item() * eye_l1_weight)
                adv_losses.append(g_adv_loss.item() * adv_weight)
                diff_scores.append(diff_score)

            # Debug / plotting
            if i % 2000 == 0:
                print(f"\n[Step {i}] Timing Breakdown:")
                print(f"  Data transfer:    {data_time * 1000:6.1f}ms")
                print(f"  Encoding:         {encoding_time * 1000:6.1f}ms")
                print(f"  Gen Forward:      {gen_forward_time * 1000:6.1f}ms")
                print(f"  Critic Update:    {critic_time * 1000:6.1f}ms")
                print(f"  Loss Compute:     {loss_compute_time * 1000:6.1f}ms")
                print(f"  Gen Backward:     {gen_backward_time * 1000:6.1f}ms")
                print(f"  TOTAL:            {total_iter_time * 1000:6.1f}ms")

                print(f"\n[Step {i}] G: {total_G_loss.item():.4f} | "
                      f"L1: {l1_loss.item() * l1_weight:.4f} | "
                      f"VGG: {vgg_loss.item() * perceptual_weight:.4f} | "
                      f"Mouth L1: {mouth_l1.item() * l1_weight:.4f} | "
                      f"D: {d_loss.item():.4f} | "
                      f"LR_G={opt_G.param_groups[0]['lr']:.2e}, LR_Enc={opt_Enc.param_groups[0]['lr']:.2e}, LR_D={opt_D.param_groups[0]['lr']:.2e}")

                with torch.no_grad():
                    show_debug_images(fake_img, driving_img, title=f"Epoch {epoch + 1} Step {i} (Training)")

                with torch.cuda.amp.autocast():
                    att_weights = G.get_attention_weights(id_embedding, motion_lat)
                print("\nAttention weights per layer (higher = more identity, lower = more motion):")
                for j, w in enumerate(att_weights):
                    print(f"Layer {j} (resolution {4 * 2 ** j}x{4 * 2 ** j}): {w:.3f}")

                # Latent visualization
                id_lat_np = to_numpy_float32(id_embedding[0])
                pose_and_expr_lat_np = to_numpy_float32(pose_and_expr_lat[0])
                visual_lat_np = to_numpy_float32(visual_lat[0])
                id_w_latent_np = to_numpy_float32(G.identity_mapping(id_embedding[0].unsqueeze(0)).squeeze(0))
                with torch.cuda.amp.autocast():
                    motion_w_latent_np = to_numpy_float32(G.motion_mapping(motion_lat[0].unsqueeze(0)).squeeze(0))

                plt.figure(figsize=(12, 8))
                plots = [
                    (id_lat_np, "Identity"),
                    (pose_and_expr_lat_np, "Pose and Expr"),
                    (visual_lat_np, "Visual Mouth"),
                    (id_w_latent_np, "Identity W"),
                    (motion_w_latent_np, "Motion W"),
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
                plt.plot(steps, l1_mouth_losses, label='Mouth L1 * w')
                plt.plot(steps, eyes_losses, label='Eyes L1 * w')
                plt.title(f"Generator Losses - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

                plt.figure(figsize=(14, 6))
                plt.plot(steps, vgg_losses, label='VGG * w')
                plt.plot(steps, adv_losses, label='Adv * w')
                plt.title(f"Generator Losses - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

                # Discriminator losses
                plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1)
                plt.plot(steps, diff_scores, label='Base')
                plt.title(f"Discriminator Diff - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

        scheduler_G.step()
        scheduler_Enc.step()
        scheduler_D.step()
        print(f"[Epoch {epoch + 1}/{epochs}] completed in {time.time() - epoch_start:.1f}s")

        # ---- Save checkpoint ----
        checkpoint = {
            'epoch': epoch + 1,
            'pose_and_expr_enc': pose_and_expr_enc.state_dict(),
            'visual_enc': visual_enc.state_dict(),
            'generator': G.state_dict(),
            'critic': D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_Enc': opt_Enc.state_dict(),
            'opt_D': opt_D.state_dict(),
        }

        torch.save(checkpoint,
                   f'weights_heatmap/last_epoch_Stylegan1_attention_HDTF_only_GAN_v2.pth')

        if epoch % 40 == 0:
            torch.save(checkpoint,
                       f'weights_heatmap/epoch_{epoch + 1}_Stylegan1_attention_HDTF_only_GAN_v2.pth')

def to_numpy_float32(tensor):
    return tensor.detach().cpu().float().numpy()



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
        cache_file="dataset_cache_HDTF_with_id_embeddings_final.pkl"
    )

    test_ratio = 0.1
    test_size = int(len(full_dataset) * test_ratio)
    train_size = len(full_dataset) - test_size

    # reproducible split
    generator_split = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator_split
    )

    print(f"Train: {len(train_dataset)},  Test: {len(test_dataset)}")

    # ---- Dataloaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    pose_and_expr_enc = HeatmapEncoder64(in_channels=175, latent_dim=256, first_layer_channels=128)
    visual_enc = HeatmapEncoder64(in_channels=66, latent_dim=128, first_layer_channels=64)
    generator = StyleGAN1Generator_256(
        identity_dim=512,  # buffalo_sc embedding size
        motion_dim=384,  # pose(128) + mouth(128) + expression(128)
        style_dim=512
    )
    D = R3GANCritic(base_channels=32)

    #Parameters for all Models
    print("Pose and Expr Encoder Parameters:", sum(p.numel() for p in pose_and_expr_enc.parameters() if p.requires_grad))
    print("Visual Encoder Parameters:", sum(p.numel() for p in visual_enc.parameters() if p.requires_grad))
    print("Generator Parameters:", sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print("Discriminator Parameters:", sum(p.numel() for p in D.parameters() if p.requires_grad))


    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.99))
    opt_encoders = torch.optim.Adam(
        list(pose_and_expr_enc.parameters()) +
        list(visual_enc.parameters()),
        lr=0.00005, betas=(0.5, 0.99)
    )
    opt_D = torch.optim.Adam(D.parameters(), lr=0.00005, betas=(0.0, 0.99))

    #Load checkpoint if needed
    checkpoint_path_D = 'weights_heatmap/last_epoch_Stylegan1_attention_HDTF_only_GAN_v2.pth'
    checkpoint_path = 'weights_heatmap/last_epoch_Stylegan1_attention_HDTF_only_GAN_v2.pth'
    checkpoint_D = torch.load(checkpoint_path_D, map_location='cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    pose_and_expr_enc.load_state_dict(checkpoint['pose_and_expr_enc'])
    visual_enc.load_state_dict(checkpoint['visual_enc'])
    D.load_state_dict(checkpoint_D['critic'])

    opt_G.load_state_dict(checkpoint['opt_G'])
    opt_encoders.load_state_dict(checkpoint['opt_Enc'])
    opt_D.load_state_dict(checkpoint['opt_D'])

    #Move optimizers to GPU
    for state in opt_G.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    for state in opt_encoders.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    for state in opt_D.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    # #
    #
    # #LR_G=2.63e-06, LR_Enc=1.86e-06, LR_D=3.17e-06
    #
    # --- Manually adjust learning rates after loading ---
    new_lr_G = 0.00001  # e.g. slightly higher (was 0.0004)
    new_lr_Enc = 0.000005 # e.g. slightly higher (was 0.0002)
    new_lr_D = 0.000005 # e.g. slightly higher (was 0.0002)

    for g in opt_G.param_groups:
       g['lr'] = new_lr_G
    for g in opt_encoders.param_groups:
       g['lr'] = new_lr_Enc
    for g in opt_D.param_groups:
         g['lr'] = new_lr_D


    train_visual(
        pose_and_expr_enc, visual_enc,
        generator, D,
        opt_G=opt_G,
        opt_D=opt_D,
        opt_Enc=opt_encoders,
        loader=train_loader,
        epochs=60,
        device=device,
        freeze_epochs=0
    )