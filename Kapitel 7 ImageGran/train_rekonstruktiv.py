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


from torch.utils.data import DataLoader, random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_first_eye_crop(img, landmarks, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=2):
    """
    Visualize the first eye crops for debugging (no resizing, normalized [-1,1] image).

    img: [B, C, H, W] tensor normalized [-1,1]
    landmarks: [B, N, 2] tensor normalized [0,1] (x,y)
    left_idxs/right_idxs: landmark index ranges for left/right eyes
    """
    import numpy as np
    import matplotlib.pyplot as plt

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
        self.perceptual_loss_fn = VGGPerceptualLoss().to(device)
        #self.lpips_loss_fn = lpips.LPIPS(net='squeeze').to(device)
        self.eye_l1_weight = eye_l1_weight



    def forward(self, fake, real, epoch, landmarks_mouth=None, landmarks_expr=None):

        # --- Global losses ---
        l1_loss = F.l1_loss(fake, real)
        # if epoch <= 60:
        #     fake_128 = F.interpolate(fake, size=(128, 128), mode='bilinear', align_corners=False)
        #     real_128 = F.interpolate(real, size=(128, 128), mode='bilinear', align_corners=False)
        #     percep_loss = self.perceptual_loss_fn(fake_128, real_128)
        # elif epoch > 60:
        fake_224 = F.interpolate(fake, size=(224, 224), mode='bilinear', align_corners=False)
        real_224 = F.interpolate(real, size=(224, 224), mode='bilinear', align_corners=False)
        percep_loss = self.perceptual_loss_fn(fake_224, real_224)

        # --- Mouth region losses ---
        if landmarks_mouth is not None and landmarks_expr is not None:
            fake_mouth = crop_mouth(fake, landmarks_mouth, padding=2)
            real_mouth = crop_mouth(real, landmarks_mouth, padding=2)
            fake_eyes = crop_eyes(fake, landmarks_expr, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=5)
            real_eyes = crop_eyes(real, landmarks_expr, left_idxs=range(37, 53), right_idxs=range(71, 86), padding=5)



            if fake_mouth is not None and real_mouth is not None:
                mouth_l1 = F.l1_loss(fake_mouth, real_mouth)
            else:
                mouth_l1 = torch.tensor(0.0, device=fake.device)

            if fake_eyes is not None and real_eyes is not None:
                eyes_l1 = F.l1_loss(fake_eyes, real_eyes)
            else:
                eyes_l1 = torch.tensor(0.0, device=fake.device)
        else:
            mouth_l1 = torch.tensor(0.0, device=fake.device)
            eyes_l1 = torch.tensor(0.0, device=fake.device)




        # --- Total loss ---
        total_loss = (self.l1_weight * l1_loss +
                      self.mouth_l1_weight * mouth_l1 +
                      self.perceptual_weight * percep_loss +
                      self.eye_l1_weight * eyes_l1)

        return total_loss, l1_loss, mouth_l1, percep_loss, eyes_l1#, lpips_loss


def train_visual(
        pose_and_expr_enc, visual_enc,
        G,
        opt_G, opt_Enc,
        loader,
        epochs=20,
        device='cuda'
):
    # ---- Move to device ----
    modules = [G, pose_and_expr_enc, visual_enc]
    for m in modules: m.to(device)

    # Set models to train mode
    for m in modules: m.train()

    # ---- AMP ----
    scaler_G = GradScaler()

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs, eta_min=1e-6)
    scheduler_Enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_Enc, T_max=epochs, eta_min=1e-6)

    # ---- Logging buffers ----
    l1_losses, vgg_losses, adv_losses, identity_losses, l1_mouth_losses, lpips_losses, eye_losses = [], [], [], [], [], [], []
    steps = []
    global_step = 0

    # loss weights
    l1_weight, adv_weight = 5.0, 0.3
    l1_weight_mouth = 2 * l1_weight
    perceptual_weight = 0.375
    lpips_weight = 0.25
    l1_weigth_eyes = 1.5 * l1_weight

    loss_fn = CustomLoss(
        l1_weight=l1_weight,
        perceptual_weight=perceptual_weight,
        mouth_l1_weight=l1_weight_mouth,
        lpips_weight=lpips_weight,
        eye_l1_weight=l1_weigth_eyes,
    )

    for epoch in range(epochs):

        epoch_start = time.time()
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for i, batch in enumerate(loader_iter):
            global_step += 1
            iter_start = time.time()

            # ---- Data ----
            t0 = time.time()
            id_embedding = batch['identity_embedding'].to(device, non_blocking=True)
            visual_mouth_i = batch['driving_mouth_hm'].to(device, non_blocking=True)
            driving_img = batch['driving_frame'].to(device, non_blocking=True)
            landmarks_mouth = batch['driving_landmarks']['mouth'].to(device, non_blocking=True)
            landmarks_expr = batch['driving_landmarks']['expr'].to(device, non_blocking=True)
            pose_and_expr_hm = batch['driving_pose_and_expr'].to(device, non_blocking=True)
            data_time = time.time() - t0

            # ---- Encode ----
            t0 = time.time()
            pose_and_expr_lat = pose_and_expr_enc(pose_and_expr_hm)
            visual_lat = visual_enc(visual_mouth_i)
            encoding_time = time.time() - t0

            motion_lat = torch.cat([pose_and_expr_lat, visual_lat], dim=1)  # (B, 512)

            t_gen_start = time.time()
            opt_G.zero_grad(set_to_none=True)
            opt_Enc.zero_grad(set_to_none=True)

            with autocast():
                t0 = time.time()
                fake = G(id_embedding, motion_lat)
                t_forward = time.time() - t0

                t0 = time.time()
                # Reconstruction losses
                total_G_loss, l1_loss, mouth_l1, vgg_loss, eye_loss = loss_fn(
                    fake, driving_img, epoch,landmarks_mouth=landmarks_mouth, landmarks_expr=landmarks_expr
                )
                t_loss = time.time() - t0

            t0 = time.time()
            scaler_G.scale(total_G_loss).backward()
            t_backward = time.time() - t0

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
                "VGG": f"{(vgg_loss.item() * perceptual_weight):.3f}",
                "MouthL1": f"{(mouth_l1.item() * l1_weight_mouth):.3f}",
                "EyeL1": f"{(eye_loss.item() * l1_weigth_eyes):.3f}",
                "T": f"{total_iter_time:.2f}s",
                "Timing": f"D={data_time:.2f}s,E={encoding_time:.2f}s,Gen={t_gen_total:.2f}s, F={t_forward:.2f}s, L={t_loss:.2f}s, B={t_backward:.2f}s"
            })

            # Logging values
            if (i % 100) == 0 and ((epoch == 0 and i > 200) or (epoch > 0)):
                steps.append(global_step)
                l1_losses.append(l1_loss.item() * l1_weight)
                vgg_losses.append(vgg_loss.item() * perceptual_weight)
                l1_mouth_losses.append(mouth_l1.item() * l1_weight_mouth)
                eye_losses.append(eye_loss.item() * l1_weigth_eyes)


            # Debug / plotting
            if i % 1000 == 0:
                print(f"[Step {i}] G: {total_G_loss.item():.4f} | "
                      f"L1: {l1_loss.item() * l1_weight:.4f} | "
                      f"VGG: {vgg_loss.item() * perceptual_weight:.4f} | "
                      f"Mouth L1: {mouth_l1.item() * l1_weight:.4f} |"
                      f"LR_G={opt_G.param_groups[0]['lr']:.2e}, LR_Enc={opt_Enc.param_groups[0]['lr']:.2e}")

                with torch.no_grad():
                    show_debug_images(fake, driving_img, title=f"Epoch {epoch + 1} Step {i} (Training)")

                att_weights = generator.get_attention_weights(id_embedding, motion_lat)
                print("\nAttention weights per layer (higher = more identity, lower = more motion):")
                for i, w in enumerate(att_weights):
                    print(f"Layer {i} (resolution {4 * 2 ** i}x{4 * 2 ** i}): {w:.3f}")

                # Latent visualization
                id_lat_np = to_numpy_float32(id_embedding[0])
                pose_and_expr_lat_np = to_numpy_float32(pose_and_expr_lat[0])
                visual_lat_np = to_numpy_float32(visual_lat[0])
                id_w_latent_np = to_numpy_float32(G.identity_mapping(id_embedding[0].unsqueeze(0)).squeeze(0))
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
                plt.plot(steps, vgg_losses, label='Vgg * w')
                plt.title(f"Generator Losses - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()

                plt.figure(figsize=(14, 6))
                plt.subplot(1, 2, 1)
                plt.plot(steps, l1_mouth_losses, label='Mouth L1 * w')
                plt.plot(steps, eye_losses, label='Eye L1 * w')
                plt.title(f"Generator Losses - Epoch {epoch + 1} Step {i}")
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()


        scheduler_G.step()
        scheduler_Enc.step()
        print(f"[Epoch {epoch + 1}/{epochs}] completed in {time.time() - epoch_start:.1f}s")

        # ---- Save checkpoint ----
        checkpoint = {
            'epoch': epoch + 1,
            'pose_and_expr_enc': pose_and_expr_enc.state_dict(),
            'visual_enc': visual_enc.state_dict(),
            'generator': G.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_Enc': opt_Enc.state_dict(),
        }

        torch.save(checkpoint, f'weights_heatmap/last_epoch_Stylegan1_attention_refined_HDTF_only.pth')

        if epoch % 40 == 0:
            torch.save(checkpoint, f'weights_heatmap/epoch_{epoch + 1}_Stylegan1_attention_refined_HDTF_only.pth')


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
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
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

    #Parameters for all Models
    print("Pose and Expr Encoder Parameters:", sum(p.numel() for p in pose_and_expr_enc.parameters() if p.requires_grad))
    print("Visual Encoder Parameters:", sum(p.numel() for p in visual_enc.parameters() if p.requires_grad))
    print("Generator Parameters:", sum(p.numel() for p in generator.parameters() if p.requires_grad))


    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.0, 0.99))
    opt_encoders = torch.optim.Adam(
        list(pose_and_expr_enc.parameters()) +
        list(visual_enc.parameters()),
        lr=0.00005, betas=(0.5, 0.99) #Ehemals 0.00008
    )

    #Load checkpoint if needed
    checkpoint_path = 'weights_heatmap/last_epoch_Stylegan1_attention_pose_and_expr_enc_256_v2.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    pose_and_expr_enc.load_state_dict(checkpoint['pose_and_expr_enc'])
    visual_enc.load_state_dict(checkpoint['visual_enc'])

    #opt_G.load_state_dict(checkpoint['opt_G'])
    #opt_encoders.load_state_dict(checkpoint['opt_Enc'])

    #Move optimizers to GPU
    #for state in opt_G.state.values():
    #    for k, v in state.items():
    #        if isinstance(v, torch.Tensor):
    #            state[k] = v.to(device)
    #for state in opt_encoders.state.values():
    #    for k, v in state.items():
    #        if isinstance(v, torch.Tensor):
    #            state[k] = v.to(device)



    train_visual(
        pose_and_expr_enc, visual_enc,
        generator,
        opt_G=opt_G,
        opt_Enc=opt_encoders,
        loader=train_loader,
        epochs=160,
        device=device
    )