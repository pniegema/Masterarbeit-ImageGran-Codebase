import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import IdentityDecoder, PoseEncoder, IdentityEncoder, ExpressionEncoder, VisualEncoder
from utils import VGGPerceptualLoss
from matplotlib import pyplot as plt
from Talking-Head-kH-Datensatz import TalkingHeadDataset
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
    lm_0[:, 0] = np.clip(lm_0[:, 0] * W, 0, W-1)
    lm_0[:, 1] = np.clip(lm_0[:, 1] * H, 0, H-1)

    x_min, y_min = int(lm_0[:,0].min()) -1, int(lm_0[:,1].min()) -1
    x_max, y_max = int(lm_0[:,0].max()) +1, int(lm_0[:,1].max()) +1

    # Crop
    crop = img_0[:, :, y_min:y_max, x_min:x_max]
    crop_np = crop.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    # Denormalize [-1,1] -> [0,1] for matplotlib
    crop_np = (crop_np + 1) / 2.0
    crop_np = crop_np.clip(0,1)

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

    def crop_mouth(self, img, landmarks):
        """
        img: [B, C, H, W] tensor
        landmarks: [B, N, 2] tensor normalized [0,1] (x,y)
        """
        B, C, H, W = img.shape
        crops = []

        for b in range(B):
            lm = landmarks[b].detach().cpu().numpy()
            # Convert normalized [0,1] -> pixel coordinates
            lm[:, 0] = np.clip(lm[:, 0] * W, 0, W - 1)
            lm[:, 1] = np.clip(lm[:, 1] * H, 0, H - 1)

            x_min, y_min = int(lm[:, 0].min()), int(lm[:, 1].min())
            x_max, y_max = int(lm[:, 0].max()), int(lm[:, 1].max())


            if x_max <= x_min or y_max <= y_min:
                # invalid crop → fallback
                crop = torch.zeros((1, C, 128, 128), device=img.device, dtype=img.dtype)
            else:
                crop = img[b:b + 1, :, y_min:y_max, x_min:x_max]
                crop = F.interpolate(crop, size=(128, 128), mode='bilinear', align_corners=False)

            crops.append(crop)

        return torch.cat(crops, dim=0) if crops else None

    def forward(self, fake, real, id_lat_mean, landmarks=None):
        # --- Global losses ---
        l1_loss = F.l1_loss(fake, real)

        fake_128 = F.interpolate(fake, size=(128, 128), mode='bilinear', align_corners=False)
        real_128 = F.interpolate(real, size=(128, 128), mode='bilinear', align_corners=False)
        fake_norm = normalize_vgg(fake_128)
        real_norm = normalize_vgg(real_128)
        percep_loss = self.perceptual_loss_fn(fake_norm, real_norm)

        fake_id_lat = self.identity_enc(fake).view(fake.size(0), -1)
        identity_loss = F.mse_loss(fake_id_lat, id_lat_mean)

        # --- Mouth region losses ---
        mouth_l1 = mouth_percep = torch.tensor(0.0, device=fake.device)
        if landmarks is not None:
            fake_mouth = self.crop_mouth(fake, landmarks)
            real_mouth = self.crop_mouth(real, landmarks)

            if fake_mouth is not None and real_mouth is not None:
                mouth_l1 = F.l1_loss(fake_mouth, real_mouth)

                #fake_mouth_norm = normalize_vgg(fake_mouth)
                #real_mouth_norm = normalize_vgg(real_mouth)
                #mouth_percep = self.perceptual_loss_fn(fake_mouth_norm, real_mouth_norm)

        # --- Total loss ---
        total_loss = (self.l1_weight * l1_loss +
                      self.perceptual_weight * percep_loss +
                      self.identity_weight * identity_loss +
                      self.mouth_l1_weight * mouth_l1
                      #self.mouth_perceptual_weight * mouth_percep
                      )

        return total_loss, l1_loss, percep_loss, identity_loss, mouth_l1



def train_visual(
    identity_enc, pose_enc, expr_enc, visual_enc,
    G,
    opt_G,
    loader,
    epochs=20,
    device='cuda'
):
    # ---- Move to device ----
    modules = [identity_enc, pose_enc, expr_enc, visual_enc, G]
    for m in modules: m.to(device)


    # ---- AMP ----
    scaler_G = GradScaler()

    steps_per_epoch = len(loader)


    # ---- Logging buffers ----
    l1_losses, vgg_losses, adv_losses, identity_losses, l1_mouth_losses = [], [], [], [], []
    steps = []
    global_step = 0

    # ---- Learning rate schedulers ----
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
        opt_G,
        max_lr=4e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )

    for epoch in range(epochs):
        # ---- Dynamic loss weights schedule ----
        # if epoch < int(epochs/4):
        #     l1_weight, adv_weight = 15.0, 0.05
        # elif epoch < int(epochs/2):
        #     l1_weight, adv_weight = 12.0, 0.1
        # elif epoch < int(3*epochs/4):
        #     l1_weight, adv_weight = 10.0, 0.2
        # else:
        l1_weight, adv_weight = 8.0, 0.3
        l1_weight_mouth = 2.5 * l1_weight
        perceptual_weight = 0.2
        identity_weight = 1

        loss_fn = CustomLoss(
                    identity_enc,
                    l1_weight=l1_weight,
                    perceptual_weight=perceptual_weight,
                    identity_weight=identity_weight,
                    mouth_l1_weight=l1_weight_mouth,
                    mouth_perceptual_weight=perceptual_weight
                )

        epoch_start = time.time()
        loader_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for i, batch in enumerate(loader_iter):
            global_step += 1
            iter_start = time.time()

            # ---- Data ----
            t0 = time.time()
            id_imgs     = batch['identity_frames'  ].to(device)  # (B,K,C,H,W)
            pose_i      = batch['driving_pose'     ].to(device)
            expr_i      = batch['driving_expression'].to(device)
            visual_mouth_i = batch['driving_mouth_hm'    ].to(device)
            driving_img = batch['driving_frame'    ].to(device)
            landmarks_mouth   = batch['driving_landmarks'        ]['mouth'].to(device)  # (B, N, 2)
            data_time = time.time() - t0

            # ---- Encode ----
            t0 = time.time()

            B, K, C, H, W = id_imgs.shape
            id_flat = id_imgs.view(B * K, C, H, W)
            id_lat = identity_enc(id_flat).view(B, K, -1)
            id_lat_mean = id_lat.mean(dim=1)  # (B, D_id)  # ID averaging


            pose_lat  = pose_enc(pose_i)
            expr_lat  = expr_enc(expr_i)
            visual_lat = visual_enc(visual_mouth_i)
            z = torch.cat([id_lat_mean, pose_lat, expr_lat, visual_lat], dim=1)
            encoding_time = time.time() - t0


            # ==============
            #   Generator
            # ==============
            t_gen_start = time.time()

            with autocast():
                fake = G(z)


            with autocast():
                total_G_loss, l1_loss, percep_loss, identity_loss, mouth_l1 = loss_fn(
                    fake, driving_img, id_lat_mean, landmarks=landmarks_mouth
                )


            opt_G.zero_grad(set_to_none=True)
            scaler_G.scale(total_G_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            if scheduler_G is not None:
                scheduler_G.step()

            t_gen_total = time.time() - t_gen_start
            total_iter_time = time.time() - iter_start

            # ---- tqdm ----
            loader_iter.set_postfix({
                "G": f"{total_G_loss.item():.3f}",
                "L1": f"{(l1_loss.item()*l1_weight):.3f}",
                "Perc": f"{(percep_loss.item()*perceptual_weight):.3f}",
                "ID": f"{(identity_loss.item()*identity_weight):.3f}",
                "T": f"{total_iter_time:.2f}s",
            })

            if i < 3:
                print(f"[Step {i}] DL={data_time:.2f}s, Enc={encoding_time:.2f}s, "
                      f"Gen={t_gen_total:.2f}s, Total={total_iter_time:.2f}s")

            # ---- Logging series ----
            if (i % 10) == 0:
                steps.append(global_step)
                l1_losses.append(l1_loss.item() * l1_weight)
                vgg_losses.append(percep_loss.item() * perceptual_weight)
                identity_losses.append(identity_loss.item() * identity_weight)
                l1_mouth_losses.append(mouth_l1.item() * l1_weight_mouth)


            # ---- Debug plots/images (optional) ----
            if i % 500 == 0:
                print(f"[Step {i}] G: {total_G_loss.item():.4f} | "
                      f"L1: {l1_loss.item()*l1_weight:.4f} | "
                      f"Perc: {percep_loss.item()*perceptual_weight:.4f}"
                      f" | ID: {identity_loss.item()*identity_weight:.4f} | "
                      f"Mouth L1: {mouth_l1.item()*l1_weight:.4f} | ")




                with torch.no_grad():
                    show_debug_images(fake, driving_img, title=f"Epoch {epoch+1} Step {i}")

                # Latent plots (first item)
                id_lat_np    = to_numpy_float32(id_lat_mean[0])
                pose_lat_np  = to_numpy_float32(pose_lat[0])
                expr_lat_np  = to_numpy_float32(expr_lat[0])
                visual_lat_np = to_numpy_float32(visual_lat[0])
                z_latent_np  = to_numpy_float32(z[0])

                plt.figure(figsize=(10,6))
                for idx,(arr,title) in enumerate([
                    (id_lat_np, "Identity"), (pose_lat_np, "Pose"),
                    (expr_lat_np, "Expression"), (visual_lat_np, "Visual"),
                    (z_latent_np, "Fused z"),
                ], start=1):
                    plt.subplot(2,3,idx); plt.title(title); plt.plot(arr); plt.xlabel("Index"); plt.ylabel("Value")
                plt.tight_layout(); plt.show()

                # Loss curves
                plt.figure(figsize=(12, 6))
                plt.plot(l1_losses, label='L1 * w')
                plt.plot(vgg_losses, label='Perceptual * w')
                plt.plot(identity_losses, label='ID * w')
                plt.plot(l1_mouth_losses, label='Mouth L1 * w')
                plt.title(f"Losses - Epoch {epoch+1} Step {i}")
                plt.xlabel("Logged step"); plt.ylabel("Value"); plt.legend(); plt.grid(); plt.show()

        print(f"[Epoch {epoch+1}/{epochs}] completed in {time.time() - epoch_start:.1f}s")

        # ---- Save checkpoint ----
        torch.save({
            'epoch': epoch + 1,
            'identity_enc': identity_enc.state_dict(),
            'pose_enc': pose_enc.state_dict(),
            'expr_enc': expr_enc.state_dict(),
            'visual_enc': visual_enc.state_dict(),
            'generator': G.state_dict(),
            'opt_G': opt_G.state_dict()
        }, f'weights/last_epoch.pth')

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'identity_enc': identity_enc.state_dict(),
                'pose_enc': pose_enc.state_dict(),
                'expr_enc': expr_enc.state_dict(),
                'visual_enc': visual_enc.state_dict(),
                'generator': G.state_dict(),
                'opt_G': opt_G.state_dict()
            }, f'weights/epoch_{epoch+1}.pth')

def to_numpy_float32(tensor):
    return tensor.detach().cpu().float().numpy()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model input size
        transforms.ToTensor(),  # scale to [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1, 1]
    ])

    dataset = TalkingHeadDataset(
        root_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\dataset",
        heatmaps_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\heatmaps",
        landmarks_dir=r"C:\Users\pnieg\Documents\Masterarbeit\TalkingHead-1KH\landmarks",
        transform=transform,
        cache_file="dataset_cache_heatmap_test.pkl"
    )

    dataloader = DataLoader(dataset, batch_size=32, num_workers=4,
                            shuffle=True, pin_memory=True if device == 'cuda' else False, persistent_workers=True)

    visual_enc = VisualEncoder(in_channels=3, latent_dim=256)
    pose_enc = PoseEncoder(in_channels=3, latent_dim=12)
    expression_enc = ExpressionEncoder(in_channels=3, latent_dim=256)
    identity_enc = IdentityEncoder(in_channels=3, latent_dim=512)
    G              = IdentityDecoder(input_dimension=1036)

    opt_G = torch.optim.Adam(
        list(G.parameters()) +
        list(identity_enc.parameters()) +
        list(pose_enc.parameters()) +
        list(expression_enc.parameters()) +
        list(visual_enc.parameters()),
        lr=2e-4,
        betas=(0.5, 0.99)
    )


    train_visual(
        identity_enc, pose_enc, expression_enc, visual_enc,
        G,
        opt_G=opt_G,
        loader=dataloader,
        epochs=60,
        device=device)