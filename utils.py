import torch
import torch.nn as nn
from torchvision import models

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo normalization: (C,H,W) tensor -> (C,H,W) tensor in [0,1]"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def display_mel_feature(mel_segment, sr=16000, hop_length=160, title=None, show_colorbar=True):
    """
    Display a single mel-spectrogram segment (shape: [80, T])
    """

    if 'torch' in str(type(mel_segment)):
        mel_segment = mel_segment.cpu().numpy()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_segment,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='magma',
        fmax=sr // 2
    )
    if show_colorbar:
        plt.colorbar(format="%+2.0f dB")
    if title:
        plt.title(title)
    else:
        plt.title("Mel-spectrogram")
    plt.tight_layout()
    plt.show()


def show_sample(sample):
    """
    Displays identity frames and driving frames from a dataset sample dictionary.
    Args:
        sample: dict with keys:
            'identity_frames': (8, C, H, W) tensor
            'driving_pose': (C, H, W) tensor
            'driving_expression': (C, H, W) tensor
            'driving_mouth': (C, H, W) tensor
            'mel_feature': (freq, time) tensor or array (optional display)
    """

    # Convert tensors to numpy arrays and transpose (C,H,W) -> (H,W,C)
    def tensor_to_img(t):
        return t.permute(1, 2, 0).cpu().numpy()

    id_imgs = [tensor_to_img(f) for f in sample['identity_frames']]
    driving_pose = tensor_to_img(sample['driving_pose'])
    driving_expr = tensor_to_img(sample['driving_expression'])
    driving_mouth = tensor_to_img(sample['driving_mouth'])

    n_id = len(id_imgs)
    n_cols = 4  # number of columns for identity grid
    n_rows = (n_id + n_cols - 1) // n_cols

    # Create one figure with rows: identity rows + 1 for driving images
    fig_height = n_rows * 3 + 4  # additional space for driving images
    plt.figure(figsize=(15, fig_height))

    # Plot identity frames on the top rows
    for i, img in enumerate(id_imgs):
        plt.subplot(n_rows + 1, n_cols, i + 1)
        plt.imshow(img)
        plt.title(f"Identity {i + 1}")
        plt.axis('off')

    # Plot driving images in bottom row spanning n_cols columns
    # Pose
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 1)
    plt.imshow(driving_pose)
    plt.title("Driving Pose")
    plt.axis('off')

    # Expression
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 2)
    plt.imshow(driving_expr)
    plt.title("Driving Expression")
    plt.axis('off')

    # Mouth
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 3)
    plt.imshow(driving_mouth)
    plt.title("Driving Mouth")
    plt.axis('off')

    # Leave the remaining spots in bottom row empty or optionally add mel spectrogram here

    plt.tight_layout()
    plt.show()

    # Optionally display mel spectrogram (if you want)
    if 'mel_feature' in sample:
        display_mel_feature(
            sample['mel_feature'],
            sr=16000,  # Assuming mel features are sampled at 16kHz
            hop_length=160,  # Typical hop length for mel features
            title="Mel-spectrogram Feature")


def show_first_batch_sample(batch):
    """
    Displays the first sample in a batch from a DataLoader.
    Args:
        batch: dict containing batched tensors with keys:
            'identity_frames': (B, 8, C, H, W)
            'driving_pose': (B, C, H, W)
            'driving_expression': (B, C, H, W)
            'driving_mouth': (B, C, H, W)
            'mel_feature': (B, freq, time) (optional)
    """

    # Helper: convert tensor to numpy image (C,H,W) -> (H,W,C)
    def tensor_to_img(t):
        t = t.clone()  # avoid modifying original tensor
        return t.permute(1, 2, 0).cpu().numpy()

    # Select first entry in batch (index 0)
    sample = {
        'identity_frames': batch['identity_frames'][0],      # (8,C,H,W)
        'driving_pose': batch['driving_pose'][0],            # (C,H,W)
        'driving_expression': batch['driving_expression'][0],
        'driving_mouth': batch['driving_mouth'][0],
        'driving_frame': batch['driving_frame'][0],
    }
    if 'mel_feature' in batch:
        sample['mel_feature'] = batch['mel_feature'][0]      # (freq,time)

    # Convert images
    id_imgs = [tensor_to_img(f) for f in sample['identity_frames']]
    driving_pose = tensor_to_img(sample['driving_pose'])
    driving_expr = tensor_to_img(sample['driving_expression'])
    driving_mouth = tensor_to_img(sample['driving_mouth'])
    driving_frame = tensor_to_img(sample['driving_frame'])

    n_id = len(id_imgs)
    n_cols = 4
    n_rows = (n_id + n_cols - 1) // n_cols

    fig_height = n_rows * 3 + 4
    plt.figure(figsize=(15, fig_height))

    # Identity frames
    for i, img in enumerate(id_imgs):
        plt.subplot(n_rows + 1, n_cols, i + 1)
        plt.imshow(img)
        plt.title(f"Identity {i + 1}")
        plt.axis('off')

    # Driving images (bottom row)
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 1)
    plt.imshow(driving_pose)
    plt.title("Driving Pose")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 2)
    plt.imshow(driving_expr)
    plt.title("Driving Expression")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 3)
    plt.imshow(driving_mouth)
    plt.title("Driving Mouth")
    plt.axis('off')
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 4)
    plt.imshow(driving_frame)
    plt.title("Driving Frame")
    plt.axis('off')



    plt.tight_layout()
    plt.show()

    # Optionally display mel spectrogram
    if 'mel_feature' in sample:
        display_mel_feature(
            sample['mel_feature'],
            sr=16000,
            hop_length=160,
            title="Mel-spectrogram Feature"
        )


def show_first_batch_sample_heatmaps(batch, alpha=0.5):
    """
    Displays the first sample in a batch from a DataLoader, compatible with heatmap inputs.

    Args:
        batch: dict containing batched tensors with keys:
            'identity_frames': (B, 8, C, H, W)
            'driving_pose': (B, Np, Hh, Wh)
            'driving_expression': (B, Ne, Hh, Wh)
            'driving_mouth': (B, Cm, H, W)
            'driving_frame': (B, C, H, W)
            'mel_feature': (B, freq, time) (optional)
        alpha: blending factor for heatmap overlay
    """
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import numpy as np

    # Helper: tensor to numpy image (C,H,W) -> (H,W,C)
    # def tensor_to_img(t):
    #     t = t.clone()
    #     return t.permute(1, 2, 0).cpu().numpy()
    def tensor_to_img(t):
        t = t.clone()
        t = (t + 1) / 2.0  # convert from [-1,1] -> [0,1]
        t = torch.clamp(t, 0, 1)
        return t.permute(1, 2, 0).cpu().numpy()

    # Helper: convert heatmap tensor to colored image
    def heatmaps_to_rgb(heatmaps):
        """
        heatmaps: torch.Tensor or np.ndarray, shape (N,H,W)
        returns: HxW RGB image
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().numpy()
        combined = heatmaps.sum(axis=0)
        combined = combined / (combined.max() + 1e-8)
        heatmap_uint8 = (combined * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[..., ::-1]  # BGR -> RGB

    # Select first entry in batch
    sample = {
        'identity_frames': batch['identity_frames'][0],  # (8,C,H,W)
        'driving_pose': batch['driving_pose'][0],  # (Np,H,W)
        'driving_expression': batch['driving_expression'][0],
        'driving_mouth': batch['driving_mouth'][0],
        'driving_frame': batch['driving_frame'][0],
    }
    if 'mel_feature' in batch:
        sample['mel_feature'] = batch['mel_feature'][0]  # (freq,time)

    # Convert images
    id_imgs = [tensor_to_img(f) for f in sample['identity_frames']]
    driving_frame = tensor_to_img(sample['driving_frame'])
    driving_mouth = tensor_to_img(sample['driving_mouth'])

    # Convert heatmaps to RGB and resize to match driving frame
    H, W = driving_frame.shape[:2]
    driving_pose = cv2.resize(heatmaps_to_rgb(sample['driving_pose']), (W, H))
    driving_expr = cv2.resize(heatmaps_to_rgb(sample['driving_expression']), (W, H))

    n_id = len(id_imgs)
    n_cols = 4
    n_rows = (n_id + n_cols - 1) // n_cols
    fig_height = n_rows * 3 + 4
    plt.figure(figsize=(15, fig_height))

    # Identity frames
    for i, img in enumerate(id_imgs):
        plt.subplot(n_rows + 1, n_cols, i + 1)
        plt.imshow(img)
        plt.title(f"Identity {i + 1}")
        plt.axis('off')

    # Driving images (bottom row)
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 1)
    plt.imshow(driving_pose)
    plt.title("Driving Pose")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 2)
    plt.imshow(driving_expr)
    plt.title("Driving Expression")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 3)
    plt.imshow(driving_mouth)
    plt.title("Driving Mouth")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 4)
    plt.imshow(driving_frame)
    plt.title("Driving Frame")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally display mel spectrogram
    if 'mel_feature' in sample:
        display_mel_feature(
            sample['mel_feature'],
            sr=16000,
            hop_length=160,
            title="Mel-spectrogram Feature"
        )

def show_first_batch_sample_heatmaps_HDTF(batch, alpha=0.5):
    """
    Displays the first sample in a batch from a DataLoader, compatible with new dataset:
    - driving_expression is an RGB image
    - driving_mouth uses the mouth heatmap converted to RGB
    - driving_pose is a heatmap
    - identity frames and driving frame are images
    """
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import numpy as np

    # Helper: tensor to numpy image (C,H,W) -> (H,W,C)
    def tensor_to_img(t):
        t = t.clone()
        t = (t + 1) / 2.0  # convert from [-1,1] -> [0,1]
        t = torch.clamp(t, 0, 1)
        return t.permute(1, 2, 0).cpu().numpy()

    # Helper: convert heatmap tensor to colored image
    def heatmaps_to_rgb(heatmaps):
        """
        heatmaps: torch.Tensor or np.ndarray, shape (N,H,W)
        returns: HxW RGB image
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().numpy()
        combined = heatmaps.sum(axis=0)
        combined = combined / (combined.max() + 1e-8)
        heatmap_uint8 = (combined * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[..., ::-1]  # BGR -> RGB

    # Select first entry in batch
    sample = {
        'identity_frames': batch['identity_frames'][0],  # (8,C,H,W)
        'driving_pose': batch['driving_pose'][0],        # (Np,H,W)
        'driving_expression': batch['driving_expression'][0],  # image
        #'driving_mouth_hm': batch['driving_mouth_hm'][0],      # (66,H,W)
        'driving_frame': batch['driving_frame'][0],
    }
    if 'mel_feature' in batch:
        sample['mel_feature'] = batch['mel_feature'][0]  # (freq,time)

    # Convert images
    id_imgs = [tensor_to_img(f) for f in sample['identity_frames']]
    driving_frame = tensor_to_img(sample['driving_frame'])
    driving_expr = tensor_to_img(sample['driving_expression'])  # expression image

    # Convert heatmaps to RGB and resize to match driving frame
    H, W = driving_frame.shape[:2]
    driving_pose = cv2.resize(heatmaps_to_rgb(sample['driving_pose']), (W, H))
    driving_mouth = cv2.resize(heatmaps_to_rgb(sample['driving_mouth_hm']), (W, H))

    # Layout
    n_id = len(id_imgs)
    n_cols = 4
    n_rows = (n_id + n_cols - 1) // n_cols
    fig_height = n_rows * 3 + 4
    plt.figure(figsize=(15, fig_height))

    # Identity frames
    for i, img in enumerate(id_imgs):
        plt.subplot(n_rows + 1, n_cols, i + 1)
        plt.imshow(img)
        plt.title(f"Identity {i + 1}")
        plt.axis('off')

    # Driving images (bottom row)
    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 1)
    plt.imshow(driving_pose)
    plt.title("Driving Pose")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 2)
    plt.imshow(driving_expr)
    plt.title("Driving Expression")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 3)
    plt.imshow(driving_mouth)
    plt.title("Driving Mouth")
    plt.axis('off')

    plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 4)
    plt.imshow(driving_frame)
    plt.title("Driving Frame")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally display mel spectrogram
    if 'mel_feature' in sample:
        display_mel_feature(
            sample['mel_feature'],
            sr=16000,
            hop_length=160,
            title="Mel-spectrogram Feature"
        )

def show_first_batch_sample_heatmaps_HDTF_single_identity(batch, alpha=0.5):
    """
    Displays the first sample in a batch from a DataLoader.
    Compatible with new dataset:
    - driving_expression is an RGB image
    - driving_mouth uses the mouth heatmap converted to RGB
    - driving_pose is a heatmap
    - identity frame and driving frame are images
    Only one identity frame is assumed: shape (C,H,W)
    """
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import numpy as np

    # --- Helper: tensor to numpy image ---
    def tensor_to_img(t):
        t = t.clone()
        t = (t + 1) / 2.0  # [-1,1] -> [0,1]
        t = torch.clamp(t, 0, 1)
        return t.permute(1, 2, 0).cpu().numpy()

    # --- Helper: convert heatmap tensor to RGB ---
    def heatmaps_to_rgb(heatmaps):
        """
        heatmaps: torch.Tensor or np.ndarray, shape (N,H,W)
        returns: HxW RGB image
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().numpy()
        combined = heatmaps.sum(axis=0)
        combined = combined / (combined.max() + 1e-8)
        heatmap_uint8 = (combined * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[..., ::-1]  # BGR -> RGB

    # --- Select first sample ---
    sample = {
        'identity_frames': batch['identity_frames'][0],  # (C,H,W)
        'driving_pose': batch['driving_pose'][0],        # (Np,H,W)
        'driving_expression': batch['driving_expression'][0],
        'driving_mouth_hm': batch['driving_mouth_hm'][0],
        'driving_frame': batch['driving_frame'][0],
    }
    if 'mel_feature' in batch:
        sample['mel_feature'] = batch['mel_feature'][0]

    # --- Convert images ---
    id_img = tensor_to_img(sample['identity_frames'])  # single image
    driving_frame = tensor_to_img(sample['driving_frame'])
    driving_expr = tensor_to_img(sample['driving_expression'])

    # --- Convert heatmaps to RGB and resize to match driving frame ---
    H, W = driving_frame.shape[:2]
    driving_pose = cv2.resize(heatmaps_to_rgb(sample['driving_pose']), (W, H))
    driving_mouth = cv2.resize(heatmaps_to_rgb(sample['driving_mouth_hm']), (W, H))

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # Identity frame
    plt.subplot(2, 4, 1)
    plt.imshow(id_img)
    plt.title("Identity")
    plt.axis('off')

    # Driving images (bottom row)
    plt.subplot(2, 4, 5)
    plt.imshow(driving_pose)
    plt.title("Driving Pose")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(driving_expr)
    plt.title("Driving Expression")
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(driving_mouth)
    plt.title("Driving Mouth")
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(driving_frame)
    plt.title("Driving Frame")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Optionally display mel spectrogram
    if 'mel_feature' in sample:
        display_mel_feature(
            sample['mel_feature'],
            sr=16000,
            hop_length=160,
            title="Mel-spectrogram Feature"
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model, model_name="Model"):
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params} of {model_name}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    return total_params


def denormalize_img(img_tensor):
    """
    Convert tensor from [-1, 1] to [0, 1] and prepare for matplotlib
    Args:
        img_tensor: Tensor of shape (B, C, H, W) in range [-1, 1]
    Returns:
        numpy array of shape (H, W, C) in range [0, 1] with dtype float32
    """
    # Take first image from batch
    if len(img_tensor.shape) == 4:
        img = img_tensor[0]  # (C, H, W)
    else:
        img = img_tensor

    # Convert from [-1, 1] to [0, 1]
    img = (img + 1.0) / 2.0

    # Clamp to valid range
    img = torch.clamp(img, 0.0, 1.0)

    # Convert to CPU and numpy
    img_np = img.detach().cpu().numpy()

    # Convert from (C, H, W) to (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))

    # Ensure float32 dtype for matplotlib
    img_np = img_np.astype(np.float32)

    return img_np


def show_debug_images(fake, real, title="Debug"):
    """Displays generated vs real image side-by-side during training inside PyCharm SciView."""
    try:

        # Convert tensor images to numpy arrays with channels last for plt.imshow
        def tensor_to_img(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float()  # ensure float32
            # If 4D (batch), select first image
            if x.ndim == 4:
                x = x[0]
            # Now x is (C, H, W) tensor
            if x.ndim == 3:
                x = x.permute(1, 2, 0).numpy()  # (H,W,C)
            else:
                x = x.numpy()
            x = (x + 1.0) / 2.0
            x = np.clip(x, 0, 1)  # Ensure valid range for imshow
            return x

        fake_img = tensor_to_img(fake)
        real_img = tensor_to_img(real)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(fake_img)
        plt.title("Generated Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(real_img)
        plt.title("Real/Driving Image")
        plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.pause(0.001)  # updates SciView instead of blocking
        plt.close()  # clear to avoid stacking figures

    except Exception as e:
        print(f"Error displaying debug images: {e}")
        print(f"Fake shape: {fake_img.shape if 'fake_img' in locals() else 'N/A'}")
        print(f"Real shape: {real_img.shape if 'real_img' in locals() else 'N/A'}")

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features.eval()
        for param in vgg_pretrained.parameters():
            param.requires_grad = False
        self.vgg = vgg_pretrained
        self.layer_ids = [3, 8, 17, 26]  # relu1_2, relu2_2, relu3_3, relu4_3
        #self.layer_ids = [3,8]  # relu1_2, relu2_2
        self.crit = nn.L1Loss()
        # ImageNet mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                features.append(x)
        return features

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std


        # Extract features only once for each input
        x_features = self._extract_features(x)
        with torch.no_grad():
            y_features = self._extract_features(y)

        # Sum L1 losses between corresponding layers
        loss = 0.0
        for xf, yf in zip(x_features, y_features):
            loss += self.crit(xf, yf)
        return loss


def normalize_vgg(x):
    """Normalize tensor for VGG input"""
    # First convert from [-1,1] to [0,1] since StyleGAN2 uses tanh()
    x = (x + 1) / 2  # Convert from [-1,1] to [0,1]

    # Then apply VGG normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std

