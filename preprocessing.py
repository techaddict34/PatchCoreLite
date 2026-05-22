import cv2
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# --- SEEDING ---

def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# --- IMAGE PREPROCESSING ---

def resize_with_pad(img, needed_size=224):
    height, width = img.shape[:2]
    scale = needed_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top = (needed_size - new_height) // 2
    bottom = needed_size - new_height - top
    left = (needed_size - new_width) // 2
    right = needed_size - new_width - left

    return cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

def get_resnet_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_image(img_path, device, verbose=False):
    if verbose:
        print("--- USING BLACK PADDING ---")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded = resize_with_pad(img, needed_size=224)
    transform = get_resnet_transform()
    return transform(Image.fromarray(padded)).unsqueeze(0).to(device)


# --- FEATURE EXTRACTION HOOK ---

def hook_fn(module, input, output, feature_list):
    feature_list.append(output.detach())


# --- PATCHCORE LITE CORE ---

class PatchCoreLite:
    @staticmethod
    def aggregate_features(captured_features):
        """
        Combine layer2 + layer3 into a flat patch matrix [N, 384].
        Uses neighbourhood-averaged pooling (kernel=3, stride=1) to make
        each patch descriptor spatially aware of its surroundings.
        Raw (un-normalised) features — normalising onto a unit sphere
        collapses the magnitude signal that distinguishes anomalous patches.
        """
        features = sorted(captured_features, key=lambda x: x.shape[-1], reverse=True)
        f2, f3 = features[0], features[1]   # layer2: 28×28, layer3: 14×14
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)          # [B, 384, H, W]
        B, C, H, W = combined.shape
        return combined.reshape(B, C, -1).transpose(1, 2).reshape(-1, C)  # [B*H*W, 384]

    @staticmethod
    def coreset_subsample(features, percentage=0.1):
        """Uniform random subsample — 10% gives dense enough coverage."""
        n = features.shape[0]
        keep = max(1, int(n * percentage))
        return features[torch.randperm(n)[:keep]]

    @staticmethod
    def compute_score(test_patches, memory_bank, temperature=0.1):
        """
        PatchCore-style soft-max (log-sum-exp) anomaly score.

        Hard max over patch distances is too sensitive to a single noisy
        patch — one outlier in a good image spikes the score into defect
        territory. Log-sum-exp is a smooth approximation of the max that
        rewards *consistently* high distances (true anomalies) while
        suppressing single-patch noise.

        temperature: lower → closer to hard max (more sensitive but noisier).
                     0.1 is the value used in the original PatchCore paper.
        """
        chunk = 256
        min_dists = []
        for i in range(0, test_patches.shape[0], chunk):
            d = torch.cdist(test_patches[i:i + chunk], memory_bank, p=2)
            min_dists.append(d.min(dim=1).values)

        patch_scores = torch.cat(min_dists)               # [N_patches]

        # log-sum-exp: log( sum( exp(s / T) ) ) * T
        score = temperature * torch.logsumexp(patch_scores / temperature, dim=0)
        return score.item()