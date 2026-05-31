import cv2
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# SEEDING
''' We do seeding with a certain
seed value to ensure we get the same results throughout,
guaranteeing consistency and reproducibility, every single run
will not experience change'''

# seed value = 99 gives best results

def seed_everything(seed=99):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# IMAGE PREPROCESSING

# 1) RESIZING AND PADDING
'''This is done to ensure the images don't shrink 
(ensuring strict transistor scanning), and the images' size 
meet the standards of ImageNet's typical training img size (224),
while perserving transistor ratio'''

# PatchCore relies on ResNet-18, and ResNet is trained on ImageNet
# We use zero padding to avoid adding any additional noises

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


# 2) TENSOR CONVERSION AND NORMALIZATION
'''To ensure the model really understands the images, 
the images are converted to tensors (since models only understand numbers).
Normalization is done to stabilize gradients, bring data and activations
to a common scale, allowing faster convergence and not letting the model
output biased, unstable results'''

# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] are the means and stds
# of the large ImageNet training dataset

def get_resnet_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# 3) PREPARING THE IMAGES WITH THE PREPROCESSING FUNCTIONS

def prepare_image(img_path, device, verbose=False):
    if verbose:
        print("--- USING BLACK PADDING ---")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded = resize_with_pad(img, needed_size=224)
    transform = get_resnet_transform()
    return transform(Image.fromarray(padded)).unsqueeze(0).to(device)


# FEATURE EXTRACTION HOOK
'''Hooks are used to capture insights regarding the behaviour of neural
networks. They provide a way to manipulate the individual layers of the nn'''

def hook_fn(module, input, output, feature_list):
    # detach the output tensor from PyTorch's built computational graph
    feature_list.append(output.detach()) 


# INSIDE PATCHCORE LITE

class PatchCoreLite:
    @staticmethod
    def aggregate_features(captured_features):
        
        # 1) FEATURE AGGREGATION
        """
        Step where the model takes the disconnected and raw feature maps
        from different layers of the ResNet-18 and combines them into a
        single cohesive feature map
        ("patch matrix of localized "patch decsriptors")
        """

        # We combine layers 2 and 3 into a flat patch matrix [N, 384], cus:
        # - layer 2 (128): captures textures, shapes, corners and other structural arrangements
        # - layer 3 (256): captures precise spacing and parallel alignemnt of the pins

        # kernel=3, stride=1 to make each patch descriptor aware of its surroundings

        features = sorted(captured_features, key=lambda x: x.shape[-1], reverse=True)
        f2, f3 = features[0], features[1]   # layer2: 28×28, layer3: 14×14

        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        
        # Resize f3's feature map tensors to the size of f2's 
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        combined = torch.cat([f2, f3_up], dim=1)         
        B, C, H, W = combined.shape
        return combined.reshape(B, C, -1).transpose(1, 2).reshape(-1, C)

    @staticmethod
    def coreset_subsample(features, percentage=0.1):

        # 2) CORESET SUBSAMPLING
        """Carefully pick the most representative sample of the
        entire dataset for faster training, achieving low memory,
        maximizing efficiency"""

        # Uniform random subsample, 10% alr gives dense enough coverage,
        # remove the 90% redundant ones

        n = features.shape[0]
        keep = max(1, int(n * percentage))
        return features[torch.randperm(n)[:keep]]

    @staticmethod
    def compute_score(test_patches, memory_bank, temperature=0.1):
        """
        Log-sum-exp is a smooth approximation of the max func that
        rewards consistently high distances (true anomalies) while
        suppressing single-patch noise
        """

        # temperature: lower means closer to hard max (more sensitive but noisier),
        # 0.1 is the value used in the original PatchCore paper, 
        # 0.0 is extremely sensitive

        # seperate the patches into chunks to go easy on the cpu
        chunk = 256
        min_dists = []
        for i in range(0, test_patches.shape[0], chunk):
            # calculates the Pairwise Euclidean Dist (p=2)
            d = torch.cdist(test_patches[i:i + chunk], memory_bank, p=2)

            # append based on smallest dist values (using Nearest Neighbor approach)
            min_dists.append(d.min(dim=1).values)

        patch_scores = torch.cat(min_dists)         

        score = temperature * torch.logsumexp(patch_scores / temperature, dim=0)
        return score.item()