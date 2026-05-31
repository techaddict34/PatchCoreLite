import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from preprocessing import seed_everything, prepare_image, hook_fn, PatchCoreLite

# Configurations
GOOD_FOLDER = "transistor/train/good"
SAVE_PATH   = "transistor_lite_model.pt"
CORESET_PCT = 0.1
TEMPERATURE = 0.1

def build_backbone(device):
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Lock ResNet in eval mode so it uses the static mean and variance vals,
    # learned from the original ImageNet training
    # ensuring robustness, consistency and determinism
    resnet.eval()
    
    for param in resnet.parameters():
        # Turns off gradient tracking to save RAM
        param.requires_grad = False 
    return resnet

# Forward hooks = executed after the forward pass through a layer is completed,
# but before the output is returned

# We use forward hooks to extract feature maps from layers 2 and 3

def register_hooks(resnet, captured_features):
    resnet.layer2[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured_features))
    resnet.layer3[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured_features))

def extract_all_patches(good_folder, resnet, device, captured_features):
    all_patches = []
    image_files = sorted([f for f in os.listdir(good_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} training images in '{good_folder}'")
    for fname in image_files:
        img_path = os.path.join(good_folder, fname)
        input_tensor = prepare_image(img_path, device)
        captured_features.clear()
        with torch.no_grad():
            _ = resnet(input_tensor)
        all_patches.append(PatchCoreLite.aggregate_features(captured_features))
    return torch.cat(all_patches, dim=0)

# PUTTING IT ALL TGT

if __name__ == "__main__":
    seed_everything(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"|| Engine running on: {device} ||")

    resnet = build_backbone(device)
    captured_features = []
    register_hooks(resnet, captured_features)

    print("Starting Feature Extraction...")
    full_raw_bank = extract_all_patches(GOOD_FOLDER, resnet, device, captured_features)
    print(f"Full bank: {full_raw_bank.shape[0]} patches  |  dim: {full_raw_bank.shape[1]}")

    memory_bank = PatchCoreLite.coreset_subsample(full_raw_bank, percentage=CORESET_PCT)

    # Score every training image against the bank to learn what "normal" looks like
    # statistically, rather than picking a threshold by hand
    print("Computing threshold from training scores...")
    train_scores = []
    image_files = sorted([f for f in os.listdir(GOOD_FOLDER)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in image_files:
        img_path = os.path.join(GOOD_FOLDER, fname)
        input_tensor = prepare_image(img_path, device)
        captured_features.clear()
        with torch.no_grad():
            _ = resnet(input_tensor)
        patches = PatchCoreLite.aggregate_features(captured_features).to(device)
        s = PatchCoreLite.compute_score(patches, memory_bank.to(device), temperature=TEMPERATURE)
        train_scores.append(s)

    train_scores_t = torch.tensor(train_scores)
    mu, sigma = train_scores_t.mean().item(), train_scores_t.std().item()

    # mean + 3*std captures 99.7% of normal variation, 
    # anything above that is considered anomalous
    threshold = mu + 3 * sigma

    print(f"Training scores  →  mean={mu:.4f}  std={sigma:.4f}")
    print(f"Auto threshold   →  {threshold:.4f}  (mean + 3*std)")

    torch.save({
        'memory_bank': memory_bank.cpu(),
        'threshold':   threshold,
        'train_mu':    mu,
        'train_sigma': sigma,
    }, SAVE_PATH)
    print(f"|| SUCCESS || Model saved to '{SAVE_PATH}'")
    print(f"   Bank size: {memory_bank.shape[0]} patches  |  dim: {memory_bank.shape[1]}")