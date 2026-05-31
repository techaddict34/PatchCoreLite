import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from preprocessing import seed_everything, prepare_image, hook_fn, PatchCoreLite

# Configurations
MODEL_PATH  = "transistor_lite_model.pt"
TEST_ROOT   = "transistor/test"
TEMPERATURE = 0.1

# Redefine some of the functions from train.py to really isolate their environment
# and just so that test.py can be an independent script

def build_backbone(device):
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False
    return resnet

def register_hooks(resnet, captured_features):
    resnet.layer2[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured_features))
    resnet.layer3[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured_features))

def score_image(img_path, resnet, device, memory_bank, captured_features):
    input_tensor = prepare_image(img_path, device)
    captured_features.clear()
    with torch.no_grad():
        _ = resnet(input_tensor)
    patches = PatchCoreLite.aggregate_features(captured_features).to(device)
    return PatchCoreLite.compute_score(patches, memory_bank, temperature=TEMPERATURE)

def run_evaluation(test_root, resnet, device, memory_bank, captured_features):
    results = {}
    for cat in sorted(os.listdir(test_root)):
        cat_path = os.path.join(test_root, cat)
        if not os.path.isdir(cat_path):
            continue
        scores = []
        for fname in sorted(os.listdir(cat_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                s = score_image(os.path.join(cat_path, fname),
                                resnet, device, memory_bank, captured_features)
                scores.append((fname, s))
        if scores:
            results[cat] = scores
    return results

def print_report(results, threshold, mu, sigma):
    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION")
    print(f"  threshold={threshold:.4f}  (train mean={mu:.4f}, std={sigma:.4f})")
    print(f"{'='*60}")

    correct = total = 0
    for cat, entries in results.items():
        is_defect = (cat != "good")
        print(f"\n  [{cat.upper()}]")
        for fname, s in entries:
            detected = s > threshold
            ok = detected == is_defect
            correct += ok
            total += 1
            mark   = "✓" if ok else "✗"
            status = "ANOMALY" if detected else "NORMAL "
            print(f"    {mark} {fname}  score={s:.4f}  → {status}")
        vals = [s for _, s in entries]
        print(f"    avg={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    print(f"\n{'='*60}")
    print(f"  ACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}\n")

# PUT IT ALL TGT

if __name__ == "__main__":
    seed_everything(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"|| Engine running on: {device} ||")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at '{MODEL_PATH}'. Run train.py first.")

    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    memory_bank = checkpoint['memory_bank'].to(device)
    threshold   = checkpoint['threshold']
    mu          = checkpoint['train_mu']
    sigma       = checkpoint['train_sigma']

    print(f"|| Model loaded from '{MODEL_PATH}'")
    print(f"   Bank size: {memory_bank.shape[0]} patches  |  dim: {memory_bank.shape[1]}")
    print(f"   Auto threshold: {threshold:.4f}  (mean={mu:.4f}, std={sigma:.4f})")

    resnet = build_backbone(device)
    captured_features = []
    register_hooks(resnet, captured_features)

    if not os.path.exists(TEST_ROOT):
        raise FileNotFoundError(f"Test folder not found at '{TEST_ROOT}'.")

    print(f"\n--- STARTING EVALUATION on '{TEST_ROOT}' ---")
    results = run_evaluation(TEST_ROOT, resnet, device, memory_bank, captured_features)
    print_report(results, threshold, mu, sigma)