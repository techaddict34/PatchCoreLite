import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

from preprocessing import PatchCoreLite, prepare_image, hook_fn, seed_everything, resize_with_pad

# Page Config
st.set_page_config(
    page_title="PatchCore Lite — Transistor Inspector",
    page_icon="🔎",
    layout="wide",
)

st.title("PatchCore Lite ~ Transistor Defect Inspector")
st.caption("Upload a transistor image to detect anomalies using PatchCore feature matching.")

# Model Loading
MODEL_PATH  = "transistor_lite_model.pt"
TEMPERATURE = 0.1

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at '{MODEL_PATH}'. Please run train.py first.")
        st.stop()

    device     = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    memory_bank = checkpoint["memory_bank"].to(device)
    threshold   = checkpoint["threshold"]
    mu          = checkpoint["train_mu"]
    sigma       = checkpoint["train_sigma"]

    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    captured = []
    # Also capture spatial feature maps for the heatmap (before flattening)
    spatial  = []
    resnet.layer2[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured))
    resnet.layer3[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, captured))

    return resnet, memory_bank, threshold, mu, sigma, captured, spatial, device

resnet, memory_bank, threshold, mu, sigma, captured, spatial, device = load_model()

# Heatmap Generation
def generate_heatmap(captured_features, memory_bank, img_display_rgb):
    """
    For every spatial patch position compute its nearest-neighbour distance
    in the memory bank, reshape to a 2-D map, upsample to image size, and
    blend as a colour overlay.
    """
    features = sorted(captured_features, key=lambda x: x.shape[-1], reverse=True)
    f2, f3 = features[0], features[1]
    H, W = f2.shape[-2], f2.shape[-1] # 28×28

    f2_pooled = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
    f3_pooled = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
    f3_up     = F.interpolate(f3_pooled, size=(H, W), mode='bilinear', align_corners=False)
    combined  = torch.cat([f2_pooled, f3_up], dim=1)   # [1, 384, 28, 28]

    # Flatten to patches, compute per-patch NN distance
    patches = combined.squeeze(0).reshape(384, -1).T   # [784, 384]
    chunk   = 256
    min_dists = []
    for i in range(0, patches.shape[0], chunk):
        d = torch.cdist(patches[i:i+chunk], memory_bank, p=2)
        min_dists.append(d.min(dim=1).values)
    patch_scores = torch.cat(min_dists).reshape(H, W).cpu().numpy()   # [28, 28]

    # Normalise to [0, 1]
    s_min, s_max = patch_scores.min(), patch_scores.max()
    if s_max > s_min:
        norm = (patch_scores - s_min) / (s_max - s_min)
    else:
        norm = np.zeros_like(patch_scores)

    # Upsample to 224×224
    heatmap_224 = cv2.resize(norm, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_224 = np.clip(heatmap_224, 0, 1)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(
        (heatmap_224 * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend with the padded image
    base = img_display_rgb.astype(np.float32)
    heat = heatmap_color.astype(np.float32)
    blended = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
    return blended.astype(np.uint8), patch_scores

# Sidebar
with st.sidebar:
    with st.sidebar:
        st.markdown("## Model Info")
        st.metric(
            label="System Threshold", 
            value="2.2263",
            help=(
                "Calculated dynamically via the Three-Sigma Rule using the training distribution baseline:\n\n"
                "Threshold = Mean + (3 × Std)\n\n"
                "Threshold = 1.9664 + (3 × 0.0866) = 2.2263"
            )
        )
    st.divider()
    st.header("Heatmap")
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    colormap_name = st.selectbox("Colormap", ["JET", "HOT", "INFERNO", "MAGMA"])
    colormap_map  = {
        "JET":     cv2.COLORMAP_JET,
        "HOT":     cv2.COLORMAP_HOT,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA":   cv2.COLORMAP_MAGMA,
    }
    selected_cmap = colormap_map[colormap_name]

# Upload & Inference Method Selection
input_method = st.radio(
    "Choose Input Method:", 
    ["Upload File", "Take Live Photo"], 
    horizontal=True
)

if input_method == "Upload File":
    uploaded = st.file_uploader(
        "Drop a transistor image here",
        type=["png", "jpg", "jpeg"],
        help="Supports .png / .jpg / .jpeg"
    )
else:
    uploaded = st.camera_input(
        "Center the transistor in the frame and snap a photo",
        help="For best results, ensure the background and lighting match your training setup."
    )

if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    # Prepare padded display image (same 224×224 used for inference)
    raw_img  = cv2.imread(tmp_path)
    raw_rgb  = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    padded   = resize_with_pad(raw_rgb, needed_size=224)   # 224×224 RGB

    # Run inference
    with st.spinner("Running PatchCore inference..."):
        seed_everything(99)
        tensor = prepare_image(tmp_path, device)
        captured.clear()
        with torch.no_grad():
            resnet(tensor)

        # Image-level score
        patches = PatchCoreLite.aggregate_features(captured).to(device)
        score   = PatchCoreLite.compute_score(patches, memory_bank, temperature=TEMPERATURE)

        # Heatmap (uses the still-populated `captured` list)
        heatmap_rgb, patch_scores = generate_heatmap(captured, memory_bank, padded)

    os.unlink(tmp_path)

    # Re-blend with user-selected alpha and colormap
    norm_scores = (patch_scores - patch_scores.min()) / max(patch_scores.max() - patch_scores.min(), 1e-8)
    heat_up     = cv2.resize(norm_scores, (224, 224), interpolation=cv2.INTER_CUBIC)
    heat_color  = cv2.applyColorMap((np.clip(heat_up, 0, 1) * 255).astype(np.uint8), selected_cmap)
    heat_color  = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended     = cv2.addWeighted(padded.astype(np.float32), 1 - alpha,
                                  heat_color.astype(np.float32), alpha, 0).astype(np.uint8)

    is_anomaly = score > threshold

    # Score Gauge
    col_m1, col_m2, col_m3 = st.columns(3)

    col_m1.markdown("### Anomaly Score")
    col_m1.metric("", f"{score:.4f}")

    col_m2.markdown("### Threshold")
    col_m2.metric("", f"{threshold:.4f}",
                  delta=f"{score - threshold:+.4f}",
                  delta_color="inverse")

    if is_anomaly:
        col_m3.markdown("### Status\n<span style='color:#ff4d4d; font-weight:bold; font-size:24px;'>ANOMALY</span>", 
                    unsafe_allow_html=True)
    else:
        col_m3.markdown("### Status\n<span style='color:#2ecc71; font-weight:bold; font-size:24px;'>NORMAL</span>", 
                    unsafe_allow_html=True)

    st.progress(
        min(score / (threshold * 1.5), 1.0),
        text=f"Score {score:.4f} / display ceiling {threshold * 1.5:.4f}"
    )

    # Image Panels
    st.subheader("Inspection Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        st.image(padded, use_container_width=True)

    with col2:
        st.markdown("**Anomaly Heatmap**")
        heat_only = cv2.applyColorMap(
            (np.clip(heat_up, 0, 1) * 255).astype(np.uint8), selected_cmap
        )
        st.image(cv2.cvtColor(heat_only, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col3:
        st.markdown("**Overlay**")
        st.image(blended, use_container_width=True)

    # Patch Score Stats
    with st.expander("Patch-level score statistics"):
        flat = patch_scores.flatten()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Defect Intensity",  f"{flat.max():.4f}",
                  help="The maximum anomaly score found in a single localized area. " 
                  "If this exceeds the threshold, the component fails.")
        
        c2.metric("Average Structural Deviation", f"{flat.mean():.4f}",
                  help="The overall variance across the entire component surface. " 
                  "High values mean the whole object looks different from the training data.")
        
        c3.metric("Anomaly Spread",  f"{flat.std():.4f}", 
                  help="How clustered or scattered the anomalies are. " 
                  "A higher value indicates a sharp, localized defect rather " 
                  "than uniform background noise.")

else:
    st.info("Make sure to send a transistor image to begin inspection.")