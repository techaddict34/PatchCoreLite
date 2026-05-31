# PatchCore Lite — Transistor Defect Inspector

PatchCore Lite is a lightweight and energy-efficient anomaly detection pipeline for detecting defects on transistor images using **feature memory + nearest-neighbor matching**.

It includes:
- **Training** to build a patch feature memory bank from *normal* (“good”) samples
- **Automatic thresholding** using training score statistics
- **CLI evaluation** to compute detection accuracy on a labeled test set
- **Streamlit app** to inspect images and visualize **patch-level anomaly heatmaps**

---

## What this project does (high level)
1. Load **ResNet-18** pretrained on ImageNet.
2. Capture intermediate spatial feature maps from `layer2[-1]` and `layer3[-1]` using **forward hooks**.
3. Convert spatial features into a set of patch descriptors and build a **memory bank** (optionally subsampled via a coreset strategy).
4. For each test image, compute its patch descriptors and score them by **nearest-neighbor distance** to the memory bank.
5. Aggregate patch distances into a single image anomaly score using a **log-sum-exp** formulation.
6. Decide **NORMAL vs ANOMALY** using an auto-computed threshold:

> **threshold = mean(train_scores) + 3 × std(train_scores)**

---

## Repository structure
- `preprocessing.py`
  - Image preprocessing (resize + padding to 224, ImageNet normalization)
  - Feature hook function
  - `PatchCoreLite` implementation:
    - `aggregate_features(...)`
    - `coreset_subsample(...)`
    - `compute_score(...)`
- `train.py`
  - Builds the full patch bank from `transistor/train/good/*`
  - Subsamples it into a memory bank
  - Computes the anomaly threshold from training scores
  - Saves checkpoint as `transistor_lite_model.pt`
- `test.py`
  - Runs evaluation on `transistor/test/<category>/*`
  - Treats `good` as normal; any other folder name is treated as defect
  - Reports accuracy
- `app.py`
  - Streamlit UI for uploading images / capturing a photo
  - Loads `transistor_lite_model.pt`
  - Produces anomaly score + overlay heatmap

---

## Data format
The code expects this folder layout:

### Training (normal only)
```
transistor/train/good/
  *.png
  *.jpg
  *.jpeg
```

### Testing (labeled)
```
transistor/test/
  good/
    *.png
    *.jpg
  defect_class_1/
    *.png
    *.jpg
  defect_class_2/
    *.png
    *.jpg
  ...
```

Notes:
- `good/` is used as the **normal** reference during evaluation.
- The evaluation script labels `cat != "good"` as **defect**.

---

## Setup

First, clone the repository and navigate to the root directory:
```bash
git clone <your-repo-url>
cd PatchCoreLite

### Requirements
See `requirements.txt`:
- `streamlit`
- `opencv-python-headless`
- `numpy`
- `torch`
- `torchvision`
- `Pillow`

### Install
```bash
pip install -r requirements.txt
```

---

## Training
Build the memory bank and auto-threshold, then save the model checkpoint.

```bash
python train.py
```

Outputs:
- `transistor_lite_model.pt` containing:
  - `memory_bank` (subsampled patch descriptors)
  - `threshold` (three-sigma rule from training scores)
  - `train_mu`, `train_sigma`

---

## Evaluation (accuracy report)
Run inference on the test dataset folders and compute accuracy.

```bash
python test.py
```

Expected model file:
- `transistor_lite_model.pt` must exist (run `train.py` first).

---

## Streamlit inference & heatmap visualization
Start the web app:

```bash
streamlit run app.py
```

You can:
- Upload an image (`png/jpg/jpeg`) or use the camera
- Get:
  - anomaly score
  - threshold and score delta
  - NORMAL/ANOMALY status
  - patch-level anomaly heatmap overlay

Heatmap logic (summary):
- Uses captured spatial features from `layer2` and `layer3`
- Computes nearest-neighbor distances per patch position
- Normalizes distances to `[0, 1]`
- Resizes to `224×224` and blends onto the padded input image

---

## Key configuration knobs (in code)
These are currently defined as module-level constants:

### Training (`train.py`)
- `GOOD_FOLDER = "transistor/train/good"`

- `SAVE_PATH = "transistor_lite_model.pt"`

- `CORESET_PCT = 0.1` (Subsample ratio. Retains a diverse 10% slice of patches, shrinking the saved model size by 90% to maximize inference efficiency)

- `TEMPERATURE = 0.1` (Log-sum-exp scale. Approximates a hard maximum to isolate defects while smoothing out random pixel noise)

### Inference (`app.py` and `test.py`)
- `MODEL_PATH = "transistor_lite_model.pt"`
- `TEMPERATURE = 0.1`

### Threshold
The app and test scripts use the saved:
- `threshold`
- `train_mu` / `train_sigma`

---

## How scoring works (PatchCore Lite)
Implemented in `preprocessing.py` (`PatchCoreLite`):

1. **Feature aggregation** (`aggregate_features`)
   - Takes hooked outputs from `layer2[-1]` and `layer3[-1]`
   - Applies pooling to form local patch descriptors
   - Upsamples `layer3` to match `layer2` spatial resolution
   - Concatenates descriptors from both layers
   - Flattens descriptors into a patch matrix of shape `[num_patches, 384]`

2. **Memory bank creation**
   - Builds descriptors over all training “good” images
   - Subsamples patches using a coreset strategy (`coreset_subsample`)

3. **Anomaly scoring** (`compute_score`)
   - For each test patch descriptor, computes Euclidean distance to all memory bank descriptors
   - Uses the **minimum distance** per patch (nearest-neighbor distance)
   - Aggregates patch distances into one image score with:
     - `temperature * logsumexp(patch_distances / temperature)`

---

## Troubleshooting
- **`Model not found at 'transistor_lite_model.pt'`**
  - Run `python train.py` first.
- **Dataset folder errors**
  - Ensure `transistor/train/good` exists and contains normal images.
  - Ensure `transistor/test/...` folders exist.
- **Poor performance**
  - Increase `CORESET_PCT` for a larger memory bank (slower + more accurate).
  - Adjust `TEMPERATURE` (smaller = more sensitive, potentially noisier).

---

## Docker (optional)
A `Dockerfile` is included in this repo. If you use it, ensure the container has access to your dataset and that the `transistor_lite_model.pt` checkpoint is available for inference.

---

## License
Add your license information here (or remove this section).
