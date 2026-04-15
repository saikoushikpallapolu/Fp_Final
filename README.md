# CRC-Scan Clinical AI Workspace

A clinical-grade histopathology AI diagnostic tool built on the [STARC-9](https://huggingface.co/datasets/Path2AI/STARC-9) dataset from Stanford University.

## Features

- 🔬 **Sliding-Window Tile Inference** — Processes large biopsy slides using 256×256 tiles
- 🎨 **Interactive AI Heatmap** — Color-coded tissue overlay with clinical legend
- ⚠️ **Uncertainty Filtering** — Tiles below 70% confidence are flagged for human review
- 📊 **Tissue Composition Charts** — Donut and bar charts for all 9 tissue classes
- 🧠 **Clinical Insights** — TIL ratio, necrotic load, and automated diagnostic summaries
- ⚡ **Grad-CAM Explainability** — Click any tile to generate an activation map

## Tissue Classes

| Code | Tissue |
|------|--------|
| ADI | Adipose |
| LYM | Lymphocyte |
| MUC | Mucin |
| MUS | Muscle |
| NCS | Necrotic Debris |
| NOR | Normal |
| BLD | Red Blood Cells |
| FCT | Loose Connective Tissue |
| TUM | Tumor |

## Setup

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit timm grad-cam streamlit-image-coordinates altair pandas pillow scikit-learn seaborn
```

### 2. Download Model Weights
```bash
mkdir weights
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Path2AI/STARC-9', filename='Model_weights/best_custom_cnn.pth', repo_type='dataset', local_dir='./weights')
"
```

### 3. Clone Evaluation Repository
```bash
git clone https://github.com/rathinaraja/STARC-9-Evaluation.git
```

### 4. Run the App
```bash
python -m streamlit run app.py
```

## Project Structure

```
FP_REF/
├── app.py                    # Streamlit Clinical Dashboard
├── multi_scale_model.py      # Dual-branch context-aware model
├── STARC-9-Evaluation/       # Stanford baseline evaluation scripts
│   ├── config.py
│   ├── main.py
│   ├── evaluate_model.py
│   ├── CNN_model.py          # STARC-9 CustomCNN architecture
│   └── ...
└── weights/                  # Downloaded model checkpoints (not tracked in git)
    └── Model_weights/
        └── best_custom_cnn.pth
```

## Enhancements over Baseline

1. **Confidence Thresholding** — 0.70 Softmax filter flags uncertain tiles
2. **Dynamic Class Weights** — Balanced loss function fixing rare class accuracy
3. **Multi-Scale Architecture** — Dual-branch ResNet50 for contextual awareness
4. **Grad-CAM Explainability** — Visual justification for every prediction
5. **Clinical Dashboard** — Streamlit frontend with real-time tissue analytics

## Citation

```
Subramanian, B., et al. (2025). STARC-9: A large-scale dataset for multi-class tissue 
classification for CRC histopathology. NeurIPS 2025 Datasets and Benchmarks Track.
```
