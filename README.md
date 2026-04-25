# CRC-SCAN | Clinical AI Workspace
=================================
A clinical-grade histopathology diagnostic workstation built on the [STARC-9](https://huggingface.co/datasets/Path2AI/STARC-9) dataset. This workstation provides automated tissue decomposition, spatial discovery, and structured pathology reporting.

## 🚀 Key Features
- **Calibrated CNN Inference**: Post-processed confidence scores (Temperature Scaling $T=1.5$) for realistic diagnostic filtering.
- **TME Fingerprint (Radar)**: Visualizes the Tumor Microenvironment (Tumor, Immune, Stroma, Vascular, Necrosis) against healthy benchmarks.
- **Pathology Lens**: Dual-layer magnification allowing pathologists to inspect original high-res slide tiles under the AI heatmap.
- **Clinical PDF Reports**: Automatically generate hospital-ready reports with drawn microscope branding, quantitative bars, and TME visualizations.
- **Permissive H&E Validation**: Intelligent image verification that ensures only legitimate tissue slides are processed.

## 🛠️ Architecture
The system has been upgraded to a modern, decoupled high-performance stack:
- **Backend**: FastAPI (Python) · PyTorch · Grad-CAM · FPDF2
- **Frontend**: Vanilla ES6 JS · Chart.js · CSS3 (Glassmorphism & Modern Aesthetics)
- **Model**: STARC-9 Custom CNN (ResNet-based architecture)

## 📦 Setup & Execution

### 1. Requirements
Ensure you have Python 3.9+ and the STARC-9 dependencies installed:
```bash
pip install torch torchvision timm grad-cam fastapi uvicorn fpdf2 pillow opencv-python numpy
```

### 2. weights & Evaluation Repo
The workstation requires the `STARC-9-Evaluation` submodule and trained weights:
```bash
# Weights should be located at ./weights/Model_weights/best_custom_cnn.pth
# Submodule should be at ./STARC-9-Evaluation/
```

### 3. Launching the Workstation
Run the centralized launcher script:
```bash
python run_server.py
```
Then navigate to **http://localhost:8000** in your browser.

## 🔬 Classification Glossary
| Code | Tissue Description |
|---|---|
| **ADI** | Adipose tissue (fat cells) |
| **LYM** | Lymphocytes (immune infiltration) |
| **MUC** | Mucin (secretory tissue) |
| **MUS** | Smooth muscle layers |
| **NCS** | Necrotic debris (cell death) |
| **NOR** | Normal mucosal architecture |
| **BLD** | Red blood cells (vascularity) |
| **FCT** | Loose connective tissue (stroma) |
| **TUM** | Malignant Adenocarcinoma cells |

---
**Disclaimer**: This tool is for research and clinical decision support purposes. Final diagnosis must be verified by a human pathologist.
