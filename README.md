# 🩺 SonoScribe-Pro: A Hybrid Transformer Framework for Trustworthy Breast Cancer Diagnosis  

SonoScribe-Pro is an advanced deep learning framework that assists radiologists in detecting and interpreting breast tumors from ultrasound images.  
Unlike traditional black-box AI models, it combines **explainable segmentation**, **tumor classification**, and **uncertainty quantification**, offering not only accurate predictions but also insights into *why* and *how confident* the system is about each diagnosis.  

---

## 📘 Abstract  
Medical AI systems often act as opaque black boxes, providing predictions without clarity or confidence estimation. SonoScribe-Pro introduces a **hybrid Swin Transformer–U-Net model** capable of segmenting lesion boundaries, classifying tumor types (benign, malignant, or normal), and quantifying prediction uncertainty.  
This creates a clinically reliable AI tool that mimics expert radiologists’ reasoning—focusing on the lesion area, making data-driven classifications, and flagging uncertain cases for expert review.  

---

## 🎯 Problem Statement  
Conventional CNN-based classifiers can predict tumor types but fail to explain *why* a decision was made or *how confident* it is. In critical medical contexts, lack of interpretability and uncertainty estimation reduces clinical trust.  

SonoScribe-Pro bridges this gap by integrating three key capabilities:
- **Hybrid Transformer Architecture** for global feature extraction.  
- **Segmentation-guided Explainability** for transparent decision-making.  
- **Uncertainty Quantification** to identify ambiguous predictions safely.  

---

## 🧠 Core Features  
- **Swin-UNet Backbone** — Combines Swin Transformer encoder and CNN decoder for effective feature extraction and localization.  
- **Dual-Head Network** — Simultaneous segmentation and classification from the same feature space.  
- **Monte Carlo Dropout** — Estimates model uncertainty during inference.  
- **Explainable AI Integration** — Provides segmentation maps highlighting decision regions.  
- **End-to-End Framework** — Outputs diagnosis, confidence score, and lesion mask in one pipeline.  

---

## ⚙️ Methodology & Workflow  
**Stage 1:** Baseline Classifier  
- Train a standard CNN (e.g., ResNet-34) for benign vs malignant classification.  
- Generate Grad-CAM maps for interpretability comparison.  

**Stage 2:** Building SonoScribe-Pro  
- Implement Swin-UNet hybrid model.  
- Use dual losses: Dice Loss (segmentation) + Cross Entropy Loss (classification).  

**Stage 3:** Uncertainty Quantification  
- Apply Monte Carlo Dropout to compute uncertainty across multiple forward passes.  

**Stage 4:** Evaluation  
- Compare baseline vs SonoScribe-Pro on AUC, F1, Dice Score.  
- Visualize segmentation, Grad-CAM, and uncertainty maps for analysis.  

---

## 🧩 Model Architecture  
- **Encoder:** Swin Transformer for hierarchical self-attention and global feature capture.  
- **Decoder:** CNN layers for fine-grained reconstruction of segmentation masks.  
- **Outputs:**  
  1. Classification (Benign / Malignant / Normal)  
  2. Segmentation Mask  
  3. Confidence Score (Uncertainty)  

---

## 🧪 Dataset  
- **Source:** [Breast Ultrasound Images Dataset (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)  
- **Classes:** Normal, Benign, Malignant  
- **Images:** 780 ultrasound scans with annotated lesion masks  

---

## 🧰 Technical Stack  
- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:**  
  - `timm` – Pre-trained Swin Transformer models  
  - `segmentation-models-pytorch` – Building blocks for hybrid architectures  
  - `scikit-learn`, `numpy`, `opencv-python`, `matplotlib`, `albumentations`, `tqdm`  
- **Platform:** Google Colab (T4 GPU)  

---

## 🧾 Installation  

```bash
# Clone the repository
git clone https://github.com/yourusername/SonoScribe-Pro.git
cd SonoScribe-Pro

# Install dependencies
pip install -r requirements.txt
