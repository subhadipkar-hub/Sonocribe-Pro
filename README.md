# Sonocribe-Pro
SonoScribe-Pro is a deep learning framework for breast ultrasound analysis that combines explainable lesion segmentation, tumor classification, and uncertainty quantification — helping radiologists understand not just the diagnosis, but also the reasoning and confidence behind each prediction.

Student: Subhadip Kar  
Guides: Prof. Kalyan Chatterjee & Prof. Sayanti Banerjee

## Project summary
SonoScribe-Pro is a Swin-based hybrid segmentation + classification model for breast ultrasound images.
It outputs:
- Segmentation mask of lesion (pixel-level)
- Classification: `normal` / `benign` / `malignant`
- Model confidence (uncertainty score using Monte Carlo Dropout)

## Repo contents
- `src/model.py` : model definition (Swin encoder + CNN decoder + classification head)
- `src/dataset.py` : data loader + augmentations
- `src/train.py` : training script for joint optimization (Dice + Cross-Entropy)
- `src/inference.py` : inference with MC Dropout (confidence estimation)
- `requirements.txt` : packages for Colab

## Dataset
Download BUS dataset (Kaggle): https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset  
Place images, masks and labels under `dataset/` or mount them in Colab.

## How to run (Colab)
1. Upload your `kaggle.json` or download the dataset manually and unzip to `/content/dataset/`.
2. Install requirements: `pip install -r requirements.txt`
3. Run training:
