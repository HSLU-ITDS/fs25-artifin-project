# Breast Cancer Detection – Fine-Tuning with EfficientNet and Multi-Task Learning

This repository provides a complete pipeline for fine-tuning image classification models to detect breast cancer using mammogram images from the RSNA Breast Cancer Detection Dataset (512x512 PNG version). It includes both a baseline CNN model and a multi-task learning model that incorporates tabular auxiliary targets.

### Project Summary

Project Summary is written on a separate document to maintin clean README documentation. Please refer to `ARTIFIN-Project-Report.pdf` to see detailed summary and background of project

## Dataset

Download the following folders/files from Kaggle:
(Link: https://www.kaggle.com/datasets/awsaf49/rsnabcd-512-png-v2-dataset?select=train_images)

- `train_images/` containing 512×512 resolution PNG images
- `train.csv` containing metadata and target labels

Folder structure:

```
train_images/
    {patient_id}/
        {image_id}.png
        ...
train.csv
```

## Setup and Installation

### 1. Install Required Packages

To install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Using GPUHub (Lab Services - PyTorch Environment)

In this environment, only the following packages are needed:

```bash
pip install opencv-python timm
```

Note: This project requires a CUDA-enabled GPU

## Fine-Tuning Experiment Pipeline

### Step 1: Dataset Trimming

Run the `ds_trim.ipynb` script to reduce and clean the dataset as necessary.

### Step 2: ROI Image Generation

Run the `generate_roi_images.ipynb` script to extract and save region-of-interest (ROI) images from the dataset.

To skip this step, you may use pre-generated ROI images. Zip file available at the following drive link:

https://hsluzern-my.sharepoint.com/:u:/g/personal/idanandrei_paguio_stud_hslu_ch/EWeK-957d6FDk2QKQ8DmUVwB4BTuQZ7hESeFfxANboIFwQ?e=zvUJxl

Required dataset directory structure:

```
/input/
    Trimmed_train.csv
    /roi/
        /trimmed_train_images_roi/
            {patient_id}/
                {image_id}.png
                ...
```

## Models

### 1. CNN Model (Baseline)

This model uses a pre-trained EfficientNet-B4 as a backbone and is fine-tuned only on image data.

### 2. CNN_AUX_4 Model (Multi-Task Learning)

This model uses EfficientNet-B4 for image encoding and includes auxiliary targets from `train.csv` such as patient age, view, and laterality. The auxiliary loss is weighted and the weight can be adjusted using the `aux_loss_weight` parameter in the configuration class (set as 40%).

## Output Files

Both models produce the following output files upon training:

| File Name                                  | Description                                               |
| ------------------------------------------ | --------------------------------------------------------- |
| `cv_epoch_results.csv`                     | Epoch-wise training and validation metrics for each fold  |
| `cv_oof_results.csv`                       | Out-of-fold evaluation metrics per fold                   |
| `model_fold{fold}.pth`                     | Best model checkpoint saved per fold                      |
| `metrics_per_fold.png`                     | Visualization of training and validation metrics per fold |
| `test_predictions.csv`                     | Test predictions along with corresponding true labels     |
| `test_sample_predictions_with_gradcam.png` | Grad-CAM visualizations for selected test images          |
