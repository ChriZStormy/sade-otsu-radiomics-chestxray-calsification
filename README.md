# Medical Image Segmentation and Classification using Multi-Otsu Optimization and Radiomics

This repository contains a comprehensive pipeline for the segmentation, feature extraction, and classification of medical images, specifically targeting **Brain MRI** and **Chest X-Ray** datasets. 

The core of the segmentation approach relies on Multi-level Otsu thresholding optimized via metaheuristic algorithms, such as Differential Evolution (DE) and unconstrained Self-Adaptive Differential Evolution (uSADE). This is followed by radiomic feature extraction and evaluation using a Support Vector Machine (SVM) classifier.

## 🚀 Features

* **Advanced Segmentation**: Utilizes optimized Multi-Otsu thresholding to segment regions of interest in complex medical images.
* **Metaheuristic Optimization**: Implements optimization algorithms (like DE and uSADE) to efficiently find optimal thresholds.
* **Radiomic Feature Extraction**: Extracts meaningful quantitative features from the segmented images, including GLCM (Gray-Level Co-occurrence Matrix) features, area, and other statistical properties.
* **Machine Learning Classification**: Employs a linear SVM classifier with Repeated Stratified K-Fold cross-validation to robustly evaluate the predictive power of the extracted radiomics.
* **Statistical Analysis**: Performs rigorous statistical testing (e.g., Friedman and Nemenyi tests) to compare the performance across different threshold levels and optimization techniques.

## 📊 Datasets Supported

The project is structured to seamlessly process two main categories of medical images:

1. **Brain X-Ray / MRI (`dataset/Brain-X-Ray`)**:
   * Alzheimer MRI Preprocessed Dataset
   * Brain Stroke CT Dataset
   * Brain Tumor MRI Dataset

2. **Thorax / Chest X-Ray (`dataset/Torax-X-Ray`)**:
   * Chest X-Ray COVID-19
   * Chest X-Ray Pneumonia
   * Chest X-Ray Tuberculosis

## ⚙️ Workflow Overview

The system operates in two primary phases:

### Phase 1: Segmentation and Feature Extraction
1. Load the corresponding datasets (Thorax or Brain).
2. Separate the dataset into classes (e.g., Healthy vs. Pathological).
3. For each image:
   - Convert to grayscale and compute the histogram.
   - Apply Multi-Otsu optimization using uSADE / DE to determine $D$ optimal thresholds.
   - Generate the segmented image.
   - Extract radiomic characteristics.
4. Save the extracted features into structured CSV files.

### Phase 2: Classification and Evaluation
1. Load the generated Radiomics CSV files.
2. Split the data using Repeated Stratified K-Fold cross-validation.
3. Standardize the data using `StandardScaler`.
4. Train a Linear SVM classifier.
5. Evaluate model performance using metrics such as **Accuracy**, **F1-Score**, and **AUC**.
6. Conduct statistical tests (Friedman and Nemenyi) to evaluate significance.
7. Generate visual results, plots, and comparative tables.

## 🛠️ Usage

The repository includes specific controller scripts to handle the different datasets:
- Run `datasets_brain.py` to execute the pipeline on the Brain MRI datasets.
- Run `datasets_torax.py` to execute the pipeline on the Chest X-Ray datasets.
- Run `clasificacion.py` for training the SVM and evaluating the extracted features.

*(Additional scripts like `ejemplos_segmentacion.py` are provided to visualize the segmentation process independently.)*

## 📝 License

Please refer to the repository for more details regarding licensing and usage conditions.
