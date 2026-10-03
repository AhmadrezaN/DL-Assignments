## Executive Summary & Overview

This document provides a comprehensive structured summary of the report for **Assignment: Neural Networks and Deep Learning. The report evaluates deep learning techniques across two distinct computer vision benchmarks:

1. **Task 1: Rice Leaf Disease Detection** — Comparing a baseline CNN architecture trained from scratch against a pre-trained Transfer Learning model.


2. **Task 2: Remote Sensing Scene Classification** — Evaluating a hybrid CNN-MLP framework on high-resolution satellite imagery with hyperparameter, optimizer, and fine-tuning experiments.
---

## Task 1: Rice Leaf Disease Detection

### 1.1 Dataset Preparation & Augmentation

* **Source Dataset:** Kaggle `rice-disease-dataset` (6 total classes).


* **Selected Classes:** `Healthy Rice Leaf`, `Bacterial Leaf Blight`, `Leaf Blast`, `Sheath Blight`[cite: 66, 78, 79].
* **Dataset Splitting:** Train: 70% (~442–457 images/class), Validation: 15% (~95–98 images/class), Test: 15% (~95–98 images/class)[cite: 66, 79].
* **Augmentation Pipeline (Training Set Only):**
* Resizing to $224 \times 224$ pixels[cite: 67, 80].
* Random Horizontal/Vertical Flips ($p = 0.5$)[cite: 80].
* Random Rotation (up to 15°) & Random Affine Translation[cite: 80].
* Color Jitter (brightness, contrast, saturation, hue)[cite: 80].



### 1.2 Model 1: AlexNet (From Scratch)

* **Architecture Specs:** Custom AlexNet implementation outputting 4 logits[cite: 67, 82].
* Total Parameters: $58,297,732$ (Fully Trainable)[cite: 82].


* **Training Setup:** Trained for 30 epochs, SGD Optimizer ($lr=0.01$, Momentum$=0.9$, Weight Decay$=5\times 10^{-4}$), Batch Size $= 32$, CrossEntropy Loss[cite: 83].
* **Performance Summary:**
* **Overall Test Accuracy:** $87.27\%$[cite: 84].
* **Mean Sensitivity (Recall):** $74.48\%$[cite: 84].
* **Mean Specificity:** $91.53\%$[cite: 84].
* **Mean Precision:** $75.12\%$ | **Mean F1-Score:** $74.68\%$[cite: 84].


* **Key Observations:** Best performance achieved on `Healthy Rice Leaf` ($94.81\%$ Accuracy, $85.71\%$ Recall)[cite: 84]. Significant inter-class confusion occurred between `Leaf Blast` and `Bacterial Leaf Blight` due to visual similarity in spot patterns[cite: 86].

### 1.3 Model 2: INC-VGGN (Transfer Learning)

* **Architecture Specs:** First 4 convolutional blocks of pre-trained VGG19 (frozen) integrated with 2 custom Inception modules and Global Average Pooling (GAP)[cite: 68, 87, 88].
* Total Parameters: $30,887,684$[cite: 88].
* Trainable Parameters: $1,562,116$[cite: 88].
* Non-trainable Parameters: $29,325,568$[cite: 88].


* **Preprocessing:** Input normalized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)[cite: 68, 88].
* **Performance Summary:**
* **Overall Test Accuracy:** $97.01\%$[cite: 89].
* **Mean Sensitivity (Recall):** $94.00\%$[cite: 89].
* **Mean Specificity:** $98.01\%$[cite: 89].
* **Mean Precision:** $94.15\%$ | **Mean F1-Score:** $94.02\%$[cite: 89].



### 1.4 Comparative Analysis & Technical Q&A

| Metric / Feature | AlexNet (From Scratch)[cite: 82, 83, 84] | INC-VGGN (Transfer Learning)[cite: 88, 89] |
| --- | --- | --- |
| **Trainable Parameters** | ~58.3 Million[cite: 82] | ~1.56 Million[cite: 88] |
| **Test Accuracy** | $87.27\%$[cite: 84] | **$97.01\%$**[cite: 89] |
| **F1-Score** | $74.68\%$[cite: 84] | **$94.02\%$**[cite: 89] |
| **Convergence Speed** | Gradual improvements through 30 epochs[cite: 83, 94] | Rapid convergence by Epoch ~10–20[cite: 89, 92, 94] |

#### Key Insights from Responses:

1. **Impact of Transfer Learning:** Pre-trained VGG features provide rich low-level geometric and color representations learned from ImageNet, reducing optimization complexity and preventing overfitting on small plant datasets[cite: 94, 95].
2. **Role of Inception Modules:** Replaces static single-scale filters with multi-scale parallel receptive fields ($1\times 1$, $3\times 3$, $5\times 5$), enabling effective capture of plant lesion spots of varying shapes and sizes[cite: 96].
3. **Global Average Pooling (GAP) vs. FC Layers:** Replacing heavy fully connected layers with GAP reduces parameters from millions down to thousands, preserves spatial feature relationships, and acts as a strong regularizer against overfitting[cite: 97].

---

## Task 2: Remote Sensing Scene Classification

### 2.1 Dataset & Data Augmentation

* **Dataset:** `UC-Merced Land Use Dataset` (21 land-use scene classes)[cite: 70, 99].
* **Split Ratio:** $80\%$ Training, $20\%$ Testing.


* **Augmentations:** $90^\circ / 180^\circ$ Rotations, Horizontal/Vertical Flips, Random Zooming[cite: 71, 98, 99]. Rotational invariance is crucial for aerial satellite imagery because top-down views lack a fixed vertical orientation[cite: 71, 98].

### 2.2 Model Architecture & Hyperparameters

* **Feature Extractor:** Pre-trained **Xception** network (frozen base extracting 2,048-dimensional features)[cite: 71, 99].
* **Classifier:** Enhanced MLP consisting of `Feature Normalization` $\rightarrow$ `Sigmoid` activation $\rightarrow$ `Dropout (0.4)` $\rightarrow$ `Linear (2048 → 21)`[cite: 72, 99].
* **Optimal Baseline Hyperparameters:** Batch Size $= 200$, Epochs $= 50$, Optimizer $=$ Adagrad ($lr=0.005$, label smoothing $= 0.05$)[cite: 101].
* **Baseline Accuracy:** **$91.43\%$** Test Accuracy[cite: 100, 101].

### 2.3 Ablation Studies & Optimizations

#### 1. Impact of Data Augmentation

* **With Augmentation:** Test Accuracy = **$91.43\%$**[cite: 100, 108].
* **Without Augmentation (Raw Data):** Test Accuracy = **$87.14\%$**[cite: 107, 109].
* **Verdict:** Data augmentation yields a **$+4.29\%$ accuracy boost**, preventing severe overfitting and smoothing out training loss curves[cite: 109, 110].

#### 2. Fully Connected (FC) Layers vs. Custom MLP Classifier

* Replacing the lightweight MLP classifier with standard heavy FC layers increased training accuracy to $99.90\%$ but led to overfitting, achieving **$93.33\%$** test accuracy with larger train-test generalization gaps[cite: 104].

#### 3. Optimizer Comparison (Adagrad vs. Adam)

* **Adagrad ($lr=0.005$):** Test Accuracy = **$91.43\%$**[cite: 100, 112].
* **Adam ($lr=0.005$):** Test Accuracy = **$92.38\%$**[cite: 111, 112].
* **Verdict:** Adam outperformed Adagrad by maintaining momentum and preventing learning rates from decaying too rapidly[cite: 113].

#### 4. Fine-Tuning Analysis

* **Strategy:** Unfreezing the last 10 convolutional layers of the Xception base with Adam optimizer ($lr=0.001$, label smoothing $= 0.1$)[cite: 73, 113, 114].
* **Results:** Reached **$92.38\%$ Test Accuracy** and **$93.05\%$ Train Accuracy**[cite: 113, 114]. Fine-tuning allowed top-level filters to adapt specifically to complex aerial textures[cite: 73, 113].
