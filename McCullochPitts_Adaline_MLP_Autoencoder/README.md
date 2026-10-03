## Overview

This repository contains the full theoretical derivations, implementation, and analysis for **Assignment 1 of Neural Networks and Deep Learning** at the University of Tehran (Department of Electrical and Computer Engineering).

---

## Task Explanations

### Question 1: McCulloch-Pitts Neurons & Logical Gate Separability

* **Task Description:**
1. Design single McCulloch-Pitts threshold neurons (determining weights $w$ and bias $b$) for basic boolean logic functions (`AND`, `OR`, `NOT`) without coding.


2. Explain intuitively using the geometry of decision boundaries why a single threshold neuron can only express linearly separable boolean functions.


3. Geometrically explain why a 3-input `Parity` function (`3-way XOR`) is linearly non-separable.


4. Design the smallest 2-layer threshold neural network architecture (with exact numerical weights and biases) to accurately represent the 3-input `Parity` function.


5. Design a network/neuron model for a 5-input `Majority` function and analyze the required hidden units.





---

### Question 2: Adaline & Madaline (Rule II) Models

* **Task Description:**
1. **Theory (Adaline):** Derive the batch Least Mean Squares (LMS) update rule from Sum of Squared Errors (SSE). Explain its relation to Normal Equations, write the online LMS rule, and explain how adding $L_2$ regularization leads to Ridge regression.


2. **Adaline Implementation:** Train Adaline from scratch on the **Breast Cancer Wisconsin** dataset. Split data (70% train, 30% test) using fixed seed = 0. Perform hyperparameter tuning on learning rates ($\eta$) and $L_2$ regularization. Evaluate performance on test data using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC. Visualize decision boundaries in 2D space reduced via PCA. Compare Adaline against standard probabilistic baselines (Logistic Regression & LDA).


3. **Madaline (Rule II) Implementation:** Implement a 3-layer Madaline network (with $m$ hidden Adaline units and a `sign`/`tanh` output activation).


4. **Non-linear Classification:** Train Madaline on synthetic non-linear `make_moons` and `make_circles` datasets ($N=600$, noise = $0.25$ and $0.10$, seed = $0$). Analyze hyperparameter choices across multiple seeds ($0, 1, 2$), plot decision surfaces, and show how hyperplanes combine to model complex boundaries.





---

### Question 3: California Housing Price Regression using MLP

* **Task Description:**
1. **Data Preprocessing:** Clean and preprocess the **California Housing Prices** dataset:


* **Missing Values:** Impute `total_bedrooms` missing values by fitting a simple linear regression model with the highly correlated `households` column ($r \approx 0.98$).


* **Outliers:** Filter out target (`median_house_value`) outliers using the Interquartile Range (IQR) method.


* **Categorical Encoding:** Convert categorical features (`ocean_proximity`) into numeric values via One-Hot Encoding.


* **Data Splitting:** Divide into Train/Validation/Test sets ($80\% / 10\% / 10\%$).




2. **Simple MLP Architecture:** Implement a baseline MLP network (Input $\rightarrow 8 \rightarrow \text{ReLU} \rightarrow 1$) consisting of 89 parameters. Train for 60 epochs using Adam optimizer ($\eta = 0.1$) and MSE Loss.


3. **Complex MLP Architecture:** Design a deeper MLP (Input $\rightarrow 32 \rightarrow 64 \rightarrow 16 \rightarrow 1$) with 3,617 parameters to outperform the baseline network.


4. **Evaluation:** Custom-implement and evaluate metrics: $MSE$, $RMSE$, $MAE$, and $R^2$ Score on all splits. Generate comparative Loss Curves and Actual vs. Predicted scatter plots.





---

### Question 4: AutoEncoder & Unsupervised Clustering on Fashion MNIST

* **Task Description:**
1. **Preprocessing:** Normalize Fashion-MNIST image data to $[0, 1]$. Explain the theoretical importance of normalization in deep learning.


2. **AutoEncoder Implementation:** Build a Fully Connected AutoEncoder with 3 encoder layers ($784 \rightarrow 512 \rightarrow 256 \rightarrow 64$) and 3 decoder layers ($64 \rightarrow 256 \rightarrow 512 \rightarrow 784$) using ReLU, Tanh bottleneck, Sigmoid output, and AdamW optimizer.


3. **Reconstruction Evaluation:** Train the AutoEncoder and plot Train/Validation MSE Loss. Visualize original vs. reconstructed images across 5 distinct fashion classes.


4. **Latent Space Clustering:** Encode test data to the 64-dimensional latent space. Cluster using K-Means ($K=10$).


5. **Comparative Analysis:** Compute Confusion Matrices, plot 2D t-SNE projections, and evaluate cluster quality using **Adjusted Rand Index (ARI)** and **Adjusted Mutual Information (AMI)** for both encoded representations and raw image features.





---

## Summary of Results and Analysis

### Question 1: McCulloch-Pitts & Logic Gates

* **Linear Separability:** A single threshold neuron calculates $y = f(w^T x + b)$, creating a flat hyperplane ($w^T x + b = 0$). It cannot create curved decision boundaries, limiting single neurons to linearly separable functions.


* **Parity Non-Separability:** In 3D space, the points outputting `1` and `0` form alternating vertices on a unit cube, making linear separation impossible. A minimum 2-layer network with 3 hidden neurons is required to solve 3-input Parity.


* **Majority Function:** A 5-input `Majority` function can be computed directly by a single threshold neuron with weights $w_i = 1$ and bias $b = -2.5$ without requiring any hidden layer.



---

### Question 2: Adaline & Madaline (Breast Cancer, Moons, Circles)

* **Breast Cancer (Adaline):**
* **Best Hyperparameters:** Learning rate $\eta = 0.0005$, $L_2$ penalty $\lambda = 0.12$.


* **Test Performance:** Accuracy = **$92.40\%$**, Precision = **$89.83\%$**, Recall = **$99.07\%$**, F1-Score = **$94.22\%$**, ROC-AUC = **$0.9740$**.


* **Comparison:** Logistic Regression ($95.91\%$ Accuracy) and LDA ($95.32\%$ Accuracy) outperformed Adaline. Probabilistic models provide better-calibrated decisions on overlapping distributions compared to simple MSE-based Adaline updates.




* **Moons & Circles (Madaline Rule II):**
* **Moons Dataset:** 5 hidden units achieved Test Accuracy = **$90.56\%$**, F1 = **$0.8994$**, ROC-AUC = **$0.9607$**.


* **Circles Dataset:** 5 hidden units achieved Test Accuracy = **$98.33\%$**, F1 = **$0.9829$**, ROC-AUC = **$0.9975$** (Reaching up to $100\%$ accuracy under optimal seed configuration).


* **Decision Surface Logic:** The hidden layer forms intersecting linear hyperplanes, which are aggregated non-linearly to cleanly isolate non-convex shapes.





---

### Question 3: California Housing MLP Regression

| Model Metric | Simple MLP (89 params)

 | Complex MLP (3,617 params)

 | Improvement / Result |
| --- | --- | --- | --- |
| **Test MSE** | $3.4936 \times 10^9$<br> | **$2.2894 \times 10^9$**<br> | Significant Error Reduction

 |
| **Test RMSE** | $59,107.04$<br> | **$47,847.18$**<br> | $\sim 11,260$ units lower error

 |
| **Test MAE** | $44,065.17$<br> | **$32,991.28$**<br> | $\sim 11,074$ units lower error

 |
| **Test $R^2$ Score** | $0.6185$<br> | **$0.7500$**<br> | Met requirement ($R^2 \ge 0.70$)

 |

* **Analysis:** The complex MLP captures non-linear feature interactions significantly better than the baseline. Scatter plots of Actual vs. Predicted values show that predictions from the complex MLP cluster far more tightly along the ideal $y=x$ diagonal line.



---

### Question 4: AutoEncoder & Clustering on Fashion MNIST

* **Reconstruction:** The AutoEncoder successfully learned compressed representations. Reconstructed images smoothed out fine noise while retaining essential global structures.


* **Clustering Analysis:** K-Means clustering formed distinct groupings in the 64D encoded space. Visual classes with distinct silhouettes (e.g., `Trouser`, `Bag`, `Ankle Boot`) separated cleanly, whereas visually similar items (e.g., `Shirt` vs. `Pullover` vs. `Coat`) exhibited expected sub-cluster overlaps.



#### Clustering Metrics Evaluation ($K=10$)

| Dataset Representation | Adjusted Rand Index (ARI) | Adjusted Mutual Information (AMI) | Requirement Status |
| --- | --- | --- | --- |
| **Raw High-Dim Images** | $0.3917$<br> | $0.8445$<br> | — |
| **Encoded Latent Space (AutoEncoder)** | **$0.3992$**<br> | **$0.8590$**<br> | **Passed** ($\text{ARI} \ge 0.30$, $\text{AMI} \ge 0.50$)|
| **Absolute Improvement** | **$+0.0075$**<br> | **$+0.0145$**<br> | Latent space reduces noise and improves separation|

* **Conclusion:** Dimensionality reduction via the AutoEncoder's non-linear bottleneck successfully removed irrelevant pixel redundancy, producing a cleaner manifold for unsupervised clustering algorithms.
