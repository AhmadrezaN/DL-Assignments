# Deep Learning & Neural Networks — Assignment: Sequence Dynamics & Time-Series Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A rigorous evaluation and implementation of sequence modeling paradigms developed for **Assignment 4** in the **Neural Networks and Deep Learning** course at the **University of Tehran** (Faculty of Electrical and Computer Engineering)[cite: 160, 174]. 

This project explores sequence modeling across two domains:
1. **Spoken Language Understanding (SLU)**: Joint intent classification and slot filling using recurrent and sequence-to-sequence neural architectures[cite: 163, 165].
2. **Financial Time-Series Forecasting**: Stationarity analysis and multi-step recursive stock price forecasting using hybrid convolutional-recurrent networks[cite: 170, 171].

---

## 📚 Table of Contents
- [Project Architecture & Structure]
- [Question 1: Spoken Language Understanding (ATIS)]
  - [Architectural Design]
  - [Evaluation & Benchmark Results]
  - [Key Findings & Analysis]
- [Question 2: S&P 500 Stock Price Prediction]
  - [Data Pipeline & Stationarity Analysis]
  - [Model Benchmarks & Computational Complexity]
  - [Ablation Study & Visualizations]

---

## 🗣️ Question 1: Spoken Language Understanding (ATIS)

### Architectural Design

Spoken Language Understanding (SLU) processes raw user utterances into structured frame representations by executing two complementary tasks simultaneously:

* **Intent Detection**: Sentence-level classification task identifying the overall user objective (e.g., `atis_flight`).


* **Slot Filling**: Token-level sequence labeling task using the Inside-Outside-Beginning (IOB) tagging format (e.g., `B-fromloc.city_name`).



We compare three primary architectures:

1. **Independent BiRNN Baseline**:
* Uses separate bidirectional Recurrent Neural Networks for intent classification and slot filling.


* Lacks parameter sharing between tasks.




2. **Joint BiLSTM (Shared Encoder)**:
* Utilizes a shared bidirectional LSTM encoder.


* **Slot Head**: Employs a Token-level Linear Layer mapping each hidden state $h_t$ to IOB tag space.


* **Intent Head**: Applies Max-Pooling over time across all sequence hidden states, feeding the resulting global representation into an Intent Linear Layer.


* **Joint Loss Function**:

$$\mathcal{L}_{\text{joint}} = \alpha \mathcal{L}_{\text{intent}} + (1 - \alpha) \mathcal{L}_{\text{slot}}$$




3. **Encoder-Decoder (Non-aligned Joint Model)**:
* Uses an Encoder LSTM to compress input text into context vectors.


* Uses an Autoregressive Decoder LSTM to generate slot tag sequences alongside a sequence-level intent prediction.


* Does not enforce strict 1-to-1 temporal alignment between input tokens and output tags.





---

### Evaluation & Benchmark Results

All models were trained on the official ATIS dataset and evaluated using standard metrics: **Accuracy** for intent detection and **Entity-Level F1-Score** (via `seqeval`) for slot filling.

| Model Architecture | Intent Accuracy (%) | Slot F1-Score (%) | Alignment Constraint | Loss Convergence |
| --- | --- | --- | --- | --- |
| **BiRNN (Independent)** | 96.25% | 94.79% | Strictly Aligned| Moderate
| **BiLSTM (Joint Architecture)** | **97.44%**<br> | **94.06%**<br> | Strictly Aligned| **Fastest / Stable**<br> |
| **Encoder-Decoder (Seq2Seq)** | 96.42%| 54.27%| Non-aligned / Generative| Slow / Error Propagation|

---

### Key Findings & Analysis

* **Task Coupling Benefit**: Joint training (BiLSTM) leverages feature sharing between global sentence semantics and local word contexts, achieving the highest overall Intent Accuracy (97.44%).


* **Alignment Sensitivity in Sequence Tagging**: Non-aligned Encoder-Decoder architectures struggle on slot extraction (54.27% F1). Because token alignment is implicit in slot filling tasks, forcing a sequence-to-sequence generative structure introduces exposure bias and search errors without providing explicit alignment guarantees.


* **Gradient Bottlenecks**: Vanilla BiRNN models suffer from gradient vanishing across longer utterances, whereas LSTM gating mechanisms stabilize long-term dependencies.



---

## 📈 Question 2: S&P 500 Stock Price Prediction

### Data Pipeline & Stationarity Analysis

The S&P 500 index (`^GSPC`) dataset was retrieved using `yfinance` covering daily records from **January 1, 2000, through 2025**.

1. **Statistical Stationarity Testing**:
* **Raw Close Prices**: Non-stationary exhibiting strong deterministic and stochastic trends.


* *Augmented Dickey-Fuller (ADF) Test*: $p = 0.9892$ (Failed to reject null hypothesis $H_0$ of a unit root).




* **Log-Return Transformation**:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$


* *ADF Test on Log Returns*: $p < 0.0001$ (Stationarity achieved at $>99\%$ confidence).




* **Autocorrelation (ACF) & Partial Autocorrelation (PACF)**:
* Confirmed significant short-term memory (lags 1 to 5), validating an autoregressive feature window length of $L = 30$ days.






2. **Data Splitting & Scaling**:
* **Train**: 2000–2021
* **Validation**: 2022–2023
* **Test**: 2024–2025


* *Data Leakage Prevention*: Fit `MinMaxScaler` parameters exclusively on the training set before transforming validation and test splits.





---

### Model Benchmarks & Computational Complexity

Models were trained to execute recursive multi-step forecasting across 30-day sliding input windows.

| Architectural Model | Trainable Parameters | Test MSE Loss | $R^2$ Score | MAPE (%) | MAE | RMSE | Computational Cost (FLOPs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline MLP (NARMAX)** | 20,548| 0.00253| 0.9917| 1.28%| 0.1159| 0.1572| **~39.9 K**<br> |
| **CNN-LSTM** | 69,060| **0.00089**<br> | **0.9969**<br> | 0.73%| 0.0064| **0.0096**<br> | ~1.22 M|
| **CNN-GRU** | **52,420**<br> | 0.00091| **0.9969**<br> | **0.72%**<br> | **0.0063**<br> | 0.0097| **~0.92 M**<br> |
| **FCN (CNN-Only Bonus)** | 14,818| 0.00312| 0.9884| 1.45%| 0.1302| 0.1766| ~84.2 K|

---

### Ablation Study & Visualizations

```text
Input (30, 1) ---> [1D Conv (32 filters, K=3)] ---> [Max Pooling (P=2)]
                                                              |
                                                              v
Output (t+1) <--- [FC Dense Layer] <--- [Recurrent Unit (LSTM/GRU, H=64)]

```

* **Feature Extraction Strategy**: Combining 1D Convolutional layers with recurrent units allows the network to first extract local invariant temporal patterns (edges/moving averages) before handing the representation to LSTM/GRU layers for long-term dependence modeling.


* **GRU Efficiency**: The **CNN-GRU** network achieves performance virtually identical to CNN-LSTM ($R^2 = 0.9969$) while requiring **24.1% fewer parameters** and approximately **24.5% fewer FLOPs** per forward pass.






```

```
