## Question 1: Brain Tissue Segmentation (M-Net / U-Net)

### Task Description

* **Goal**: Perform pixel-by-pixel 2D multi-class segmentation on 3D T1-weighted brain MRI scans to separate brain tissues into 4 target classes: Background, Cerebrospinal Fluid (CSF), Gray Matter (GM), and White Matter (WM).


* **Dataset**: IBSR dataset containing skull-stripped, pre-normalized 3D MRI volumes of 18 patients (`IBSR_{patient_no}_ana_strip.nii.gz` and ground truth `IBSR_{patient_no}_segTRI_fill_ana.nii.gz`).


* **Preprocessing Pipeline**:
1. Extracts slices along Axial ($xy$), Coronal ($xz$), and Sagittal ($yz$) axes starting at index 10 with step size 3 (up to 48 slices per axis, totaling 144 slices/patient).


2. Applies 90-degree rotation and center padding to uniform $256 \times 256$ dimensions.


3. Divides each slice into four non-overlapping $128 \times 128$ patches (576 patches/patient).


4. Data Split: $80\%$ Training (12 patients / 6,528 patches) and $20\%$ Validation (6 patients / 3,264 patches).




* **Model Architecture & Training Parameters**:
* **Architecture**: M-Net (U-Net variant) with Encoder-Decoder paths and skip connections. Total Trainable Parameters: **7,705,476**; Non-trainable: **0**.


* **Hyperparameters**: Batch size = 1, Optimizer = SGD, Learning Rate = 0.0001, Loss = Categorical Cross-Entropy, Epochs = 10, Weight Initialization = LeCun Normal Initialization ( $ std = \sqrt{1/\text{fan\_in}} $ ).





---

### Evaluation Results & Summary

After 10 training epochs, evaluation metrics were computed on the entire validation set across all patches:

| Class | Precision | Recall | Jaccard (IoU) | Dice (DSC) | Target Dice Met? |
| --- | --- | --- | --- | --- | --- |
| **Background** | 0.9997 | 0.9973 | 0.9970 | 0.9985 | — |
| **CSF** | 0.4798 | 0.9342 | 0.4641 | 0.6340 | Yes ($\ge 0.55$)|
| **GM (Gray Matter)** | 0.8831 | 0.7697 | 0.6985 | 0.8225 | Yes ($\ge 0.73$)|
| **WM (White Matter)** | 0.8273 | 0.8740 | 0.7391 | 0.8500 | Yes ($\ge 0.70$)|
| **Overall Mean** | — | — | — | **0.8262**<br> | Pass |

#### Result Analysis

1. **Target Performance**: Model performance surpassed all assignment threshold requirements (WM Dice reached 0.85 vs target 0.70; GM reached 0.82 vs target 0.73; CSF reached 0.63 vs target 0.55).


2. **CSF Class Challenge**: CSF achieved a high Recall (0.9342) but lower Precision (0.4798). This indicates the network successfully locates almost all CSF regions, but over-predicts boundaries into adjacent structures (false positives).


3. **Primary Reasons for CSF Degradation**:
* **Thin & Dispersed Structure**: CSF appears in thin, highly fragmented boundaries across brain ventricles.


* **Class Imbalance**: Total CSF pixels are significantly fewer than GM and WM in MRI scans.


* **Partial Volume Effects**: Inter-tissue boundaries suffer from intensity blur across neighboring voxel borders.





---

## Question 2: Vehicle Detection (Enhanced Faster R-CNN)

### Task Description

* **Goal**: Implement object detection to detect and bounding-box localize road vehicles on highway camera footage.


* **Dataset**: Large-Scale Vehicle Detection Dataset (LSVH).


* **Data Preprocessing**:
1. Mapped raw bounding boxes (`Car`, `Bus`, `Van`, `DontCare`) into a unified target class (`vehicle`).


2. Filtered out noisy/small human-unidentifiable objects tagged as `DontCare` and dropped images containing no vehicles.


3. Image tensor normalization applied matching MobileNet backbone expectations.




* **Architectural Concepts**:
* **Soft-NMS vs NMS**: Soft-NMS linearly decays confidence scores of overlapping proposals rather than completely suppressing them, retaining adjacent and closely grouped vehicles.


* **MobileNetV2 vs VGG16**: MobileNetV2 uses depthwise separable convolutions, reducing parameters ($4.2\text{M}$ vs $138\text{M}$) and compute cost ($569\text{M}$ vs $15,300\text{M}$ MAdd) to enable rapid inference.


* **Context-Aware RoI Pooling (CARoI)**: Standard RoI pooling distorts very small proposals during downsampling; CARoI dynamically uses Deconvolution for small proposals to preserve fine spatial details of distant vehicles.




* **Training Parameters**:
* **Model Backbone**: MobileNetV2 pretrained backbone with Faster R-CNN framework.


* **Hyperparameters**: Epochs = 25, Batch size = 8, Optimizer = Adam with LR Scheduler, Loss = Fast R-CNN joint loss (RPN Objectness + RPN Box Reg + Classifier Loss + Box Reg Loss).





---

### Evaluation Results & Summary

Training was executed over 25 epochs with loss tracking and $mAP@0.50$ evaluation across validation and test splits.

| Metric Split | Target | Achieved Value | Result |
| --- | --- | --- | --- |
| **Validation $mAP@0.50$** | $\ge 0.60$<br> | **0.7402**<br> | Passed |
| **Test Set $mAP@0.50$** | — | **0.6749**<br> | Passed |
| **Convergence** | Stable loss curve | Losses stabilized after epoch ~5

 | Passed |

#### Result Analysis

1. **Convergence & Generalization**: Total training and validation loss curves rapidly converged within initial epochs without signs of overfitting, proving effective data preprocessing and learning rate scheduling.


2. **Detection Performance**: Standard Faster R-CNN achieves $67.49\% \text{ mAP}@0.50$ on test data.


3. **Qualitative Insight**: Visual evaluation confirms that while nearby and medium-distance vehicles are accurately bounded and classified, plain Faster R-CNN struggles on extremely small, highly distant objects or densely overlapping vehicle clusters—underlining the theoretical necessity of Context-Aware RoI Pooling and Soft-NMS.
