-----

# Multi-modal Fusion Framework for Head and Neck Cancer Prognosis

This repository implements a deep learning-based multi-modal fusion framework designed to predict the prognosis (e.g., Overall Survival, OS) of Head and Neck Cancer (HNC) patients.

The framework integrates **CT Imaging (3D CNN)**, **Clinical Text (PubMedBERT)**, and **Graph Structural Features (GNN)** using a **Late Fusion** strategy (Gaussian Naive Bayes) to achieve robust and accurate predictions.

-----

## 🏗️ Architecture

The framework consists of three independent branch networks processing different modalities, followed by a fusion layer:

1.  **3D CNN Branch (CT Imaging)**:

      * **Input**: Preprocessed 3D CT scans (target size: 64x224x224) + Masks.
      * **Backbone**: Custom 3D Convolutional Network with 4 `ConvBlock` layers and `AdaptiveAvgPool3d`.
      * **Purpose**: Extracts deep spatial visual features from the tumor region.

2.  **GNN Branch (Graph Features)**:

      * **Input**: Graph data constructed from ROIs (feature dimension: 1302).
      * **Backbone**: Utilizes `ResGatedGraphConv` (Residual Gated Graph Convolutions) and `TopKPooling`.
      * **Purpose**: Captures topological relationships and feature interactions between different ROIs.

3.  **Text Branch (Clinical Reports)**:

      * **Input**: Clinical text descriptions and tabular clinical features.
      * **Backbone**: **PubMedBERT** (`BiomedNLP-PubMedBERT`) for semantic embedding + MLP for classification.
      * **Purpose**: Leverages unstructured semantic information from clinical descriptions.

4.  **Decision Fusion**:

      * **Method**: **Gaussian Naive Bayes (GNB)** fusion strategy.
      * **Mechanism**: Aggregates the probability outputs from the three individual models to generate the final prediction.

-----

## 📂 Dataset

The dataset is derived from 7 centers, including a multi-center development cohort and an independent external validation cohort.

| Cohort | Sample Size (n) | Description |
| :--- | :--- | :--- |
| **Model Development Cohort** | **1321** | Aggregated from 6 centers (TCIA Public Dataset: CHUS, CHUM, HMR, HN1, OPC1, OPC2). |
| - *Training Set* | 925 | Used for model training. |
| - *Validation Set* | 134 | Used for hyperparameter tuning and early stopping. |
| - *Internal Test Set* | 262 | Used for internal performance evaluation. |
| **External Validation Cohort** | **181** | Independent external data from ** External_set **. |

*Note: Strict exclusion criteria were applied to ensure data quality (e.g., excluding patients with undefined RTstructs or missing clinical data).*

-----

## 🛠️ Requirements

Please ensure the following Python libraries are installed:

```bash
torch>=1.8.0
torch_geometric
monai
transformers  # For PubMedBERT
scikit-learn
pandas
numpy
SimpleITK     # For 3D image processing
tqdm
```

-----

## 📂 File Structure

  * **`Model_.py`**: Defines the core model architectures, including `Model_CNN`, `Model_GNN`, `Model_TEXT`, and the `PubMedBertTextEncoder`.
  * **`data_utils.py`**: Contains the `NPC_OS_Dataset` class, handling CSV loading, SimpleITK-based CT preprocessing, and graph data construction.
  * **`trainer.py`**: The main training script including the training loop, validation, early stopping mechanism, and logging.
  * **`late_fusion_GNB.py`**: The fusion script. It loads pre-trained weights for CNN, GNN, and Text models, extracts probability vectors, trains the Gaussian Naive Bayes fusion model, and outputs evaluation metrics (AUC/Accuracy).

-----

## 🚀 Usage

### 1\. Data Preparation

Ensure your data directory structure matches the configuration in `data_utils.py`:

  * `dataset/`: Directory for processed graph datasets.
  * `csv_data/`: Directory for file paths and indices (e.g., `file_paths_CT.csv`).
  * `Text_data/`: Directory for clinical text CSVs.
  * `all_clin_data.csv`: Tabular clinical features.

### 2\. Training Single-Modality Models

Use `trainer.py` to train each sub-model individually. You need to instantiate the corresponding class (`Model_CNN`, `Model_GNN`, or `Model_TEXT`) within the `main()` function in `trainer.py`.

```bash
# Example: Run the training script
python trainer.py
```

### 3\. Multi-modal Fusion & Evaluation

Once all three models are trained and the best weights (`.pth` files) are saved, run the fusion script:

```bash
python late_fusion_GNB.py
```

This script will:

1.  Load the pre-trained weights for CNN, GNN, and Text models.
2.  Extract prediction probabilities for both Training and Test sets.
3.  Fit a **Gaussian Naive Bayes** classifier on the training predictions.
4.  Evaluate the fusion model on the test set and print the **Classification Report** (AUC, Accuracy, Precision, Recall).

-----

## 📊 Results

The `late_fusion_GNB.py` script outputs a detailed performance report:

  * **Individual Models**: Accuracy and AUC for CNN, GNN, and Text models independently.
  * **Fusion Model**: Accuracy and AUC for the GNB-fused result.

