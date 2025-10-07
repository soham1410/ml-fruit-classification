# Ml-Fruit-classification

**Project Title:** A Comparative Analysis of SVM, CNN, and Ensemble Models for Fruit Classification

## Introduction

This project focuses on the classification of fruits from the Fruits-360 dataset. It reproduces the methodology of a DAML 2023 paper and extends it by implementing an ensemble model for improved classification accuracy. The project includes a Streamlit web application that allows users to upload fruit images and receive predictions from various trained models.

## Proposed Work/Methodology

The project explores and compares three primary machine learning approaches for fruit classification:

1.  **SVM with PCA:** This method utilizes Principal Component Analysis (PCA) for dimensionality reduction, followed by a Support Vector Machine (SVM) for classification. The implementation experiments with different numbers of principal components (k=2, 5, 8) and uses k-fold cross-validation to ensure robustness.

2.  **Convolutional Neural Network (CNN):** A 6-layer CNN architecture, as described in the DAML 2023 paper, is implemented using PyTorch. This deep learning model is designed to automatically learn and extract features from the fruit images.

3.  **Ensemble Model:** To enhance prediction accuracy, a soft-voting ensemble model is created. This model combines the predictions from the CNN and the best-performing SVM+PCA model, leveraging the strengths of both approaches.

## System Architecture

**Components:**

1.  **Data Layer:**
    *   **Kaggle Dataset (Fruits-360):** The primary source of data.

2.  **Data Preparation Layer:**
    *   **Data Downloader (`src/data.py`):** Downloads the dataset from Kaggle.
    *   **Image Transformation:** Applies transformations such as resizing and conversion to tensors.

3.  **Model Training Layer (`train.py`):**
    *   **Configuration (`config.json`):** A JSON file that stores hyperparameters for the training process.
    *   **SVM+PCA Pipeline:**
        *   **PCA:** Reduces the dimensionality of the image data.
        *   **SVM:** A Support Vector Machine classifier trained on the PCA-transformed data.
    *   **CNN Pipeline:**
        *   **CNN:** A 6-layer Convolutional Neural Network trained on the image data.
    *   **Ensemble Pipeline:**
        *   **Ensemble Model:** A soft-voting ensemble that combines the predictions of the CNN and the best SVM+PCA model.

4.  **Model Storage:**
    *   **`models/` directory:** Stores the trained models (`.pkl` for SVM/PCA, `.pth` for CNN).

5.  **Inference Layer (`src/inference.py`):**
    *   **Model Loader:** Loads the trained models from the `models/` directory.
    *   **Prediction Engine:** Takes an input image and generates predictions from all models.

6.  **Presentation Layer (`app.py`):**
    *   **Streamlit Web App:** Provides a user interface for uploading images and viewing predictions.

**Flows:**

1.  The **Data Downloader** retrieves the **Kaggle Dataset**.
2.  The **Image Transformation** component processes the data for training.
3.  The **Model Training Layer** reads the **Configuration** from `config.json`.
4.  The training layer then uses the prepared data and hyperparameters to train the **SVM+PCA**, **CNN**, and **Ensemble** models.
5.  The trained models are saved to the **Model Storage**.
6.  The **Streamlit Web App** receives an image from the user.
7.  The **Prediction Engine** in the **Inference Layer** loads the models from **Model Storage** and uses them to classify the image.
8.  The **Streamlit Web App** displays the predictions to the user.

## Data Sets

-   **Description:** The project utilizes the Fruits-360 dataset, a large dataset of fruit images.
-   **Source:** The dataset is automatically downloaded from Kaggle using the `moltean/fruits` dataset slug. The `src/data.py` script handles the download and extraction process.
-   **Exploration and Loading:** The `train.py` script is responsible for loading the data. The implementation has been corrected to use the full training and testing sets to ensure model robustness and accuracy. The `visualize.py` script generates PCA plots for further data exploration.

## Algorithms

The following algorithms are implemented and evaluated in this project:

-   **Principal Component Analysis (PCA):** For dimensionality reduction.
-   **Support Vector Machine (SVM):** For classification.
-   **Convolutional Neural Network (CNN):** For deep learning-based classification.
-   **Ensemble Learning (Soft-Voting):** To combine predictions from multiple models.

## Implementation (Steps)

The project follows these implementation steps:

1.  **Configuration:** Before running the training, review and set the desired hyperparameters in `config.json`.
2.  **Data Preparation:** The `src/data.py` script downloads and prepares the Fruits-360 dataset for training.
3.  **Model Training:** The `train.py` script reads the configuration and trains the SVM+PCA, CNN, and ensemble models on the prepared data.
4.  **Model Evaluation:** After training, the `train.py` script evaluates the performance of each model and saves the results.
5.  **Visualization:** The `visualize.py` script generates various plots to visualize the model performance and data characteristics.
6.  **Inference:** The `app.py` script provides a user-friendly interface to test the trained models with new fruit images.

## Experimental Setup

-   **Configuration (`config.json`):** Key hyperparameters for the training process are defined in `config.json`. This includes:
    -   `pca_components`: A list of component numbers for PCA.
    -   `fold_values`: A list of fold numbers for cross-validation.
    -   `random_state`: The seed for reproducible random operations.
    -   `cnn_epochs`: The number of epochs for CNN training.
    -   `batch_size`: The batch size for data loaders.
    -   `learning_rate`: The learning rate for the Adam optimizer.
-   **Tools:** The Kaggle API is used for downloading the dataset.
-   **Libraries:**
    -   `streamlit`: For creating the web application.
    -   `pandas`: For data manipulation and analysis.
    -   `Pillow (PIL)`: For image processing.
    -   `torch` & `torchvision`: For implementing and training the CNN.
    -   `scikit-learn`: For implementing the SVM and PCA models.
    -   `matplotlib` & `seaborn`: For data visualization.
    -   `tqdm`: For displaying progress bars.
-   **Framework:** PyTorch is the primary deep learning framework used for the CNN model.
-   **Language:** The entire project is implemented in Python.

## Libraries and Functions Used

-   **`streamlit`**: `st.title`, `st.markdown`, `st.sidebar`, `st.file_uploader`, `st.image`, `st.spinner`, `st.success`, `st.metric`
-   **`pandas`**: `pd.read_csv`, `pd.DataFrame`
-   **`torch`**: `nn.Module`, `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`, `nn.Dropout`, `optim.Adam`, `nn.CrossEntropyLoss`
-   **`torchvision`**: `transforms.Compose`, `transforms.Resize`, `transforms.ToTensor`, `datasets.ImageFolder`
-   **`sklearn`**: `PCA`, `SVC`, `StratifiedKFold`, `accuracy_score`, `classification_report`, `confusion_matrix`
-   **`matplotlib`**: `plt.figure`, `plt.scatter`, `plt.title`, `plt.savefig`, `plt.close`
-   **`seaborn`**: `sns.color_palette`, `sns.heatmap`
-   **`joblib`**: `joblib.dump`, `joblib.load`
-   **`PIL`**: `Image.open`
