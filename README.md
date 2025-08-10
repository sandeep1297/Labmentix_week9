# 🐟 Multiclass Fish Image Classification

This project focuses on building, evaluating, and deploying a deep learning model to classify fish images into multiple categories.  
It serves as a comprehensive guide to the **machine learning project lifecycle**, from data preparation to a functional web application.

---

## 📚 Skills Acquired
- **Deep Learning**: Building and training models with Python and TensorFlow/Keras.
- **Transfer Learning**: Utilizing pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3) for feature extraction and fine-tuning.
- **Data Preprocessing**: Implementing image rescaling and data augmentation.
- **Model Evaluation**: Analyzing accuracy, precision, recall, and F1-score.
- **Visualization**: Creating plots for training history and confusion matrices.
- **Model Deployment**: Building an interactive web application with Streamlit.

---

## 🎯 Problem Statement
The objective is to classify fish images into **one of 11 distinct species**.  
The project compares a custom CNN with multiple transfer learning models to identify the **most effective architecture** and deploy it as a real-time prediction tool.

---

## 💼 Business Objectives
- **Enhanced Accuracy**: Identify the best model for fish species classification, useful in quality control, environmental monitoring, and fisheries.
- **Deployment-Ready**: Provide real-time predictions accessible to non-technical users.
- **Informed Model Selection**: Offer a clear comparison of different architectures for practical decision-making.

---

## ⚠️ Challenges
- **Class Imbalance**: Some species have far fewer images than others, hurting per-class performance.
- **Data Drift**: Real-world variations (lighting, background, quality) could degrade performance.
- **Scalability**: Current Streamlit app is a prototype; scaling requires more robust deployment.

---

## 🛠️ Methodology

### 1. Data Preprocessing & Augmentation
- Images loaded via **TensorFlow’s ImageDataGenerator**.
- Rescaling to **[0,1]**.
- Augmentation: Rotation, shifting, and horizontal flipping to improve robustness.

### 2. Model Training
- **Custom CNN**: Baseline model with convolutional and pooling layers.
- **Transfer Learning**: Fine-tuning VGG16, ResNet50, MobileNet, InceptionV3.
- Base layers frozen initially, then partially unfrozen for fine-tuning.

### 3. Model Evaluation
- **Test Accuracy & Loss**
- **Classification Reports**: Per-class precision, recall, F1-score.
- **Training Curves**: Accuracy/loss plots to detect overfitting.

---

## 📊 Results & Findings

> **Key Insight**: High overall accuracy does not always mean good per-class performance — class imbalance was a major issue.

| Model Name  | Overall Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|-------------|-----------------------|--------------------|-----------------|-------------------|
| CustomCNN   | 98.17%                | 0.10               | 0.10            | 0.10              |
| VGG16       | 97.32%                | 0.10               | 0.10            | 0.10              |
| ResNet50    | 31.47%                | 0.10               | 0.11            | 0.09              |
| **MobileNet** | **99.68%**          | **0.09**           | **0.09**        | **0.09**          |
| InceptionV3 | 98.61%                | 0.11               | 0.11            | 0.11              |

✅ **MobileNet** was selected for deployment due to its top accuracy and efficiency.

---

## 🚀 Deployment
The best model (`MobileNet_best_model.h5`) was deployed in a **Streamlit web app** with:
- 📂 File uploader for images.
- 📌 Real-time fish species predictions.
- 📈 Confidence scores + bar chart visualization.

---

## 🖥️ How to Run Locally

### Prerequisites
- Python **3.7+**
- `pip` package manager

### Setup
```bash
# Clone repo
git clone <repository_url>

# Navigate to project directory
cd <project_directory>

# Install dependencies
pip install -r requirements.txt

# Ensure trained model is saved
mkdir trained_models
# Place your MobileNet_best_model.h5 here

# Run the app
streamlit run app.py
