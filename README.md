# Cats vs. Dogs Classification

This project leverages machine learning techniques to classify images of cats and dogs from the Kaggle dataset [Cats and Dogs Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

## Project Overview

The project involves:
- Dimensionality reduction using PCA to analyze variance preservation.
- Classification model development to distinguish between cat and dog images.
- Exploration of clustering and visualization techniques for deeper insights.

## Key Features

1. **Dataset**
   - Images sourced from Kaggle.
   - Data preprocessed and resized for analysis and modeling.

2. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) applied to determine the number of components required to preserve 90% variance.

3. **Clustering and Visualization**
   - t-SNE, KMeans, and Gaussian Mixture Model used for exploratory clustering.
   - Visualization techniques for cluster separation and overlap analysis.

4. **Classification**
   - Built and evaluated multiple machine learning models:
     - **Logistic Regression**
     - **Support Vector Machines (SVM)**
     - **Random Forest**
     - **Convolutional Neural Networks (CNN)** for advanced image analysis.

5. **Performance Metrics**
   - Accuracy and confusion matrix analysis for model evaluation.
   - Visualizations of results using libraries like matplotlib and seaborn.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
## Future Improvements
- Enhance CNN architecture for better accuracy.
- Experiment with data augmentation techniques.
- Integrate real-time prediction capabilities.

---

## References
- [Cats and Dogs Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- Scikit-learn and TensorFlow Documentation
