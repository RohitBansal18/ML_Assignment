# 📱 Mobile Price Classification using Machine Learning

## BITS Pilani – M.Tech (Artificial Intelligence & Machine Learning)

### Course
Machine Learning

### Assignment
Assignment 2 – Machine Learning Model Development and Deployment

---

# Project Overview

This project predicts the **price range of mobile phones** using supervised Machine Learning algorithms.

The application is developed using **Python**, **Scikit-learn**, and **Streamlit**. It allows users to upload a CSV file containing mobile phone specifications and predicts the corresponding price category.

The project compares the performance of multiple Machine Learning models using several evaluation metrics and deploys the best-performing models through a Streamlit web application.

---

# Dataset

Dataset Used:

**Mobile Price Classification**

Features include:

- battery_power
- blue
- clock_speed
- dual_sim
- fc
- four_g
- int_memory
- m_dep
- mobile_wt
- n_cores
- pc
- px_height
- px_width
- ram
- sc_h
- sc_w
- talk_time
- three_g
- touch_screen
- wifi

Target Variable:

- price_range

Classes:

- 0 – Low Cost
- 1 – Medium Cost
- 2 – High Cost
- 3 – Very High Cost

---

# Machine Learning Models Implemented

The following Machine Learning algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

# Project Workflow

1. Import libraries
2. Load dataset
3. Exploratory Data Analysis (EDA)
4. Data cleaning
5. Missing value analysis
6. Duplicate record analysis
7. Feature selection
8. Train/Test Split
9. Feature Scaling
10. Model Training
11. Model Evaluation
12. Model Comparison
13. Save trained models
14. Streamlit Deployment

---

# Evaluation Metrics

The following metrics were used to evaluate the models:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Area Under Curve (AUC)

---

# Project Structure

```
ML_Assignment/
│
├── app.py
├── train_models.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── mobile_price.csv
│   └── kaggle_test_unseen.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── outputs/
│
└── images/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/RohitBansal18/ML_Assignment.git
```

Move into the project directory

```bash
cd ML_Assignment
```

Install required packages

```bash
pip install -r requirements.txt
```

---

# Run the Jupyter Notebook

```bash
jupyter notebook
```

Open

```
train_models.ipynb
```

Train all Machine Learning models.

---

# Run the Streamlit Application

```bash
streamlit run app.py
```

The application will launch in your default browser.

---

# Streamlit Features

The application provides:

- Upload CSV dataset
- Select Machine Learning model
- Predict mobile price range
- Dataset preview
- Dataset summary
- Performance comparison
- Confusion Matrix
- Classification Report
- Download prediction results

---

# Saved Models

The following trained models are stored in the **models** directory.

- logistic_regression.pkl
- decision_tree.pkl
- knn.pkl
- naive_bayes.pkl
- random_forest.pkl
- scaler.pkl

---

# Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# Results

The Machine Learning models were compared using multiple evaluation metrics.

The best-performing model can be selected from the Streamlit interface for making predictions.

---

# Future Improvements

Possible enhancements include:

- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Model explainability using SHAP
- Deployment on Streamlit Community Cloud
- Docker containerization

---

# Author

**Rohit Bansal**

M.Tech (Artificial Intelligence & Machine Learning)

BITS Pilani

---

# License

This project is developed for academic purposes as part of the BITS Pilani Machine Learning Assignment.