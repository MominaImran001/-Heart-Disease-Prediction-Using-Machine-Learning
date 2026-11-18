# Heart-Disease-Prediction-ML
# Heart Disease Prediction Using Machine Learning

**A Machine Learning project that uses a Random Forest Classifier to predict heart disease based on selected patient health features.**

##  Project Overview**

This project focuses on building a Heart Disease Prediction model using Python and Machine Learning. The model analyzes selected medical attributes of patients to classify heart disease outcomes.

A Random Forest Classifier is trained on the heart disease dataset, evaluated using standard classification metrics, and saved using Pickle for future use.

##  Project Objectives**

- Develop a machine learning model for heart disease prediction.
- Identify relevant features from the dataset.
- Train and evaluate a Random Forest Classification model.
- Save the trained model for future predictions.

## Technologies & Libraries**

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Model training and evaluation
- **Pickle** – Model serialization
- **Jupyter Notebook** – Development environment

##  Dataset & Features**

The project uses a heart disease dataset (`heart.csv`).

The following six features are selected for model training:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender of the patient |
| `cp` | Chest pain type |
| `exang` | Exercise-induced angina |
| `thalach` | Maximum heart rate achieved |
| `oldpeak` | ST depression induced by exercise |

**Target Variable:** `target`

##  Project Workflow**

1. **Data Loading:** Import the heart disease dataset using Pandas.
2. **Feature Selection:** Select six relevant medical features.
3. **Target Preparation:** Transform the target labels for the intended classification output.
4. **Data Splitting:** Divide the dataset into 80% training and 20% testing data using stratified sampling.
5. **Model Training:** Train a Random Forest Classifier.
6. **Model Evaluation:** Evaluate predictions using accuracy, confusion matrix, precision, recall, and F1-score.
7. **Model Saving:** Export the trained model using Pickle.

##  Machine Learning Model**

**Algorithm:** Random Forest Classifier

The model is implemented using Scikit-learn's `RandomForestClassifier` with a fixed random state for reproducibility.

##  Model Evaluation**

The model's performance is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

##  Model Serialization**

The trained model is saved in `.pkl` format using Python's Pickle library.

**Model File:** `heart_model_fixed_selected_features11.pkl`

This allows the trained model to be loaded and reused without training it again.



##  Future Improvements**

- Perform hyperparameter tuning to improve model performance.
- Compare Random Forest with other classification algorithms.
- Implement a user-friendly prediction interface.
- Explore additional medical features for improved model evaluation.

##  Disclaimer**

This project is developed for educational and machine learning practice purposes only. It is not intended to provide medical advice or replace professional medical diagnosis.
