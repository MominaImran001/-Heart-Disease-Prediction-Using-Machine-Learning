# Heart-Detection-Model
Heart Disease Prediction Using Machine Learning

#Project Overview

This project focuses on developing a Machine Learning model to predict heart disease using patient health data. It uses the Random Forest Classifier to analyze selected medical features and classify heart disease outcomes.

#Technologies & Libraries

Python

Pandas

NumPy

Scikit-learn

Pickle

Jupyter Notebook

#Features Used:

The model uses the following six features:

Age

Sex

Chest Pain Type (cp)

Exercise-Induced Angina (exang)

Maximum Heart Rate Achieved (thalach)

ST Depression (oldpeak)

Project Workflow

Load the heart disease dataset.

Select relevant features for model training.

Prepare and transform target labels.

Split the dataset into training and testing sets using stratified sampling.

Train a Random Forest Classifier.

Evaluate model performance using accuracy, confusion matrix, and classification report.

Save the trained model using Pickle for future use.

Model

Random Forest Classifier

The dataset is divided into 80% training data and 20% testing data.

Model Evaluation

The model is evaluated using:

Accuracy Score

Confusion Matrix

Precision

Recall

F1-Score

Model Export

The trained model is saved as a .pkl file using Pickle, allowing it to be reused without retraining.

Disclaimer

This project is developed for educational and machine learning practice purposes only. It is not intended to replace professional medical diagnosis.
