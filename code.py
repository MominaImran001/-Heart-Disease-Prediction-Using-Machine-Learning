# ========================================
#  HEART MODEL TRAINING & SAVE WITH PICKLE
# ========================================
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------
# 1️ Load dataset
# ----------------------------------------
df = pd.read_csv("Downloads/heart.csv")
print(" Dataset Loaded Successfully!\n")
print(df.head())

# ----------------------------------------
# 2️ Select features
# ----------------------------------------
selected_features = ['age', 'sex', 'cp', 'exang', 'thalach', 'oldpeak']
X = df[selected_features]

# Flip target labels to fix reversed predictions
y = df['target'].apply(lambda x: 0 if x==1 else 1)

# ----------------------------------------
# 3️ Split data
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------
# 4️ Train model
# ----------------------------------------
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train, y_train)


# 5️ Save model using pickle
# ----------------------------------------
MODEL_FILE = "heart_model_fixed_selected_features11.pkl"
with open(MODEL_FILE, "wb") as f:
    pickle.dump(heart_model, f)
print(f" Model trained and saved as '{MODEL_FILE}'")

# ----------------------------------------
# 6️ Evaluate model
# ----------------------------------------
y_pred = heart_model.predict(X_test)
print("\n Model Performance on Test Data:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

