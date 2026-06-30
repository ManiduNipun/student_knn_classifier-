# ==========================================
# Student Pass/Fail Prediction using KNN
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# STEP 1 - Load Dataset
# ==========================================

df = pd.read_csv("student_performance_dataset.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# ==========================================
# STEP 2 - Convert Pass/Fail to Numbers
# ==========================================

df["Result"] = df["Result"].map({
    "Pass": 1,
    "Fail": 0
})

# ==========================================
# STEP 3 - Features and Target
# ==========================================

X = df[[
    "Study_Time",
    "Attendance",
    "Assignment_Score",
    "Sleep_Hours"
]]

y = df["Result"]

# ==========================================
# STEP 4 - Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# STEP 5 - Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# STEP 6 - Create KNN Model
# ==========================================

knn = KNeighborsClassifier(n_neighbors=5)

# ==========================================
# STEP 7 - Train Model
# ==========================================

knn.fit(X_train, y_train)

# ==========================================
# STEP 8 - Predictions
# ==========================================

y_pred = knn.predict(X_test)

# ==========================================
# STEP 9 - Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# STEP 10 - Predict New Student
# ==========================================

print("\n" + "=" * 50)
print("NEW STUDENT PREDICTION")
print("=" * 50)

new_student = pd.DataFrame(
    [[4, 70, 55, 6]],
    columns=[
        "Study_Time",
        "Attendance",
        "Assignment_Score",
        "Sleep_Hours"
    ]
)

new_student_scaled = scaler.transform(new_student)

prediction = knn.predict(new_student_scaled)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")

# ==========================================
# STEP 11 - Find Best K Value
# ==========================================

print("\n" + "=" * 50)
print("K VALUE COMPARISON")
print("=" * 50)

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = accuracy_score(y_test, pred)

    print(f"K = {k}  --> Accuracy = {score:.2%}")

print("\nProject Completed Successfully!")