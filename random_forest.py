"""
Random Forest Classifier - Complete Implementation
For Version Control System Practical Assignment
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

print("="*60)
print("RANDOM FOREST CLASSIFIER - MODEL EVALUATION")
print("="*60)

# Step 1: Load dataset
print("\n[1] Loading dataset...")
data = load_breast_cancer()
X = data.data
y = data.target
print(f"Dataset: Breast Cancer Wisconsin")
print(f"Total samples: {len(X)}")
print(f"Features: {len(data.feature_names)}")
print(f"Classes: {data.target_names}")

# Step 2: Split data
print("\n[2] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Step 3: Create and train Random Forest
print("\n[3] Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
rf_model.fit(X_train, y_train)
print("Training complete!")

# Step 4: Make predictions
print("\n[4] Making predictions on test data...")
y_pred = rf_model.predict(X_test)

# Step 5: Calculate metrics
print("\n[5] Calculating performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Step 6: Display results
print("\n" + "="*60)
print("RANDOM FOREST - PERFORMANCE METRICS")
print("="*60)
print(f"\nACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"PRECISION: {precision:.4f}")
print(f"RECALL:    {recall:.4f}")
print(f"F1-SCORE:  {f1:.4f}")

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)
print("            Predicted")
print("           Benign  Malignant")
print(f"Actual")
print(f"  Benign    {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"  Malignant {cm[1,0]:5d}    {cm[1,1]:5d}")

print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(f"Algorithm: Random Forest Classifier")
print(f"Number of Trees: 100")
print(f"Max Depth: 10")
print(f"Dataset: Breast Cancer Wisconsin")
print(f"Test Size: 30%")
print("="*60)