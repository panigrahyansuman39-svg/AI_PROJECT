"""
Smart Crop Recommendation System — Model Training Script
-----------------------------------------------------------
Trains a Random Forest classifier on soil (N, P, K, pH) and climate
(temperature, humidity, rainfall) data to recommend the best-suited crop.

Run:
    python train_model.py

Outputs:
    crop_model.pkl       -> trained model
    dashboard_data.json  -> metrics + charts data used by the dashboard
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

DATA_PATH = "Cleaned_Dataset.xlsx"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# 1. Load data ----------------------------------------------------------
df = pd.read_excel(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

# 2. Train / test split (80/20, stratified so every crop is represented) -
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model ----------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Evaluate --------------------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")

print(f"Accuracy:  {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")

classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=classes)
report = classification_report(y_test, y_pred, output_dict=True)

# 5. Save the trained model ------------------------------------------------
joblib.dump(model, "crop_model.pkl", compress=3)
joblib.dump(FEATURES, "features.pkl")

# 6. Export everything the dashboard needs ---------------------------------
dashboard_data = {
    "accuracy": round(acc, 4),
    "f1_macro": round(f1, 4),
    "precision_macro": round(prec, 4),
    "recall_macro": round(rec, 4),
    "n_samples": len(df),
    "n_features": len(FEATURES),
    "n_classes": len(classes),
    "classes": classes,
    "feature_importance": dict(zip(FEATURES, model.feature_importances_.round(4).tolist())),
    "class_distribution": y.value_counts().to_dict(),
    "confusion_matrix": cm.tolist(),
    "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items() if k in classes},
    "train_size": len(X_train),
    "test_size": len(X_test),
}

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print("\nSaved crop_model.pkl, features.pkl and dashboard_data.json")
