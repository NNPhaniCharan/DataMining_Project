# train_croprec_model.py
import json
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_models import KNNClassifier

# Resolve paths relative to the repository root so the script works
# whether it's run from Backend/ or the project root.
BASE = Path(__file__).resolve().parent
# The cleaned CSV sits inside Backend/DataAndCleaning/Data/CleanedData
DATA = BASE / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"
MODEL = BASE / "models" / "croprec_model.joblib"
META = BASE / "models" / "meta_croprec.json"

NUM = ["N","P","K","pH","rainfall","temperature"]
CAT = ["State_Name","Season"]
LABEL = "Crop"    # predict crop

def main():
    df = pd.read_csv(DATA)
    X = df[NUM + CAT].copy()
    y = df[LABEL].astype(str)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    clf = KNNClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    
    # Get prediction probabilities for top-k accuracy
    pred_proba = pipe.predict_proba(Xte)
    
    # Calculate main metrics
    top1_acc = float(accuracy_score(yte, pred))
    top3_acc = float(top_k_accuracy_score(yte, pred_proba, k=3, labels=pipe.named_steps["clf"].classes_))
    f1 = float(f1_score(yte, pred, average="macro"))
    
    # Calculate confusion matrix
    cm = confusion_matrix(yte, pred, labels=pipe.named_steps["clf"].classes_)
    
    # Baseline: Most frequent crop per state
    # Calculate most frequent crop per state from training data
    train_df = pd.concat([Xtr, ytr.rename('Crop_label')], axis=1)
    state_mode_crop = train_df.groupby('State_Name')['Crop_label'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
    
    # Predict using baseline
    baseline_pred = Xte['State_Name'].map(state_mode_crop)
    # For unseen states, use global most frequent crop
    global_mode = ytr.mode()[0] if len(ytr.mode()) > 0 else ytr.iloc[0]
    baseline_pred = baseline_pred.fillna(global_mode)
    
    baseline_acc = float(accuracy_score(yte, baseline_pred))
    baseline_f1 = float(f1_score(yte, baseline_pred, average="macro"))

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL)
    META.write_text(json.dumps({
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "classes": pipe.named_steps["clf"].classes_.tolist(),
        "baseline_most_frequent_per_state": {
            "top1_accuracy": baseline_acc,
            "f1_macro": baseline_f1
        },
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features_num": NUM,
        "features_cat": CAT,
        "label": LABEL,
        "algorithm": "Custom k-NN Classifier (distance-weighted)"
    }, indent=2))
    print("Saved:", MODEL)
    print(f"\nMetrics:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print(f"  Macro-F1: {f1:.4f}")
    print(f"\nBaseline (Most Frequent Crop per State):")
    print(f"  Top-1 Accuracy: {baseline_acc:.4f}")
    print(f"  Macro-F1: {baseline_f1:.4f}")

if __name__ == "__main__":
    main()
