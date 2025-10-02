# train_yield_model.py
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_models import RidgeRegressionCV

# Resolve paths relative to the repository root so the script works
# whether it's run from Backend/ or the project root.
BASE = Path(__file__).resolve().parent
# The cleaned CSV sits inside Backend/DataAndCleaning/Data/CleanedData
DATA = BASE / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"
MODEL = BASE / "models" / "yield_model.joblib"
META = BASE / "models" / "meta_yield.json"

NUM = ["N","P","K","pH","rainfall","temperature"]
CAT = ["State_Name","Season","Crop"]
TARGET = "Yield_ton_per_hec"

def main():
    df = pd.read_csv(DATA)
    X = df[NUM + CAT].copy()
    y = df[TARGET].astype(float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    model = RidgeRegressionCV(alphas=np.logspace(-3,3,25), cv=5)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    metrics = {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(((yte - pred)**2).mean())),
        "mae": float(mean_absolute_error(yte, pred)),
        "chosen_alpha": float(pipe.named_steps["model"].alpha_),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features_num": NUM,
        "features_cat": CAT,
        "target": TARGET,
        "algorithm": "Custom Ridge Regression (closed-form, L2-regularized)"
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL)
    META.write_text(json.dumps(metrics, indent=2))
    print("Saved:", MODEL, "\nMetrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
