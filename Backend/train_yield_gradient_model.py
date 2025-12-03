# train_yield_gradient_model.py
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_models import GradientBoostingRegressorCV

# Resolve paths relative to the repository root so the script works
# whether it's run from Backend/ or the project root.
BASE = Path(__file__).resolve().parent
# The cleaned CSV sits inside Backend/DataAndCleaning/Data/CleanedData
DATA = BASE / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"
MODEL = BASE / "models" / "GradientBoosting_yield_model.joblib"
META = BASE / "models" / "meta_GradientBoosting_yield.json"

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
    
    # Custom Gradient Boosting with cross-validation
    model = GradientBoostingRegressorCV(
        n_estimators=300,          # Max number of trees
        learning_rate=0.05,        # Shrinkage factor
        max_depth=8,               # Tree depth
        min_samples_split=20,      # Min samples to split
        min_samples_leaf=10,       # Min samples in leaf
        subsample=0.75,            # Stochastic GB (75% samples per tree)
        cv=8,                      # 8-fold cross-validation
        random_state=42,
        verbose=1                  # Show progress
    )
    
    pipe = Pipeline([("pre", pre), ("model", model)])
    
    print("Training Gradient Boosting model with CV...")
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    
    # Calculate baseline metrics
    # Baseline 1: Global mean yield
    global_mean = ytr.mean()
    global_mean_pred = np.full_like(yte, global_mean)
    baseline_global = {
        "r2": float(r2_score(yte, global_mean_pred)),
        "rmse": float(np.sqrt(((yte - global_mean_pred)**2).mean())),
        "mae": float(mean_absolute_error(yte, global_mean_pred))
    }
    
    # Baseline 2: Per-state mean yield
    # Create a mapping of state to mean yield from training data
    train_df = pd.concat([Xtr, ytr], axis=1)
    train_state_means = train_df.groupby('State_Name')[TARGET].mean().to_dict()
    state_mean_pred = Xte['State_Name'].map(train_state_means)
    # For unseen states, use global mean
    state_mean_pred = state_mean_pred.fillna(global_mean).values
    baseline_state = {
        "r2": float(r2_score(yte, state_mean_pred)),
        "rmse": float(np.sqrt(((yte - state_mean_pred)**2).mean())),
        "mae": float(mean_absolute_error(yte, state_mean_pred))
    }
    
    metrics = {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(((yte - pred)**2).mean())),
        "mae": float(mean_absolute_error(yte, pred)),
        "baseline_global_mean": baseline_global,
        "baseline_per_state_mean": baseline_state,
        "best_n_estimators": int(pipe.named_steps["model"].best_n_estimators_),
        "learning_rate": float(pipe.named_steps["model"].learning_rate),
        "max_depth": int(pipe.named_steps["model"].max_depth),
        "subsample": float(pipe.named_steps["model"].subsample),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features_num": NUM,
        "features_cat": CAT,
        "target": TARGET,
        "algorithm": "Custom Gradient Boosting (from scratch, decision trees with gradient descent)"
    }

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL)
    META.write_text(json.dumps(metrics, indent=2))
    print("\n" + "="*60)
    print("Model trained successfully!")
    print("="*60)
    print(f"Saved model: {MODEL}")
    print(f"\nTest Set Performance:")
    print(f"  R² Score:  {metrics['r2']:.4f}")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"\nOptimal hyperparameters (from CV):")
    print(f"  n_estimators: {metrics['best_n_estimators']}")
    print(f"  learning_rate: {metrics['learning_rate']}")
    print(f"  max_depth: {metrics['max_depth']}")
    print("="*60)

if __name__ == "__main__":
    main()