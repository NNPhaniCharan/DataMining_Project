from pathlib import Path
import joblib

# Base directory is the Backend folder where this file resides
BASE_DIR = Path(__file__).resolve().parent.parent

# Central models directory used by API routers
MODELS_DIR = BASE_DIR / "models"

def load_artifact(name: str):
    """
    Load a serialized artifact (joblib) from the models directory.

    Parameters:
    - name: file name inside the models directory (e.g., 'GradientBoosting_yield_model.joblib')
    """
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


