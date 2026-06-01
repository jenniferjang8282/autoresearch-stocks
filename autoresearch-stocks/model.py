from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )),
    ])
