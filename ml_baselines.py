import os
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import loguniform, randint, uniform
import joblib

# Optional dependencies
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from graph_features import get_or_build_cache, compute_pair_features


def build_features(df: pd.DataFrame, topology_xlsx: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    # Core engineered features from existing approach
    buyer = df["Buyer"].astype(int).values
    seller = df["Seller"].astype(int).values
    y = df["Sold_wec"].astype(float).values

    base = np.stack([
        buyer,
        seller,
        np.abs(buyer - seller),
        buyer % 10,
        seller % 10,
        (buyer + seller) % 20,
    ], axis=1).astype(float)

    extra = []
    meta = {"used_graph": False}
    if topology_xlsx and os.path.exists(topology_xlsx):
        try:
            _, G = get_or_build_cache(topology_xlsx)
            rows = []
            for b, s in zip(buyer, seller):
                pf = compute_pair_features(G, int(b), int(s))
                rows.append([
                    pf["distance"], pf["upstream_downstream"],
                    pf["buyer_flow_out"], pf["buyer_area"], pf["buyer_length"], pf["buyer_slope"], pf["buyer_width"], pf["buyer_depth"],
                    pf["seller_flow_out"], pf["seller_area"], pf["seller_length"], pf["seller_slope"], pf["seller_width"], pf["seller_depth"],
                ])
            extra = np.array(rows, dtype=float)
            meta["used_graph"] = True
        except Exception:
            extra = []
            meta["used_graph"] = False
    X = base if extra == [] else np.concatenate([base, extra], axis=1)
    return X, y, meta


def kfold_train_predict(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Tuple[float, float, float, np.ndarray]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr = y[train_idx]
        clone = joblib.loads(joblib.dumps(model))
        clone.fit(Xtr, ytr)
        preds[val_idx] = clone.predict(Xva)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2, preds


def train_lgbm(train_csv: str, topology_xlsx: Optional[str] = None, models_dir: str = "models", random_state: int = 42) -> Dict[str, Any]:
    X, y, meta = build_features(pd.read_csv(train_csv), topology_xlsx)
    if not LGB_AVAILABLE:
        return {"error": "lightgbm not installed"}
    model = lgb.LGBMRegressor(objective="regression", random_state=random_state, n_estimators=800, learning_rate=0.05, num_leaves=63)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    # simple randomized search
    param_dist = {
        "model__num_leaves": randint(31, 255),
        "model__learning_rate": loguniform(1e-3, 2e-1),
        "model__n_estimators": randint(300, 1500),
        "model__min_child_samples": randint(10, 200),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=20, cv=3, random_state=random_state, n_jobs=-1, verbose=0)
    search.fit(X, y)
    best = search.best_estimator_
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best, os.path.join(models_dir, "lgbm_model.pkl"))
    mae, rmse, r2, preds = kfold_train_predict(best, X, y)
    unc = float(np.std(preds - y))
    registry_path = os.path.join(models_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path):
        try:
            registry = json.load(open(registry_path, "r", encoding="utf-8"))
        except Exception:
            registry = {}
    registry["lgbm"] = {"metrics": {"mae": mae, "rmse": rmse, "r2": r2, "uncertainty": unc}, "used_graph": meta["used_graph"]}
    json.dump(registry, open(registry_path, "w", encoding="utf-8"), indent=2)
    return registry["lgbm"]


def train_xgb(train_csv: str, topology_xlsx: Optional[str] = None, models_dir: str = "models", random_state: int = 42) -> Dict[str, Any]:
    X, y, meta = build_features(pd.read_csv(train_csv), topology_xlsx)
    if not XGB_AVAILABLE:
        return {"error": "xgboost not installed"}
    model = xgb.XGBRegressor(random_state=random_state, n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    param_dist = {
        "model__n_estimators": randint(300, 1500),
        "model__learning_rate": loguniform(1e-3, 2e-1),
        "model__max_depth": randint(3, 12),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-8, 1e-1),
        "model__reg_lambda": loguniform(1e-8, 1e-1),
    }
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=20, cv=3, random_state=random_state, n_jobs=-1, verbose=0)
    search.fit(X, y)
    best = search.best_estimator_
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best, os.path.join(models_dir, "xgb_model.pkl"))
    mae, rmse, r2, preds = kfold_train_predict(best, X, y)
    unc = float(np.std(preds - y))
    registry_path = os.path.join(models_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path):
        try:
            registry = json.load(open(registry_path, "r", encoding="utf-8"))
        except Exception:
            registry = {}
    registry["xgb"] = {"metrics": {"mae": mae, "rmse": rmse, "r2": r2, "uncertainty": unc}, "used_graph": meta["used_graph"]}
    json.dump(registry, open(registry_path, "w", encoding="utf-8"), indent=2)
    return registry["xgb"]


def load_model(model_name: str, models_dir: str = "models") -> Optional[Any]:
    path = os.path.join(models_dir, f"{model_name}_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def predict(model_name: str, buyer: int, seller: int, topology_xlsx: Optional[str], models_dir: str = "models") -> Dict[str, Any]:
    model = load_model(model_name, models_dir)
    if model is None:
        return {"error": f"model {model_name} not found"}
    # build a single row of features
    base = np.array([[buyer, seller, abs(buyer - seller), buyer % 10, seller % 10, (buyer + seller) % 20]], dtype=float)
    extra = []
    if topology_xlsx and os.path.exists(topology_xlsx):
        try:
            _, G = get_or_build_cache(topology_xlsx)
            pf = compute_pair_features(G, int(buyer), int(seller))
            extra = np.array([[pf["distance"], pf["upstream_downstream"], pf["buyer_flow_out"], pf["buyer_area"], pf["buyer_length"], pf["buyer_slope"], pf["buyer_width"], pf["buyer_depth"], pf["seller_flow_out"], pf["seller_area"], pf["seller_length"], pf["seller_slope"], pf["seller_width"], pf["seller_depth"]]], dtype=float)
        except Exception:
            extra = []
    X = base if extra == [] else np.concatenate([base, extra], axis=1)
    yhat = float(model.predict(X)[0])
    return {"prediction": round(max(0.0, yhat), 2)}