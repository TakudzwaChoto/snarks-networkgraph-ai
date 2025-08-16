import os
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
import joblib

# Optional PyTorch backend. If unavailable, we will fallback to sklearn MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    optim = None  # type: ignore


def _build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build numeric features and derive max segment id.

    Expected columns in df: 'Buyer', 'Seller', 'Sold_wec'
    """
    if not {"Buyer", "Seller", "Sold_wec"}.issubset(df.columns):
        raise ValueError("train_tradedata.csv must contain columns: Buyer, Seller, Sold_wec")

    buyer = df["Buyer"].astype(int).values
    seller = df["Seller"].astype(int).values
    target = df["Sold_wec"].astype(float).values

    # Hand-crafted numeric features
    numeric_features = np.stack([
        buyer,
        seller,
        np.abs(buyer - seller),
        buyer % 10,
        seller % 10,
        (buyer + seller) % 20,
    ], axis=1).astype(float)

    max_segment_id = int(max(buyer.max(initial=0), seller.max(initial=0)))
    return numeric_features, target, max_segment_id


if TORCH_AVAILABLE:
    class _TorchRegressor(nn.Module):
        def __init__(self, num_segments: int, numeric_dim: int):
            super().__init__()
            embed_dim = 16
            hidden = 64

            self.buyer_emb = nn.Embedding(num_embeddings=num_segments + 1, embedding_dim=embed_dim)
            self.seller_emb = nn.Embedding(num_embeddings=num_segments + 1, embedding_dim=embed_dim)

            self.numeric_bn = nn.BatchNorm1d(numeric_dim)

            self.backbone = nn.Sequential(
                nn.Linear(embed_dim * 2 + numeric_dim, hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, 1),
            )

        def forward(self, buyer_ids: torch.Tensor, seller_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
            be = self.buyer_emb(buyer_ids)
            se = self.seller_emb(seller_ids)
            if numeric.ndim == 1:
                numeric = numeric.unsqueeze(1)
            numeric = self.numeric_bn(numeric)
            x = torch.cat([be, se, numeric], dim=1)
            out = self.backbone(x)
            return out.squeeze(1)
else:
    _TorchRegressor = None  # type: ignore


class DeepPredictor:
    """Deep learning predictor with optional PyTorch backend and sklearn fallback.

    Artifacts:
      - models/deep_model.pt (torch) or models/mlp_sklearn.pkl
      - models/scaler.pkl
      - models/meta.json
    """

    FEATURE_NAMES = [
        "buyer",
        "seller",
        "abs_distance",
        "buyer_mod10",
        "seller_mod10",
        "sum_mod20",
    ]

    def __init__(self, train_csv: str = "train_tradedata.csv", models_dir: str = "models") -> None:
        self.train_csv = train_csv
        self.models_dir = models_dir
        self.scaler: Optional[StandardScaler] = None
        self.max_segment_id: int = 0

        # sklearn fallback
        self.sklearn_mlp: Optional[MLPRegressor] = None

        # torch backend
        self.net: Optional["_TorchRegressor"] = None
        self.device = None

        os.makedirs(self.models_dir, exist_ok=True)

        if not self._load():
            self.train()
            self._load()

    # ---------------------- Persistence ----------------------
    def _artifact_paths(self) -> Dict[str, str]:
        return {
            "torch_model": os.path.join(self.models_dir, "deep_model.pt"),
            "sk_mlp": os.path.join(self.models_dir, "mlp_sklearn.pkl"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "meta": os.path.join(self.models_dir, "meta.json"),
        }

    def _load(self) -> bool:
        paths = self._artifact_paths()
        try:
            # Load meta and scaler first
            if os.path.exists(paths["meta"]) and os.path.exists(paths["scaler"]):
                with open(paths["meta"], "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.max_segment_id = int(meta.get("max_segment_id", 0))
                self.scaler = joblib.load(paths["scaler"])
            else:
                return False

            # Prefer torch model if available and torch present
            if TORCH_AVAILABLE and os.path.exists(paths["torch_model"]):
                numeric_dim = int(self.scaler.mean_.shape[0])
                self.net = _TorchRegressor(self.max_segment_id, numeric_dim)
                self.device = torch.device("cpu")
                self.net.load_state_dict(torch.load(paths["torch_model"], map_location=self.device))
                self.net.eval()
                return True

            # Fallback to sklearn MLP
            if os.path.exists(paths["sk_mlp"]):
                self.sklearn_mlp = joblib.load(paths["sk_mlp"])
                return True

            return False
        except Exception:
            return False

    def _save(self) -> None:
        paths = self._artifact_paths()
        meta = {"max_segment_id": self.max_segment_id}
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f)
        if self.scaler is not None:
            joblib.dump(self.scaler, paths["scaler"])
        if TORCH_AVAILABLE and self.net is not None:
            torch.save(self.net.state_dict(), paths["torch_model"])
        elif self.sklearn_mlp is not None:
            joblib.dump(self.sklearn_mlp, paths["sk_mlp"])

    # ---------------------- Training ----------------------
    def train(self, random_state: int = 42, max_epochs: int = 300) -> Dict[str, Any]:
        df = pd.read_csv(self.train_csv)
        X_numeric, y, max_seg = _build_features(df)
        self.max_segment_id = int(max_seg)

        # Standardize numeric features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_numeric)

        # Train/val split
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        idx_train, idx_val = indices[:split], indices[split:]

        # Pack arrays
        X_tr, X_va = X_scaled[idx_train], X_scaled[idx_val]
        y_tr, y_va = y[idx_train], y[idx_val]
        buyer = df["Buyer"].astype(int).values
        seller = df["Seller"].astype(int).values
        b_tr, b_va = buyer[idx_train], buyer[idx_val]
        s_tr, s_va = seller[idx_train], seller[idx_val]

        history: Dict[str, Any] = {}

        if TORCH_AVAILABLE:
            # Torch training
            self.device = torch.device("cpu")
            self.net = _TorchRegressor(self.max_segment_id, X_tr.shape[1]).to(self.device)
            optimizer = optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            criterion = nn.MSELoss()

            # Convert to tensors
            def to_tensor(arr):
                return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            def to_long(arr):
                return torch.as_tensor(arr, dtype=torch.long, device=self.device)

            Xtr_t, Xva_t = to_tensor(X_tr), to_tensor(X_va)
            ytr_t, yva_t = to_tensor(y_tr), to_tensor(y_va)
            btr_t, bva_t = to_long(b_tr), to_long(b_va)
            str_t, sva_t = to_long(s_tr), to_long(s_va)

            best_val = float("inf")
            best_state = None
            patience = 30
            bad = 0

            for epoch in range(max_epochs):
                self.net.train()
                optimizer.zero_grad(set_to_none=True)
                pred = self.net(btr_t, str_t, Xtr_t)
                loss = criterion(pred, ytr_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(bva_t, sva_t, Xva_t)
                    val_loss = criterion(val_pred, yva_t).item()

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if bad >= patience:
                    break

            if best_state is not None:
                self.net.load_state_dict(best_state)

            self._save()

            # Compute metrics
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(bva_t, sva_t, Xva_t).cpu().numpy()
            metrics = self._compute_metrics(y_va, val_pred)
            history.update(metrics)
            return history

        # sklearn fallback
        self.sklearn_mlp = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.2,
        )
        X_tr_full = np.concatenate([X_tr], axis=1)
        self.sklearn_mlp.fit(X_tr_full, y_tr)
        self._save()
        val_pred = self.sklearn_mlp.predict(X_va)
        metrics = self._compute_metrics(y_va, val_pred)
        history.update(metrics)
        return history

    # ---------------------- Prediction ----------------------
    def predict(self, buyer_segment: int, seller_segment: int) -> float:
        buyer_segment = int(buyer_segment)
        seller_segment = int(seller_segment)
        # Build single-row features
        numeric = np.array([
            buyer_segment,
            seller_segment,
            abs(buyer_segment - seller_segment),
            buyer_segment % 10,
            seller_segment % 10,
            (buyer_segment + seller_segment) % 20,
        ], dtype=float).reshape(1, -1)
        if self.scaler is None:
            raise RuntimeError("Model not initialized")
        X = self.scaler.transform(numeric)

        if TORCH_AVAILABLE and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                b = torch.as_tensor([buyer_segment], dtype=torch.long)
                s = torch.as_tensor([seller_segment], dtype=torch.long)
                x = torch.as_tensor(X, dtype=torch.float32)
                yhat = self.net(b, s, x).cpu().numpy()[0]
            return max(0.0, float(np.round(yhat, 2)))

        if self.sklearn_mlp is not None:
            yhat = self.sklearn_mlp.predict(X)[0]
            return max(0.0, float(np.round(yhat, 2)))

        return 0.0

    def _predict_batch(self, X_scaled: np.ndarray, buyers: np.ndarray, sellers: np.ndarray) -> np.ndarray:
        if TORCH_AVAILABLE and self.net is not None:
            self.net.eval()
            with torch.no_grad():
                b = torch.as_tensor(buyers.astype(int), dtype=torch.long)
                s = torch.as_tensor(sellers.astype(int), dtype=torch.long)
                x = torch.as_tensor(X_scaled, dtype=torch.float32)
                yhat = self.net(b, s, x).cpu().numpy()
            return yhat
        elif self.sklearn_mlp is not None:
            return self.sklearn_mlp.predict(X_scaled)
        else:
            raise RuntimeError("Model not available")

    # ---------------------- Evaluation ----------------------
    def evaluate(self) -> Dict[str, Any]:
        df = pd.read_csv(self.train_csv)
        X_numeric, y, _ = _build_features(df)
        X = self.scaler.transform(X_numeric) if self.scaler is not None else X_numeric

        yhat = self._predict_batch(X, df["Buyer"].values, df["Seller"].values)
        return self._compute_metrics(y, yhat)

    def feature_importance(self, n_repeats: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """Permutation importance style analysis on validation split.
        Returns importance as increase in RMSE when a feature is permuted.
        """
        rng = np.random.default_rng(random_state)
        df = pd.read_csv(self.train_csv)
        X_numeric, y, _ = _build_features(df)
        X = self.scaler.transform(X_numeric) if self.scaler is not None else X_numeric
        buyers = df["Buyer"].values
        sellers = df["Seller"].values

        baseline_pred = self._predict_batch(X, buyers, sellers)
        baseline_rmse = float(np.sqrt(mean_squared_error(y, baseline_pred)))

        importances = []
        for j, name in enumerate(self.FEATURE_NAMES):
            rmses = []
            for _ in range(n_repeats):
                Xp = X.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, j] = Xp[perm, j]
                pred = self._predict_batch(Xp, buyers, sellers)
                rmse = float(np.sqrt(mean_squared_error(y, pred)))
                rmses.append(rmse)
            avg_rmse = float(np.mean(rmses))
            delta = avg_rmse - baseline_rmse
            importances.append({
                "feature": name,
                "delta_rmse": round(delta, 6),
                "baseline_rmse": round(baseline_rmse, 6),
                "permuted_rmse": round(avg_rmse, 6),
            })

        importances.sort(key=lambda d: d["delta_rmse"], reverse=True)
        return {"importances": importances, "baseline_rmse": round(baseline_rmse, 6)}

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        # Convert to a simple binary classification task: whether trade > 0
        y_true_cls = (y_true > 0.0).astype(int)
        y_pred_cls = (y_pred > 0.0).astype(int)
        acc = float(accuracy_score(y_true_cls, y_pred_cls))
        prec = float(precision_score(y_true_cls, y_pred_cls, zero_division=0))
        rec = float(recall_score(y_true_cls, y_pred_cls, zero_division=0))
        f1 = float(f1_score(y_true_cls, y_pred_cls, zero_division=0))

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }