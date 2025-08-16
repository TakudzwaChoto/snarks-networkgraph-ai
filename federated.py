import os
import json
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from audit_log import AuditLog
from deep_predictor import _build_features


def build_features_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X, y, _ = _build_features(df)
    return X.astype(float), y.astype(float)


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42) -> Tuple:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1.0 - val_ratio))
    tr, va = idx[:split], idx[split:]
    return X[tr], y[tr], X[va], y[va]


@dataclass
class FLConfig:
    n_clients: int = 5
    client_fraction: float = 1.0
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-2
    clip_norm: float = 1.0
    seed: int = 42


class FedDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_clients: int, seed: int = 42) -> None:
        self.clients: List[Tuple[np.ndarray, np.ndarray]] = []
        n = X.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        parts = np.array_split(idx, n_clients)
        for p in parts:
            self.clients.append((X[p], y[p]))


class FedAvgRegressor:
    def __init__(self, dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # weights and bias combined: last term is bias
        self.w = rng.normal(scale=0.01, size=(dim + 1,))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_bias(X)
        return Xb @ self.w

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        Xb = self._add_bias(X)
        pred = Xb @ self.w
        err = pred - y
        loss = float(np.mean(err ** 2))
        grad = (2.0 / Xb.shape[0]) * (Xb.T @ err)
        return loss, grad

    def client_update(self, X: np.ndarray, y: np.ndarray, batch_size: int, lr: float, epochs: int, clip_norm: float, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        w0 = self.w.copy()
        n = X.shape[0]
        for _ in range(epochs):
            idx = np.arange(n)
            rng.shuffle(idx)
            for i in range(0, n, batch_size):
                j = idx[i:i + batch_size]
                _, g = self.loss_and_grad(X[j], y[j])
                # clip gradient
                g_norm = float(np.linalg.norm(g) + 1e-12)
                if g_norm > clip_norm:
                    g = g * (clip_norm / g_norm)
                self.w = self.w - lr * g
        delta = self.w - w0
        # prepare summary (no raw data)
        update_hash = hashlib.sha256(delta.tobytes()).hexdigest()
        return {
            'delta': delta,
            'delta_norm': float(np.linalg.norm(delta)),
            'samples': int(n),
            'update_hash': update_hash,
        }

    def apply_delta(self, delta: np.ndarray) -> None:
        self.w = self.w + delta


class FederatedSimulator:
    def __init__(self, audit: AuditLog, train_csv: str = 'train_tradedata.csv') -> None:
        self.audit = audit
        self.train_csv = train_csv
        self.config: Optional[FLConfig] = None
        self.dataset: Optional[FedDataset] = None
        self.global_model: Optional[FedAvgRegressor] = None
        self.round: int = 0
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None

    def init(self, cfg: FLConfig) -> Dict[str, Any]:
        X, y = build_features_from_csv(self.train_csv)
        X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_ratio=0.2, seed=cfg.seed)
        self.dataset = FedDataset(X_tr, y_tr, n_clients=cfg.n_clients, seed=cfg.seed)
        self.global_model = FedAvgRegressor(dim=X.shape[1], seed=cfg.seed)
        self.round = 0
        self.config = cfg
        self.X_val, self.y_val = X_va, y_va
        evt = self.audit.append('fl.init', {
            'n_clients': cfg.n_clients,
            'client_fraction': cfg.client_fraction,
            'local_epochs': cfg.local_epochs,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'clip_norm': cfg.clip_norm,
            'n_train': int(X_tr.shape[0]),
            'n_val': int(X_va.shape[0]),
        })
        return {'status': 'ok', 'audit_head': evt['head']}

    def _metrics(self) -> Dict[str, float]:
        if self.X_val is None or self.y_val is None or self.global_model is None:
            return {}
        yhat = self.global_model.predict(self.X_val)
        mse = float(np.mean((yhat - self.y_val) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(yhat - self.y_val)))
        return {'mae': round(mae, 4), 'rmse': round(rmse, 4)}

    def run_round(self) -> Dict[str, Any]:
        assert self.dataset is not None and self.global_model is not None and self.config is not None
        cfg = self.config
        n = len(self.dataset.clients)
        m = max(1, int(math.ceil(cfg.client_fraction * n)))
        # sample first m clients (deterministic for reproducibility)
        selected = list(range(m))
        self.audit.append('fl.round.start', {'round': self.round + 1, 'selected_clients': m})
        updates: List[Tuple[np.ndarray, int, str]] = []
        for cid in selected:
            Xi, yi = self.dataset.clients[cid]
            # copy model to client
            client_model = FedAvgRegressor(dim=self.global_model.w.shape[0] - 1)
            client_model.w = self.global_model.w.copy()
            summary = client_model.client_update(Xi, yi, batch_size=cfg.batch_size, lr=cfg.lr, epochs=cfg.local_epochs, clip_norm=cfg.clip_norm, seed=cfg.seed + cid)
            self.audit.append('fl.client.update', {
                'round': self.round + 1,
                'client_id': cid,
                'samples': summary['samples'],
                'delta_norm': round(summary['delta_norm'], 6),
                'update_hash': summary['update_hash'],
            })
            updates.append((summary['delta'], summary['samples'], summary['update_hash']))
        # aggregate
        total_samples = sum(s for _, s, _ in updates)
        if total_samples == 0:
            agg_delta = np.zeros_like(self.global_model.w)
        else:
            agg = np.zeros_like(self.global_model.w)
            for delta, samples, _ in updates:
                agg += (samples / total_samples) * delta
            agg_delta = agg
        self.global_model.apply_delta(agg_delta)
        self.round += 1
        metrics = self._metrics()
        evt = self.audit.append('fl.round.end', {'round': self.round, 'metrics': metrics, 'selected_clients': m})
        return {'status': 'ok', 'round': self.round, 'metrics': metrics, 'audit_head': evt['head']}

    def status(self) -> Dict[str, Any]:
        return {
            'round': self.round,
            'config': asdict(self.config) if self.config else None,
            'metrics': self._metrics(),
            'audit_head': self.audit.head(),
        }

    def predict(self, buyer: int, seller: int) -> float:
        # Build a single-row feature as in deep predictor
        x = np.array([[buyer, seller, abs(buyer - seller), buyer % 10, seller % 10, (buyer + seller) % 20]], dtype=float)
        if self.global_model is None:
            return 0.0
        yhat = float(self.global_model.predict(x)[0])
        return round(max(0.0, yhat), 2)