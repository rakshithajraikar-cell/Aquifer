"""Forecasting models compared in the project.

Every model exposes the same two methods:
    fit(train, val=None)      learn from a pandas Series (monthly, MS index)
    forecast(history, steps)  return `steps` future values after `history`
so one evaluation routine (evaluate.py) treats all of them identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from xgboost import XGBRegressor

from . import config
from .features import FEATURE_COLUMNS, make_features

torch.set_num_threads(2)


def _monthly(s: pd.Series) -> pd.Series:
    return s.asfreq("MS")


class BaseForecaster:
    name = "base"
    family = "base"

    def fit(self, train: pd.Series, val: pd.Series | None = None):
        raise NotImplementedError

    def forecast(self, history: pd.Series, steps: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- baselines
class NaiveForecaster(BaseForecaster):
    name = "Naive (last observed value)"
    family = "baseline"

    def fit(self, train, val=None):
        return self

    def forecast(self, history, steps):
        return np.full(steps, float(history.iloc[-1]))


class SeasonalNaiveForecaster(BaseForecaster):
    name = "Seasonal naive (same month last year)"
    family = "baseline"

    def fit(self, train, val=None):
        return self

    def forecast(self, history, steps):
        last_year = history.values[-12:]
        return np.array([last_year[h % 12] for h in range(steps)], dtype=float)


# ---------------------------------------------------------- statistical
class _StatsForecaster(BaseForecaster):
    family = "statistical"
    res = None

    def forecast(self, history, steps):
        # Keep the fitted parameters, re-run the Kalman filter over the history
        # available at the forecast origin, then forecast from its end.
        updated = self.res.apply(_monthly(history), refit=False)
        return np.asarray(updated.forecast(steps), dtype=float)


class ArimaForecaster(_StatsForecaster):
    """Non-seasonal ARIMA(p,1,q). Order chosen by AIC on the training data only."""
    name = "ARIMA (AIC-selected order)"

    def fit(self, train, val=None):
        y = _monthly(train)
        best = None
        for p in range(3):
            for q in range(3):
                try:
                    fitted = SARIMAX(y, order=(p, 1, q), enforce_stationarity=False,
                                     enforce_invertibility=False).fit(disp=False)
                except Exception:
                    continue
                if best is None or fitted.aic < best[0]:
                    best = (fitted.aic, (p, 1, q), fitted)
        self.order = best[1]
        self.res = best[2]
        return self


class SarimaForecaster(_StatsForecaster):
    """Seasonal ARIMA(1,1,1)(1,0,1)[12]; the same specification as the original project."""
    name = "SARIMA (1,1,1)(1,0,1)[12]"

    def fit(self, train, val=None):
        self.res = SARIMAX(_monthly(train), order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return self


# ------------------------------------------------------- tree ensembles
class TreeForecaster(BaseForecaster):
    """Random Forest / XGBoost on lag features, forecasting recursively."""
    family = "machine learning"

    def __init__(self, name: str, estimator, feature_columns=None):
        self.name = name
        self.estimator = estimator
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)

    def fit(self, train, val=None):
        X = make_features(_monthly(train))[self.feature_columns]
        mask = X.notna().all(axis=1).to_numpy()
        self.model_ = clone(self.estimator).fit(X.to_numpy()[mask], train.to_numpy()[mask])
        return self

    def _predict_next(self, s: pd.Series) -> float:
        X = make_features(s)[self.feature_columns].iloc[[-1]].to_numpy()
        return float(self.model_.predict(X)[0])

    def forecast(self, history, steps):
        s = _monthly(history)
        out = []
        for _ in range(steps):
            nxt = s.index[-1] + pd.offsets.MonthBegin(1)
            placeholder = pd.concat([s, pd.Series([np.nan], index=[nxt])])
            value = self._predict_next(placeholder)
            out.append(value)
            s = pd.concat([s, pd.Series([value], index=[nxt])])
        return np.array(out)


def make_random_forest(max_depth=None, min_samples_leaf=3):
    return RandomForestRegressor(n_estimators=300, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=config.SEED, n_jobs=2)


def make_xgboost(max_depth=3, learning_rate=0.03, n_estimators=300):
    return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=learning_rate, subsample=0.8,
                        objective="reg:squarederror", random_state=config.SEED, n_jobs=2)


# ------------------------------------------------------- deep learning
class _RecurrentNet(nn.Module):
    def __init__(self, kind: str, hidden: int):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cell(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class RecurrentForecaster(BaseForecaster):
    """LSTM or GRU on the last LOOKBACK months (scaled with training-set statistics)."""
    family = "deep learning"

    def __init__(self, kind: str, hidden: int = 32, lookback: int = config.LOOKBACK,
                 epochs: int = 300, patience: int = 25, lr: float = 1e-3):
        self.kind = kind
        self.name = kind.upper()
        self.hidden, self.lookback = hidden, lookback
        self.epochs, self.patience, self.lr = epochs, patience, lr
        self.best_epochs_ = None

    def _windows(self, scaled: np.ndarray):
        L = self.lookback
        X = np.stack([scaled[i - L:i] for i in range(L, len(scaled))])[..., None]
        y = scaled[L:]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def fit(self, train, val=None):
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        self.mu = float(train.mean())
        self.sd = float(train.std(ddof=0)) or 1.0
        tr = (train.to_numpy() - self.mu) / self.sd
        Xtr, ytr = self._windows(tr)

        Xv = yv = None
        if val is not None:
            joined = np.concatenate([tr[-self.lookback:], (val.to_numpy() - self.mu) / self.sd])
            Xv, yv = self._windows(joined)

        self.net_ = _RecurrentNet(self.kind, self.hidden)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
        max_epochs = self.epochs if val is not None or self.best_epochs_ is None else self.best_epochs_
        gen = torch.Generator().manual_seed(config.SEED)
        for epoch in range(max_epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]
                opt.zero_grad()
                loss_fn(self.net_(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            if Xv is not None:
                self.net_.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self.net_(Xv), yv))
                if vloss < best_loss - 1e-6:
                    best_loss, best_epoch, wait = vloss, epoch + 1, 0
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if Xv is not None:
            self.net_.load_state_dict(best_state)
            self.best_epochs_ = best_epoch
        self.net_.eval()
        return self

    def forecast(self, history, steps):
        scaled = (history.to_numpy() - self.mu) / self.sd
        window = list(scaled[-self.lookback:])
        out = []
        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(window[-self.lookback:], dtype=torch.float32).reshape(1, -1, 1)
                p = float(self.net_(x)[0])
                window.append(p)
                out.append(p * self.sd + self.mu)
        return np.array(out)
