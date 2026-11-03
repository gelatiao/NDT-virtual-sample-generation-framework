"""Training-only versions of local TC outlier and RF feature processing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression


class Preprocessor:
    def __init__(self, model, normalize_oxides=False):
        self.model = model
        self.normalize_oxides = normalize_oxides

    def normalize(self, x):
        x = x.astype(float).copy()
        if self.normalize_oxides:
            columns = ['SiO2', 'Al2O3', 'K2O', 'CaO', 'Fe2O3']
            totals = x[columns].sum(axis=1)
            if (totals <= 0).any():
                raise ValueError('Oxide totals must be positive')
            x[columns] = x[columns].div(totals, axis=0) * 100
        return x

    def clean(self, x):
        x = self.normalize(x)
        if self.model == 'rf':
            z = ((x - self.mean) / self.std.replace(0, 1)).abs()
            extreme = (x <= self.q1 - 3 * self.iqr) | (x >= self.q3 + 3 * self.iqr) | (z > 4)
            mild_low = (x < self.q1 - 1.5 * self.iqr) & ~extreme
            mild_high = (x > self.q3 + 1.5 * self.iqr) & ~extreme
            return x.mask(mild_low, self.q1, axis=1).mask(mild_high, self.q3, axis=1).mask(extreme, self.median, axis=1)
        return x.clip(self.mean - 4 * self.std, self.mean + 4 * self.std, axis=1)

    def fit(self, x, y):
        normalized = self.normalize(x)
        self.mean, self.std = normalized.mean(), normalized.std(ddof=1).fillna(0)
        self.q1, self.q3 = normalized.quantile(.25), normalized.quantile(.75)
        self.iqr, self.median = self.q3 - self.q1, normalized.median()
        cleaned = self.clean(x)
        self.poly = PolynomialFeatures(2, include_bias=False) if self.model == 'rf' else None
        values = self.poly.fit_transform(cleaned) if self.poly else cleaned.to_numpy()
        self.scaler = StandardScaler().fit(values)
        values = self.scaler.transform(values)
        self.indices = np.arange(values.shape[1])
        self.selector = None
        if self.model == 'rf':
            # Local script: low-variance removal, correlated-column removal, then KBest(6).
            kept = list(np.flatnonzero(values.var(axis=0) > .01))
            corr = pd.DataFrame(values).corr().to_numpy()
            drop = {j for offset, i in enumerate(kept) for j in kept[offset + 1:] if abs(corr[i, j]) > .95}
            self.indices = np.array([i for i in kept if i not in drop], dtype=int)
            if not len(self.indices):
                raise ValueError('No RF features remain after filtering')
            self.selector = SelectKBest(f_regression, k=min(6, len(self.indices))).fit(values[:, self.indices], y)
        return self

    def transform(self, x):
        values = self.clean(x)
        values = self.poly.transform(values) if self.poly else values.to_numpy()
        values = self.scaler.transform(values)[:, self.indices]
        if self.selector is not None:
            values = self.selector.transform(values)
        return pd.DataFrame(values, columns=[f'f{i}' for i in range(values.shape[1])])
