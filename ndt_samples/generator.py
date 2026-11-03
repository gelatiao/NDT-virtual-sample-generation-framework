"""Grouped interpolation with correlated continuous noise."""
import numpy as np
import pandas as pd


def generate_virtual_samples(X, y, noise_ratio=0.01, n_virtual=100,
                             random_state=None, categorical_columns=()):
    """Return (features, targets); DataFrame input retains columns.

    categorical_columns contains column names for DataFrames, indices for arrays.
    Samples are interpolated only within the same observed category combination.
    Targets must represent one continuous regression outcome.
    """
    if isinstance(n_virtual, bool) or not isinstance(n_virtual, (int, np.integer)) or n_virtual < 0:
        raise ValueError('n_virtual must be a nonnegative integer')
    if not np.isscalar(noise_ratio) or not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError('noise_ratio must be finite and nonnegative')
    is_frame = isinstance(X, pd.DataFrame)
    if is_frame and not X.columns.is_unique:
        raise ValueError('feature column names must be unique')
    a = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError('X must be a nonempty 2-D numeric table')
    if target.ndim != 1 or len(target) != len(a):
        raise ValueError('y must have one target per input row')
    if not np.isfinite(a).all() or not np.isfinite(target).all():
        raise ValueError('X and y must contain only finite values')
    cats = []
    for column in categorical_columns:
        if is_frame:
            if column not in X.columns:
                raise ValueError(f'unknown categorical column: {column}')
            index = X.columns.get_loc(column)
        else:
            if isinstance(column, bool) or not isinstance(column, (int, np.integer)) or not 0 <= column < a.shape[1]:
                raise ValueError('array categorical columns must be valid integer indices')
            index = int(column)
        if index not in cats:
            cats.append(index)
    continuous = [i for i in range(a.shape[1]) if i not in cats]
    rng = np.random.default_rng(random_state)
    result = np.empty((n_virtual, a.shape[1]), dtype=float)
    targets = np.empty(n_virtual, dtype=float)
    if cats:
        _, group_ids = np.unique(a[:, cats], axis=0, return_inverse=True)
    else:
        group_ids = np.zeros(len(a), dtype=int)
    # Select groups in proportion to their observed row counts.
    chosen_groups = group_ids[rng.integers(len(a), size=n_virtual)]
    for group in np.unique(chosen_groups):
        positions = np.flatnonzero(chosen_groups == group)
        rows = np.flatnonzero(group_ids == group)
        features, labels = a[rows], target[rows]
        values = features[:, continuous]
        std = values.std(axis=0)
        scale = np.where(std > 0, std, 1.0)
        normalized = (values - values.mean(axis=0)) / scale
        delta = normalized[:, None, :] - normalized[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        density = np.sum(1.0 / (1.0 + distances), axis=1)
        base_weights = 1.0 / density
        base_weights /= base_weights.sum()
        # Construct a PSD noise transform without scalar-correlation or constant-column issues.
        covariance = normalized.T @ normalized / len(rows)
        if continuous:
            eigvals, eigvecs = np.linalg.eigh(covariance)
            transform = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
        for out in positions:
            base = rng.choice(len(rows), p=base_weights)
            candidates = np.flatnonzero(np.arange(len(rows)) != base)
            if len(candidates):
                lo, hi = np.percentile(distances[base, candidates], [30, 70])
                middle = candidates[(distances[base, candidates] >= lo) &
                                    (distances[base, candidates] <= hi)]
                pool = middle if len(middle) >= 2 else candidates
                others = rng.choice(pool, size=min(2, len(pool)), replace=False)
                indices = np.r_[base, others]
            else:
                indices = np.array([base])
            weights = rng.dirichlet(np.full(len(indices), 2.0))
            result[out] = features[base]
            if continuous:
                interpolated = weights @ values[indices]
                noise = (transform @ rng.normal(size=len(continuous))) * std * noise_ratio
                result[out, continuous] = np.clip(interpolated + noise,
                                                   values.min(axis=0), values.max(axis=0))
            targets[out] = np.clip(weights @ labels[indices] +
                                   rng.normal() * labels.std() * noise_ratio,
                                   labels.min(), labels.max())
    if is_frame:
        result = pd.DataFrame(result, columns=X.columns)
        for index in cats:
            column = X.columns[index]
            result[column] = result[column].astype(X[column].dtype)
    return result, targets
