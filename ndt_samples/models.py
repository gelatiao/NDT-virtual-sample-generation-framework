"""Factories using paper Table 2 and local TC parameter ranges."""
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

NAMES = ('mlp', 'svr', 'rf', 'gbdt', 'catboost', 'xgboost')


def make_model(name, params, seed):
    options = dict(params)
    if name == 'mlp':
        if 'hidden_layer_sizes' in options:
            options['hidden_layer_sizes'] = tuple(options['hidden_layer_sizes'])
        return MLPRegressor(random_state=seed, **options)
    if name == 'svr':
        return SVR(**options)
    if name == 'rf':
        return RandomForestRegressor(random_state=seed, n_jobs=1, **options)
    if name == 'gbdt':
        return GradientBoostingRegressor(random_state=seed, **options)
    if name == 'catboost':
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError('Install requirements-models.txt to run CatBoost') from exc
        return CatBoostRegressor(random_seed=seed, verbose=False,
                                 allow_writing_files=False, thread_count=1, **options)
    if name == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError('Install requirements-models.txt to run XGBoost') from exc
        return XGBRegressor(random_state=seed, n_jobs=1,
                            objective='reg:squarederror', **options)
    raise ValueError(f'Unknown model: {name}')
