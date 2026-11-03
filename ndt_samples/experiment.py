"""Real predictions for each progressive step; no selection on held-out scores."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .models import NAMES, make_model
from .preprocessing import Preprocessor
from .generator import generate_virtual_samples
from . import local_generators

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['Ultrasonic velocity', 'SiO2', 'Al2O3', 'K2O', 'CaO', 'Fe2O3', 'saturated', 'Hardness']
ALIASES = {'Ultrasonic velocity (m/s)': 'Ultrasonic velocity', 'SiO2(%)': 'SiO2',
           'Al2O3(%)': 'Al2O3', 'K2O(%)': 'K2O', 'CaO(%)': 'CaO',
           'Fe2O3(%)': 'Fe2O3', 'Saturated': 'saturated', 'Hardness(HRC)': 'Hardness', 'TC(W/mK)': 'TC'}


def load_data(path):
    data = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig').rename(columns=ALIASES)
    if not data.columns.is_unique or not set(FEATURES + ['TC']).issubset(data.columns):
        raise ValueError('Input requires the eight named features and TC; see data/README.md')
    ids = data['No.'].to_numpy() if 'No.' in data else np.arange(1, len(data) + 1)
    if len(set(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique')
    numeric = data[FEATURES + ['TC']].apply(pd.to_numeric, errors='raise')
    if len(numeric) < 10 or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Need at least 10 finite numeric rows')
    if not numeric.saturated.isin([0, 1]).all():
        raise ValueError('saturated must contain only 0 or 1')
    return numeric[FEATURES], numeric.TC.to_numpy(), ids


def partitions(x, protocol, folds, seed):
    if protocol == 'local':
        train, test = train_test_split(np.arange(len(x)), test_size=.2, random_state=seed)
        yield train, test
    else:
        chemistry = x[['SiO2', 'Al2O3', 'K2O', 'CaO', 'Fe2O3']].to_numpy()
        _, groups = np.unique(chemistry, axis=0, return_inverse=True)
        if len(np.unique(groups)) < folds:
            raise ValueError('Not enough chemistry groups for requested folds')
        for train_groups, test_groups in KFold(folds, shuffle=True, random_state=seed).split(np.unique(groups)):
            train = np.flatnonzero(np.isin(groups, train_groups))
            test = np.flatnonzero(np.isin(groups, test_groups))
            assert not set(groups[train]) & set(groups[test])
            yield train, test


def run_fold(x, y, ids, train, test, name, config, seed, fold):
    pre = Preprocessor(name, config['normalize_oxides']).fit(x.iloc[train], y[train])
    tx, vx = pre.transform(x.iloc[train]), pre.transform(x.iloc[test])
    targets = y[train].copy()
    raw = x.iloc[train].reset_index(drop=True)
    augmented = tx.copy()
    records, predictions, warning_records = [], [], []
    total = len(train)
    sizes = [total] + list(range(total + config['step_size'], config['max_samples'] + 1, config['step_size']))
    if config['max_samples'] > total and sizes[-1] != config['max_samples']:
        sizes.append(config['max_samples'])
    for step, size in enumerate(sizes):
        if step:
            count = size - len(targets)
            step_seed = seed * 10000 + fold * 1000 + step
            if config['generator'] == 'local':
                # Preserve local np.random semantics without leaking state to callers.
                state = np.random.get_state()
                try:
                    np.random.seed(step_seed)
                    generated, generated_y = getattr(local_generators, name + '_samples')(
                        augmented, targets, noise_ratio=config['noise_ratio'], n_virtual=count)
                finally:
                    np.random.set_state(state)
                augmented = pd.concat([augmented, generated], ignore_index=True)
            else:
                # Improved variant generates only from real rows, before feature engineering.
                generated, generated_y = generate_virtual_samples(
                    raw, y[train], n_virtual=count, noise_ratio=config['noise_ratio'],
                    random_state=step_seed, categorical_columns=['saturated'])
                augmented = pd.concat([augmented, pre.transform(generated)], ignore_index=True)
            targets = np.r_[targets, generated_y]
        params = dict(config['models'][name])
        if name == 'mlp' and step:
            params['alpha'] *= config['mlp_augmented_alpha_multiplier']
        model = make_model(name, params, seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(augmented, targets)
        warning_records.extend({'model': name, 'seed': seed, 'fold': fold, 'step': step,
                                'category': type(w.message).__name__, 'message': str(w.message)} for w in captured)
        for split, features, truth, row_ids in [('train', tx, y[train], ids[train]), ('test', vx, y[test], ids[test])]:
            predicted = model.predict(features)
            if not np.isfinite(predicted).all():
                raise ValueError('Model returned nonfinite predictions')
            meta = dict(model=name, seed=seed, fold=fold, step=step, split=split,
                        n_real=len(train), n_virtual=size-total, n_training=size)
            records.append(dict(meta, R2=float(r2_score(truth, predicted)),
                                RMSE=float(np.sqrt(mean_squared_error(truth, predicted))),
                                MAE=float(mean_absolute_error(truth, predicted))))
            predictions.extend(dict(meta, sample_id=int(i), actual=float(a), prediction=float(b))
                               for i, a, b in zip(row_ids, truth, predicted))
    return records, predictions, warning_records


def validate(config):
    for field in ['step_size', 'max_samples', 'folds']:
        if isinstance(config[field], bool) or not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if config['folds'] < 2 or config['folds'] > 33:
        raise ValueError('folds must be between 2 and 33')
    if not config['seeds'] or any(type(s) is not int or s < 0 or s > 100000 for s in config['seeds']):
        raise ValueError('seeds must be nonempty integers between 0 and 100000')
    if not np.isfinite(config['noise_ratio']) or config['noise_ratio'] < 0:
        raise ValueError('noise_ratio must be finite and nonnegative')
    if not np.isfinite(config['mlp_augmented_alpha_multiplier']) or config['mlp_augmented_alpha_multiplier'] <= 0:
        raise ValueError('MLP alpha multiplier must be finite and positive')
    if config['protocol'] not in ['local', 'grouped'] or config['generator'] not in ['local', 'improved']:
        raise ValueError('Unknown protocol or generator')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run progressive TC regression experiments based on local scripts')
    parser.add_argument('--model', choices=[*NAMES, 'all'], default='mlp')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local.json')
    parser.add_argument('--input', type=Path, default=ROOT / 'data/samples.tsv')
    parser.add_argument('--protocol', choices=['local', 'grouped'])
    parser.add_argument('--generator', choices=['local', 'improved'])
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        for key in ['protocol', 'generator', 'max_samples']:
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        validate(config)
        x, y, ids = load_data(args.input)
        names = list(NAMES) if args.model == 'all' else [args.model]
        # Validate model dependencies and parameters before allocating an output directory.
        for name in names:
            make_model(name, config['models'][name], config['seeds'][0])
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    metadata = dict(config=config, selected_models=names, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    python=platform.python_version(), target='TC', source_config=str(args.config),
                    packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
                    note='Refactored local workflow; fixed parameters, no exact-paper-result claim. Test curves do not select a best step.')
    metadata['code_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT / 'ndt_samples').glob('*.py')}
    for package in ['xgboost', 'catboost']:
        try:
            metadata['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    metrics, predictions, messages, failures = [], [], [], []
    for name in names:
        for seed in config['seeds']:
            for fold, (train, test) in enumerate(partitions(x, config['protocol'], config['folds'], seed)):
                try:
                    m, p, w = run_fold(x, y, ids, train, test, name, config, seed, fold)
                    metrics.extend(m); predictions.extend(p); messages.extend(w)
                    print(f'{name} seed={seed} fold={fold}: {len(m)//2} stages complete', flush=True)
                except Exception:
                    failures.append(dict(model=name, seed=seed, fold=fold, traceback=traceback.format_exc()))
                # Checkpoint completed folds even when a later model fails.
                pd.DataFrame(metrics).to_csv(args.output / 'metrics.csv', index=False)
                pd.DataFrame(predictions).to_csv(args.output / 'predictions.csv', index=False)
                (args.output / 'warnings.json').write_text(json.dumps(messages, indent=2), encoding='utf-8')
                (args.output / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    summary = dict(completed_stage_count=len(metrics) // 2, failed_folds=len(failures),
                   note='All real test scores, including negative R2, are retained. No test-set best-step selection.')
    if metrics:
        frame = pd.DataFrame(metrics)
        summary['mean_test_metrics_by_step'] = frame[frame.split == 'test'].groupby(['model', 'step'])[['R2', 'RMSE', 'MAE']].mean().reset_index().to_dict('records')
        comparisons = []
        for (name, seed, fold), rows in frame[frame.split == 'test'].groupby(['model', 'seed', 'fold']):
            rows = rows.sort_values('step')
            baseline, final = rows.iloc[0], rows.iloc[-1]
            comparisons.append(dict(model=name, seed=int(seed), fold=int(fold),
                                    n_virtual=int(final.n_virtual), baseline_R2=baseline.R2,
                                    final_R2=final.R2, R2_change=final.R2-baseline.R2,
                                    baseline_RMSE=baseline.RMSE, final_RMSE=final.RMSE,
                                    RMSE_reduction_percent=(1-final.RMSE/baseline.RMSE)*100 if baseline.RMSE else None,
                                    baseline_MAE=baseline.MAE, final_MAE=final.MAE,
                                    MAE_reduction_percent=(1-final.MAE/baseline.MAE)*100 if baseline.MAE else None))
        pd.DataFrame(comparisons).to_csv(args.output / 'comparisons.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} fold(s) failed; see {args.output}/failures.json')
    print(f'Results saved to {args.output.resolve()}')
