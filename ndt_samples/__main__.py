"""Run with python -m ndt_samples."""
import argparse
from pathlib import Path
import pandas as pd
from .generator import generate_virtual_samples


def main():
    parser = argparse.ArgumentParser(description='Generate virtual NDT regression samples')
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'samples.tsv')
    parser.add_argument('--target', default='TC(W/mK)')
    parser.add_argument('--categorical', nargs='*', default=['Saturated'])
    parser.add_argument('--drop', nargs='*', default=['No.'])
    parser.add_argument('--n-virtual', type=int, default=100)
    parser.add_argument('--noise-ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('generated/virtual_samples.csv'))
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.input.resolve():
            raise ValueError('output must differ from input')
        if args.output.exists():
            raise ValueError('output already exists; choose another --output path')
        data = pd.read_csv(args.input, sep=None, engine='python', encoding='utf-8-sig')
        if not data.columns.is_unique:
            raise ValueError('input column names must be unique')
        if args.target not in data.columns:
            raise ValueError(f'target column not found: {args.target}')
        if args.target in args.drop or args.target in args.categorical:
            raise ValueError('target cannot be dropped or categorical')
        features = data.drop(columns=[args.target] + args.drop)
        virtual, targets = generate_virtual_samples(
            features, data[args.target], args.noise_ratio, args.n_virtual,
            args.seed, args.categorical)
        virtual[args.target] = targets
        args.output.parent.mkdir(parents=True, exist_ok=True)
        virtual.to_csv(args.output, index=False, encoding='utf-8-sig')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f'Loaded {len(data)} rows; generated {len(virtual)} rows: {args.output.resolve()}')


if __name__ == '__main__':
    main()
