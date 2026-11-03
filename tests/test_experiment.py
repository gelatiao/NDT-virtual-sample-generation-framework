import json
import unittest
import numpy as np
from ndt_samples.experiment import ROOT, load_data, partitions, run_fold, validate
from ndt_samples.preprocessing import Preprocessor


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.ids = load_data(ROOT / 'data/samples.tsv')
        self.config = json.loads((ROOT / 'configs/local.json').read_text())

    def test_grouped_partition_prevents_pair_leakage(self):
        visited = []
        for train, test in partitions(self.x, 'grouped', 5, 42):
            pairs = lambda indices: {tuple(row) for row in self.x.iloc[indices, 1:6].to_numpy()}
            self.assertFalse(pairs(train) & pairs(test))
            visited.extend(test)
        self.assertEqual(sorted(visited), list(range(66)))

    def test_scaler_uses_training_only(self):
        train = self.x.iloc[:40]
        processor = Preprocessor('svr').fit(train, self.y[:40])
        before = processor.scaler.mean_.copy()
        changed = self.x.iloc[40:].copy() * 100
        processor.transform(changed)
        np.testing.assert_array_equal(processor.scaler.mean_, before)
        np.testing.assert_allclose(processor.transform(train).mean(axis=0), 0, atol=1e-12)

    def test_progression_predictions_and_reproducibility(self):
        train, test = next(partitions(self.x, 'local', 5, 42))
        self.config['max_samples'] = 80
        a = run_fold(self.x, self.y, self.ids, train, test, 'svr', self.config, 42, 0)
        b = run_fold(self.x, self.y, self.ids, train, test, 'svr', self.config, 42, 0)
        self.assertEqual(a[:2], b[:2])
        self.assertEqual([r['n_training'] for r in a[0] if r['split']=='test'], [52, 77, 80])
        self.assertEqual({r['sample_id'] for r in a[1] if r['split']=='test'}, set(self.ids[test]))
        test_rows = [r for r in a[1] if r['split']=='test' and r['step']==2]
        error = np.array([r['prediction']-r['actual'] for r in test_rows])
        self.assertAlmostEqual(a[0][-1]['RMSE'], np.sqrt(np.mean(error**2)))

    def test_improved_rf_progression(self):
        self.config.update(generator='improved', max_samples=60)
        train, test = next(partitions(self.x, 'local', 5, 42))
        metrics, _, _ = run_fold(self.x, self.y, self.ids, train, test, 'rf', self.config, 42, 0)
        self.assertEqual(metrics[-1]['n_virtual'], 8)
        self.assertTrue(np.isfinite(metrics[-1]['RMSE']))

    def test_invalid_config(self):
        self.config['step_size'] = 0
        with self.assertRaises(ValueError):
            validate(self.config)


if __name__ == '__main__':
    unittest.main()
