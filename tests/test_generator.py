import unittest
import numpy as np
import pandas as pd
from ndt_samples import generate_virtual_samples as generate


class GeneratorTests(unittest.TestCase):
    def test_single_feature_and_column_target(self):
        x, y = generate(np.arange(4.).reshape(-1, 1), np.arange(4.).reshape(-1, 1), random_state=1)
        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(y.shape, (100,))
        self.assertTrue(np.isfinite(x).all())

    def test_single_row_and_constant(self):
        x, y = generate(np.array([[2., 3.]]), [4.], n_virtual=5)
        np.testing.assert_array_equal(x, np.tile([2., 3.], (5, 1)))
        np.testing.assert_array_equal(y, np.full(5, 4.))

    def test_empty_output(self):
        for x in [np.eye(3), pd.DataFrame(np.eye(3))]:
            a, b = generate(x, [1, 2, 3], n_virtual=0)
            self.assertEqual(a.shape, (0, 3))
            self.assertEqual(b.shape, (0,))

    def test_categories_and_group_bounds(self):
        x = pd.DataFrame({'value': [0., 1., 100., 101.], 'state': [0, 0, 1, 1]})
        a, y = generate(x, [0., 1., 100., 101.], n_virtual=200,
                        categorical_columns=['state'], random_state=3)
        self.assertEqual(set(a.state), {0, 1})
        for state, lo in [(0, 0), (1, 100)]:
            mask = a.state == state
            self.assertTrue(a.loc[mask, 'value'].between(lo, lo + 1).all())
            self.assertTrue(((y[mask] >= lo) & (y[mask] <= lo + 1)).all())

    def test_reproducible(self):
        args = (np.eye(3), [1., 2., 3.])
        a, b = generate(*args, random_state=42)
        c, d = generate(*args, random_state=42)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)

    def test_all_categorical(self):
        a, _ = generate(np.array([[0, 1], [1, 0]]), [1., 2.], categorical_columns=[0, 1])
        self.assertTrue(all(tuple(row) in {(0., 1.), (1., 0.)} for row in a))

    def test_invalid_inputs(self):
        for x, y, options in [([], [], {}), ([[1]], [1, 2], {}),
                               ([[float('nan')]], [1], {}),
                               ([[1]], [1], {'n_virtual': -1}),
                               ([[1]], [1], {'noise_ratio': -1}),
                               ([[1]], [1], {'categorical_columns': [2]})]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate(x, y, **options)


if __name__ == '__main__':
    unittest.main()
