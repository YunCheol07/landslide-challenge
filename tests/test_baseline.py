import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.baseline import decision, extract, feature, fit, write_predictions
from src.evaluation import read_predictions

CONFIG = {'version': 'ndvi_prior_v1', 'seed': 42, 'ring_inner_px': 3.0,
          'ring_outer_px': 8.0, 'min_valid_pixels': 1,
          'positive_rule': 'inside_minus_ring_ndvi_lte_threshold', 'boundary': 'unchanged_prior'}
PRIOR = [[[[100, 100], [104, 100], [104, 104], [100, 104]]]]


class BaselineTests(unittest.TestCase):
    def test_ndvi_float_arithmetic_and_invalid_pixels(self):
        arr = np.full((256, 256, 4), 1000, dtype=np.uint16)
        inside = np.zeros((256, 256), dtype=bool)
        inside[0, 0] = True
        ring = np.zeros_like(inside)
        ring[0, 1] = True
        arr[0, 0, 3] = 500
        arr[0, 1, 3] = 3000
        self.assertAlmostEqual(feature(arr, inside, ring, 1), -1/3 - 0.5)
        arr[0, 0, 0] = 0
        self.assertIsNone(feature(arr, inside, ring, 1))

    def test_fit_ignores_missing_and_ignore_and_uses_train_only(self):
        rows = [(('w', '1', 's'), -0.5, PRIOR), (('w', '2', 's'), 0.5, PRIOR),
                (('w', '3', 's'), -100, PRIOR), (('w', '4', 's'), 100, PRIOR)]
        truth = {rows[0][0]: {'visible': 1}, rows[1][0]: {'visible': 0},
                 rows[2][0]: {'ignore': True, 'visible': 0}}
        model = fit(rows, truth, CONFIG)
        self.assertEqual(model['fit_samples'], 2)
        self.assertEqual(model['fit_macro_f1'], 1)
        self.assertTrue(decision(-0.5, model))
        self.assertFalse(decision(0.5, model))
        self.assertFalse(decision(None, model))

    def test_label_free_inference_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'input'
            window = root / 'ls_scene_01_w01'
            (window / 'bands').mkdir(parents=True)
            (window / 'sites.json').write_text(json.dumps([{'site_id': 's', 'prior_polygon': PRIOR}]))
            np.save(window / 'bands/20260101.npy', np.full((256, 256, 4), 1000, dtype=np.uint16))
            rows = extract(root, CONFIG)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], 0)
            # Even an unreadable label must never be read by inference.
            (window / 'labels').mkdir()
            (window / 'labels/20260101.json').write_text('not JSON')
            self.assertEqual(extract(root, CONFIG), rows)
            model = {'threshold': 0.1, 'fallback_positive': False}
            output = Path(tmp) / 'pred.csv'
            write_predictions(output, rows, model)
            predictions = read_predictions(output, {rows[0][0]})
            self.assertEqual(predictions[rows[0][0]].area, 16)
            with self.assertRaises(FileExistsError):
                write_predictions(output, rows, model)


if __name__ == '__main__':
    unittest.main()
