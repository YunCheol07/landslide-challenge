"""Only synthetic fixtures; no competition data are copied into tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scripts.audit_data import Audit


def square(size=4):
    return [[[[0, 0], [size, 0], [size, size], [0, size], [0, 0]]]]


class AuditTests(unittest.TestCase):
    def fixture(self, root, labels):
        window = root / 'ls_scene_01_w01'
        for name in ('bands', 'images', 'labels'):
            (window / name).mkdir(parents=True)
        sites = [{'site_id': sid, 'prior_polygon': square()} for sid in ['a', 'b', 'c', 'd']]
        (window / 'sites.json').write_text(json.dumps(sites))
        (window / 'labels/20260101.json').write_text(json.dumps(labels))
        arr = np.ones((256, 256, 4), dtype=np.uint16)
        arr[:128, :, 3] = 0
        np.save(window / 'bands/20260101.npy', arr)
        Image.new('RGB', (256, 256)).save(window / 'images/20260101.png')
        return window

    def test_missing_ignore_and_negative_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, [{'site_id': 'a', 'visible': 1, 'polygon': square()},
                                {'site_id': 'b', 'visible': 0},
                                {'site_id': 'c', 'visible': 0, 'ignore': True}])
            report = Audit().run(root)
            self.assertEqual(report['status'], 'pass')
            for key in ['visible_positive', 'visible_negative', 'ignore_excluded', 'missing_labels_excluded', 'shape_eligible']:
                self.assertEqual(report['counts'][key], 1)
            self.assertEqual(report['valid_pixel_fraction']['mean'], 0.5)
            self.assertNotIn(str(root), json.dumps(report))

    def test_duplicate_unknown_and_invalid_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, [{'site_id': 'a', 'visible': 0}, {'site_id': 'a', 'visible': 1},
                                {'site_id': 'unknown', 'visible': 0}, {'site_id': 'b', 'visible': '0'}])
            report = Audit().run(root)
            self.assertEqual(report['status'], 'fail')
            self.assertEqual(report['errors']['labels_duplicate_site_id'], 1)
            self.assertEqual(report['errors']['labels_unknown_site'], 1)
            self.assertEqual(report['errors']['invalid_visible'], 1)
            self.assertEqual(report['counts'].get('visible_negative', 0), 0)

    def test_missing_labels_never_become_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = self.fixture(root, [])
            (window / 'labels/20260101.json').unlink()
            report = Audit().run(root)
            self.assertEqual(report['counts']['missing_labels_excluded'], 4)
            self.assertEqual(report['counts'].get('visible_negative', 0), 0)

    def test_bad_array_and_corrupt_json_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = self.fixture(root, [])
            np.save(window / 'bands/20260101.npy', np.ones((2, 2)))
            (window / 'labels/20260101.json').write_text('{')
            report = Audit().run(root)
            self.assertEqual(report['errors']['bands_shape_or_dtype'], 1)
            self.assertEqual(report['errors']['labels_unreadable_or_invalid_json'], 1)
            self.assertEqual(report['counts']['unreadable_label_slots_excluded'], 4)

    def test_continuous_area_holes_threshold_and_invalid_topology(self):
        audit = Audit()
        coords = square()
        coords[0].append([[1, 1], [1, 3], [3, 3], [3, 1], [1, 1]])
        self.assertEqual(audit.geometry(coords, 'test'), 12)
        self.assertEqual(audit.geometry([[[[0, 0], [4, 0], [4, 4], [0, 4]]]], 'test'), 16)
        self.assertEqual(audit.geometry(square(0.5), 'test'), 0.25)
        self.assertIsNone(audit.geometry([[[[0, 0], [4, 4], [0, 4], [4, 0], [0, 0]]]], 'test'))
        self.assertIsNone(audit.geometry(square(257), 'test'))
        self.assertIsNone(audit.geometry(square(float('nan')), 'test'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, [{'site_id': 'a', 'visible': 1, 'polygon': square(3)},
                                {'site_id': 'b', 'visible': 1, 'polygon': [[[[0, 0], [5, 0], [5, 2], [0, 2], [0, 0]]]]}])
            report = Audit().run(root)
            self.assertEqual(report['counts']['shape_below_10px2'], 1)
            self.assertEqual(report['counts']['shape_eligible'], 1)


if __name__ == '__main__':
    unittest.main()
