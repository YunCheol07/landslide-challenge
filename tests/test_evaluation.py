import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation import EvaluationError, evaluate, polygon_geometry, read_predictions


def box(x=0, y=0, w=4, h=4):
    return [[[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]]]


def label(coords=None, visible=1):
    return {'visible': visible, 'polygon': box() if coords is None else coords}


def key(site='s', date='20260101', location='loc'):
    return location, date, site


class EvaluationTests(unittest.TestCase):
    def test_perfect_and_missing_prediction(self):
        truth = {key(): label(), key('n'): label(visible=0)}
        result = evaluate(truth, {key(): polygon_geometry(box())})
        self.assertEqual(result['combined_score'], 1)
        result = evaluate(truth, {})
        self.assertAlmostEqual(result['presence']['macro_f1'], 1/3)
        self.assertEqual(result['site_miou'], 0)
        self.assertAlmostEqual(result['combined_score'], 0.2)

    def test_macro_f1_uses_both_classes(self):
        truth = {key('a'): label(), key('b'): label(),
                 key('c'): label(visible=0), key('d'): label(visible=0)}
        preds = {key('a'): polygon_geometry(box()), key('c'): polygon_geometry(box())}
        r = evaluate(truth, preds)
        self.assertEqual(r['presence']['macro_f1'], 0.5)
        for name in ('tp', 'tn', 'fp', 'fn'):
            self.assertEqual(r['counts'][name], 1)

    def test_site_average_not_frame_average_and_location_identity(self):
        truth = {key('s', '1', 'a'): label(), key('s', '2', 'a'): label(),
                 key('s', '1', 'b'): label()}
        preds = {key('s', '1', 'b'): polygon_geometry(box())}
        self.assertEqual(evaluate(truth, preds)['site_miou'], 0.5)

    def test_continuous_iou_and_holes(self):
        truth = {key(): label(box(w=4, h=4))}
        r = evaluate(truth, {key(): polygon_geometry(box(x=0.5))})
        self.assertAlmostEqual(r['site_miou'], 14/18)
        coords = box()
        coords[0].append([[1, 1], [1, 3], [3, 3], [3, 1]])
        self.assertEqual(polygon_geometry(coords).area, 12)
        self.assertEqual(evaluate({key(): label(coords)}, {key(): polygon_geometry(box())})['site_miou'], 0.75)

    def test_area_threshold_and_negative_shapes_excluded(self):
        truth = {key('a'): label(box(w=5, h=2)), key('b'): label(box(w=3, h=3)),
                 key('c'): label(visible=0)}
        r = evaluate(truth, {})
        self.assertEqual(r['counts']['shape_eligible'], 1)
        self.assertEqual(r['counts']['shape_below_10px2'], 1)
        self.assertEqual(r['counts']['fn'], 2)

    def test_ignore_missing_and_no_labels(self):
        truth = {key('a'): None, key('b'): {'ignore': True}, key('c'): label(visible=0)}
        r = evaluate(truth, {key('a'): polygon_geometry(box()), key('b'): polygon_geometry(box())})
        self.assertEqual(r['counts']['tn'], 1)
        self.assertEqual(r['counts']['fp'], 0)
        self.assertEqual(r['presence']['macro_f1'], 0.5)
        self.assertIsNone(r['combined_score'])
        with self.assertRaises(EvaluationError):
            evaluate({key(): None}, {})

    def test_invalid_truth_blocks_shape_but_retains_presence(self):
        bowtie = [[[[0, 0], [4, 4], [0, 4], [4, 0]]]]
        r = evaluate({key(): label(bowtie), key('b'): label()}, {}, diagnostic=True)
        self.assertEqual(r['counts']['fn'], 2)
        self.assertEqual(r['counts']['shape_unresolved'], 1)
        self.assertIsNone(r['site_miou'])
        self.assertIsNone(r['combined_score'])
        self.assertEqual(r['diagnostic_valid_geometry_only_site_miou'], 0)

    def test_malformed_geometry_and_labels_rejected(self):
        for coords in [None, [], box(x=-1), box(w=257), box(x=float('nan'))]:
            with self.assertRaises(EvaluationError):
                polygon_geometry(coords)
        for row in [{'visible': '0'}, {'visible': True}, {'visible': 0, 'ignore': 'false'}]:
            with self.assertRaises(EvaluationError):
                evaluate({key(): row}, {})

    def test_csv_empty_duplicate_unknown_and_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pred.csv'
            def write(rows):
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['location_id', 'date', 'site_id', 'polygons'])
                    writer.writerows(rows)
            for text in ['', '[]']:
                write([(*key(), text)])
                self.assertEqual(read_predictions(path, {key()}), {key(): None})
            write([(*key(), json.dumps(box()))])
            self.assertEqual(read_predictions(path, {key()})[key()].area, 16)
            for rows in [[(*key(), ''), (*key(), '')], [(*key('unknown'), '')],
                         [(*key(), '{')], [(*key(), 'null')], [(*key(), json.dumps(box(x=-1)))]]:
                write(rows)
                with self.assertRaises(EvaluationError):
                    read_predictions(path, {key()})


if __name__ == '__main__':
    unittest.main()
