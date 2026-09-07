"""Split invariants and immutability, using synthetic metadata only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.make_split import fingerprint, freeze, select_scenes


class SplitTests(unittest.TestCase):
    def test_order_independence_disjointness_and_class_coverage(self):
        counts = {'a': (500, 900), 'b': (23, 24), 'c': (35, 3),
                  'd': (0, 11), 'e': (13, 1), 'f': (9, 2), 'g': (20, 10)}
        split = select_scenes(counts, 42, 2, 0.2)
        self.assertEqual(split, select_scenes(dict(reversed(list(counts.items()))), 42, 2, 0.2))
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train']) | set(split['validation']), set(counts))
        self.assertEqual(len(split['validation']), 2)
        self.assertIn('a', split['train'])
        self.assertEqual(split['validation'], ['b', 'c'])
        for scenes in split.values():
            self.assertGreater(sum(counts[s][0] for s in scenes), 0)
            self.assertGreater(sum(counts[s][1] for s in scenes), 0)

    def test_impossible_splits_rejected(self):
        for counts, n in [({'a': (1, 0), 'b': (0, 1)}, 1),
                          ({'a': (1, 1)}, 1), ({'a': (1, 1), 'b': (1, 1)}, 0)]:
            with self.assertRaises(ValueError):
                select_scenes(counts, 42, n, 0.2)

    def test_freeze_verifies_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            manifest = {'spec': {'seed': 42}, 'dataset_sha256': 'synthetic'}
            self.assertEqual(freeze(manifest, path), 'created')
            original = path.read_bytes()
            self.assertEqual(freeze(manifest, path), 'verified')
            changed = copy.deepcopy(manifest)
            changed['spec']['seed'] = 43
            with self.assertRaises(ValueError):
                freeze(changed, path)
            self.assertEqual(path.read_bytes(), original)

    def test_fingerprint_detects_content_and_inventory_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'synthetic' / 'labels'
            labels.mkdir(parents=True)
            path = labels / '20260101.json'
            path.write_text('[]')
            first = fingerprint(root)
            self.assertEqual(first, fingerprint(root))
            path.write_text('[{}]')
            self.assertNotEqual(first, fingerprint(root))
            path.write_text('[]')
            self.assertEqual(first, fingerprint(root))
            path.rename(labels / '20260102.json')
            self.assertNotEqual(first, fingerprint(root))


if __name__ == '__main__':
    unittest.main()
