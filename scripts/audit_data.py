"""Read-only training-data audit. Output contains aggregate counts, never coordinates/IDs."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from shapely.geometry import shape
from shapely.errors import ShapelyError


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.errors = Counter()
        self.warnings = Counter()
        self.scenes = set()
        self.dates = set()
        self.valid_fractions = []

    def read_records(self, path, kind):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
                raise ValueError('expected record list')
            return value
        except (OSError, ValueError, TypeError):
            self.errors[kind + '_unreadable_or_invalid_json'] += 1
            return None

    def geometry(self, coords, kind):
        try:
            # MultiPolygon coordinates, including holes. Dataset rings omit the
            # repeated endpoint; Shapely closes rings in memory without changing files.
            if not isinstance(coords, list) or not coords:
                raise ValueError('empty geometry')
            for polygon in coords:
                if not isinstance(polygon, list) or not polygon:
                    raise ValueError('empty polygon')
                for ring in polygon:
                    a = np.asarray(ring, dtype=float)
                    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
                        raise ValueError('invalid ring')
                    if not np.isfinite(a).all() or (a < 0).any() or (a > 256).any():
                        raise ValueError('invalid coordinates')
                    if not np.array_equal(a[0], a[-1]):
                        self.counts[kind + '_implicitly_closed_rings'] += 1
            geom = shape({'type': 'MultiPolygon', 'coordinates': coords})
            if geom.is_empty or not geom.is_valid or geom.area <= 0:
                raise ValueError('invalid topology')
            self.counts[kind + '_valid'] += 1
            return geom.area
        except (ValueError, TypeError, IndexError, OverflowError, ShapelyError):
            self.errors[kind + '_invalid'] += 1
            return None

    def indexed_records(self, records, kind):
        indexed = {}
        duplicates = set()
        for record in records:
            sid = record.get('site_id')
            if not isinstance(sid, str) or not sid:
                self.errors[kind + '_invalid_site_id'] += 1
                continue
            if sid in indexed:
                self.errors[kind + '_duplicate_site_id'] += 1
                duplicates.add(sid)
            indexed[sid] = record
        # Ambiguous labels must not silently become training examples.
        for sid in duplicates:
            del indexed[sid]
        return indexed

    def bands(self, path):
        try:
            arr = np.load(path, allow_pickle=False)
            if arr.shape != (256, 256, 4) or arr.dtype != np.uint16:
                self.errors['bands_shape_or_dtype'] += 1
                return
            valid = (arr[:, :, 0] > 0) & (arr[:, :, 3] > 0)
            fraction = float(valid.mean())
            self.valid_fractions.append(fraction)
            self.counts['bands_valid'] += 1
            self.counts['valid_pixels'] += int(valid.sum())
            self.counts['total_pixels'] += int(valid.size)
            if fraction == 0:
                self.warnings['bands_no_valid_pixels'] += 1
            # Values above 10000 are reported, not clipped or offset-corrected.
            self.counts['band_values_above_10000'] += int((arr > 10000).sum())
        except (OSError, ValueError, TypeError, EOFError):
            self.errors['bands_unreadable'] += 1

    def png(self, path):
        try:
            with Image.open(path) as im:
                if im.size != (256, 256) or im.format != 'PNG':
                    self.errors['image_shape_or_format'] += 1
                im.verify()
            with Image.open(path) as im:
                im.load()
            self.counts['images_readable'] += 1
        except (OSError, ValueError, SyntaxError):
            self.errors['image_unreadable'] += 1

    def window(self, window):
        self.counts['windows'] += 1
        match = re.fullmatch(r'(ls_scene_\d+)_w\d+', window.name)
        if match:
            self.scenes.add(match[1])
        else:
            self.errors['unrecognized_scene_name'] += 1
        records = self.read_records(window / 'sites.json', 'sites')
        sites = self.indexed_records(records, 'sites') if records is not None else None
        if sites is not None:
            self.counts['sites'] += len(sites)
            if not sites:
                self.errors['sites_empty'] += 1
            for site in sites.values():
                self.geometry(site.get('prior_polygon'), 'prior_polygon')
        files = {}
        for folder, suffix in [('bands', '.npy'), ('images', '.png'), ('labels', '.json')]:
            directory = window / folder
            if not directory.is_dir():
                self.errors[folder + '_directory_missing'] += 1
            files[folder] = {p.stem: p for p in directory.glob('*' + suffix) if p.is_file()}
            self.counts[folder + '_files'] += len(files[folder])
        dates = set().union(*(set(v) for v in files.values()))
        if not dates:
            self.errors['window_without_dates'] += 1
        self.dates.update(dates)
        for date in sorted(dates):
            try:
                if not re.fullmatch(r'\d{8}', date):
                    raise ValueError('date')
                datetime.strptime(date, '%Y%m%d')
            except ValueError:
                self.errors['invalid_date'] += 1
            for kind in ('bands', 'images'):
                if date not in files[kind]:
                    self.errors[kind + '_missing_for_date'] += 1
            if date in files['bands']:
                self.bands(files['bands'][date])
            if date in files['images']:
                self.png(files['images'][date])
            if sites is None:
                continue
            self.counts['site_date_slots'] += len(sites)
            if date not in files['labels']:
                self.counts['missing_labels_excluded'] += len(sites)
                self.warnings['label_file_missing'] += 1
                continue
            labels = self.read_records(files['labels'][date], 'labels')
            if labels is None:
                self.counts['unreadable_label_slots_excluded'] += len(sites)
                continue
            indexed = self.indexed_records(labels, 'labels')
            self.counts['missing_labels_excluded'] += len(set(sites) - set(indexed))
            self.errors['labels_unknown_site'] += len(set(indexed) - set(sites))
            for sid, label in indexed.items():
                if sid not in sites:
                    continue
                ignore = label.get('ignore', False)
                if type(ignore) not in (bool, int) or ignore not in (0, 1):
                    self.errors['invalid_ignore'] += 1
                    continue
                if ignore:
                    self.counts['ignore_excluded'] += 1
                    continue
                visible = label.get('visible')
                if type(visible) is not int or visible not in (0, 1):
                    self.errors['invalid_visible'] += 1
                    continue
                self.counts['visible_positive' if visible else 'visible_negative'] += 1
                if visible:
                    area = self.geometry(label.get('polygon'), 'label_polygon')
                    if area is not None:
                        self.counts['shape_eligible' if area >= 10 else 'shape_below_10px2'] += 1
                elif label.get('polygon'):
                    self.warnings['negative_with_polygon'] += 1

    def run(self, root):
        if not root.is_dir():
            raise ValueError('train_dir is not a readable directory')
        windows = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        if not windows:
            raise ValueError('no window directories found')
        for window in windows:
            self.window(window)
        self.counts['scenes'] = len(self.scenes)
        self.counts['unique_dates'] = len(self.dates)
        fractions = self.valid_fractions
        return {
            'schema_version': 1,
            'status': 'fail' if any(self.errors.values()) else 'pass',
            'counts': dict(sorted(self.counts.items())),
            'valid_pixel_fraction': {
                'min': min(fractions), 'mean': sum(fractions) / len(fractions),
                'max': max(fractions),
            } if fractions else None,
            'errors': {k: v for k, v in sorted(self.errors.items()) if v},
            'warnings': {k: v for k, v in sorted(self.warnings.items()) if v},
            'scope': 'Aggregate audit only; no split, training, geometry repair, or file writes. '
                     'Scene inferred from ls_scene_<number>_w<number>. '
                     'Label counts are not a ready-to-train manifest.',
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/local.json')
    parser.add_argument('--train-dir', type=Path, help='Override local config; never included in report')
    args = parser.parse_args()
    try:
        root = args.train_dir
        if root is None:
            config = json.loads(args.config.read_text(encoding='utf-8'))
            root = Path(config['train_dir']).expanduser()
        report = Audit().run(root)
    except (OSError, ValueError, KeyError, TypeError):
        # Exception messages can contain private paths or coordinates.
        print('Audit could not start: check config and train directory.', file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if report['status'] == 'fail' else 0


if __name__ == '__main__':
    raise SystemExit(main())
