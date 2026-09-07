"""Read-only baseline error analysis; stdout contains aggregates only."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from shapely import contains_xy
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_split import PROJECT, MANIFEST, build_manifest, fingerprint
from scripts.evaluate import load_truth
from src.baseline import feature, decision
from src.evaluation import polygon_geometry, EvaluationError, read_predictions


def stats(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'median': float(np.median(values)),
            'min': float(min(values)), 'max': float(max(values))} if values else {'n': 0}


def collect(root, scenes, model):
    truth = load_truth(root, scenes)
    yy, xx = np.mgrid[0:256, 0:256].astype(float)
    xx += 0.5
    yy += 0.5
    rows = []
    config = model['config']
    for window in sorted(root.iterdir()):
        if not window.is_dir() or window.name.rsplit('_w', 1)[0] not in scenes:
            continue
        sites = json.loads((window / 'sites.json').read_text())
        geoms = [polygon_geometry(s['prior_polygon']) for s in sites]
        occupied = unary_union(geoms)
        masks = []
        for geom in geoms:
            ring = geom.buffer(config['ring_outer_px']).difference(geom.buffer(config['ring_inner_px'])).difference(occupied)
            masks.append((contains_xy(geom, xx, yy), contains_xy(ring, xx, yy)))
        for path in sorted((window / 'bands').glob('*.npy')):
            arr = np.load(path, allow_pickle=False)
            valid = (arr[:, :, 0] > 0) & (arr[:, :, 3] > 0)
            red, nir = arr[:, :, 0].astype(float), arr[:, :, 3].astype(float)
            ndvi = np.divide(nir-red, nir+red, out=np.zeros_like(red), where=valid)
            brightness = arr[:, :, :3].astype(float).mean(axis=2) / 10000
            for site, geom, (inside, ring) in zip(sites, geoms, masks):
                key = (window.name, path.stem, site['site_id'])
                label = truth.get(key)
                if label is None or label.get('ignore', False):
                    continue
                value = feature(arr, inside, ring, config['min_valid_pixels'])
                present = decision(value, model)
                target = label['visible']
                outcome = ('tp' if present else 'fn') if target else ('fp' if present else 'tn')
                a, b = inside & valid, ring & valid
                row = {'key': key, 'target': target, 'predicted': present, 'outcome': outcome,
                       'score': value, 'margin': None if value is None else value-model['threshold'],
                       'inside_ndvi': float(ndvi[a].mean()) if a.any() else None,
                       'ring_ndvi': float(ndvi[b].mean()) if b.any() else None,
                       'inside_brightness': float(brightness[a].mean()) if a.any() else None,
                       'inside_pixels': int(a.sum()), 'ring_pixels': int(b.sum()),
                       'inside_valid_fraction': float(a.sum()/inside.sum()) if inside.any() else None,
                       'prior_area': geom.area, 'gt_area': None, 'prior_iou': None,
                       'gt_coverage_by_prior': None, 'prior_coverage_by_gt': None}
                if target:
                    try:
                        gt = polygon_geometry(label.get('polygon'))
                        overlap = gt.intersection(geom).area
                        row.update(gt_area=gt.area, prior_iou=overlap/gt.union(geom).area,
                                   gt_coverage_by_prior=overlap/gt.area,
                                   prior_coverage_by_gt=overlap/geom.area)
                    except EvaluationError:
                        pass
                rows.append(row)
    return rows


def summarize(rows):
    by_site = defaultdict(list)
    for row in rows:
        by_site[(row['key'][0], row['key'][2])].append(row)
    outcome_counts = Counter(r['outcome'] for r in rows)
    fields = ['score', 'margin', 'inside_ndvi', 'ring_ndvi', 'inside_brightness',
              'inside_pixels', 'ring_pixels', 'inside_valid_fraction', 'prior_area',
              'gt_area', 'prior_iou', 'gt_coverage_by_prior', 'prior_coverage_by_gt']
    groups = {outcome: {field: stats([r[field] for r in rows if r['outcome']==outcome])
                        for field in fields} for outcome in ['tp','fn','fp','tn']}
    concentration = {}
    for outcome in ['fn', 'fp']:
        counts = sorted([sum(r['outcome']==outcome for r in group) for group in by_site.values()], reverse=True)
        concentration[outcome] = {'sites_with_errors': sum(x>0 for x in counts),
                                  'top_site_errors': counts[0] if counts else 0,
                                  'top_two_sites_errors': sum(counts[:2])}
    transitions = Counter()
    for group in by_site.values():
        ordered = sorted(group, key=lambda r:r['key'][1])
        for previous, current in zip(ordered, ordered[1:]):
            transitions['adjacent_labeled_pairs'] += 1
            if previous['target'] == current['target']:
                transitions['same_label_pairs'] += 1
                if previous['predicted'] != current['predicted']:
                    transitions['prediction_flip_same_label'] += 1
    bins = {}
    for name, predicate in [
        ('prior_area_lt10', lambda r:r['prior_area']<10),
        ('prior_area_10_to100', lambda r:10<=r['prior_area']<100),
        ('prior_area_ge100', lambda r:r['prior_area']>=100),
        ('abs_margin_le002', lambda r:r['margin'] is not None and abs(r['margin'])<=0.02),
        ('abs_margin_gt002', lambda r:r['margin'] is not None and abs(r['margin'])>0.02),
        ('inside_pixels_lt10', lambda r:r['inside_pixels']<10),
        ('inside_pixels_ge10', lambda r:r['inside_pixels']>=10),
        ('positive_prior_iou_lt025', lambda r:r['prior_iou'] is not None and r['prior_iou']<0.25),
        ('positive_prior_iou_ge025', lambda r:r['prior_iou'] is not None and r['prior_iou']>=0.25),
    ]:
        subset = [r for r in rows if predicate(r)]
        bins[name] = {'n':len(subset), 'errors':sum(r['outcome'] in ['fp','fn'] for r in subset)}
    fp_sites = {site for site, group in by_site.items() if any(r['outcome']=='fp' for r in group)}
    matched = [r for site in fp_sites for r in by_site[site]]
    fp_site_comparison = {outcome: {field:stats([r[field] for r in matched if r['outcome']==outcome])
                                  for field in ['score','inside_ndvi','ring_ndvi','inside_brightness']}
                          for outcome in ['fp','tn']}
    return {'fp_site_comparison':fp_site_comparison, 'counts': dict(outcome_counts), 'sites':len(by_site), 'groups':groups,
            'error_concentration':concentration, 'temporal':dict(transitions), 'bins':bins}


def main():
    try:
        root = Path(json.loads((PROJECT/'configs/local.json').read_text())['train_dir'])
        frozen = json.loads(MANIFEST.read_text())
        model = json.loads((PROJECT/'private/baseline_v1/model.json').read_text())
        if (build_manifest(root, json.loads((PROJECT/'configs/split.json').read_text())) != frozen or
                model['dataset_sha256'] != frozen['dataset_sha256']):
            raise EvaluationError('Dataset drift')
        report = {'analysis_version':1, 'model_changed':False, 'partitions':{}}
        for name in ['train','validation']:
            scenes = set(frozen['scene_partitions'][name])
            rows = collect(root, scenes, model)
            if name=='validation':
                predictions = read_predictions(PROJECT/'private/baseline_v1/predictions.csv', set(load_truth(root,scenes)))
                if any((predictions.get(r['key']) is not None) != r['predicted'] for r in rows):
                    raise EvaluationError('Prediction mismatch')
            report['partitions'][name] = summarize(rows)
        if fingerprint(root) != frozen['dataset_sha256']:
            raise EvaluationError('Dataset changed')
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError):
        print('Analysis failed: inspect data/model consistency. Private details suppressed.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
