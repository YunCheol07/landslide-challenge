"""Independent NDVI/prior baseline. Images and masks exist only in memory."""
import csv
import json
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.ops import unary_union

from src.evaluation import EvaluationError, polygon_geometry


def validate_config(config):
    if (config.get('version') != 'ndvi_prior_v1' or
            config.get('positive_rule') != 'inside_minus_ring_ndvi_lte_threshold' or
            config.get('boundary') != 'unchanged_prior' or
            not 0 <= config['ring_inner_px'] < config['ring_outer_px'] or
            type(config['min_valid_pixels']) is not int or config['min_valid_pixels'] < 1):
        raise EvaluationError('Unsupported baseline configuration')


def feature(arr, inside, ring, minimum):
    if arr.shape != (256, 256, 4) or arr.dtype != np.uint16:
        raise EvaluationError('Invalid bands array')
    valid = (arr[:, :, 0] > 0) & (arr[:, :, 3] > 0)
    a, b = inside & valid, ring & valid
    if a.sum() < minimum or b.sum() < minimum:
        return None
    red = arr[:, :, 0].astype(np.float64) / 10000.0
    nir = arr[:, :, 3].astype(np.float64) / 10000.0
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=valid)
    return float(ndvi[a].mean() - ndvi[b].mean())


def extract(root, config, scenes=None):
    """Read only sites.json and bands; labels cannot influence prediction features."""
    validate_config(config)
    yy, xx = np.mgrid[0:256, 0:256].astype(float)
    xx += 0.5
    yy += 0.5
    records = []
    windows = sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith('.'))
    for window in windows:
        if scenes is not None and window.name.rsplit('_w', 1)[0] not in scenes:
            continue
        sites = json.loads((window / 'sites.json').read_text())
        ids = [s['site_id'] for s in sites]
        if not sites or len(set(ids)) != len(ids):
            raise EvaluationError('Missing or duplicate sites')
        geoms = [polygon_geometry(s['prior_polygon']) for s in sites]
        occupied = unary_union(geoms)
        masks = []
        for geom in geoms:
            ring = geom.buffer(config['ring_outer_px']).difference(
                geom.buffer(config['ring_inner_px'])).difference(occupied)
            masks.append((contains_xy(geom, xx, yy), contains_xy(ring, xx, yy)))
        files = sorted((window / 'bands').glob('*.npy'))
        if not files:
            raise EvaluationError('Missing bands')
        for path in files:
            arr = np.load(path, allow_pickle=False)
            for site, (inside, ring) in zip(sites, masks):
                score = feature(arr, inside, ring, config['min_valid_pixels'])
                records.append(((window.name, path.stem, site['site_id']), score,
                                site['prior_polygon']))
    if not records:
        raise EvaluationError('No prediction inputs')
    return records


def decision(value, model):
    return bool(model['fallback_positive']) if value is None else value <= model['threshold']


def fit(records, truth, config):
    """Train-only exact threshold search; fixed tie-break picks smallest threshold."""
    rows = []
    for key, score, _ in records:
        label = truth.get(key)
        if label is None or label.get('ignore', False):
            continue
        visible = label.get('visible')
        if type(visible) is not int or visible not in (0, 1):
            raise EvaluationError('Invalid training label')
        rows.append((score, visible))
    labels = [y for _, y in rows]
    if set(labels) != {0, 1}:
        raise EvaluationError('Training requires both classes')
    values = sorted({x for x, _ in rows if x is not None})
    if not values:
        raise EvaluationError('No usable training features')
    fallback = sum(labels) > len(labels) / 2
    candidates = [float(np.nextafter(values[0], -np.inf))] + values
    best = None
    for threshold in candidates:
        tp = tn = fp = fn = 0
        for value, target in rows:
            pred = fallback if value is None else value <= threshold
            tp += int(pred and target == 1)
            tn += int(not pred and target == 0)
            fp += int(pred and target == 0)
            fn += int(not pred and target == 1)
        macro = (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn)) / 2
        if best is None or macro > best[0]:
            best = macro, threshold
    return {'version': 'ndvi_prior_v1', 'config': config, 'threshold': best[1],
            'fallback_positive': fallback, 'fit_samples': len(rows),
            'fit_missing_features': sum(x is None for x, _ in rows),
            'fit_macro_f1': best[0]}


def write_predictions(path, records, model):
    """Never overwrite an existing result. Caller chooses a private output directory."""
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['location_id', 'date', 'site_id', 'polygons'])
        for key, value, prior in records:
            writer.writerow([*key, json.dumps(prior, separators=(',', ':'))
                             if decision(value, model) else ''])
