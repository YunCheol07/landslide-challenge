"""Local evaluator for the documented competition rules; no geometry repair."""
from collections import Counter, defaultdict
import csv
import json

import numpy as np
from shapely.errors import ShapelyError
from shapely.geometry import shape


class EvaluationError(ValueError):
    """Messages intentionally omit coordinates, IDs, and file paths."""


def polygon_geometry(coords):
    """Parse nonempty MultiPolygon coordinates; implicit ring closure is allowed."""
    try:
        if not isinstance(coords, list) or not coords:
            raise ValueError()
        for polygon in coords:
            if not isinstance(polygon, list) or not polygon:
                raise ValueError()
            for ring in polygon:
                a = np.asarray(ring, dtype=float)
                if (a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or
                        not np.isfinite(a).all() or (a < 0).any() or (a > 256).any()):
                    raise ValueError()
        geom = shape({'type': 'MultiPolygon', 'coordinates': coords})
        if geom.is_empty or not geom.is_valid or geom.area <= 0:
            raise ValueError()
        return geom
    except (ValueError, TypeError, IndexError, OverflowError, ShapelyError):
        raise EvaluationError('Invalid polygon geometry') from None


def read_predictions(path, allowed_keys):
    """Strict CSV adapter; blank and [] mean absent, missing rows also mean absent."""
    predictions = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['location_id', 'date', 'site_id', 'polygons']:
            raise EvaluationError('Unexpected prediction CSV header')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise EvaluationError('Malformed prediction CSV row')
            key = (row['location_id'], row['date'], row['site_id'])
            if key not in allowed_keys:
                raise EvaluationError('Prediction outside selected partition or unknown input')
            if key in predictions:
                raise EvaluationError('Duplicate prediction row')
            raw = row['polygons'].strip()
            try:
                coords = json.loads(raw) if raw else []
            except ValueError:
                raise EvaluationError('Invalid prediction JSON') from None
            predictions[key] = None if coords == [] else polygon_geometry(coords)
    return predictions


def evaluate(truth, predictions, diagnostic=False):
    """truth maps (location,date,site) to record or None (missing).

    Predictions map these keys to validated Shapely geometries or None.
    Presence uses all explicit non-ignored binary labels, even invalid GT shapes.
    """
    if set(predictions) - set(truth):
        raise EvaluationError('Unknown prediction key')
    for geom in predictions.values():
        if geom is not None and (geom.geom_type != 'MultiPolygon' or geom.is_empty or
                                 not geom.is_valid or geom.area <= 0 or
                                 not all(np.isfinite(geom.bounds)) or
                                 min(geom.bounds) < 0 or max(geom.bounds) > 256):
            raise EvaluationError('Invalid prediction geometry')
    counts = Counter(tp=0, tn=0, fp=0, fn=0, missing_excluded=0, ignore_excluded=0,
                     shape_unresolved=0, shape_eligible=0, shape_below_10px2=0)
    site_ious = defaultdict(list)
    for key, label in truth.items():
        if label is None:
            counts['missing_excluded'] += 1
            continue
        ignore = label.get('ignore', False)
        if type(ignore) not in (bool, int) or ignore not in (0, 1):
            raise EvaluationError('Invalid ignore label')
        if ignore:
            counts['ignore_excluded'] += 1
            continue
        visible = label.get('visible')
        if type(visible) is not int or visible not in (0, 1):
            raise EvaluationError('Invalid visible label')
        prediction = predictions.get(key)
        present = prediction is not None
        counts[('tp' if present else 'fn') if visible else ('fp' if present else 'tn')] += 1
        if not visible:
            continue
        try:
            target = polygon_geometry(label.get('polygon'))
        except EvaluationError:
            counts['shape_unresolved'] += 1
            continue
        if target.area < 10:
            counts['shape_below_10px2'] += 1
            continue
        counts['shape_eligible'] += 1
        try:
            iou = (target.intersection(prediction).area / target.union(prediction).area
                   if present else 0.0)
        except ShapelyError:
            raise EvaluationError('Geometry operation failed') from None
        site_ious[(key[0], key[2])].append(iou)
    tp, tn, fp, fn = (counts[x] for x in ('tp', 'tn', 'fp', 'fn'))
    if not tp + tn + fp + fn:
        raise EvaluationError('No evaluable presence labels')
    def f1(correct, other):
        denominator = 2 * correct + other
        return 2 * correct / denominator if denominator else 0.0
    positive_f1, negative_f1 = f1(tp, fp + fn), f1(tn, fp + fn)
    macro_f1 = (positive_f1 + negative_f1) / 2
    valid_miou = (sum(sum(values) / len(values) for values in site_ious.values()) /
                  len(site_ious)) if site_ious else None
    miou = None if counts['shape_unresolved'] else valid_miou
    result = {
        'evaluator_version': 'documented_rules_v1',
        'status': 'shape_blocked' if counts['shape_unresolved'] else
                  ('no_shape_targets' if miou is None else 'ok'),
        'counts': dict(counts), 'shape_sites': len(site_ious),
        'presence': {'positive_f1': positive_f1, 'negative_f1': negative_f1,
                     'macro_f1': macro_f1},
        'site_miou': miou,
        'combined_score': 0.6 * macro_f1 + 0.4 * miou if miou is not None else None,
        'official_server_parity_verified': False,
    }
    if diagnostic:
        result['diagnostic_valid_geometry_only_site_miou'] = valid_miou
    return result
