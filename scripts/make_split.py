"""Freeze a scene-level split locally; publish only aggregate counts."""
import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data import Audit

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / 'private' / 'scene_split.json'


def select_scenes(scene_counts, seed, validation_count, target_fraction):
    """Choose by label counts only, never model predictions or scores."""
    scenes = sorted(scene_counts)
    if type(seed) is not int or type(validation_count) is not int:
        raise ValueError('seed/count must be integers')
    if not 0 < validation_count < len(scenes) or not 0 < target_fraction < 1:
        raise ValueError('both partitions must contain scenes')
    total_positive = sum(scene_counts[s][0] for s in scenes)
    total_negative = sum(scene_counts[s][1] for s in scenes)
    if not total_positive or not total_negative:
        raise ValueError('both classes required')
    candidates = []
    for group in combinations(scenes, validation_count):
        positive = sum(scene_counts[s][0] for s in group)
        negative = sum(scene_counts[s][1] for s in group)
        if not (0 < positive < total_positive and 0 < negative < total_negative):
            continue
        # Equal class weight avoids choosing a split based only on the majority class.
        distance = (abs(positive / total_positive - target_fraction) +
                    abs(negative / total_negative - target_fraction))
        tie = hashlib.sha256(json.dumps([seed, group]).encode()).hexdigest()
        candidates.append((distance, tie, group))
    if not candidates:
        raise ValueError('no scene split retains both classes')
    validation = set(min(candidates)[2])
    return {'train': sorted(set(scenes) - validation), 'validation': sorted(validation)}


def fingerprint(root):
    """Hash expected input inventory and content; no data or hashes printed."""
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   (p.name == 'sites.json' or p.parent.name in {'bands', 'images', 'labels'})
                   and not p.name.startswith('.'))
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, 'big'))
        digest.update(relative)
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                content.update(chunk)
        digest.update(content.digest())
    return digest.hexdigest()


def build_manifest(root, spec):
    if (spec.get('schema_version') != 1 or spec.get('algorithm') != 'class_count_balanced_scene_v1'):
        raise ValueError('unsupported split specification')
    if not root.is_dir():
        raise ValueError('missing input directory')
    before = fingerprint(root)
    windows = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    scene_windows = {}
    for window in windows:
        match = re.fullmatch(r'(ls_scene_\d+)_w\d+', window.name)
        if match is None:
            raise ValueError('unknown scene naming scheme')
        scene_windows.setdefault(match[1], []).append(window)
    scene_counts = {}
    for scene, group in scene_windows.items():
        audit = Audit()
        for window in group:
            audit.window(window)
        scene_counts[scene] = (audit.counts['visible_positive'], audit.counts['visible_negative'])
    partitions = select_scenes(scene_counts, spec['seed'], spec['validation_scene_count'],
                               spec['target_validation_fraction'])
    summaries = {}
    for name, scenes in partitions.items():
        audit = Audit()
        for scene in scenes:
            for window in scene_windows[scene]:
                audit.window(window)
        blocking = {key: value for key, value in audit.errors.items()
                    if value and key != 'label_polygon_invalid'}
        if blocking:
            raise ValueError('data audit failed; run audit_data.py for aggregate errors')
        if not audit.counts['visible_positive'] or not audit.counts['visible_negative']:
            raise ValueError('partition lacks a class; review split specification explicitly')
        summaries[name] = {
            'scenes': len(scenes),
            **{key: audit.counts[key] for key in (
                'windows', 'sites', 'bands_files', 'visible_positive', 'visible_negative',
                'missing_labels_excluded', 'ignore_excluded', 'shape_eligible', 'shape_below_10px2')},
            'shape_unresolved': audit.errors['label_polygon_invalid'],
        }
    if before != fingerprint(root):
        raise ValueError('input changed during audit')
    return {
        'schema_version': 1, 'spec': spec, 'dataset_sha256': before,
        'scene_partitions': partitions, 'summary': summaries,
        'geometry_policy': 'retain_positive_presence; unresolved_geometry_blocks_official_shape_score',
    }


def freeze(manifest, path=MANIFEST):
    """Existing manifests are verified exactly, never silently updated."""
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('frozen split differs; review data/spec changes explicitly')
        return 'verified'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return 'created'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/local.json')
    parser.add_argument('--spec', type=Path, default=PROJECT / 'configs/split.json')
    parser.add_argument('--check', action='store_true', help='Require and verify the frozen local manifest')
    args = parser.parse_args()
    try:
        if args.check and not MANIFEST.is_file():
            raise ValueError('no frozen split exists; run without --check first')
        config = json.loads(args.config.read_text())
        spec = json.loads(args.spec.read_text())
        manifest = build_manifest(Path(config['train_dir']).expanduser(), spec)
        state = freeze(manifest)
    except (OSError, ValueError, KeyError, TypeError):
        print('Split failed: inspect configuration/data audit or frozen manifest drift. Existing split was not overwritten.', file=sys.stderr)
        return 1
    print(json.dumps({'state': state, 'summary': manifest['summary'],
                      'official_shape_score_ready': not any(
                          p['shape_unresolved'] for p in manifest['summary'].values())}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
