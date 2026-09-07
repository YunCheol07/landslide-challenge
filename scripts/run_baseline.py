"""Fit train-only baseline, predict validation and evaluate; local artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_split import PROJECT, MANIFEST, build_manifest, fingerprint
from scripts.evaluate import load_truth
from src.baseline import extract, fit, write_predictions
from src.evaluation import EvaluationError, evaluate, read_predictions


def source_digest():
    h = hashlib.sha256()
    for folder in ['src', 'scripts', 'configs']:
        for path in sorted((PROJECT / folder).glob('*')):
            if path.is_file() and path.name != 'local.json' and path.suffix in {'.py', '.json'}:
                h.update(path.relative_to(PROJECT).as_posix().encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='baseline_v1')
    args = parser.parse_args()
    try:
        if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.run_name):
            raise EvaluationError('Invalid run name')
        output = PROJECT / 'private' / args.run_name
        if output.exists():
            raise EvaluationError('Run already exists')
        root = Path(json.loads((PROJECT / 'configs/local.json').read_text())['train_dir']).expanduser()
        config = json.loads((PROJECT / 'configs/baseline.json').read_text())
        frozen = json.loads(MANIFEST.read_text())
        spec = json.loads((PROJECT / 'configs/split.json').read_text())
        if build_manifest(root, spec) != frozen:
            raise EvaluationError('Split changed')
        start = time.perf_counter()
        train_scenes = set(frozen['scene_partitions']['train'])
        validation_scenes = set(frozen['scene_partitions']['validation'])
        train_records = extract(root, config, train_scenes)
        model = fit(train_records, load_truth(root, train_scenes), config)
        # Threshold frozen before reading validation features or validation labels here.
        model['dataset_sha256'] = frozen['dataset_sha256']
        model['source_sha256'] = source_digest()
        model['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).decode().strip()
        model['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT))
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        validation_records = extract(root, config, validation_scenes)
        inference_seconds = time.perf_counter() - start
        if fingerprint(root) != frozen['dataset_sha256']:
            raise EvaluationError('Data changed')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'model.json').write_text(json.dumps(model, indent=2, allow_nan=False) + '\n')
        prediction_path = output / 'predictions.csv'
        write_predictions(prediction_path, validation_records, model)
        truth = load_truth(root, validation_scenes)
        report = evaluate(truth, read_predictions(prediction_path, set(truth)), diagnostic=True)
        report.update({'experiment_id': args.run_name, 'seed': config['seed'],
                       'git_commit': model['git_commit'], 'git_dirty': model['git_dirty'],
                       'source_sha256': model['source_sha256'], 'training_seconds': training_seconds,
                       'inference_seconds': inference_seconds,
                       'validation_input_rows': len(validation_records),
                       'validation_missing_features': sum(r[1] is None for r in validation_records),
                       'fit_samples': model['fit_samples'], 'fit_missing_features': model['fit_missing_features']})
        (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError):
        print('Baseline failed: inspect configuration, frozen split, or existing run directory. Private details suppressed.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
