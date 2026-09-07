"""Evaluate local predictions against a verified frozen partition, aggregate output only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_split import MANIFEST, PROJECT, build_manifest, fingerprint
from src.evaluation import EvaluationError, evaluate, read_predictions


def load_truth(root, scenes):
    truth = {}
    for window in sorted(root.iterdir()):
        if not window.is_dir() or window.name.rsplit('_w', 1)[0] not in scenes:
            continue
        sites = json.loads((window / 'sites.json').read_text())
        ids = {r['site_id'] for r in sites}
        dates = set()
        for folder, extension in [('bands', '*.npy'), ('images', '*.png'), ('labels', '*.json')]:
            dates.update(p.stem for p in (window / folder).glob(extension))
        for date in sorted(dates):
            path = window / 'labels' / (date + '.json')
            records = json.loads(path.read_text()) if path.exists() else []
            indexed = {}
            for record in records:
                sid = record['site_id']
                if sid in indexed or sid not in ids:
                    raise EvaluationError('Duplicate or unknown ground truth site')
                indexed[sid] = record
            for sid in ids:
                truth[(window.name, date, sid)] = indexed.get(sid)
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/local.json')
    parser.add_argument('--partition', choices=['train', 'validation'], default='validation')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions', type=Path)
    inputs.add_argument('--empty-predictions', action='store_true', help='No-model sanity check, all absent')
    parser.add_argument('--diagnostic', action='store_true', help='Also report valid-GT-only diagnostic mIoU')
    args = parser.parse_args()
    try:
        root = Path(json.loads(args.config.read_text())['train_dir']).expanduser()
        frozen = json.loads(MANIFEST.read_text())
        spec = json.loads((PROJECT / 'configs/split.json').read_text())
        if build_manifest(root, spec) != frozen:
            raise EvaluationError('Frozen split drift')
        truth = load_truth(root, set(frozen['scene_partitions'][args.partition]))
        predictions = {} if args.empty_predictions else read_predictions(args.predictions, set(truth))
        report = evaluate(truth, predictions, diagnostic=args.diagnostic)
        if fingerprint(root) != frozen['dataset_sha256']:
            raise EvaluationError('Input changed during evaluation')
        report['partition'] = args.partition
        report['prediction_source'] = 'all_absent_sanity_check' if args.empty_predictions else 'csv'
    except (OSError, ValueError, KeyError, TypeError):
        print('Evaluation failed: check frozen split, data audit, and prediction format. No private details printed.', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
