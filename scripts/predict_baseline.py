"""Offline inference: needs bands and sites only, never labels or split manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline import extract, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        # Do not put predictions into provided input data.
        if args.input_dir.resolve() in args.output.resolve().parents:
            raise ValueError()
        model = json.loads(args.model.read_text())
        if model['version'] != 'ndvi_prior_v1':
            raise ValueError()
        rows = extract(args.input_dir, model['config'])
        write_predictions(args.output, rows, model)
        print(json.dumps({'prediction_rows': len(rows), 'missing_features': sum(r[1] is None for r in rows)}))
    except (OSError, ValueError, KeyError, TypeError):
        print('Inference failed: check model, inputs, and unused output path.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
