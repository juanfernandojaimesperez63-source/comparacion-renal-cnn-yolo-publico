"""Reconstruye las métricas desde predicciones y comprueba la trazabilidad."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ('cyst', 'normal', 'stone', 'tumor')
SEEDS = (17, 27, 37, 47, 57)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--require-checkpoints', action='store_true')
    args = parser.parse_args()
    fingerprint = sha256(args.manifest.read_bytes()).hexdigest()
    assert fingerprint == 'c2a3b9888bae729b5f2c50289e4d2d8918fa15e699761a20178b7c3d1b77f8dd', 'El manifiesto no es el congelado para este experimento.'
    rows = read_csv(args.manifest)
    active = [row for row in rows if row['split'] in ('train', 'val', 'test')]
    for field in ('group', 'pixel_sha256', 'sha256'):
        assigned = {}
        for row in active:
            assigned.setdefault(row[field], set()).add(row['split'])
        assert all(len(splits) == 1 for splits in assigned.values()), f'Fuga entre conjuntos: {field}'
    expected = {row['path']: row['class'] for row in rows if row['split'] == 'test'}
    assert len(expected) == 2633, 'Tamaño de prueba inesperado.'
    hardware, backends, checked = set(), set(), []
    for model in ('cnn', 'yolo'):
        for seed in SEEDS:
            folder = args.results/'full'/f'{model}-{seed}'
            config = json.loads((folder/'trained.json').read_text())
            saved = json.loads((folder/'evaluated.json').read_text())
            assert config['manifest'] == saved['manifest'] == fingerprint
            assert saved['model'] == config['model'] == model
            assert saved['seed'] == config['seed'] == seed
            assert not saved['smoke'] and not config['smoke'] and saved['split'] == 'test'
            predictions = read_csv(folder/'predictions.csv')
            assert len(predictions) == len(expected) == saved['images']
            assert len({row['path'] for row in predictions}) == len(expected)
            assert {row['path']: row['true'] for row in predictions} == expected
            probabilities = np.array([[float(row[f'p_{name}']) for name in CLASSES] for row in predictions])
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-6)
            predicted = [CLASSES.index(row['predicted']) for row in predictions]
            np.testing.assert_array_equal(probabilities.argmax(axis=1), predicted)
            truth = [CLASSES.index(row['true']) for row in predictions]
            report = classification_report(truth, predicted, labels=range(4), target_names=CLASSES,
                                           output_dict=True, zero_division=0)
            for metric, value in [('accuracy', report['accuracy']),
                                  ('f1_macro', report['macro avg']['f1-score']),
                                  ('precision_macro', report['macro avg']['precision']),
                                  ('recall_macro', report['macro avg']['recall'])]:
                np.testing.assert_allclose(saved[metric], value, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(np.loadtxt(folder/'confusion.csv', delimiter=',', skiprows=1),
                                          confusion_matrix(truth, predicted, labels=range(4)))
            for row in read_csv(folder/'class_metrics.csv'):
                for metric in ('precision', 'recall', 'f1-score', 'support'):
                    np.testing.assert_allclose(float(row[metric]), report[row['class']][metric], rtol=0, atol=1e-12)
            history = read_csv(folder/'history.csv')
            best_index = int(np.argmax([float(row['val_f1_macro']) for row in history]))
            assert int(history[best_index]['epoch']) == config['best_epoch']
            assert len(history) == config['final_epoch'] <= 20
            if args.require_checkpoints:
                assert (folder/'best.pt').is_file() or (folder/'best.weights.h5').is_file()
            hardware.add(saved['hardware'])
            backends.add(config.get('backend', 'original'))
            checked.append(f'{model}-{seed}')
    assert len(hardware) == len(backends) == 1, 'Se mezclaron hardware o backends.'
    result = {'passed': True, 'runs_checked': checked, 'test_images_per_run': len(expected),
              'checkpoints_required': args.require_checkpoints, 'manifest': fingerprint, 'hardware': sorted(hardware), 'backends': sorted(backends),
              'checks': ['frozen_manifest', 'no_group_or_hash_overlap', 'test_paths_and_labels', 'probabilities_and_argmax', 'metrics_recomputed',
                         'class_metrics', 'confusion_counts', 'best_validation_epoch']}
    (args.results/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
