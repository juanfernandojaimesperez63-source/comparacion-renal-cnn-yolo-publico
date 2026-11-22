"""Ejecuta el protocolo en procesos aislados y conserva las corridas terminadas."""
import argparse
from pathlib import Path
import subprocess
import sys

SEEDS = (17, 27, 37, 47, 57)
MODELS = ('cnn', 'yolo')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('original', 'shared_torch'), default='shared_torch')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    options = ['--root', str(args.root.resolve()), '--manifest', str(args.manifest.resolve()),
               '--output', str(args.output.resolve()), '--backend', args.backend]

    def execute(phase, model, seed, smoke=False):
        label = f"{'smoke' if smoke else 'full'}-{model}-{seed}-{phase}"
        command = [sys.executable, '-u', str(source/'run.py'), phase, model, str(seed), *options]
        if smoke:
            command.append('--smoke')
        print(f'INICIO {label}', flush=True)
        with (logs/f'{label}.log').open('a') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
        print(f'FINAL {label}', flush=True)

    for model in MODELS:
        for phase in ('train', 'evaluate'):
            execute(phase, model, SEEDS[0], smoke=True)
    for phase in ('train', 'evaluate'):
        for seed in SEEDS:
            for model in MODELS:
                execute(phase, model, seed)
    subprocess.run([sys.executable, str(source/'verify_results.py'), '--require-checkpoints', '--manifest', str(args.manifest.resolve()), '--results', str(args.output.resolve())], check=True)
    subprocess.run([sys.executable, str(source/'aggregate.py'), str(args.output.resolve()), '--excel-fallback'], check=True)


if __name__ == '__main__':
    main()
