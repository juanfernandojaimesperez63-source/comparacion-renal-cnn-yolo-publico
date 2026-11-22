"""Regenera ejemplos de error desde las predicciones guardadas, sin inferencia."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    for folder in sorted((args.results/'full').iterdir()):
        file = folder/'predictions.csv'
        if not file.is_file():
            continue
        with file.open(newline='', encoding='utf-8') as stream:
            predictions = list(csv.DictReader(stream))
        chosen = sorted((row for row in predictions if row['true'] != row['predicted']),
                        key=lambda row: -float(row['p_'+row['predicted']]))[:6]
        if not chosen:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(9, 7.2), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        for ax, row in zip(axes.flat, chosen):
            with Image.open(args.root/row['path']) as source:
                image = source.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            ax.imshow(image)
            confidence = float(row['p_'+row['predicted']])
            ax.set_title(f"{row['true']} → {row['predicted']}\n{confidence:.4f}", fontsize=9, pad=8)
        fig.savefig(folder/'errors.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
        with (folder/'error_examples.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
            writer.writeheader()
            writer.writerows(chosen)
        print(folder.name, len(chosen), 'ejemplos de error')


if __name__ == '__main__':
    main()
