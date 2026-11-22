"""Resume las diez evaluaciones de prueba y genera figuras y tablas."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import SEEDS, dump_csv, dump_json

METRICS = ("f1_macro", "accuracy", "precision_macro", "recall_macro", "training_seconds",
           "latency_ms_p50", "latency_ms_p95")
MODELS = ("cnn", "yolo")


def table(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def export_excel(path, sheets):
    # En Colab no está disponible el runtime de hojas de cálculo de Codex.
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        sheet = workbook.create_sheet(title)
        sheet.append(list(rows[0]))
        for row in rows:
            sheet.append(list(row.values()))
        for column in sheet.columns:
            letter = get_column_letter(column[0].column)
            sheet.column_dimensions[letter].width = min(66, max(len(str(c.value)) for c in column) + 3)
            column[0].font = Font(bold=True, color="FFFFFF")
            column[0].fill = PatternFill("solid", fgColor="17365D")
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--excel-fallback", action="store_true", help="Exportación portable para entornos sin el runtime de hojas de cálculo de Codex.")
    args = parser.parse_args()
    out = args.results
    pairs = {(model, seed): out / "full" / f"{model}-{seed}" for model in MODELS for seed in SEEDS}
    if any(not (path / "evaluated.json").is_file() for path in pairs.values()):
        parser.error("Se necesitan las diez evaluaciones completas.")
    runs = {key: json.loads((path / "evaluated.json").read_text()) for key, path in pairs.items()}
    if len({run["manifest"] for run in runs.values()}) != 1:
        parser.error("Los manifiestos de las ejecuciones difieren.")
    if any(run["split"] != "test" or run["smoke"] or
           (run["model"], run["seed"]) != key for key, run in runs.items()):
        parser.error("Resultados incompatibles con el protocolo.")
    if len({run.get("backend", "original") for run in runs.values()}) != 1:
        parser.error("Las ejecuciones mezclan backends diferentes.")
    if len({run["hardware"] for run in runs.values()}) > 1:
        print("Hubo cambios de hardware entre semillas. Revisa los tiempos antes de compararlos.")

    plt.rcParams.update({"font.size": 9, "figure.figsize": (3.5, 2.8)})
    summary = []
    for model in MODELS:
        for metric in METRICS:
            values = [runs[model, seed][metric] for seed in SEEDS]
            summary.append({"model": model, "metric": metric, "mean": float(np.mean(values)),
                            "sample_sd": float(np.std(values, ddof=1)), "n": len(values)})
    classes = []
    for (model, seed), path in pairs.items():
        for row in table(path / "class_metrics.csv"):
            classes.append({"model": model, "seed": seed, "class": row["class"],
                            **{key: float(row[key]) for key in ("precision", "recall", "f1-score")},
                            "support": int(float(row["support"]))})

    delta = np.array([runs["yolo", seed]["f1_macro"] - runs["cnn", seed]["f1_macro"] for seed in SEEDS])
    observed = float(delta.mean())
    null = np.array([np.mean(delta * np.array(signs)) for signs in product((-1, 1), repeat=len(delta))])
    stats = {"test": "paired_exact_sign_flip_two_sided", "pairs": len(delta),
             "delta_mean_yolo_minus_cnn": observed,
             "p": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
             "minimum_possible_p": 2 / len(null)}
    differences = [{"seed": seed, "delta_f1_yolo_minus_cnn": float(value)} for seed, value in zip(SEEDS, delta)]
    dump_json(out / "paired_test.json", stats)
    dump_csv(out / "all_runs.csv", list(runs.values()))
    dump_csv(out / "summary.csv", summary)
    if args.excel_fallback:
        export_excel(out / "consolidado.xlsx", {"Ejecuciones": list(runs.values()), "Resumen": summary,
                     "Por clase": classes, "Diferencias": differences, "Prueba": [stats]})

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for ax, model in zip(axes, MODELS):
        for seed in SEEDS:
            history = table(pairs[model, seed] / "history.csv")
            ax.plot([int(row["epoch"]) for row in history],
                    [float(row["val_f1_macro"]) for row in history], label=str(seed))
        ax.set(title=model.upper(), xlabel="Época", ylabel="F1 macro en validación", ylim=(0, 1))
        ax.legend(title="Semilla")
    fig.savefig(out / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for ax, model in zip(axes, MODELS):
        for seed in SEEDS:
            history = table(pairs[model, seed] / "history.csv")
            epochs = [int(row["epoch"]) for row in history]
            line, = ax.plot(epochs, [float(row["train_loss"]) for row in history], label=str(seed))
            ax.plot(epochs, [float(row["val_loss"]) for row in history],
                    linestyle="--", color=line.get_color())
        ax.set(title=model.upper(), xlabel="Época", ylabel="Entropía cruzada")
        ax.legend(title="Semilla")
    fig.suptitle("Pérdida de entrenamiento (continua) y validación (discontinua)")
    fig.savefig(out / "loss_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for seed in SEEDS:
        ax.plot([0, 1], [runs[model, seed]["f1_macro"] for model in MODELS],
                marker="o", label=str(seed))
    ax.set(xticks=[0, 1], xticklabels=["CNN", "YOLO26n-cls"], ylabel="F1 macro en prueba")
    ax.legend(title="Semilla")
    fig.savefig(out / "paired_f1.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for model in MODELS:
        ax.scatter([runs[model, seed]["latency_ms_p50"] for seed in SEEDS],
                   [runs[model, seed]["f1_macro"] for seed in SEEDS], label=model.upper())
    ax.set(xlabel="Latencia mediana (ms por imagen)", ylabel="F1 macro en prueba")
    ax.legend()
    fig.savefig(out / "f1_latency.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
