"""Entrena y evalúa un modelo por proceso sobre el manifiesto congelado."""
import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score



CLASSES = ("cyst", "normal", "stone", "tumor")
SEEDS = (17, 27, 37, 47, 57)
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5


def dump_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)


def batches(rows, root, rng=None, limit=None):
    indexes = np.arange(len(rows))
    if rng is not None:
        rng.shuffle(indexes)
    if limit:
        indexes = indexes[:limit]
    for start in range(0, len(indexes), BATCH_SIZE):
        examples = [rows[i] for i in indexes[start:start + BATCH_SIZE]]
        images = []
        for row in examples:
            image = decode(root / row["path"])
            if rng is not None:
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image = image.rotate(float(rng.uniform(-108, 108)), Image.Resampling.BILINEAR)
            images.append(np.asarray(image))
        yield np.stack(images), np.array([CLASSES.index(row["class"]) for row in examples]), examples


def score(model, rows, root, limit=None):
    predictions, targets, paths = [], [], []
    for images, labels, examples in batches(rows, root, limit=limit):
        probabilities = model.predict(images)
        predictions.extend(probabilities)
        targets.extend(labels)
        paths.extend(row["path"] for row in examples)
    return np.asarray(targets), np.stack(predictions), paths


def train(model, args, rows, out, fingerprint):
    rng = np.random.default_rng(args.seed)
    history = []
    best_f1, best_epoch = -1.0, 0
    started = time.perf_counter()
    limit = BATCH_SIZE * 2 if args.smoke else None
    for epoch in range(1, (1 if args.smoke else MAX_EPOCHS) + 1):
        batch_losses = [(model.train(x, y), len(y)) for x, y, _ in batches(
            rows["train"], args.root, rng=rng, limit=limit
        )]
        loss, batch_counts = zip(*batch_losses)
        y, probs, _ = score(model, rows["val"], args.root, limit)
        val_f1 = float(f1_score(y, probs.argmax(axis=1), labels=range(4),
                               average="macro", zero_division=0))
        val_loss = -float(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-8)).mean())
        history.append({"epoch": epoch, "train_loss": float(np.average(loss, weights=batch_counts)),
                        "val_loss": val_loss, "val_f1_macro": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            model.save(out)
        dump_csv(out / "history.csv", history)
        print(history[-1], flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    configuration = {
        "backend": args.backend, "model": args.model, "seed": args.seed, "smoke": args.smoke, "manifest": fingerprint,
        "best_epoch": best_epoch, "final_epoch": epoch, "best_val_f1_macro": best_f1,
        "training_seconds": time.perf_counter() - started, "hardware": model.hardware,
        "platform": platform.platform(), "python": platform.python_version(), "parameters": model.parameters,
        "optimizer": "Adam", "learning_rate": 0.001, "batch_size": BATCH_SIZE,
        "patience": PATIENCE, "max_epochs": 1 if args.smoke else MAX_EPOCHS,
        "image_size": 224, "pretrained": False, "loss": "cross_entropy",
        "augmentation": "p=0.5 horizontal, p=0.5 vertical, rotación uniforme ±108°, relleno negro",
        "versions": {name: version(name) for name in (
            "numpy", "Pillow", "scikit-learn", "tensorflow" if args.model == "cnn" and args.backend == "original" else "torch",
            *(("torch", "torchvision") if args.model == "yolo" or args.backend == "shared_torch" else ()),
        )},
    }
    dump_json(out / "trained.json", configuration)


def evaluate(model, args, rows, out, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.figsize": (3.5, 3.1)})

    model.load(out)
    split = "val" if args.smoke else "test"
    test_rows = rows[split]
    if args.smoke:
        test_rows = test_rows[:BATCH_SIZE * 2]
    y, probabilities, paths = score(model, test_rows, args.root)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(y, predicted, labels=range(4), target_names=CLASSES,
                                   zero_division=0, output_dict=True)
    prediction_rows = [
        {"path": path, "true": CLASSES[target], "predicted": CLASSES[prediction],
         **{f"p_{label}": float(p) for label, p in zip(CLASSES, scores)}}
        for path, target, prediction, scores in zip(paths, y, predicted, probabilities)
    ]
    dump_csv(out / "predictions.csv", prediction_rows)
    dump_csv(out / "class_metrics.csv", [{"class": name, **report[name]} for name in CLASSES])
    matrix = confusion_matrix(y, predicted, labels=range(4))
    np.savetxt(out / "confusion.csv", matrix, delimiter=",", fmt="%d",
               header=",".join(CLASSES), comments="")
    fig, ax = plt.subplots(layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicción", ylabel="Etiqueta real")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    fig.savefig(out / "confusion.png", dpi=160)
    plt.close(fig)

    selected = np.linspace(0, len(paths) - 1, 100, dtype=int)
    decoded = [np.array(decode(args.root / paths[i]))[None] for i in selected]
    for _ in range(20):
        model.predict(decoded[0])
    timings = []
    for image in decoded:
        start = time.perf_counter_ns()
        model.predict(image)
        timings.append((time.perf_counter_ns() - start) / 1e6)

    wrong = np.flatnonzero(y != predicted)
    # Misma regla para los dos modelos: errores con mayor confianza.
    chosen = sorted(wrong, key=lambda i: -probabilities[i, predicted[i]])[:6]
    if chosen:
        fig, axes = plt.subplots(2, 3, figsize=(9, 7.2), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, chosen):
            ax.imshow(decode(args.root / paths[i]))
            ax.set_title(f"{CLASSES[y[i]]} → {CLASSES[predicted[i]]}\n{probabilities[i, predicted[i]]:.4f}", fontsize=9, pad=8)
        fig.savefig(out / "errors.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    if args.model == "cnn":
        i = chosen[0] if chosen else 0
        image = np.asarray(decode(args.root / paths[i]))
        heatmap = model.gradcam(image, int(predicted[i]))
        fig, ax = plt.subplots(layout="constrained")
        ax.imshow(image)
        ax.imshow(heatmap, extent=(-0.5, 223.5, 223.5, -0.5), cmap="inferno",
                  alpha=0.45, interpolation="bilinear", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{CLASSES[y[i]]} → {CLASSES[predicted[i]]}")
        fig.savefig(out / "gradcam.png", dpi=160)
        plt.close(fig)
        dump_json(out / "gradcam.json", {"path": paths[i], "layer": "conv3",
                                         "target": CLASSES[predicted[i]], "max": float(heatmap.max())})

    summary = {
        **{key: config[key] for key in ("model", "seed", "best_epoch", "final_epoch", "training_seconds",
                                        "hardware", "manifest", "smoke", "backend")},
        "split": split, "images": len(y), "accuracy": float(np.mean(y == predicted)),
        "precision_macro": report["macro avg"]["precision"], "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"], "latency_ms_p50": float(np.median(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
    }
    dump_csv(out / "metrics.csv", [summary])
    dump_json(out / "evaluated.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("model", choices=("cnn", "yolo"))
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--backend", choices=("original", "shared_torch"), default="original")
    args = parser.parse_args()
    if args.backend == "shared_torch":
        from models_mps import ReferenceCNN, YOLOClassifier
    else:
        from models import ReferenceCNN, YOLOClassifier
    output = args.output / ("smoke" if args.smoke else "full")
    out = output / f"{args.model}-{args.seed}"
    fingerprint = sha256(args.manifest.read_bytes()).hexdigest()
    if args.phase == "evaluate" and not args.smoke:
        for model in ("cnn", "yolo"):
            for seed in SEEDS:
                checkpoint = output / f"{model}-{seed}" / "trained.json"
                if not checkpoint.is_file() or json.loads(checkpoint.read_text())["manifest"] != fingerprint or json.loads(checkpoint.read_text()).get("backend", "original") != args.backend:
                    parser.error("Primero deben finalizar los diez entrenamientos de este manifiesto.")
    trained = out / "trained.json"
    config = json.loads(trained.read_text()) if trained.exists() else None
    if config and (config["manifest"] != fingerprint or config.get("backend", "original") != args.backend):
        parser.error("El directorio contiene otro manifiesto. Usa otra carpeta de resultados.")
    if args.phase == "train" and config:
        print(f"Ya entrenado: {out}")
        return
    if args.phase == "evaluate" and (out / "evaluated.json").exists():
        print(f"Ya evaluado: {out}")
        return
    if args.phase == "evaluate" and not config:
        parser.error("Entrena primero la ejecución solicitada.")
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    with args.manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] in rows:
                rows[row["split"]].append(row)
    for split in rows:
        if {row["class"] for row in rows[split]} != set(CLASSES):
            parser.error(f"Faltan clases en {split}.")
        # Orden intercalado determinista para que la prueba breve incluya las cuatro clases.
        rows[split].sort(key=lambda row: sha256(row["path"].encode()).hexdigest())
    model = (ReferenceCNN if args.model == "cnn" else YOLOClassifier)(args.seed)
    if args.phase == "train":
        train(model, args, rows, out, fingerprint)
    else:
        evaluate(model, args, rows, out, config)


if __name__ == "__main__":
    main()
