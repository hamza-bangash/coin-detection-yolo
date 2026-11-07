"""Train YOLOv8 and YOLO26 (n/s/m/l) on the coin dataset, plot the
train/validation curves, build the comparison table, and report the
per-class AP of the Top-3 models by mAP@50.

Run it from VS Code (select the .venv interpreter, press the Run button)
or from a terminal:

    python scripts/train_compare.py                  # train + report
    python scripts/train_compare.py --stage train    # only train
    python scripts/train_compare.py --stage report   # only tables + graphs

Training settings match the earlier YOLOv10 / YOLOv11 runs
(imgsz=448, batch=16, epochs=100, patience=100, seed=0, default augmentation)
so the numbers are directly comparable.

Output layout (each model keeps its own folder, weights included):

    runs/
      yolo8/model_n/weights/best.pt, last.pt
                    results.csv, train_val_curves.png, metrics.json, ...
      yolo8/model_s/ ...
      yolo26/model_n/ ...
      comparison.csv / comparison.md / comparison_curves.png
      top3_per_class_ap.csv / top3_per_class_ap.md
"""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# CONFIG - edit here if your paths differ
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]          # ...\coins
REPO = Path(__file__).resolve().parents[1]          # ...\coins\coin-detection-yolo
DATA = ROOT / "final_dataset" / "coins.yaml"
OUT_DIR = ROOT / "runs"
EXISTING_RESULTS = REPO / "results" / "model_comparison.csv"   # v10 / v11 numbers
EXISTING_WEIGHTS = ROOT / "existing_weights"    # optional: yolo10_n.pt ... yolo11_l.pt

FAMILIES = ["yolo8", "yolo26"]
SIZES = ["n", "s", "m", "l"]
EPOCHS = 100
IMGSZ = 448
BATCH = 16
WORKERS = 4
SHOW_PLOTS = True        # open the graphs at the end (they are always saved)

WEIGHT_PREFIX = {"yolo8": "yolov8", "yolo26": "yolo26"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run_dir(family, size):
    return OUT_DIR / family / f"model_{size}"


def model_name(family, size):
    return f"{family}_{size}"


def read_results_csv(path):
    """Read an Ultralytics results.csv into a list of dicts (keys stripped)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def num(row, key):
    return float(row[key])


def is_complete(rd):
    csv_path = rd / "results.csv"
    if not (rd / "weights" / "best.pt").exists() or not csv_path.exists():
        return False
    return len(read_results_csv(csv_path)) >= EPOCHS


def format_table(headers, rows):
    """Plain-text table that prints nicely in any terminal."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [line, "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |", line]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    out.append(line)
    return "\n".join(out)


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def evaluate_best(family, size, weights=None, out_json=None):
    """Validate a checkpoint (default: this run's best.pt) and store overall +
    per-class metrics as json."""
    from ultralytics import YOLO

    rd = run_dir(family, size)
    best = Path(weights) if weights else rd / "weights" / "best.pt"
    out_json = Path(out_json) if out_json else rd / "metrics.json"
    model = YOLO(str(best))
    m = model.val(data=str(DATA), imgsz=IMGSZ, batch=BATCH, workers=WORKERS,
                  plots=False, verbose=False, project=str(out_json.parent),
                  name="final_val", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[m.names[int(c)]] = {
            "ap50": float(m.box.ap50[i]),
            "ap50_95": float(m.box.ap[i]),
        }
    data = {
        "model": model_name(family, size),
        "best_precision": float(m.box.mp),
        "best_recall": float(m.box.mr),
        "best_map50": float(m.box.map50),
        "best_map50_95": float(m.box.map),
        "per_class": per_class,
    }
    out_json.write_text(json.dumps(data, indent=2))
    return data


def train_all(families, sizes, epochs):
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_dir = OUT_DIR / "_pretrained"
    pretrained_dir.mkdir(exist_ok=True)
    os.chdir(pretrained_dir)        # pretrained .pt files download here, not in the repo

    for family in families:
        for size in sizes:
            rd = run_dir(family, size)
            tag = model_name(family, size)

            if is_complete(rd):
                print(f"\n[skip] {tag}: already trained")
            else:
                last = rd / "weights" / "last.pt"
                csv_path = rd / "results.csv"
                if last.exists() and csv_path.exists() and len(read_results_csv(csv_path)) < epochs:
                    print(f"\n[resume] {tag}")
                    YOLO(str(last)).train(resume=True)
                else:
                    weights = f"{WEIGHT_PREFIX[family]}{size}.pt"
                    print(f"\n[train] {tag}  <-  {weights}")
                    YOLO(weights).train(
                        data=str(DATA), epochs=epochs, imgsz=IMGSZ, batch=BATCH,
                        patience=100, seed=0, deterministic=True, workers=WORKERS,
                        device=0, project=str(OUT_DIR / family), name=f"model_{size}",
                        exist_ok=True, plots=True,
                    )

            if not (rd / "metrics.json").exists() and (rd / "weights" / "best.pt").exists():
                print(f"[eval] {tag}: validating best.pt")
                evaluate_best(family, size)


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
def plot_model_curves(family, size):
    """Train-vs-validation loss and accuracy graphs for one model."""
    rd = run_dir(family, size)
    rows = read_results_csv(rd / "results.csv")
    ep = [num(r, "epoch") for r in rows]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{model_name(family, size)} - training vs validation", fontsize=14)

    # YOLOv8/10/11 log a DFL loss; YOLO26 dropped DFL and logs an L1 loss instead.
    third = ("DFL loss", "dfl_loss") if "train/dfl_loss" in rows[0] else ("L1 loss", "l1_loss")
    for a, (name, key) in zip(ax[0], [("Box loss", "box_loss"), ("Class loss", "cls_loss"), third]):
        a.plot(ep, [num(r, f"train/{key}") for r in rows], label="train")
        a.plot(ep, [num(r, f"val/{key}") for r in rows], label="validation")
        a.set_title(name); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()

    a = ax[1][0]
    a.plot(ep, [num(r, "metrics/precision(B)") for r in rows], label="precision")
    a.plot(ep, [num(r, "metrics/recall(B)") for r in rows], label="recall")
    a.set_title("Precision / Recall"); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()

    a = ax[1][1]
    a.plot(ep, [num(r, "metrics/mAP50(B)") for r in rows], color="tab:green")
    a.set_title("mAP@50 (validation)"); a.set_xlabel("epoch"); a.grid(alpha=0.3)

    a = ax[1][2]
    a.plot(ep, [num(r, "metrics/mAP50-95(B)") for r in rows], color="tab:red")
    a.set_title("mAP@50-95 (validation)"); a.set_xlabel("epoch"); a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(rd / "train_val_curves.png", dpi=130)
    return fig


def plot_comparison(results):
    """Overlay of validation mAP@50 curves + final mAP@50 bar chart."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for r in results:
        if r["curve"]:
            ax[0].plot(r["curve"], label=r["model"])
    ax[0].set_title("Validation mAP@50 per epoch"); ax[0].set_xlabel("epoch")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, ncol=2)

    ordered = sorted(results, key=lambda r: r["map50"])
    ax[1].barh([r["model"] for r in ordered], [r["map50"] for r in ordered], color="tab:blue")
    lo = min(r["map50"] for r in results)
    ax[1].set_xlim(max(0.0, lo - 0.02), 1.0)
    ax[1].set_title("Final-epoch mAP@50"); ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_curves.png", dpi=130)
    return fig


# --------------------------------------------------------------------------- #
# report: comparison table, Top-3, per-class AP
# --------------------------------------------------------------------------- #
def collect_results(families, sizes):
    """New runs (final-epoch row, like the repo's comparison.md) + existing v10/v11."""
    results = []
    for family in families:
        for size in sizes:
            csv_path = run_dir(family, size) / "results.csv"
            if not csv_path.exists():
                continue
            rows = read_results_csv(csv_path)
            last = rows[-1]
            results.append({
                "model": model_name(family, size), "family": family, "size": size,
                "epochs": len(rows), "new": True,
                "precision": num(last, "metrics/precision(B)"),
                "recall": num(last, "metrics/recall(B)"),
                "map50": num(last, "metrics/mAP50(B)"),
                "map50_95": num(last, "metrics/mAP50-95(B)"),
                "curve": [num(r, "metrics/mAP50(B)") for r in rows],
            })
    if EXISTING_RESULTS.exists():
        for r in read_results_csv(EXISTING_RESULTS):
            results.append({
                "model": r["model"], "family": r["model"].split("_")[0],
                "size": r["model"].split("_")[1], "epochs": int(r["epochs_recorded"]), "new": False,
                "precision": float(r["metrics/precision(B)"]), "recall": float(r["metrics/recall(B)"]),
                "map50": float(r["metrics/mAP50(B)"]), "map50_95": float(r["metrics/mAP50-95(B)"]),
                "curve": [],
            })
    return results


def per_class_for(r):
    """Per-class AP from metrics.json (computed from best.pt)."""
    if not r["new"]:
        # v10/v11 weights are not in the repo: use them only if you put
        # e.g. existing_weights/yolo11_n.pt next to the dataset.
        w = EXISTING_WEIGHTS / f"{r['model']}.pt"
        mj = EXISTING_WEIGHTS / f"{r['model']}.json"
        if not w.exists():
            return None
        if not mj.exists():
            evaluate_best(r["family"], r["size"], weights=w, out_json=mj)
        return json.loads(mj.read_text())["per_class"]
    mj = run_dir(r["family"], r["size"]) / "metrics.json"
    if not mj.exists() and (run_dir(r["family"], r["size"]) / "weights" / "best.pt").exists():
        evaluate_best(r["family"], r["size"])
    return json.loads(mj.read_text())["per_class"] if mj.exists() else None


def report(families, sizes):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_results(families, sizes)
    if not results:
        print("No results found yet - run the training stage first.")
        return []
    figs = []

    # per-model train / validation graphs
    for r in results:
        if r["new"]:
            figs.append(plot_model_curves(r["family"], r["size"]))

    # comparison table (all models, sorted by mAP@50)
    results.sort(key=lambda r: r["map50"], reverse=True)
    headers = ["Rank", "Model", "Epochs", "Precision", "Recall", "mAP@50", "mAP@50-95"]
    rows = [[i + 1, r["model"], r["epochs"], f"{r['precision']:.4f}", f"{r['recall']:.4f}",
             f"{r['map50']:.4f}", f"{r['map50_95']:.4f}"] for i, r in enumerate(results)]
    print("\nModel comparison (final-epoch metrics, sorted by mAP@50)\n")
    print(format_table(headers, rows))
    with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    (OUT_DIR / "comparison.md").write_text(markdown_table(headers, rows) + "\n", encoding="utf-8")

    figs.append(plot_comparison(results))

    # Top-3 by mAP@50 and their per-class AP
    top3 = results[:3]
    print("\nTop-3 by mAP@50:", ", ".join(f"{r['model']} ({r['map50']:.4f})" for r in top3))
    pcs = {r["model"]: per_class_for(r) for r in top3}
    missing = [m for m, v in pcs.items() if v is None]
    if missing:
        print(f"  (no local weights for {', '.join(missing)} - per-class AP skipped for them)")
    shown = [r["model"] for r in top3 if pcs[r["model"]]]
    if shown:
        classes = list(next(iter(v for v in pcs.values() if v)).keys())
        for key, label in [("ap50", "AP@50"), ("ap50_95", "AP@50-95")]:
            headers = ["Class"] + shown
            rows = [[c] + [f"{pcs[m][c][key]:.4f}" if c in pcs[m] else "-" for m in shown] for c in classes]
            rows.append(["mean"] + [f"{sum(pcs[m][c][key] for c in pcs[m]) / len(pcs[m]):.4f}" for m in shown])
            print(f"\nPer-class {label} (best.pt weights)\n")
            print(format_table(headers, rows))
            suffix = "" if key == "ap50" else "_50_95"
            with open(OUT_DIR / f"top3_per_class_ap{suffix}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(headers); w.writerows(rows)
            (OUT_DIR / f"top3_per_class_ap{suffix}.md").write_text(markdown_table(headers, rows) + "\n", encoding="utf-8")

    print(f"\nTables and graphs saved in: {OUT_DIR}")
    return figs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "report", "all"], default="all")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=list(WEIGHT_PREFIX))
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    if args.stage in ("train", "all"):
        train_all(args.families, args.sizes, args.epochs)
    if args.stage in ("report", "all"):
        figs = report(args.families, args.sizes)
        if SHOW_PLOTS and figs:
            plt.show()


if __name__ == "__main__":      # required on Windows (dataloader workers)
    main()
