"""
Pascal VOC → YOLO Format Converter for Coin Dataset
=====================================================
Structure expected:
  dataset/
    1/
      images/   (*.jpg, *.jpeg, *.png)
      xml/      (*.xml  — one per image, same stem)
    2/
      ...
    99/
      ...

Output (mirrors the same subfolder layout inside --output_dir):
  yolo_dataset/
    1/
      images/
      labels/    (*.txt  YOLO annotations)
    ...
  classes.txt   (one class name per line, ordered by class index)

Label normalisation
-------------------
The scraper introduced noisy labels.  The mapping below collapses every
known-bad variant to the canonical 20-class set:
  I A L C R S T V O M E N F P B X D Q G H
"""

import os
import xml.etree.ElementTree as ET
import argparse
import shutil
from pathlib import Path

# ── Canonical classes (order defines the YOLO class index) ─────────────────
CLASSES = ["I", "A", "L", "C", "R", "S", "T", "V", "O", "M",
           "E", "N", "F", "P", "B", "X", "D", "Q", "G", "H"]

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# ── Noise → canonical mapping ───────────────────────────────────────────────
# Handles: trailing punctuation, brackets, lowercase equivalents, doubled chars
RAW_NOISE_MAP = {
    # punctuation / bracket variants
    "M.": "M",
    "M]": "M",
    # lowercase → uppercase (straight mapping for letters present in CLASSES)
    "c": "C",
    "d": "D",
    "f": "F",
    "h": "H",
    "i": "I",
    "l": "L",
    "o": "O",
    "s": "S",
    "x": "X",
    # doubled / garbled
    "ww": None,   # garbage noise – dropped
    # period-only or empty strings
    ".": None,
}


def normalise_label(raw: str) -> str | None:
    """Return canonical label string, or None if the annotation should be dropped."""
    label = raw.strip()
    if label in CLASS_INDEX:           # already canonical
        return label
    if label in RAW_NOISE_MAP:
        return RAW_NOISE_MAP[label]    # may be None (drop)
    # Try simple uppercase as a last resort
    up = label.upper()
    if up in CLASS_INDEX:
        return up
    return None                        # unknown → drop with a warning


def convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """Pascal VOC pixel bbox → YOLO normalised (cx, cy, w, h)."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh


def process_xml(xml_path: Path, img_w: int, img_h: int):
    """
    Parse a single XML annotation file.
    Returns list of (class_index, cx, cy, bw, bh) tuples and a list of warnings.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    warnings = []

    # Allow the XML to carry its own size (overrides caller-supplied dims if present)
    size_el = root.find("size")
    if size_el is not None:
        try:
            img_w = int(size_el.findtext("width", default=str(img_w)))
            img_h = int(size_el.findtext("height", default=str(img_h)))
        except ValueError:
            pass

    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="").strip()
        canonical = normalise_label(raw_name)

        if canonical is None:
            warnings.append(f"  DROPPED unknown label '{raw_name}' in {xml_path.name}")
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            warnings.append(f"  SKIPPED object '{raw_name}' (no bndbox) in {xml_path.name}")
            continue

        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            warnings.append(f"  SKIPPED object '{raw_name}' (bad bbox coords) in {xml_path.name}")
            continue

        cls_idx = CLASS_INDEX[canonical]
        cx, cy, bw, bh = convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        annotations.append((cls_idx, cx, cy, bw, bh))

    return annotations, warnings


def convert_dataset(input_dir: str, output_dir: str, copy_images: bool = True,
                    default_img_size: int = 448, verbose: bool = False):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_annotations = 0
    total_dropped = 0
    all_warnings = []

    # Iterate subfolders named 1..99 (or any subfolder that exists)
    subfolders = sorted(
        [d for d in input_root.iterdir() if d.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else 0
    )

    for subfolder in subfolders:
        img_dir = subfolder / "images"
        xml_dir = subfolder / "xml"

        if not img_dir.exists() or not xml_dir.exists():
            if verbose:
                print(f"[SKIP] {subfolder.name}: missing images/ or xml/ subfolder")
            continue

        out_img_dir = output_root / subfolder.name / "images"
        out_lbl_dir = output_root / subfolder.name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [f for f in img_dir.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )

        for img_path in image_files:
            xml_path = xml_dir / (img_path.stem + ".xml")
            if not xml_path.exists():
                all_warnings.append(f"  NO XML for image {img_path}")
                continue

            annotations, warnings = process_xml(xml_path, default_img_size, default_img_size)
            all_warnings.extend(warnings)
            total_dropped += sum(1 for w in warnings if "DROPPED" in w)

            # Write YOLO label file (even if empty – keeps image in dataset)
            label_path = out_lbl_dir / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for cls_idx, cx, cy, bw, bh in annotations:
                    f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            total_annotations += len(annotations)

            # Optionally copy image
            if copy_images:
                shutil.copy2(img_path, out_img_dir / img_path.name)

            total_images += 1

        if verbose:
            print(f"[OK]  {subfolder.name}: {len(image_files)} images processed")

    # Write classes.txt at dataset root
    classes_file = output_root / "classes.txt"
    with open(classes_file, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")

    # Summary
    print("\n" + "=" * 55)
    print("  CONVERSION COMPLETE")
    print("=" * 55)
    print(f"  Images processed   : {total_images}")
    print(f"  Annotations written: {total_annotations}")
    print(f"  Annotations dropped: {total_dropped}  (unknown/noisy labels)")
    print(f"  Output directory   : {output_root.resolve()}")
    print(f"  classes.txt        : {classes_file.resolve()}")
    print("=" * 55)

    if all_warnings:
        print(f"\n  {len(all_warnings)} warnings (see below or enable --verbose):")
        for w in all_warnings[:20]:
            print(w)
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more.")

    print("\nClass index mapping:")
    for i, cls in enumerate(CLASSES):
        print(f"  {i:2d}  {cls}")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC coin dataset → YOLO format"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Root folder of the coin dataset (contains subfolders 1-99)"
    )
    parser.add_argument(
        "--output_dir", default="yolo_dataset",
        help="Destination folder for YOLO output (default: ./yolo_dataset)"
    )
    parser.add_argument(
        "--no_copy_images", action="store_true",
        help="Skip copying images (only write .txt labels)"
    )
    parser.add_argument(
        "--img_size", type=int, default=448,
        help="Fallback image size if not in XML (default: 448)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-subfolder progress"
    )

    args = parser.parse_args()
    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_images=not args.no_copy_images,
        default_img_size=args.img_size,
        verbose=args.verbose,
    )