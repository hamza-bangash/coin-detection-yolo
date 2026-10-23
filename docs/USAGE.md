
# Installation and Inference

## Requirements

- Python 3.10 or a compatible Python version supported by the selected Ultralytics release
- pip
- A trained YOLO model weights file (`.pt`)

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/coin-detection-yolo.git
cd coin-detection-yolo
```

Replace `YOUR_USERNAME` with the GitHub account that owns the repository.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Obtain model weights

Trained weights are not included in this repository.

Use a model weights file that you are authorized to access. Keep it outside the repository, for example in a directory named `models/` in your home directory.

## 5. Run inference on one image

```bash
python src/predict.py \
  --weights ~/models/coin-model.pt \
  --source examples/images/0003649.jpg
```

Replace the weights path with the actual location of your model file.

## 6. Run inference on all example images

```bash
python src/predict.py \
  --weights ~/models/coin-model.pt \
  --source examples/images/
```

## 7. Set a confidence threshold

The default confidence threshold is `0.25`. To change it, for example to `0.40`, use:

```bash
python src/predict.py \
  --weights ~/models/coin-model.pt \
  --source examples/images/0003649.jpg \
  --conf 0.40
```

## Output

Prediction images are saved under:

`runs/coin-predictions/results/`

The output directory may be created when inference runs.
