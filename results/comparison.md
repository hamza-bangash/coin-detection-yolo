# Model Comparison

Metrics below use the final recorded row in each model's results.csv.
Values are reported as recorded; higher precision, recall, and mAP generally indicate better detection performance.

| Model | Epochs recorded | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | --- | --- | --- | --- | --- |
| yolo8_n | 100 | 0.9393 | 0.9036 | 0.9467 | 0.6674 |
| yolo8_s | 100 | 0.9391 | 0.9004 | 0.9444 | 0.6642 |
| yolo8_m | 100 | 0.9260 | 0.9041 | 0.9349 | 0.6495 |
| yolo8_l | 100 | 0.9198 | 0.9095 | 0.9262 | 0.6427 |
| yolo10_n | 95 | 0.9274 | 0.9029 | 0.9398 | 0.6640 |
| yolo10_s | 77 | 0.9251 | 0.9019 | 0.9387 | 0.6629 |
| yolo10_m | 61 | 0.9384 | 0.8998 | 0.9329 | 0.6646 |
| yolo10_l | 58 | 0.9143 | 0.9087 | 0.9375 | 0.6695 |
| yolo11_n | 100 | 0.9244 | 0.9131 | 0.9495 | 0.6669 |
| yolo11_s | 84 | 0.9286 | 0.9124 | 0.9491 | 0.6704 |
| yolo11_m | 73 | 0.9385 | 0.9043 | 0.9438 | 0.6676 |
| yolo11_l | 68 | 0.9204 | 0.9146 | 0.9431 | 0.6691 |
| yolo26_n | 100 | 0.9300 | 0.9098 | 0.9366 | 0.6527 |
| yolo26_s | 100 | 0.9322 | 0.9017 | 0.9399 | 0.6597 |
| yolo26_m | 100 | 0.9376 | 0.8996 | 0.9317 | 0.6530 |
| yolo26_l | 100 | 0.9348 | 0.9025 | 0.9326 | 0.6592 |

Models are listed by family. YOLOv8 and YOLO26 were trained for the full 100 epochs with
imgsz=448, batch=16, seed=0 (`scripts/train_compare.py`).
