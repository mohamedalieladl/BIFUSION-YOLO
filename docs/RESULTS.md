# Results

All values are the ones reported in the manuscript. They are reproduced here
for convenience; the machine-readable copies are in `assets/results/`.

## Main comparison

DUT Anti-UAV test split, 640 x 640 input.

| Method | Params (M) | GFLOPs | FPS | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---|---|---|---|---|---|---|
| Faster R-CNN (R50-FPN) | 41.50 | 207.0 | 24 | 85.1 | 79.4 | 87.1 | 57.2 |
| SSD300 | 26.30 | 35.2 | 42 | 84.3 | 78.9 | 82.3 | 51.4 |
| YOLOv3-tiny | 8.70 | 12.9 | 105 | 88.2 | 80.1 | 88.7 | 55.1 |
| YOLOv5n | 1.87 | 4.5 | 110 | 90.1 | 82.4 | 89.5 | 57.9 |
| YOLOv6n | 4.23 | 11.8 | 95 | 90.4 | 82.8 | 89.4 | 58.0 |
| YOLOv7-tiny | 6.23 | 13.2 | 88 | 90.9 | 83.2 | 90.8 | 58.6 |
| YOLOv8n | 3.01 | 8.1 | 102 | 90.6 | 82.9 | 90.4 | 58.4 |
| YOLOv9t | 1.73 | 7.6 | 93 | 90.2 | 82.5 | 90.0 | 58.2 |
| YOLOv10n | 2.69 | 8.7 | 108 | 90.8 | 83.1 | 90.7 | 58.9 |
| YOLO26n | 2.58 | 8.2 | 96 | 92.9 | 85.3 | 91.1 | 60.8 |
| YOLOv12n | 2.63 | 6.5 | 88 | 93.1 | 85.7 | 92.0 | 61.5 |
| RT-DETR-R18 | 20.00 | 60.0 | 45 | 91.8 | 84.2 | 91.7 | 61.2 |
| Gao et al. (quoted) | 1.59 | 6.8 | 91 | 96.1 | 89.0 | 95.2 | 64.9 |
| **BiFusion-YOLO** | **1.78** | **7.4** | **89** | **97.2** | **90.3** | **95.6** | **66.4** |

The Gao et al. row is quoted because their code and weights were not available;
it is excluded from the paired statistical tests. FPS is TensorRT FP32 on an
RTX 3090, batch one, model inference only.

## Width-stratified mAP@0.5:0.95

| Width bin | Instances | YOLO26n | BiFusion-YOLO | Delta |
|---|---|---|---|---|
| < 10 px | 2,427 | 38.5 | 54.1 | +15.6 |
| 10-20 px | 3,140 | 53.2 | 63.5 | +10.3 |
| 20-32 px | 1,921 | 65.0 | 69.7 | +4.7 |
| 32-64 px | 1,566 | 85.0 | 78.5 | -6.5 |
| 64-96 px | 617 | 90.5 | 79.5 | -11.0 |
| > 96 px | 438 | 91.5 | 78.5 | -13.0 |
| Narrow, < 32 px | 7,488 | 51.5 | 62.0 | +10.6 |
| Mid, 32-96 px | 2,183 | 86.6 | 78.8 | -7.8 |
| Wide, > 96 px | 438 | 91.5 | 78.5 | -13.0 |
| Overall | 10,109 | 60.8 | 66.4 | +5.6 |

These are width groups, not COCO area categories.

## Cumulative ablation

| Stage | Added component | mAP@0.5 | mAP@0.5:0.95 | Delta |
|---|---|---|---|---|
| 0 | Baseline YOLO26n | 91.1 | 60.8 | - |
| 1 | Ghost convolution | 91.6 | 61.3 | +0.5 / +0.5 |
| 2 | RepVGG blocks | 92.2 | 61.9 | +0.6 / +0.6 |
| 3 | Coordinate Attention | 93.2 | 62.9 | +1.0 / +1.0 |
| 4 | Haar wavelet downsampling | 94.1 | 63.8 | +0.9 / +0.9 |
| 5 | Weighted BiFPN | 94.7 | 64.4 | +0.6 / +0.6 |
| 6 | CCFM blocks | 95.0 | 64.9 | +0.3 / +0.5 |
| 7 | P2 head replaces P5, P4 promoted | 95.6 | 66.4 | +0.6 / +1.5 |

The sequence is cumulative. A delta is the change associated with one addition
in this order, not a standalone effect of that component.

## Efficiency and quantization

| Platform | Method | Precision | Size (MB) | FPS | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---|---|---|---|---|---|
| RTX 3090 | YOLO26n | FP32 | 10.3 | 96 | 91.1 | 60.8 |
| RTX 3090 | YOLO26n | FP16 | 5.2 | 158 | 91.1 | 60.8 |
| RTX 3090 | YOLO26n | INT8 | 2.7 | 244 | 90.7 | 60.2 |
| RTX 3090 | BiFusion-YOLO | FP32 | 7.1 | 89 | 95.6 | 66.4 |
| RTX 3090 | BiFusion-YOLO | FP16 | 3.6 | 143 | 95.6 | 66.3 |
| RTX 3090 | BiFusion-YOLO | INT8 | 1.8 | 221 | 95.2 | 65.7 |
| Jetson Orin | YOLO26n | FP32 | 10.3 | 24 | 91.1 | 60.8 |
| Jetson Orin | YOLO26n | FP16 | 5.2 | 38 | 91.1 | 60.8 |
| Jetson Orin | YOLO26n | INT8 | 2.7 | 58 | 90.7 | 60.2 |
| Jetson Orin | BiFusion-YOLO | FP32 | 7.1 | 21 | 95.6 | 66.4 |
| Jetson Orin | BiFusion-YOLO | FP16 | 3.6 | 34 | 95.6 | 66.3 |
| Jetson Orin | BiFusion-YOLO | INT8 | 1.8 | 52 | 95.2 | 65.7 |

Latency is the median of 1,000 TensorRT 10 iterations at batch one and
640 x 640 input, after 100 warm-up iterations. The Jetson Orin runs in its
15 W mode. The numbers cover model inference only and exclude NMS, decoding and
host-to-device transfer.

## Cross-dataset transfer

mAP@0.5 (%). Direct transfer uses only DUT training data; +10% adds 50 epochs
on 10% of the target training split.

| Training data | Test set | YOLO26n | BiFusion-YOLO | Delta |
|---|---|---|---|---|
| DUT (in-domain) | DUT | 91.1 | 95.6 | +4.5 |
| DUT (direct) | Anti-UAV | 61.4 | 68.9 | +7.5 |
| DUT (direct) | Drone-vs-Bird | 74.2 | 79.6 | +5.4 |
| DUT (direct) | Det-Fly | 58.3 | 65.1 | +6.8 |
| DUT + 10% | Anti-UAV | 78.2 | 84.7 | +6.5 |
| DUT + 10% | Drone-vs-Bird | 85.1 | 89.3 | +4.2 |
| DUT + 10% | Det-Fly | 73.4 | 80.2 | +6.8 |

## Robustness

mAP@0.5 (%). All degradations are synthetic and applied at inference only.

| Degradation | Severity | YOLO26n | BiFusion-YOLO | Delta |
|---|---|---|---|---|
| Clean | - | 91.1 | 95.6 | +4.5 |
| Fog | medium | 86.4 | 91.8 | +5.4 |
| Motion blur | kernel 9 | 83.1 | 89.7 | +6.6 |
| Low light | x0.3 | 82.0 | 88.4 | +6.4 |
| JPEG | quality 30 | 88.9 | 93.2 | +4.3 |
| Downscale | 2x | 84.2 | 90.1 | +5.9 |
| Gaussian noise | sigma 25 | 85.6 | 91.0 | +5.4 |

## Ten-seed statistics

| Metric | YOLO26n | BiFusion-YOLO | Delta | 95% CI | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| P (%) | 92.9 +/- 0.31 | 97.2 +/- 0.22 | +4.3 | [3.9, 4.7] | < 0.001 | 0.002 |
| R (%) | 85.3 +/- 0.42 | 90.3 +/- 0.35 | +5.0 | [4.5, 5.5] | < 0.001 | 0.002 |
| mAP@0.5 (%) | 91.1 +/- 0.24 | 95.6 +/- 0.18 | +4.5 | [4.1, 4.9] | < 0.001 | 0.002 |
| mAP@0.5:0.95 (%) | 60.8 +/- 0.29 | 66.4 +/- 0.21 | +5.6 | [5.2, 6.0] | < 0.001 | 0.002 |

## Error decomposition

At the maximum-F1 operating point on the test split.

| Category | FN | % of FN | FP | % of FP |
|---|---|---|---|---|
| Tiny target (< 10 px) | 139 | 63.8 | 0 | 0.0 |
| Motion blur | 41 | 18.8 | 0 | 0.0 |
| Occlusion | 22 | 10.1 | 0 | 0.0 |
| Background clutter | 0 | 0.0 | 33 | 56.9 |
| Duplicate detection | 0 | 0.0 | 14 | 24.1 |
| Other | 16 | 7.3 | 11 | 19.0 |
| Total | 218 | 100.0 | 58 | 100.0 |

TP = 2,027, FP = 58, FN = 218, total errors = 276. These counts reproduce
P = 97.2% and R = 90.3%.
