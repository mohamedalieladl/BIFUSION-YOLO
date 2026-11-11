# Reproducibility

## Scope

This repository contains the model, the training and evaluation code, the
export path, the configuration files and the tests. It does not contain the DUT
images, the trained weights or the per-run logs. The reported numbers are the
ones in the manuscript; they are not recomputed by this repository.

## Environment

| Item | Value |
|---|---|
| Python | 3.9 or newer |
| PyTorch | 2.1 or newer |
| CUDA | 12.1 in the paper |
| Framework | plain PyTorch, no Ultralytics dependency |
| Export | ONNX opset 17, TensorRT 10 |

## Data

The main benchmark is the detection split of DUT Anti-UAV: 10,000 visible-light
images and 10,109 annotated UAVs in one class. The official split is 5,200
training, 2,600 validation and 2,200 test images.

The cross-dataset experiments use Anti-UAV, Drone-vs-Bird and Det-Fly. All three
are reduced to a single UAV class, with birds treated as distractors. The 10%
adaptation subset is drawn from each target training split, and the target test
split is used for evaluation only.

## Training protocol

The recipe is in `configs/bifusion_yolo.yaml` and is described in
[TRAINING.md](TRAINING.md). The same recipe is used for every detector that is
retrained, except for a small learning-rate search over
`{0.5, 1.0, 2.0} x 10^-3` on the validation set. Pretrained COCO weights are
used where they exist.

The random seed is fixed per run. The ten-seed analysis uses ten independent
runs with different seeds; the seed list is part of the training configuration
that accompanies a release.

## Evaluation protocol

- Images are letterboxed to 640 x 640; no test-time augmentation.
- Detections are matched greedily in descending score order at the stated IoU.
- AP uses the 101-point COCO interpolation.
- mAP@0.5:0.95 averages AP over IoU 0.50 to 0.95 in steps of 0.05.
- NMS IoU is 0.70, the AP confidence threshold is 0.001, and at most 300
  detections are kept per image.
- Precision, recall and F1 are reported at the confidence that maximises F1 on
  the evaluated split, frozen for the test split.
- Width bins are computed from the annotated box width, not from box area.
  They are not the COCO area categories.

## Reproducing the tables

| Table | Source |
|---|---|
| Main comparison | `assets/results/main_comparison.csv` |
| Cumulative ablation | `assets/results/ablation.csv` |
| Width-stratified AP | `assets/results/size_ap.csv` |
| Efficiency and quantization | `assets/results/efficiency.csv` |
| Transfer and robustness | `assets/results/transfer.csv` |
| Ten-seed statistics | `docs/RESULTS.md` |
| Error decomposition | `assets/results/errors.csv` |

To regenerate a table from a trained checkpoint:

```bash
python -m bifusion.val --checkpoint runs/train/bifusion_yolo/best.pt --split test --json runs/test_metrics.json
```

The JSON contains precision, recall, F1, mAP@0.5, mAP@0.5:0.95 and the six
width bins, in the same layout as `Metrics.compute()`.

## Parameter count

```bash
python tools/param_budget.py
python tools/count_params.py --check
```

The first command needs no PyTorch and checks the arithmetic of Table 3. The
second builds the model, folds batch normalization and the RepVGG branches, and
compares the deployed count with 1,783,110.

## Statistical protocol

Each configuration is trained ten times with different seeds. One paired
observation is one seed: run `s` of the model and run `s` of the baseline share
the seed, the split, the augmentation and the evaluation script. The null
hypothesis is a zero mean paired difference. A Shapiro-Wilk test is applied to
the ten paired differences; if normality is not rejected, a two-sided paired
t-test is used, otherwise the Wilcoxon signed-rank test. The effect size is
Cohen's d_z = mean(d) / sd(d). Confidence intervals are percentile intervals
from bootstrapping the ten paired differences 10,000 times. Significance is
claimed at alpha 0.05 after Holm-Bonferroni correction across the four metrics.

## Known gaps

- The training logs and checkpoints are not part of this repository.
- The exact 10% few-shot sampling and the per-run seeds are recorded in the
  configuration files that accompany the release, not here.
- The INT8 calibration set is described in the paper (500 training images,
  per-channel weight scales); the calibration script is not included.
- The TensorRT engines are hardware-specific and are not distributed.

## License

The code license will be finalized before public release. Until then, all
rights are reserved by the authors.
