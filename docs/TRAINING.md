# Training

## Recipe

The configuration in `configs/bifusion_yolo.yaml` follows Table 5 of the
manuscript.

| Item | Value |
|---|---|
| Input | 640 x 640 RGB |
| Epochs | 300 |
| Batch size | 16 |
| Optimizer | AdamW (beta1 0.9, beta2 0.999) |
| Initial learning rate | 1.0e-3 |
| Final learning-rate factor | 0.01 |
| Weight decay | 5.0e-4 |
| Schedule | cosine annealing |
| Warm-up | 3 epochs, linear |
| EMA decay | 0.9999 |
| Mixed precision | FP16 |
| Mosaic | 1.0, disabled for the last 10 epochs |
| HSV | h 0.015, s 0.7, v 0.4 |
| Horizontal flip | 0.5 |
| Scale jitter | 0.5 |
| Translation | 0.1 |
| Copy-paste | 0.1 |
| Loss weights | 7.5 / 0.5 / 1.5 |
| Assigner | TAL (alpha 1.0, beta 6.0, k 10) |
| DFL bins | 16 |
| NMS IoU (val/test) | 0.70 |
| AP confidence | 0.001 |
| Max detections | 300 |

## Loss

The objective is the one in Eq. (35):

```
L = 7.5 * L_CIoU + 0.5 * L_BCE + 1.5 * L_DFL
```

There is no objectness branch. Classification is binary cross-entropy with
logits over the assigned soft targets. The box term is CIoU on the decoded
boxes. The distribution focal term supervises the four discretized sides with
16 bins. Losses are normalized by the number of assigned positives; the
classification term is normalized by the sum of the target scores.

Assignment is task-aligned:

```
t = score^1.0 * IoU^6.0
```

The top 10 candidates per ground-truth box become positives, and a candidate
claimed by several boxes keeps the highest alignment.

## Augmentation

Training uses mosaic with probability 1.0, HSV jitter, horizontal flip, random
scale and translation, and small-object copy-paste. Mosaic is switched off for
the last ten epochs. Validation and test images are letterboxed to 640 x 640
and nothing else. No test-time augmentation is used.

The mosaic implementation combines four images into a 2 x 2 grid, then applies
the affine, flip and colour transforms. Copy-paste copies crops of targets
narrower than 32 pixels and adds the corresponding box.

## Data format

```
datasets/DUT/
  images/train/  labels/train/
  images/val/    labels/val/
  images/test/   labels/test/
```

Each label file has one line per box:

```
class x_center y_center width height
```

Coordinates are normalized to the image size. The single class is index 0. The
dataset loader pairs `images` and `labels` by replacing the last `images`
component of the path, so the directory names must keep that structure.

## Run

```bash
python -m bifusion.train \
  --config configs/bifusion_yolo.yaml \
  --data configs/dut.yaml \
  --device 0
```

Useful options:

- `--weights path.pt` initializes from a checkpoint.
- `--resume runs/train/bifusion_yolo/last.pt` resumes a run.
- `--epochs`, `--batch`, `--imgsz`, `--workers`, `--seed`, `--deterministic`
  override the configuration.
- `--project` and `--name` choose the output directory.

Outputs:

```
runs/train/bifusion_yolo/
  best.pt        EMA weights at the best mAP@0.5:0.95
  last.pt        last epoch, with optimizer and scheduler state
  results.csv    one row per epoch
```

## Checkpoints

`best.pt` and `last.pt` are dictionaries with `model`, `ema`, `optimizer`,
`scheduler`, `epoch`, `best`, `config` and `data`. The evaluation and export
scripts read `config` to rebuild the model, so a checkpoint is self-contained.

## Notes

- The training model keeps the RepVGG branches and batch normalization. Call
  `model.deploy()` before counting parameters or exporting.
- Mixed precision is enabled only on CUDA. On CPU the scaler is disabled.
- The learning rate is applied per parameter group; biases and normalization
  parameters use zero weight decay.
