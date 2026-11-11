# Results

Machine-readable copies of the tables in the manuscript. The values are the
reported results; they are not recomputed by this repository.

| File | Table |
|---|---|
| `headline.json` | Headline metrics, complexity, speed and the main trade-off |
| `main_comparison.csv` | Comparison with generic detectors on the DUT test split |
| `ablation.csv` | Cumulative ablation |
| `size_ap.csv` | Width-stratified mAP@0.5:0.95 |
| `efficiency.csv` | FP32, FP16 and INT8 on RTX 3090 and Jetson Orin |
| `transfer.csv` | Cross-dataset transfer and image degradations |
| `errors.csv` | Error decomposition at the maximum-F1 operating point |

The same numbers are rendered in [../docs/RESULTS.md](../../docs/RESULTS.md).

The reported model has 1.78M parameters and 7.4 GFLOPs at 640 x 640 input and
reaches 97.2% precision, 90.3% recall, 95.6% mAP@0.5 and 66.4% mAP@0.5:0.95 on
the DUT Anti-UAV test split.
