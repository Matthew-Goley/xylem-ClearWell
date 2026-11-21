# Model 1 results: water detection on Sen1Floods11

We measured pooled water IoU (water pixels only, TP, FP and FN summed over all valid pixels of a group) of the U-Net checkpoint from epoch 44 against the Otsu threshold baseline on Sen1Floods11 hand-labeled tiles: the 90-tile test split, its 14-tile India subset and the 15-tile Bolivia holdout, water threshold 0.5 on the predicted probability.

| Group | Tiles | U-Net pooled water IoU | U-Net precision | U-Net recall | Otsu pooled water IoU | Otsu precision | Otsu recall |
|---|---|---|---|---|---|---|---|
| All test | 90 | 0.663 | 0.839 | 0.760 | 0.546 | 0.681 | 0.733 |
| India test subset | 14 | 0.663 | 0.858 | 0.745 | 0.704 | 0.880 | 0.779 |
| Bolivia holdout | 15 | 0.661 | 0.801 | 0.791 | 0.697 | 0.781 | 0.867 |

Valid pixels: hand label is 0 or 1 and the Otsu label is defined. Using the hand label alone gives the same values to 3 decimals.

The same U-Net predictions under three definitions, for comparison with other papers:

| Group | Pooled water IoU | 2-class mean IoU (water, not water) | Per-tile mean water IoU |
|---|---|---|---|
| All test | 0.663 | 0.805 | 0.408 |
| India test subset | 0.663 | 0.784 | 0.423 |
| Bolivia holdout | 0.661 | 0.794 | 0.440 |

Per-tile mean leaves out tiles where the IoU is 0/0 (no water in the label and none predicted: 3 of 90 test tiles). It is low because tiles with little water score near 0 (results/figures/fig4_tile_iou_vs_water_fraction.png).

## Caveats

- The test split has been scored 3 times in total: 1 on the epoch-7 checkpoint, 2 on the epoch-44 checkpoint. The epoch-7 run (results/model1_metrics_run1.json) gave pooled water IoU 0.657 on all test tiles, 0.660 on the India subset and 0.533 on Bolivia. It ran at 2026-11-14 17:04:24 (outputs/logs/test.log), before the learning-rate sweep started at 2026-11-14 17:10:21 (outputs/logs/sweep.log). Figures 3 and 4 recomputed per-tile predictions from the epoch-44 checkpoint for plotting only.
- Otsu scores higher than the U-Net on the India test subset (n=14): pooled water IoU 0.704 vs 0.663.
- Otsu scores higher than the U-Net on the Bolivia holdout (n=15): pooled water IoU 0.697 vs 0.661.
- The U-Net scores higher only on all test tiles pooled: 0.663 vs 0.546.
- Hyperparameters (learning rates, schedule, checkpoint epoch) were chosen on the validation split only (results/model1_sweep.md, outputs/train_log.csv).
- Checkpoint: outputs/unet.pt, SHA256 b42facd16f14b86eba9c39a4b128a386f9d2510ec4447ef4b4b876eace2e1a46.
