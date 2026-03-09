# Gradient Ascent (GA) Hyperparameter Tuning

## Fixed Parameters
- EPOCHS=1, MAX_LENGTH=512, BATCH_SIZE=4, STEPS=32
- Baseline NLL: Forget=2.1117, Retain=2.4375

## Results

| LR | Retain Weight | Forget NLL | Retain NLL | MMLU Accuracy | WMDP-Bio Robust Accuracy | L2 Weight Dist | Timestamp (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | 2.112 | 2.438 | 0.4499 | 0.4309 | - | - |
| 1e-05 | 1.0 | 2.486 | 2.345 | 0.4464 | 0.2672 | 2.87 | 2026-03-08 00:36 |
| 1e-05 | 5.0 | 2.145 | 2.307 | 0.4509 | 0.2639 | 2.39 | 2026-03-08 00:48 |
| 2e-05 | 1.0 | 171.006 | 5.871 | 0.3791 | 0.2470 | 8.21 | 2026-03-08 00:59 |
| 2e-05 | 5.0 | 2.451 | 2.242 | 0.4505 | 0.2606 | 5.79 | 2026-03-08 01:11 |
| 5e-05 | 1.0 | 170.419 | 3.731 | 0.3435 | 0.2437 | 17.51 | 2026-03-08 01:23 |
