# Gradient Difference (GRAD_DIFF) Hyperparameter Tuning

## Fixed Parameters
- EPOCHS=1, MAX_LENGTH=512, BATCH_SIZE=4, STEPS=32
- Baseline NLL: Forget=2.1117, Retain=2.4375

## Results

| LR | Forget Weight | Forget NLL | Retain NLL | MMLU Accuracy | WMDP-Bio Robust Accuracy | L2 Weight Dist | Timestamp (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | 2.112 | 2.438 | 0.4499 | 0.4309 | - | - |
| 1e-05 | 1.0 | 2.486 | 2.345 | 0.4469 | 0.2704 | 2.87 | 2026-03-08 01:35 |
| 1e-05 | 5.0 | 148.233 | 24.813 | 0.3856 | 0.2557 | 5.42 | 2026-03-08 01:47 |
| 2e-05 | 1.0 | 171.024 | 5.894 | 0.3796 | 0.2446 | 8.21 | 2026-03-08 01:59 |
| 2e-05 | 5.0 | 174.817 | 8.687 | 0.2933 | 0.2343 | 8.86 | 2026-03-08 02:09 |
