# CB / CB_LAT Hyperparameter Sweep Summary

## Goal
Find hyperparameters that maximize internal weight change (measured by L2 weight distance from base model, with per-layer breakdown) while preserving MMLU (~0.45) and keeping WMDP Bio Robust near random (~0.25).

## Baseline Model (EleutherAI/deep-ignorance-unfiltered)
- MMLU: 0.4499
- WMDP Bio Robust: 0.4309
- Forget NLL: 2.112
- Retain NLL: 2.438

## Measurement Uncertainty

**Benchmark accuracy (binomial standard error):**

| Benchmark | N questions | Typical acc | Stderr (1σ) | 95% CI (±2σ) |
|---|---|---|---|---|
| MMLU | ~14,042 | 0.45 | ±0.004 | ±0.008 |
| WMDP Bio Robust | ~1,299 | 0.26 | ±0.012 | ±0.024 |
| MMLU - WMDP gap | — | 0.18 | ±0.013 | ±0.025 |

Differences in MMLU smaller than ~0.008 or WMDP smaller than ~0.024 are within noise. The MMLU-WMDP gap differences across configs (~0.01) are statistically indistinguishable.

**L2 weight distance:** Deterministic given a trained model (no sampling uncertainty). Run-to-run variance from training stochasticity appears small — near-identical objectives (GA LR=1e-05/RW=1 vs grad_diff LR=1e-05/FW=1) produce L2=2.87 in both cases.

**NLL:** Deterministic given a model and eval dataset. No statistical uncertainty beyond floating-point precision.

## Phase 0: Exploration & Bug Fix

**CB sweep (prior session):** Started with hyperparameters suggested by Lucia (EPOCHS=1, 32 steps), then swept Alpha (100, 200, 500) x Steering Coeff (5, 10, 15) at LR=1e-05, layers 5-6-7. All 9 runs hit State 4 (MMLU ~0.45, WMDP ~0.26) with identical losses. Initially this looked like a good result, so we moved on to CB_LAT to compare.

**CB_LAT sweep:** Attempted the same 9-run grid. All crashed with a multi-GPU device placement bug (`cuda:1 vs cuda:0`). Fixed `lat_loss` and `cb_lat_loss` to handle `device_map="auto"`. Re-ran all 9 -- results identical to CB.

**Key realization:** Although benchmark metrics looked good, the forget/retain NLL was unchanged from baseline and identical across all 18 runs. This indicated the weights had barely changed -- the unlearning was superficial (representation rerouting at layers 5-6-7 without meaningful weight modification). This motivated deeper exploration to find configs that actually change the model's internals.

## Phase 1: Finding the LR Boundary (Layers 5-6-7)

**Rationale:** Push LR higher to find where weights actually change.

**LR sweep (5e-05, 1e-04, 2e-04, 5e-04, plus BS=2 variant):** All hit State 1 -- both MMLU and WMDP collapsed to random, NLL exploded (forget: 8-31, retain: 7-26).

**LR boundary (2e-05, 3e-05):** Still State 1. Sharp cliff between LR=1e-05 (State 4) and LR=2e-05 (State 1) at layers 5-6-7.

**Interpretation:** Early layers are too sensitive -- any real weight change destroys general capability.

## Phase 2: Layer Sweep

**Rationale:** Test whether different layer targets allow more weight change without destroying the model. Run at LR=2e-05 (which destroyed layers 5-6-7) to see which layers tolerate it.

| Layers | MMLU | WMDP | Forget NLL | Retain NLL | State |
|---|---|---|---|---|---|
| 5,6,7 (early) | 0.237 | 0.263 | 8.68 | 7.34 | 1 |
| **13,14,15 (mid)** | **0.329** | **0.262** | **5.20** | **5.30** | **3** |
| 21,22,23 (late) | 0.289 | 0.264 | 20.13 | 16.01 | 1 |

**Interpretation:** Mid-layers (13-14-15) are the most selective -- they tolerate more weight change while preserving more MMLU. Factual knowledge is stored in middle layers, so rerouting there is more targeted.

**Baseline NLL eval:** Ran a forward-pass-only evaluation to establish ground truth -- forget=2.112, retain=2.438. Confirmed that LR=1e-05 runs barely moved from baseline (~0.04 shift).

## Phase 3: Tuning LR x BS at Layers 13-14-15

**Rationale:** Use larger batch sizes to stabilize training (smoother gradients = less collateral damage), buying headroom to push LR higher for maximum weight change.

| LR | BS | MMLU | WMDP | Forget NLL | Retain NLL |
|---|---|---|---|---|---|
| 1e-05 | 8 | 0.4445 | 0.2577 | 2.161 | 2.532 |
| 1.3e-05 | 8 | 0.4382 | 0.2524 | 2.191 | 2.695 |
| 1.3e-05 | 16 | 0.4425 | 0.2565 | 2.178 | 2.686 |
| 1.5e-05 | 4 | 0.3840 | 0.2573 | 2.504 | 4.414 |
| 1.5e-05 | 8 | 0.4207 | 0.2573 | 2.272 | 3.442 |
| 1.5e-05 | 16 | 0.4401 | 0.2618 | 2.192 | 2.767 |
| 1.5e-05 | 32 | 0.4422 | 0.2598 | 2.170 | 2.590 |
| 2e-05 | 8 | 0.3022 | 0.2704 | 4.639 | 5.965 |
| 2e-05 | 16 | 0.3700 | 0.2655 | 2.724 | 5.533 |
| 2e-05 | 32 | 0.4127 | 0.2659 | 2.332 | 4.074 |

**Key observation:** Larger BS preserves capability at the same LR (e.g. LR=1.5e-05: BS=4 MMLU=0.38, BS=8 MMLU=0.42, BS=32 MMLU=0.44).

## Phase 4: L2 Weight Distance Measurement

**Rationale:** Added per-layer L2 weight distance from base model to measure actual internal change, rather than using LR as a proxy. Re-ran top configs with L2 logging.

| LR | BS | MMLU | WMDP | Forget NLL | Retain NLL | L2 Dist |
|---|---|---|---|---|---|---|
| 1e-05 | 8 | 0.4448 | 0.2561 | 2.161 | 2.532 | 2.33 |
| **1.3e-05** | **16** | **0.4422** | **0.2565** | **2.179** | **2.686** | **2.69** |
| 1.5e-05 | 32 | 0.4423 | 0.2565 | 2.170 | 2.591 | 2.42 |
| 2e-05 | 32 | 0.4127 | 0.2659 | 2.332 | 4.074 | 3.81 |

**Key findings:**
1. Only layers 0-15 change; layers 16-31 have zero L2 distance. Gradients don't flow past the target layers.
2. Layer 13 consistently has the highest per-layer L2, confirming the target layer gets the most change.
3. LR=1.3e-05/BS=16 gives the most weight change (L2=2.69) while keeping MMLU>0.44. Larger BS smooths gradients, which preserves metrics but can actually reduce total weight change.
4. LR=2e-05/BS=32 has the highest L2 (3.81) but MMLU drops to 0.41.

## Phase 5: CB vs CB_LAT Comparison at Layers 13-14-15

**Rationale:** Test whether the LAT adversarial component adds anything at higher LRs where weights actually change.

| LR | BS | Method | MMLU | WMDP | Forget NLL | Retain NLL | L2 |
|---|---|---|---|---|---|---|---|
| 1e-05 | 8 | CB | 0.4448 | 0.2561 | 2.161 | 2.532 | 2.333 |
| 1e-05 | 8 | CB_LAT | 0.4487 | 0.2639 | 2.161 | 2.532 | 2.333 |
| 1.3e-05 | 16 | CB | 0.4422 | 0.2565 | 2.179 | 2.686 | 2.689 |
| 1.3e-05 | 16 | CB_LAT | 0.4430 | 0.2552 | 2.179 | 2.686 | 2.688 |
| 1.5e-05 | 32 | CB | 0.4423 | 0.2565 | 2.170 | 2.591 | 2.423 |
| 1.5e-05 | 32 | CB_LAT | 0.4437 | 0.2544 | 2.170 | 2.591 | 2.423 |

**Conclusion:** CB and CB_LAT produce identical results -- same L2, same NLL, same metrics within noise. The LAT adversarial perturbation adds no measurable effect even at higher LRs. CB alone is sufficient.

## Phase 6: Late Layers (21-22-23)

**Rationale:** Test whether late layers offer different weight change characteristics than mid-layers.

| LR | BS | Layers | MMLU | WMDP | Forget NLL | Retain NLL | L2 |
|---|---|---|---|---|---|---|---|
| 1e-05 | 8 | 21,22,23 | 0.4495 | 0.2696 | 2.218 | 2.618 | 2.352 |
| 1.3e-05 | 16 | 21,22,23 | ? (eval crashed) | ? | 2.289 | 2.870 | 2.633 |
| 1.5e-05 | 32 | 21,22,23 | OOM | - | - | - | - |

**Observation:** Late layers give similar L2 to mid-layers but spread changes across more layers (0-23 vs 0-15). LR=1e-05/BS=8 at late layers preserves MMLU slightly better (0.4495 vs 0.4448) but WMDP is higher (0.2696 vs 0.2561). BS=32 OOMs because gradients flow through more layers. Incomplete data due to crashes.

## Phase 7: Task Arithmetic Removal (TAR)

**Rationale:** TAR uses a fundamentally different mechanism -- fine-tune on forget data then subtract the learned direction. This directly targets weight change in the forget-data direction, potentially giving much larger L2 with preserved metrics.

**Initial sweep (8 configs, no L2):**

| tar_lr | alpha | MMLU | WMDP | State |
|---|---|---|---|---|
| 1e-05 | 0.5 | 0.4497 | 0.2667 | 4 |
| 1e-05 | 1.0 | 0.4487 | 0.2639 | 4 |
| 1e-05 | 2.0 | 0.4477 | 0.2684 | 4 |
| 1e-05 | 5.0 | 0.4295 | 0.2532 | 3-ish |
| 5e-05 | 1.0 | 0.4115 | 0.2573 | 3 |
| 5e-05 | 2.0 | 0.2695 | 0.2528 | 1 |
| 1e-04 | 1.0 | 0.2463 | 0.2376 | 1 |
| 1e-04 | 2.0 | 0.2484 | 0.2363 | 1 |

**Re-run with L2/NLL measurement (boundary exploration):**

| tar_lr | alpha | Forget NLL | Retain NLL | L2 Dist |
|---|---|---|---|---|
| 1e-05 | 2.0 | 2.297 | 2.586 | **4.49** |
| 1e-05 | 5.0 | 2.705 | 2.873 | **11.22** |
| 2e-05 | 1.0 | 2.258 | 2.565 | **5.56** |
| 2e-05 | 2.0 | 2.441 | 2.697 | **11.12** |
| 2e-05 | 5.0 | 3.466 | 3.403 | **27.77** |
| 3e-05 | 1.0 | 2.313 | 2.600 | **9.45** |
| 3e-05 | 2.0 | 2.627 | 2.816 | **18.89** |
| 5e-05 | 1.0 | 2.621 | 2.752 | **19.97** |

**MMLU/WMDP for re-runs (now extracted):**

| tar_lr | alpha | Forget NLL | Retain NLL | MMLU | WMDP | L2 Dist |
|---|---|---|---|---|---|---|
| 1e-05 | 2.0 | 2.297 | 2.586 | 0.4477 | 0.2655 | **4.49** |
| 1e-05 | 5.0 | 2.705 | 2.873 | 0.4304 | 0.2499 | **11.22** |
| 2e-05 | 1.0 | 2.258 | 2.565 | 0.4443 | 0.2709 | **5.56** |
| 2e-05 | 2.0 | 2.441 | 2.697 | 0.4313 | 0.2614 | **11.12** |
| 2e-05 | 5.0 | 3.466 | 3.403 | 0.3497 | 0.2552 | **27.77** |
| 3e-05 | 1.0 | 2.313 | 2.600 | 0.4385 | 0.2626 | **9.45** |
| 3e-05 | 2.0 | 2.627 | 2.816 | 0.4134 | 0.2594 | **18.89** |
| 5e-05 | 1.0 | 2.621 | 2.752 | 0.4118 | 0.2577 | **19.97** |

**Key findings:**
1. TAR gives **dramatically higher L2** than CB: 4.49-27.77 vs CB's best 2.69. At comparable NLL, TAR moves weights 2-4x more.
2. TAR changes weights **uniformly across all 32 layers**, unlike CB which only changes layers 0 to target layer. This means the weight change is distributed throughout the entire model.
3. The alpha parameter is a direct multiplier on weight change: alpha=2 roughly doubles L2 vs alpha=1. This gives fine-grained control.
4. TAR preserves MMLU remarkably well even at high L2: tar_lr=3e-05/alpha=1 gives L2=9.45 with MMLU=0.4385.
5. tar_lr=2e-05/alpha=1 (L2=5.56, MMLU=0.4443) gives 2x more weight change than CB's best while keeping MMLU>0.44.

## Phase 8: Gradient Ascent (GA)

**Rationale:** Test gradient-based unlearning -- gradient ascent on forget data combined with gradient descent on retain data. Unlike CB which reroutes representations at specific layers, GA directly modifies all model weights.

| LR | Retain Weight | Forget NLL | Retain NLL | MMLU | WMDP | L2 Dist |
|---|---|---|---|---|---|---|
| 1e-05 | 1.0 | 2.486 | 2.345 | 0.4464 | 0.2672 | 2.87 |
| 1e-05 | 5.0 | 2.145 | 2.307 | 0.4509 | 0.2639 | 2.39 |
| 2e-05 | 1.0 | 171.0 | 5.871 | 0.3791 | 0.2470 | 8.21 |
| **2e-05** | **5.0** | **2.451** | **2.242** | **0.4505** | **0.2606** | **5.79** |
| 5e-05 | 1.0 | 170.4 | 3.731 | 0.3435 | 0.2437 | 17.51 |

**Key findings:**
1. GA changes all 32 layers, with more change in later layers (opposite pattern to CB which changes early layers more).
2. GA LR=2e-05/RW=5.0 is the standout: L2=5.79, MMLU=0.4505, WMDP=0.2606, NLL close to baseline. Comparable to TAR tar_lr=2e-05/alpha=1 (L2=5.56, MMLU=0.4443).
3. Higher retain_weight (5.0 vs 1.0) at the same LR dramatically preserves metrics: at LR=2e-05, RW=1 destroys forget NLL (171) while RW=5 keeps it at 2.45.
4. At LR=1e-05, GA with either retain weight barely changes the model (L2~2.4-2.9) -- similar to CB at similar LR.

## Phase 9: Gradient Difference (GRAD_DIFF)

**Rationale:** Similar to GA but with explicit forget weight control. Gradient ascent on forget data (weighted by forget_weight) + gradient descent on retain data.

| LR | Forget Weight | Forget NLL | Retain NLL | MMLU | WMDP | L2 Dist |
|---|---|---|---|---|---|---|
| 1e-05 | 1.0 | 2.486 | 2.345 | 0.4469 | 0.2704 | 2.87 |
| 1e-05 | 5.0 | 148.2 | 24.81 | 0.3856 | 0.2557 | 5.42 |
| 2e-05 | 1.0 | 171.0 | 5.894 | 0.3796 | 0.2446 | 8.21 |
| 2e-05 | 5.0 | 174.8 | 8.687 | 0.2933 | 0.2343 | 8.86 |

**Key findings:**
1. grad_diff LR=1e-05/FW=1.0 is identical to GA LR=1e-05/RW=1.0 (same L2=2.87, same NLL) -- as expected, since FW=1/RW=1 reduces to the same objective.
2. Increasing forget_weight at LR=1e-05 from 1→5 destroys forget NLL (148!) while L2 only reaches 5.42. This is less efficient than GA with higher LR.
3. grad_diff is more aggressive than GA -- harder to find the sweet spot between no-change and destruction.

## Phase 10: Weight Distortion (WT_DIST)

**Rationale:** Add Gaussian noise to ALL weights, then fine-tune on retain data to recover general capability while keeping noise-induced forgetting.

| LR | Noise Std | Forget NLL | Retain NLL | MMLU | WMDP | L2 Dist |
|---|---|---|---|---|---|---|
| 1e-05 | 5e-05 | 2.135 | 2.305 | 0.4506 | 0.2655 | 4.74 |
| 2e-05 | 5e-05 | 2.141 | 2.233 | 0.4514 | 0.2626 | 7.01 |
| 1e-05 | 0.0001 | 2.135 | 2.305 | 0.4511 | 0.2577 | 9.01 |
| **2e-05** | **0.0001** | **2.140** | **2.234** | **0.4515** | **0.2659** | **10.39** |
| 1e-05 | 0.0005 | 2.136 | 2.306 | 0.4485 | 0.2630 | 41.63 |
| 2e-05 | 0.0005 | 2.141 | 2.234 | 0.4504 | 0.2594 | 41.95 |
| 1e-05 | 0.01 | 2.532 | 2.870 | 0.2498 | 0.2474 | 828 |
| 1e-05 | 0.02 | 7.603 | 7.785 | 0.2374 | 0.2511 | 1656 |
| 1e-05 | 0.05 | 14.752 | 14.900 | 0.2295 | 0.2651 | 4140 |
| 2e-05 | 0.02 | 6.891 | 6.980 | 0.2282 | 0.2651 | 1656 |
| 2e-05 | 0.05 | 10.959 | 11.009 | 0.2295 | 0.2651 | 4140 |

**Key findings:**
1. **Small noise (5e-05 to 0.0001) preserves MMLU (~0.45) and drops WMDP to ~0.26** -- viable configs exist after all.
2. noise_std=0.0001/LR=2e-05 gives L2=10.39 with MMLU=0.4515 -- comparable to TAR's best L2 but with better MMLU preservation.
3. However, the L2 is dominated by random noise (per-layer L2 is uniform ~1.5 across all layers), not targeted weight change. The WMDP drop comes from retain fine-tuning, not from the noise targeting bio knowledge.
4. noise_std=0.0005 gives L2~42 with MMLU still preserved (~0.45) -- much higher L2 than any other method, but again this is mostly random noise magnitude.
5. noise_std≥0.01 destroys MMLU to random. The boundary is between 0.0005 and 0.01.
6. The retain fine-tuning (32 steps) is sufficient to recover capability at small noise levels but not at large ones.

## Cross-Method Comparison — Best Config Per Method

| Method | Best Config | L2 Dist | MMLU | WMDP | MMLU-WMDP | Forget NLL | Retain NLL |
|---|---|---|---|---|---|---|---|
| **CB** | LR=1.3e-05, BS=16, Ly=13-14-15 | 2.69 | 0.4422 | 0.2565 | 0.1857 | 2.179 | 2.686 |
| **CB_LAT** | (identical to CB) | 2.69 | 0.4430 | 0.2552 | 0.1878 | 2.179 | 2.686 |
| **GA** | LR=2e-05, RW=5.0 | 5.79 | 0.4505 | 0.2606 | 0.1899 | 2.451 | 2.242 |
| **grad_diff** | LR=1e-05, FW=1.0 | 2.87 | 0.4469 | 0.2704 | 0.1765 | 2.486 | 2.345 |
| **TAR** | tar_lr=1e-05, alpha=5 | 11.22 | 0.4304 | 0.2499 | 0.1805 | 2.705 | 2.873 |
| **wt_dist** | LR=2e-05, noise_std=0.0001 | 10.39 | 0.4515 | 0.2659 | 0.1856 | 2.140 | 2.234 |

Note: MMLU-WMDP gap differences across methods (~0.01) are within measurement uncertainty (±0.025 at 95% CI).

**Caveat on L2 comparability across methods:** L2 weight distance measures magnitude of weight change but not its directionality. For methods like TAR, GA, grad_diff, and CB, the weight change is driven by gradients computed on forget/retain data, so the L2 reflects movement in a task-relevant direction (e.g. the forget-data gradient). For wt_dist, the L2 is dominated by injected random Gaussian noise — the same L2 magnitude represents movement in a random direction, not targeted forgetting. For example, wt_dist L2=10.39 and TAR L2=11.22 are numerically similar, but TAR's change is entirely in the forget-data direction while wt_dist's is mostly random noise with a small retain fine-tuning component. L2 values are therefore **not directly comparable between wt_dist and other methods**.

**Key takeaway:** TAR gives the most weight change (L2=11.22) while preserving MMLU>0.43 and dropping WMDP near random. GA is competitive at moderate L2 (5.79) with best MMLU preservation. CB/CB_LAT are limited to L2~2.7. grad_diff is similar to GA but harder to tune. wt_dist is not useful -- random noise destroys capability without targeted unlearning.
