## Baselines

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | 0.4316 | 0.3560 | 0.2426 |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | 0.4499 | 0.4309 | 0.3652 |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### cb

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0019 | 0.2446 | 0.2465 | 0.2268 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0053 | 0.2447 | 0.2500 | 0.2407 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0150 | 0.2465 | 0.2615 | 0.2268 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0192 | 0.2423 | 0.2615 | 0.2240 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0209 | 0.2349 | 0.2558 | 0.2296 |

### ga

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_ml2048 | 0.1173 | 0.3558 | 0.2385 | 0.2472 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | 0.3393 | 0.2408 | 0.2454 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | 0.2844 | 0.2316 | 0.2546 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep4_lr3e-05_bs32_rw1.0_ml2048 | 0.0085 | 0.2596 | 0.2512 | 0.2426 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr3e-05_bs32_rw1.0_ml2048 | -0.0113 | 0.2352 | 0.2465 | 0.2416 |

### npo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | 0.1440 | 0.3813 | 0.2373 | 0.2509 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.02_rw1.0_ml2048 | 0.1098 | 0.3413 | 0.2316 | 0.2612 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml4096 | 0.1036 | 0.3675 | 0.2638 | 0.2472 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | 0.3437 | 0.2707 | 0.2584 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | 0.3159 | 0.2431 | 0.2584 |

### rmu

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2704 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2686 |

### simnpo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1780 | 0.4315 | 0.2535 | 0.2565 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | 0.4288 | 0.2638 | 0.2602 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1278 | 0.3962 | 0.2684 | 0.2574 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1117 | 0.4205 | 0.3088 | 0.2909 |
| girishgupta_simnpo__ep3_lr3e-05_bs4_b0.01_rw1.0_ml1024 | 0.1105 | 0.4216 | 0.3111 | 0.2695 |

### tar

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | 0.2295 | 0.2673 | 0.2491 |

### wt_dist

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) |
| --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | 0.3518 | 0.2903 | 0.3550 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | 0.4256 | 0.3986 | 0.3931 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | 0.2537 | 0.2604 | 0.2388 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | 0.2301 | 0.2627 | 0.2435 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep2_lr3e-05_bs32_wn0.02_ml2048 | -0.0359 | 0.2291 | 0.2650 | 0.2110 |

