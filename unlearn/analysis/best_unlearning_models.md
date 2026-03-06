## Baselines

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | 0.4316 | 0.3560 | 0.2426 | 0.4006 |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | 0.4499 | 0.4309 | 0.3652 | 0.5263 |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### ga

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | 0.3393 | 0.2408 | 0.2454 | 0.2537 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | 0.2844 | 0.2316 | 0.2546 | 0.2404 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw1.0_ml2048 | -0.0130 | 0.2416 | 0.2546 | 0.2398 | 0.2529 |

### npo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | 0.3437 | 0.2707 | 0.2584 | 0.2490 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | 0.3159 | 0.2431 | 0.2584 | 0.2349 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0680 | 0.3330 | 0.2650 | 0.2760 | 0.2490 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.0679 | 0.3052 | 0.2373 | 0.2491 | 0.2490 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0650 | 0.3311 | 0.2661 | 0.2677 | 0.2569 |

### rmu

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2686 | 0.2467 |

### simnpo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1780 | 0.4315 | 0.2535 | 0.2565 | 0.2616 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | 0.4288 | 0.2638 | 0.2602 | 0.2765 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1278 | 0.3962 | 0.2684 | 0.2574 | 0.2687 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1117 | 0.4205 | 0.3088 | 0.2909 | 0.3331 |
| girishgupta_simnpo__ep3_lr3e-05_bs4_b0.01_rw1.0_ml1024 | 0.1105 | 0.4216 | 0.3111 | 0.2695 | 0.3401 |

### tar

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |

### wt_dist

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | 0.3518 | 0.2903 | 0.3550 | 0.3543 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | 0.4256 | 0.3986 | 0.3931 | 0.4776 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | 0.2537 | 0.2604 | 0.2388 | 0.2694 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | 0.2301 | 0.2627 | 0.2435 | 0.2474 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.02_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2193 | 0.2459 |

