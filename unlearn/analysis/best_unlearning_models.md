## Cross-Method Comparison — Best Config Per Method

*Best run per method ranked by Score = MMLU − WMDP (Robust)*

| Method | Best Config | L2 Dist | MMLU | WMDP (Robust) | MMLU−WMDP (Robust) | Forget NLL | Retain NLL | Date (UTC) | Run Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cb | **Epochs:** 1<br>**Learning Rate:** 1.3e-05<br>**Batch Size:** 16<br>**Alpha:** 200.0<br>**Steering Coeff:** 10.0<br>**Layers:** 13-14-15<br>**Max Length:** 512<br>**Max Lines:** 1024 | 2.68 | 0.4425 | 0.4136 | 0.0289 | 2.178 | 2.684 | 2026-03-09 19:31 UTC | EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15_mle512_mli1024 |
| cb_lat | **Epochs:** 1<br>**Learning Rate:** 1.3e-05<br>**Batch Size:** 16<br>**Alpha:** 200.0<br>**Steering Coeff:** 10.0<br>**LAT Epsilon:** 0.1<br>**LAT Steps:** 5<br>**Layers:** 13-14-15<br>**Max Length:** 512<br>**Max Lines:** 1024 | 2.68 | 0.4435 | 0.4124 | 0.0311 | 2.178 | 2.684 | 2026-03-09 19:43 UTC | EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.3e-05_bs16_a200.0_sc10.0_le0.1_ls5_ly13-14-15_mle512_mli1024 |
| dpo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Max Length:** 512<br>**Max Lines:** 8192 | 17.61 | 0.3840 | 0.2408 | 0.1432 | 184.176 | 3.617 | 2026-03-09 21:34 UTC | EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.01_mle512_mli8192 |
| ga | **Epochs:** 3<br>**Learning Rate:** 3e-05<br>**Batch Size:** 32<br>**Retain Weight:** 5.0<br>**Max Length:** 2048 | N/A | 0.3558 | 0.2385 | 0.1173 | N/A | N/A | 2026-03-05 14:02 UTC | unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr3e-05_bs32_rw5.0_ml2048 |
| ga_simple | **Epochs:** 1<br>**Learning Rate:** 1e-05<br>**Batch Size:** 32<br>**Max Length:** 1024 | N/A | 0.4471 | 0.4240 | 0.0231 | 2.144 | 2.169 | 2026-03-07 17:20 UTC | EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1e-05_bs32_ml1024 |
| grad_diff | **Epochs:** 3<br>**Learning Rate:** 4e-05<br>**Batch Size:** 32<br>**Forget Weight:** 1.0<br>**Max Length:** 512<br>**Max Lines:** 2048 | 14.68 | 0.3524 | 0.2719 | 0.0805 | 181.122 | 2.762 | 2026-03-09 21:21 UTC | EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.0_mle512_mli2048 |
| lat | **Epochs:** 3<br>**Learning Rate:** 3e-05<br>**Batch Size:** 32<br>**LAT Epsilon:** 0.1<br>**LAT Steps:** 5<br>**Retain Weight:** 1.0<br>**Layers:** 5-6-7<br>**Max Length:** 512<br>**Max Lines:** 2048 | N/A | 0.2898 | 0.2742 | 0.0156 | 188.541 | 2.687 | 2026-03-08 17:27 UTC | EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr3e-05_bs32_le0.1_ls5_rw1.0_ly5-6-7_mle512_mli2048 |
| npo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.5<br>**Max Length:** 512<br>**Max Lines:** 8192 | 17.52 | 0.4426 | 0.2673 | 0.1753 | 182.439 | 2.429 | 2026-03-09 19:52 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4.5e-05_bs32_b0.01_rw1.5_mle512_mli8192 |
| rmu | **Epochs:** 1<br>**Learning Rate:** 2e-05<br>**Batch Size:** 32<br>**Alpha:** 1000.0<br>**Steering Coeff:** 20.0<br>**Layers:** 11-12-13<br>**Max Length:** 2048 | N/A | 0.4030 | 0.3468 | 0.0562 | 2.233 | 3.100 | 2026-03-06 18:20 UTC | EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr2e-05_bs32_a1000.0_sc20.0_ly11-12-13_ml2048 |
| simnpo | **Epochs:** 3<br>**Learning Rate:** 3e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.0<br>**Max Length:** 2048 | N/A | 0.4315 | 0.2535 | 0.1780 | 198.902 | 2.482 | 2026-03-04 03:39 UTC | EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 |
| tar | **TAR Alpha:** 5.0<br>**TAR LR:** 1e-05<br>**TAR Epochs:** 1<br>**Max Length:** 512<br>**Max Lines:** 1024 | 11.13 | 0.4313 | 0.3998 | 0.0315 | 2.697 | 2.867 | 2026-03-09 16:53 UTC | EleutherAI_deep-ignorance-unfiltered/tar__ta5.0_tlr1e-05_tep1_mle512_mli1024 |
| wt_dist | **Epochs:** 3<br>**Learning Rate:** 3e-05<br>**Batch Size:** 32<br>**Noise Std:** 0.01<br>**Max Length:** 2048 | N/A | 0.3518 | 0.2903 | 0.0615 | N/A | N/A | 2026-03-05 08:31 UTC | unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 |

## Last 20 Runs (Most Recent First)

*All runs sorted by date, newest first*

| Method | Best Config | L2 Dist | MMLU | WMDP (Robust) | MMLU−WMDP (Robust) | Forget NLL | Retain NLL | Date (UTC) | Run Name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wt_dist | **Epochs:** 1<br>**Learning Rate:** 2e-05<br>**Batch Size:** 32<br>**Noise Std:** 0.0001<br>**Max Length:** 512<br>**Max Lines:** 1024 | 9.24 | 0.4495 | 0.4343 | 0.0152 | 2.137 | 2.235 | 2026-03-10 07:24 UTC | EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr2e-05_bs32_wn0.0001_mle512_mli1024 |
| grad_diff | **Epochs:** 1<br>**Learning Rate:** 1e-05<br>**Batch Size:** 32<br>**Forget Weight:** 1.0<br>**Max Length:** 512<br>**Max Lines:** 1024 | 1.29 | 0.4492 | 0.4343 | 0.0148 | 2.169 | 2.324 | 2026-03-10 07:05 UTC | EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs32_fw1.0_mle512_mli1024 |
| ga | **Epochs:** 1<br>**Learning Rate:** 2e-05<br>**Batch Size:** 4<br>**Retain Weight:** 5.0<br>**Max Length:** 512<br>**Max Lines:** 1024 | 5.74 | 0.4504 | 0.4424 | 0.0080 | 2.413 | 2.241 | 2026-03-10 06:55 UTC | EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr2e-05_bs4_rw5.0_mle512_mli1024 |
| cb_lat | **Epochs:** 3<br>**Learning Rate:** 1e-05<br>**Batch Size:** 32<br>**Alpha:** 1000.0<br>**Steering Coeff:** 5.0<br>**LAT Epsilon:** 0.1<br>**LAT Steps:** 5<br>**Layers:** 11-12-13<br>**Max Length:** 2048<br>**Max Lines:** 1024 | 3.91 | 0.4314 | 0.4113 | 0.0201 | 2.120 | 2.795 | 2026-03-10 05:01 UTC | EleutherAI_deep-ignorance-unfiltered/cb_lat__ep3_lr1e-05_bs32_a1000.0_sc5.0_le0.1_ls5_ly11-12-13_mle2048_mli1024 |
| dpo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.03<br>**Max Length:** 512<br>**Max Lines:** 8192 | 15.02 | 0.3549 | 0.2742 | 0.0807 | 173.997 | 4.910 | 2026-03-10 03:27 UTC | EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.03_mle512_mli8192 |
| dpo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.02<br>**Max Length:** 512<br>**Max Lines:** 8192 | 15.93 | 0.3914 | 0.2523 | 0.1391 | 178.026 | 3.987 | 2026-03-09 22:05 UTC | EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.02_mle512_mli8192 |
| cb | **Epochs:** 2<br>**Learning Rate:** 1e-05<br>**Batch Size:** 32<br>**Alpha:** 1000.0<br>**Steering Coeff:** 5.0<br>**Layers:** 11-12-13<br>**Max Length:** 2048<br>**Max Lines:** 1024 | 2.96 | 0.4390 | 0.4182 | 0.0208 | 2.057 | 2.190 | 2026-03-09 21:39 UTC | EleutherAI_deep-ignorance-unfiltered/cb__ep2_lr1e-05_bs32_a1000.0_sc5.0_ly11-12-13_mle2048_mli1024 |
| dpo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Max Length:** 512<br>**Max Lines:** 8192 | 17.61 | 0.3840 | 0.2408 | 0.1432 | 184.176 | 3.617 | 2026-03-09 21:34 UTC | EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.01_mle512_mli8192 |
| grad_diff | **Epochs:** 3<br>**Learning Rate:** 4e-05<br>**Batch Size:** 32<br>**Forget Weight:** 1.0<br>**Max Length:** 512<br>**Max Lines:** 2048 | 14.68 | 0.3524 | 0.2719 | 0.0805 | 181.122 | 2.762 | 2026-03-09 21:21 UTC | EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.0_mle512_mli2048 |
| npo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.5<br>**Max Length:** 512<br>**Max Lines:** 10240 | 18.18 | 0.4087 | 0.2615 | 0.1472 | 185.590 | 2.711 | 2026-03-09 21:04 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4.5e-05_bs32_b0.01_rw1.5_mle512_mli10240 |
| ga | **Epochs:** 3<br>**Learning Rate:** 3e-05<br>**Batch Size:** 32<br>**Retain Weight:** 5.0<br>**Max Length:** 2048<br>**Max Lines:** 1024 | 8.21 | 0.4412 | 0.4332 | 0.0080 | 83.296 | 2.286 | 2026-03-09 21:04 UTC | EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_mle2048_mli1024 |
| npo | **Epochs:** 1<br>**Learning Rate:** 4e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.5<br>**Max Length:** 512<br>**Max Lines:** 8192 | 15.73 | 0.4237 | 0.2903 | 0.1333 | 179.825 | 2.346 | 2026-03-09 20:24 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4e-05_bs32_b0.01_rw1.5_mle512_mli8192 |
| wt_dist | **Epochs:** 1<br>**Learning Rate:** 2e-05<br>**Batch Size:** 4<br>**Noise Std:** 0.0001<br>**Max Length:** 512<br>**Max Lines:** 1024 | 10.37 | 0.4534 | 0.4447 | 0.0087 | 2.141 | 2.234 | 2026-03-09 20:14 UTC | EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr2e-05_bs4_wn0.0001_mle512_mli1024 |
| grad_diff | **Epochs:** 1<br>**Learning Rate:** 1e-05<br>**Batch Size:** 4<br>**Forget Weight:** 1.0<br>**Max Length:** 512<br>**Max Lines:** 1024 | 2.80 | 0.4469 | 0.4263 | 0.0207 | 2.429 | 2.343 | 2026-03-09 20:04 UTC | EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs4_fw1.0_mle512_mli1024 |
| npo | **Epochs:** 1<br>**Learning Rate:** 4.5e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.5<br>**Max Length:** 512<br>**Max Lines:** 8192 | 17.52 | 0.4426 | 0.2673 | 0.1753 | 182.439 | 2.429 | 2026-03-09 19:52 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4.5e-05_bs32_b0.01_rw1.5_mle512_mli8192 |
| cb_lat | **Epochs:** 1<br>**Learning Rate:** 1.3e-05<br>**Batch Size:** 16<br>**Alpha:** 200.0<br>**Steering Coeff:** 10.0<br>**LAT Epsilon:** 0.1<br>**LAT Steps:** 5<br>**Layers:** 13-14-15<br>**Max Length:** 512<br>**Max Lines:** 1024 | 2.68 | 0.4435 | 0.4124 | 0.0311 | 2.178 | 2.684 | 2026-03-09 19:43 UTC | EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.3e-05_bs16_a200.0_sc10.0_le0.1_ls5_ly13-14-15_mle512_mli1024 |
| cb | **Epochs:** 1<br>**Learning Rate:** 1.3e-05<br>**Batch Size:** 16<br>**Alpha:** 200.0<br>**Steering Coeff:** 10.0<br>**Layers:** 13-14-15<br>**Max Length:** 512<br>**Max Lines:** 1024 | 2.68 | 0.4425 | 0.4136 | 0.0289 | 2.178 | 2.684 | 2026-03-09 19:31 UTC | EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15_mle512_mli1024 |
| npo | **Epochs:** 1<br>**Learning Rate:** 5e-05<br>**Batch Size:** 32<br>**Beta:** 0.015<br>**Retain Weight:** 1.25<br>**Max Length:** 512<br>**Max Lines:** 6144 | 18.46 | 0.3950 | 0.2961 | 0.0989 | 181.694 | 2.506 | 2026-03-09 17:22 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.015_rw1.25_mle512_mli6144 |
| tar | **TAR Alpha:** 5.0<br>**TAR LR:** 1e-05<br>**TAR Epochs:** 1<br>**Max Length:** 512<br>**Max Lines:** 1024 | 11.13 | 0.4313 | 0.3998 | 0.0315 | 2.697 | 2.867 | 2026-03-09 16:53 UTC | EleutherAI_deep-ignorance-unfiltered/tar__ta5.0_tlr1e-05_tep1_mle512_mli1024 |
| npo | **Epochs:** 1<br>**Learning Rate:** 4e-05<br>**Batch Size:** 32<br>**Beta:** 0.01<br>**Retain Weight:** 1.5<br>**Max Length:** 512<br>**Max Lines:** 6144 | 14.83 | 0.3719 | 0.2604 | 0.1115 | 179.378 | 2.686 | 2026-03-09 16:48 UTC | EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4e-05_bs32_b0.01_rw1.5_mle512_mli6144 |

## Baselines

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | N/A | 0.4316 | 0.3560 | 0.2499 | 0.2426 | 0.4006 | N/A | N/A | 2026-03-03 08:16 UTC |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | N/A | 0.4499 | 0.4309 | 0.2573 | 0.3652 | 0.5263 | N/A | N/A | 2026-03-03 07:30 UTC |
| EleutherAI_deep-ignorance-unfiltered | 0.0190 | N/A | 0.4499 | 0.4309 | 0.2573 | 0.3652 | 0.5263 | N/A | N/A | 2026-03-04 07:33 UTC |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### cb

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15_mle512_mli1024 | 0.0289 | 2.6794 | 0.4425 | 0.4136 | 0.2594 | 0.3504 | 0.5004 | 2.1783 | 2.6842 | 2026-03-09 19:31 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a1000.0_sc5.0_ly13-14-15_mle512_mli1024 | 0.0274 | 2.6785 | 0.4421 | 0.4147 | 0.2565 | 0.3522 | 0.5020 | 2.1783 | 2.6848 | 2026-03-09 16:10 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep2_lr1e-05_bs32_a1000.0_sc5.0_ly11-12-13_mle2048_mli1024 | 0.0208 | 2.9595 | 0.4390 | 0.4182 | 0.2487 | 0.3699 | 0.5051 | 2.0571 | 2.1897 | 2026-03-09 21:39 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep2_lr1e-05_bs32_a1000.0_sc5.0_ly11-12-13_ml2048 | 0.0165 | N/A | 0.4071 | 0.3906 | 0.2524 | 0.3606 | 0.4564 | 2.3332 | 4.3229 | 2026-03-07 07:18 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs32_a500.0_sc10.0_ly11-12-13_ml2048 | 0.0136 | N/A | 0.3154 | 0.3018 | 0.2630 | 0.3309 | 0.3064 | 4.9704 | 7.0958 | 2026-03-07 07:00 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs32_a800.0_sc8.0_ly11-12-13_ml2048 | 0.0104 | N/A | 0.3146 | 0.3041 | 0.2618 | 0.3336 | 0.3071 | 4.9594 | 7.0928 | 2026-03-07 07:37 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0019 | N/A | 0.2446 | 0.2465 | 0.2647 | 0.2268 | 0.2459 | 11.7820 | 10.4446 | 2026-03-05 16:37 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0053 | N/A | 0.2447 | 0.2500 | 0.2663 | 0.2407 | 0.2514 | 11.8866 | 10.6731 | 2026-03-05 16:43 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0150 | N/A | 0.2465 | 0.2615 | 0.2717 | 0.2268 | 0.2608 | 11.4395 | 10.4954 | 2026-03-05 16:57 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0192 | N/A | 0.2423 | 0.2615 | 0.2573 | 0.2240 | 0.2506 | 11.1575 | 10.4234 | 2026-03-05 17:03 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0209 | N/A | 0.2349 | 0.2558 | 0.2684 | 0.2296 | 0.2490 | 10.1736 | 9.5921 | 2026-03-05 15:42 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0225 | N/A | 0.2332 | 0.2558 | 0.2709 | 0.2379 | 0.2482 | 10.1675 | 9.5875 | 2026-03-05 15:49 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc15.0_ly5-6-7_ml1024 | -0.0252 | N/A | 0.2329 | 0.2581 | 0.2700 | 0.2351 | 0.2522 | 10.1761 | 9.6053 | 2026-03-05 15:36 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc20.0_ly5-6-7_ml1024 | -0.0284 | N/A | 0.2458 | 0.2742 | 0.2737 | 0.2221 | 0.2639 | 11.4968 | 10.4809 | 2026-03-05 16:50 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0308 | N/A | 0.2342 | 0.2650 | 0.2626 | 0.2454 | 0.2553 | 10.2871 | 9.4630 | 2026-03-05 16:22 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc25.0_ly5-6-7_ml1024 | -0.0312 | N/A | 0.2395 | 0.2707 | 0.2528 | 0.2268 | 0.2592 | 11.2043 | 10.4906 | 2026-03-05 17:10 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc15.0_ly5-6-7_ml1024 | -0.0330 | N/A | 0.2354 | 0.2684 | 0.2647 | 0.2481 | 0.2545 | 10.2448 | 9.4423 | 2026-03-05 16:15 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc25.0_ly5-6-7_ml1024 | -0.0335 | N/A | 0.2361 | 0.2696 | 0.2594 | 0.2481 | 0.2553 | 10.2625 | 9.4550 | 2026-03-05 16:29 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc15.0_ly5-6-7_ml1024 | -0.0346 | N/A | 0.2338 | 0.2684 | 0.2651 | 0.2444 | 0.2514 | 10.3023 | 9.4916 | 2026-03-05 15:55 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0363 | N/A | 0.2333 | 0.2696 | 0.2622 | 0.2491 | 0.2522 | 10.2902 | 9.4786 | 2026-03-05 16:09 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc20.0_ly5-6-7_ml1024 | -0.0398 | N/A | 0.2344 | 0.2742 | 0.2651 | 0.2435 | 0.2608 | 10.3004 | 9.4840 | 2026-03-05 16:02 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs1_a800.0_sc8.0_ly11-12-13_ml6144 | -0.0471 | N/A | 0.2317 | 0.2788 | 0.2676 | 0.2398 | 0.2482 | 9.7559 | 8.7581 | 2026-03-07 09:30 UTC |

### cb_lat

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.3e-05_bs16_a200.0_sc10.0_le0.1_ls5_ly13-14-15_mle512_mli1024 | 0.0311 | 2.6800 | 0.4435 | 0.4124 | 0.2573 | 0.3504 | 0.4996 | 2.1782 | 2.6841 | 2026-03-09 19:43 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.3e-05_bs16_a1000.0_sc5.0_le0.1_ls5_ly13-14-15_mle512_mli1024 | 0.0294 | 2.6796 | 0.4430 | 0.4136 | 0.2569 | 0.3485 | 0.5035 | 2.1783 | 2.6847 | 2026-03-09 16:22 UTC |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep3_lr1e-05_bs32_a1000.0_sc5.0_le0.1_ls5_ly11-12-13_mle2048_mli1024 | 0.0201 | 3.9074 | 0.4314 | 0.4113 | 0.2511 | 0.3671 | 0.4878 | 2.1196 | 2.7950 | 2026-03-10 05:01 UTC |

### dpo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.01_mle512_mli8192 | 0.1432 | 17.6103 | 0.3840 | 0.2408 | 0.2437 | 0.2491 | 0.2451 | 184.1763 | 3.6173 | 2026-03-09 21:34 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.02_mle512_mli8192 | 0.1391 | 15.9307 | 0.3914 | 0.2523 | 0.2659 | 0.2379 | 0.2506 | 178.0263 | 3.9866 | 2026-03-09 22:05 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr4e-05_bs32_b0.01_mle512_mli2048 | 0.0879 | N/A | 0.3551 | 0.2673 | 0.2511 | 0.2481 | 0.2757 | 185.5913 | 3.7995 | 2026-03-08 15:42 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr4.5e-05_bs32_b0.03_mle512_mli8192 | 0.0807 | 15.0244 | 0.3549 | 0.2742 | 0.2651 | 0.2193 | 0.2639 | 173.9975 | 4.9095 | 2026-03-10 03:27 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.01_mle512_mli2048 | 0.0473 | N/A | 0.3089 | 0.2615 | 0.2269 | 0.2221 | 0.2498 | 162.2404 | 3.4266 | 2026-03-08 15:25 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.05_mle512_mli2048 | 0.0303 | N/A | 0.3310 | 0.3007 | 0.2651 | 0.2110 | 0.3134 | 133.2267 | 4.4992 | 2026-03-08 16:31 UTC |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.1_mle512_mli2048 | 0.0238 | N/A | 0.3257 | 0.3018 | 0.2659 | 0.2500 | 0.3048 | 91.2222 | 5.3127 | 2026-03-08 16:52 UTC |

### ga

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr3e-05_bs32_rw5.0_ml2048 | 0.1173 | N/A | 0.3558 | 0.2385 | 0.2404 | 0.2472 | 0.2553 | N/A | N/A | 2026-03-05 14:02 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_ml2048 | 0.1173 | N/A | 0.3558 | 0.2385 | 0.2404 | 0.2472 | 0.2553 | 199.1775 | 2.8462 | 2026-03-05 11:52 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | N/A | 0.3393 | 0.2408 | 0.2248 | 0.2454 | 0.2537 | N/A | N/A | 2026-03-05 11:51 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | N/A | 0.3393 | 0.2408 | 0.2248 | 0.2454 | 0.2537 | 198.9943 | 2.6624 | 2026-03-05 09:41 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | N/A | 0.2844 | 0.2316 | 0.2437 | 0.2546 | 0.2404 | N/A | N/A | 2026-03-05 09:41 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | N/A | 0.2844 | 0.2316 | 0.2437 | 0.2546 | 0.2404 | 195.8933 | 2.7136 | 2026-03-05 07:30 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr2e-05_bs32_rw5.0_mle512_mli1024 | 0.0132 | 3.1391 | 0.4499 | 0.4366 | 0.2548 | 0.3550 | 0.5287 | 2.1596 | 2.2369 | 2026-03-09 16:32 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep4_lr3e-05_bs32_rw1.0_ml2048 | 0.0085 | N/A | 0.2596 | 0.2512 | 0.2520 | 0.2426 | 0.2419 | N/A | N/A | 2026-03-05 19:23 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep4_lr3e-05_bs32_rw1.0_ml2048 | 0.0085 | N/A | 0.2596 | 0.2512 | 0.2520 | 0.2426 | 0.2419 | 203.0704 | 2.9996 | 2026-03-05 16:37 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_mle2048_mli1024 | 0.0080 | 8.2079 | 0.4412 | 0.4332 | 0.2474 | 0.3634 | 0.5232 | 83.2961 | 2.2860 | 2026-03-09 21:04 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr2e-05_bs4_rw5.0_mle512_mli1024 | 0.0080 | 5.7401 | 0.4504 | 0.4424 | 0.2606 | 0.3606 | 0.5342 | 2.4126 | 2.2413 | 2026-03-10 06:55 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep1_lr3e-05_bs32_rw1.0_ml2048 | -0.0113 | N/A | 0.2352 | 0.2465 | 0.2536 | 0.2416 | 0.2435 | N/A | N/A | 2026-03-05 15:02 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr3e-05_bs32_rw1.0_ml2048 | -0.0113 | N/A | 0.2352 | 0.2465 | 0.2536 | 0.2416 | 0.2435 | 180.4826 | 3.3891 | 2026-03-05 14:03 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr3e-05_bs32_rw1.0_ml2048 | -0.0130 | N/A | 0.2416 | 0.2546 | 0.2396 | 0.2398 | 0.2529 | N/A | N/A | 2026-03-05 07:30 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw1.0_ml2048 | -0.0130 | N/A | 0.2416 | 0.2546 | 0.2396 | 0.2398 | 0.2529 | 201.5835 | 3.4513 | 2026-03-05 05:18 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep2_lr3e-05_bs32_rw1.0_ml2048 | -0.0168 | N/A | 0.2344 | 0.2512 | 0.2458 | 0.2398 | 0.2372 | N/A | N/A | 2026-03-05 16:37 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep2_lr3e-05_bs32_rw1.0_ml2048 | -0.0168 | N/A | 0.2344 | 0.2512 | 0.2458 | 0.2398 | 0.2372 | 198.0561 | 2.8619 | 2026-03-05 15:02 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr1e-05_bs32_rw1.0_ml2048 | -0.0312 | N/A | 0.2326 | 0.2638 | 0.2647 | 0.2379 | 0.2632 | N/A | N/A | 2026-03-05 21:35 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr1e-05_bs32_rw1.0_ml2048 | -0.0312 | N/A | 0.2326 | 0.2638 | 0.2647 | 0.2379 | 0.2632 | 212.4552 | 3.8803 | 2026-03-05 19:24 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_ga__ep3_lr2e-05_bs32_rw1.0_ml2048 | -0.0352 | N/A | 0.2609 | 0.2961 | 0.2643 | 0.2444 | 0.2742 | N/A | N/A | 2026-03-05 23:46 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr2e-05_bs32_rw1.0_ml2048 | -0.0352 | N/A | 0.2609 | 0.2961 | 0.2643 | 0.2444 | 0.2742 | 217.0435 | 3.1056 | 2026-03-05 21:36 UTC |

### ga_simple

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1e-05_bs32_ml1024 | 0.0231 | N/A | 0.4471 | 0.4240 | 0.2606 | 0.3727 | 0.5185 | 2.1441 | 2.1690 | 2026-03-07 17:20 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep2_lr1.5e-05_bs32_ml1024 | 0.0224 | N/A | 0.3865 | 0.3641 | 0.2491 | 0.3281 | 0.3998 | 43.6780 | 30.5354 | 2026-03-07 19:47 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep3_lr1e-05_bs32_ml1024 | 0.0216 | N/A | 0.4133 | 0.3917 | 0.2561 | 0.3467 | 0.4564 | 39.5764 | 23.9488 | 2026-03-07 18:56 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr2e-05_bs32_ml1024 | 0.0209 | N/A | 0.3723 | 0.3514 | 0.2565 | 0.3058 | 0.3826 | 34.1664 | 24.8204 | 2026-03-07 18:00 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep2_lr1e-05_bs32_ml1024 | 0.0194 | N/A | 0.4365 | 0.4171 | 0.2647 | 0.3773 | 0.4996 | 3.4951 | 2.5983 | 2026-03-07 18:19 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1.5e-05_bs32_ml1024 | 0.0153 | N/A | 0.4416 | 0.4263 | 0.2635 | 0.3727 | 0.5122 | 2.6094 | 2.3711 | 2026-03-07 19:17 UTC |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr3e-05_bs32_ml1024 | 0.0134 | N/A | 0.2485 | 0.2350 | 0.2339 | 0.2407 | 0.2482 | 90.5616 | 78.9245 | 2026-03-07 17:32 UTC |

### grad_diff

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0805 | 14.6780 | 0.3524 | 0.2719 | 0.2700 | 0.2407 | 0.2679 | 181.1218 | 2.7620 | 2026-03-09 21:21 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw0.75_mle512_mli2048 | 0.0766 | N/A | 0.3474 | 0.2707 | 0.2639 | 0.2602 | 0.2553 | 180.6188 | 2.6808 | 2026-03-08 12:17 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.5_mle512_mli2048 | 0.0566 | N/A | 0.3377 | 0.2811 | 0.2717 | 0.2444 | 0.2891 | 181.4866 | 3.0881 | 2026-03-08 13:06 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.0_ml1024 | 0.0468 | N/A | 0.3533 | 0.3065 | 0.2433 | 0.2519 | 0.3354 | 208.3995 | 3.9771 | 2026-03-07 20:23 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr4e-05_bs32_fw1.5_mle512_mli2048 | 0.0375 | N/A | 0.3220 | 0.2846 | 0.2725 | 0.2435 | 0.2773 | 178.7582 | 3.7167 | 2026-03-08 14:38 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0332 | N/A | 0.3155 | 0.2823 | 0.2655 | 0.2314 | 0.2726 | 178.4407 | 2.9857 | 2026-03-08 13:42 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw0.75_mle512_mli2048 | 0.0245 | N/A | 0.3240 | 0.2995 | 0.2487 | 0.2509 | 0.3205 | 177.1177 | 2.7526 | 2026-03-08 09:59 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs4_fw1.0_mle512_mli1024 | 0.0207 | 2.8050 | 0.4469 | 0.4263 | 0.2622 | 0.3606 | 0.5232 | 2.4294 | 2.3425 | 2026-03-09 20:04 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs32_fw1.0_mle512_mli1024 | 0.0148 | 1.2913 | 0.4492 | 0.4343 | 0.2594 | 0.3597 | 0.5279 | 2.1691 | 2.3238 | 2026-03-10 07:05 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw2.0_ml1024 | 0.0145 | N/A | 0.2807 | 0.2661 | 0.2302 | 0.2361 | 0.2702 | 167.6785 | 3.9282 | 2026-03-07 20:52 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.0_mle512_mli2048 | 0.0126 | N/A | 0.3145 | 0.3018 | 0.2598 | 0.2370 | 0.3032 | 177.4739 | 3.0196 | 2026-03-08 09:45 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0090 | N/A | 0.3223 | 0.3134 | 0.2635 | 0.2435 | 0.3307 | 172.7062 | 3.6520 | 2026-03-08 13:53 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr3e-05_bs32_fw1.0_mle512_mli2048 | 0.0058 | N/A | 0.3134 | 0.3076 | 0.2520 | 0.2463 | 0.3119 | 175.1713 | 3.3911 | 2026-03-08 13:21 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.5_mle512_mli2048 | 0.0039 | N/A | 0.2965 | 0.2926 | 0.2577 | 0.2435 | 0.2852 | 177.8307 | 3.4684 | 2026-03-08 11:00 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw3.0_ml1024 | -0.0007 | N/A | 0.2712 | 0.2719 | 0.2330 | 0.2314 | 0.2726 | 168.7651 | 4.2790 | 2026-03-07 21:21 UTC |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw2.0_mle512_mli2048 | -0.0043 | N/A | 0.3114 | 0.3157 | 0.2528 | 0.2268 | 0.3181 | 178.0312 | 4.0228 | 2026-03-08 11:22 UTC |

### lat

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr3e-05_bs32_le0.1_ls5_rw1.0_ly5-6-7_mle512_mli2048 | 0.0156 | N/A | 0.2898 | 0.2742 | 0.2507 | 0.2704 | 0.2663 | 188.5414 | 2.6867 | 2026-03-08 17:27 UTC |

### npo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4.5e-05_bs32_b0.01_rw1.5_mle512_mli8192 | 0.1753 | 17.5180 | 0.4426 | 0.2673 | 0.2659 | 0.2677 | 0.2765 | 182.4386 | 2.4286 | 2026-03-09 19:52 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4.5e-05_bs32_b0.01_rw1.5_mle512_mli10240 | 0.1472 | 18.1849 | 0.4087 | 0.2615 | 0.2503 | 0.2584 | 0.2647 | 185.5902 | 2.7112 | 2026-03-09 21:04 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | 0.1440 | N/A | 0.3813 | 0.2373 | 0.2548 | 0.2509 | 0.2506 | N/A | N/A | 2026-03-05 20:57 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | 0.1440 | N/A | 0.3813 | 0.2373 | 0.2548 | 0.2509 | 0.2506 | 205.0158 | 2.9583 | 2026-03-05 18:29 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4e-05_bs32_b0.01_rw1.5_mle512_mli8192 | 0.1333 | 15.7258 | 0.4237 | 0.2903 | 0.2585 | 0.2630 | 0.3095 | 179.8251 | 2.3462 | 2026-03-09 20:24 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw1.5_mle512_mli6144 | 0.1255 | 18.2407 | 0.3974 | 0.2719 | 0.2696 | 0.2481 | 0.2852 | 182.1695 | 2.6186 | 2026-03-09 15:53 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw1.0_mle512_mli6144 | 0.1121 | 18.1638 | 0.3495 | 0.2373 | 0.2577 | 0.2416 | 0.2537 | 182.4943 | 2.7752 | 2026-03-09 15:29 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4e-05_bs32_b0.01_rw1.5_mle512_mli6144 | 0.1115 | 14.8298 | 0.3719 | 0.2604 | 0.2565 | 0.2584 | 0.2702 | 179.3783 | 2.6856 | 2026-03-09 16:48 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.02_rw1.0_ml2048 | 0.1098 | N/A | 0.3413 | 0.2316 | 0.2577 | 0.2612 | 0.2302 | N/A | N/A | 2026-03-06 06:47 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.02_rw1.0_ml2048 | 0.1098 | N/A | 0.3413 | 0.2316 | 0.2577 | 0.2612 | 0.2302 | 203.6757 | 2.7988 | 2026-03-06 04:20 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml4096 | 0.1036 | N/A | 0.3675 | 0.2638 | 0.2676 | 0.2472 | 0.2718 | N/A | N/A | 2026-03-05 16:01 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml4096 | 0.1036 | N/A | 0.3675 | 0.2638 | 0.2676 | 0.2472 | 0.2718 | 207.2005 | 3.1744 | 2026-03-05 11:27 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw2.0_ml2048 | 0.1025 | N/A | 0.4089 | 0.3065 | 0.2667 | 0.2528 | 0.3134 | 179.9415 | 2.8352 | 2026-03-07 05:24 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.02_rw1.0_ml2048 | 0.1023 | N/A | 0.4330 | 0.3306 | 0.2655 | 0.2602 | 0.3739 | 179.5334 | 2.3986 | 2026-03-07 06:14 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.5_ml2048 | 0.0998 | N/A | 0.3959 | 0.2961 | 0.2647 | 0.2704 | 0.2993 | 180.0890 | 2.5439 | 2026-03-07 05:45 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.015_rw1.25_mle512_mli6144 | 0.0989 | 18.4561 | 0.3950 | 0.2961 | 0.2626 | 0.2435 | 0.3386 | 181.6943 | 2.5063 | 2026-03-09 17:22 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw2.0_mle512_mli6144 | 0.0929 | 18.3625 | 0.3705 | 0.2776 | 0.2643 | 0.2481 | 0.2891 | 181.3814 | 2.5338 | 2026-03-09 16:19 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw3.0_ml2048 | 0.0857 | N/A | 0.4175 | 0.3318 | 0.2692 | 0.2454 | 0.3488 | 179.0658 | 2.4161 | 2026-03-07 04:53 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | N/A | 0.3437 | 0.2707 | 0.2643 | 0.2584 | 0.2490 | N/A | N/A | 2026-03-05 06:38 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | N/A | 0.3437 | 0.2707 | 0.2643 | 0.2584 | 0.2490 | 220.5241 | 2.8078 | 2026-03-05 03:54 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | N/A | 0.3159 | 0.2431 | 0.2626 | 0.2584 | 0.2349 | N/A | N/A | 2026-03-05 00:47 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | N/A | 0.3159 | 0.2431 | 0.2626 | 0.2584 | 0.2349 | 204.6300 | 2.8058 | 2026-03-04 22:18 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep4_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0680 | N/A | 0.3330 | 0.2650 | 0.2573 | 0.2760 | 0.2490 | N/A | N/A | 2026-03-05 10:03 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0680 | N/A | 0.3330 | 0.2650 | 0.2573 | 0.2760 | 0.2490 | 222.3897 | 2.6125 | 2026-03-05 06:38 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.0679 | N/A | 0.3052 | 0.2373 | 0.2372 | 0.2491 | 0.2490 | N/A | N/A | 2026-03-05 11:27 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.0679 | N/A | 0.3052 | 0.2373 | 0.2372 | 0.2491 | 0.2490 | 196.8414 | 5.7226 | 2026-03-05 10:03 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0650 | N/A | 0.3311 | 0.2661 | 0.2598 | 0.2677 | 0.2569 | N/A | N/A | 2026-03-05 03:53 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0650 | N/A | 0.3311 | 0.2661 | 0.2598 | 0.2677 | 0.2569 | 206.2143 | 2.4163 | 2026-03-05 00:47 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw2.0_ml2048 | 0.0608 | N/A | 0.3258 | 0.2650 | 0.2585 | 0.2342 | 0.2584 | N/A | N/A | 2026-03-06 08:09 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw2.0_ml2048 | 0.0608 | N/A | 0.3258 | 0.2650 | 0.2585 | 0.2342 | 0.2584 | 194.3022 | 2.8449 | 2026-03-05 04:34 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr2e-05_bs32_b0.01_rw1.0_ml2048 | 0.0551 | N/A | 0.3454 | 0.2903 | 0.2557 | 0.2602 | 0.2844 | N/A | N/A | 2026-03-05 18:28 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr2e-05_bs32_b0.01_rw1.0_ml2048 | 0.0551 | N/A | 0.3454 | 0.2903 | 0.2557 | 0.2602 | 0.2844 | 216.4733 | 2.4112 | 2026-03-05 16:01 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw0.75_ml2048 | 0.0255 | N/A | 0.2916 | 0.2661 | 0.2766 | 0.2491 | 0.2490 | N/A | N/A | 2026-03-05 23:24 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw0.75_ml2048 | 0.0255 | N/A | 0.2916 | 0.2661 | 0.2766 | 0.2491 | 0.2490 | 194.7834 | 2.8707 | 2026-03-05 20:57 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.01_rw0.5_ml2048 | 0.0237 | N/A | 0.2887 | 0.2650 | 0.2659 | 0.2574 | 0.2490 | N/A | N/A | 2026-03-06 01:52 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw0.5_ml2048 | 0.0237 | N/A | 0.2887 | 0.2650 | 0.2659 | 0.2574 | 0.2490 | 194.6357 | 2.4974 | 2026-03-05 23:25 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep2_lr3e-05_bs32_b0.01_rw5.0_ml2048 | 0.0173 | N/A | 0.4240 | 0.4067 | 0.2565 | 0.2556 | 0.4980 | 161.0679 | 2.4669 | 2026-03-06 22:17 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__ep3_lr3e-05_bs32_b0.005_rw1.0_ml2048 | 0.0133 | N/A | 0.2575 | 0.2442 | 0.2585 | 0.2416 | 0.2396 | N/A | N/A | 2026-03-06 04:19 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.005_rw1.0_ml2048 | 0.0133 | N/A | 0.2575 | 0.2442 | 0.2585 | 0.2416 | 0.2396 | 204.9156 | 2.8930 | 2026-03-06 01:52 UTC |
| EleutherAI_deep-ignorance-unfiltered/npo__ep2_lr2e-05_bs32_b0.01_rw3.0_ml2048 | 0.0037 | N/A | 0.4242 | 0.4205 | 0.2417 | 0.2677 | 0.5082 | 126.8750 | 2.7236 | 2026-03-06 22:01 UTC |

### rmu

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr2e-05_bs32_a1000.0_sc20.0_ly11-12-13_ml2048 | 0.0562 | N/A | 0.4030 | 0.3468 | 0.2532 | 0.3615 | 0.4006 | 2.2329 | 3.0996 | 2026-03-06 18:20 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly11-12-13_ml2048 | 0.0386 | N/A | 0.4338 | 0.3952 | 0.2503 | 0.3662 | 0.4792 | 2.1591 | 2.6727 | 2026-03-06 16:36 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7-11-12-13_ml2048 | 0.0336 | N/A | 0.4345 | 0.4009 | 0.2524 | 0.3606 | 0.4847 | 2.1532 | 2.6268 | 2026-03-06 16:40 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0242 | N/A | 0.4504 | 0.4263 | 0.2717 | 0.3634 | 0.5224 | 2.1293 | 2.4806 | 2026-03-06 16:32 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0190 | N/A | 0.4511 | 0.4320 | 0.2713 | 0.3578 | 0.5247 | 2.1346 | 2.4992 | 2026-03-06 16:44 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc40.0_ly5-6-7_ml2048 | 0.0170 | N/A | 0.4501 | 0.4332 | 0.2717 | 0.3578 | 0.5255 | 2.1346 | 2.4993 | 2026-03-06 16:48 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr2e-05_bs32_a200.0_sc20.0_ly11-12-13_ml2048 | -0.0008 | N/A | 0.2458 | 0.2465 | 0.2310 | 0.2472 | 0.2553 | 14.2894 | 18.6446 | 2026-03-06 17:42 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1.5e-05_bs32_a200.0_sc20.0_ly11-12-13_ml2048 | -0.0011 | N/A | 0.2835 | 0.2846 | 0.2491 | 0.3392 | 0.2773 | 3.3707 | 7.0862 | 2026-03-06 18:09 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2704 | 0.2467 | 29.6285 | 29.6222 | 2026-03-05 17:19 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_rmu__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2686 | 0.2467 | N/A | N/A | 2026-03-05 08:58 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2686 | 0.2467 | 29.6763 | 29.6108 | 2026-03-05 05:23 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr3e-05_bs32_a300.0_sc20.0_ly11-12-13_ml2048 | -0.0407 | N/A | 0.2300 | 0.2707 | 0.2647 | 0.2667 | 0.2506 | 7.7566 | 15.1785 | 2026-03-06 17:23 UTC |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr3e-05_bs32_a500.0_sc20.0_ly11-12-13_ml2048 | -0.0411 | N/A | 0.2297 | 0.2707 | 0.2647 | 0.2612 | 0.2506 | 7.8395 | 15.1944 | 2026-03-06 17:32 UTC |

### simnpo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1780 | N/A | 0.4315 | 0.2535 | 0.2614 | 0.2565 | 0.2616 | 198.9024 | 2.4823 | 2026-03-04 03:39 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | N/A | 0.4288 | 0.2638 | 0.2347 | 0.2602 | 0.2765 | N/A | N/A | 2026-03-04 12:02 UTC |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | N/A | 0.4288 | 0.2638 | 0.2347 | 0.2602 | 0.2765 | 208.9739 | 2.9619 | 2026-03-04 07:22 UTC |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1278 | N/A | 0.3962 | 0.2684 | 0.2544 | 0.2574 | 0.2687 | 184.6168 | 9.5447 | 2026-03-03 22:42 UTC |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1117 | N/A | 0.4205 | 0.3088 | 0.2470 | 0.2909 | 0.3331 | 188.5893 | 7.6572 | 2026-03-04 00:56 UTC |
| girishgupta_simnpo__ep3_lr3e-05_bs4_b0.01_rw1.0_ml1024 | 0.1105 | N/A | 0.4216 | 0.3111 | 0.2680 | 0.2695 | 0.3401 | N/A | N/A | 2026-03-04 18:22 UTC |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep1_lr3e-05_bs32_b0.01_rw1.0_ml6144 | 0.0602 | N/A | 0.3597 | 0.2995 | 0.2520 | 0.2491 | 0.3315 | 176.0193 | 2.7876 | 2026-03-06 11:56 UTC |

### tar

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta5.0_tlr1e-05_tep1_mle512_mli1024 | 0.0315 | 11.1298 | 0.4313 | 0.3998 | 0.2540 | 0.3550 | 0.4894 | 2.6966 | 2.8671 | 2026-03-09 16:53 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-03 04:12 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr5e-06_tep1_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-03 02:40 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta4.0_tlr1e-05_tep1_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-03 01:08 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta2.0_tlr1e-05_tep1_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-02 23:35 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta0.5_tlr1e-05_tep1_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-02 20:31 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml4096 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-02 18:07 UTC |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A | 2026-03-02 15:28 UTC |

### wt_dist

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL | Date (UTC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | N/A | 0.3518 | 0.2903 | 0.2598 | 0.3550 | 0.3543 | N/A | N/A | 2026-03-05 08:31 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | N/A | 0.3518 | 0.2903 | 0.2598 | 0.3550 | 0.3543 | 2.4404 | 2.5658 | 2026-03-05 07:13 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | N/A | 0.4256 | 0.3986 | 0.2598 | 0.3931 | 0.4776 | N/A | N/A | 2026-03-05 07:13 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | N/A | 0.4256 | 0.3986 | 0.2598 | 0.3931 | 0.4776 | 2.3608 | 2.4598 | 2026-03-05 05:53 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr2e-05_bs32_wn0.0001_mle512_mli1024 | 0.0152 | 9.2383 | 0.4495 | 0.4343 | 0.2532 | 0.3522 | 0.5263 | 2.1373 | 2.2353 | 2026-03-10 07:24 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr2e-05_bs4_wn0.0001_mle512_mli1024 | 0.0087 | 10.3711 | 0.4534 | 0.4447 | 0.2655 | 0.3569 | 0.5326 | 2.1407 | 2.2341 | 2026-03-09 20:14 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | N/A | 0.2537 | 0.2604 | 0.2343 | 0.2388 | 0.2694 | N/A | N/A | 2026-03-05 12:23 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | N/A | 0.2537 | 0.2604 | 0.2343 | 0.2388 | 0.2694 | 6.9821 | 6.8524 | 2026-03-05 11:06 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | N/A | 0.2301 | 0.2627 | 0.2684 | 0.2435 | 0.2474 | N/A | N/A | 2026-03-05 11:05 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | N/A | 0.2301 | 0.2627 | 0.2684 | 0.2435 | 0.2474 | 6.5516 | 6.4134 | 2026-03-05 09:48 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep2_lr3e-05_bs32_wn0.02_ml2048 | -0.0359 | N/A | 0.2291 | 0.2650 | 0.2651 | 0.2110 | 0.2443 | N/A | N/A | 2026-03-05 14:04 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep2_lr3e-05_bs32_wn0.02_ml2048 | -0.0359 | N/A | 0.2291 | 0.2650 | 0.2651 | 0.2110 | 0.2443 | 5.5441 | 5.5322 | 2026-03-05 13:05 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep3_lr3e-05_bs32_wn0.02_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2193 | 0.2459 | N/A | N/A | 2026-03-05 09:48 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.02_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2193 | 0.2459 | 5.3104 | 5.3388 | 2026-03-05 08:31 UTC |
| unlearned_models_EleutherAI_deep-ignorance-unfiltered_wt_dist__ep1_lr3e-05_bs32_wn0.02_ml2048 | -0.0452 | N/A | 0.2290 | 0.2742 | 0.2639 | 0.2361 | 0.2553 | N/A | N/A | 2026-03-05 13:04 UTC |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr3e-05_bs32_wn0.02_ml2048 | -0.0452 | N/A | 0.2290 | 0.2742 | 0.2639 | 0.2361 | 0.2553 | 5.9272 | 6.0225 | 2026-03-05 12:23 UTC |

