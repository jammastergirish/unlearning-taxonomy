#!/usr/bin/env bash
# Ninth sweep: Optimizing `cb` (Concept-Bottleneck) based on Sweep results.
#
# Current best cb:
#   ep3_lr3e-05_bs4_a100.0_sc20.0_ly5-6-7 -> Score 0.0921 (MMLU 0.3502, WMDP 0.2581)
#
# Key observations from prior sweeps:
#   - ly5-6-7 (tight span) works far better than ly5-10-15-20-25-30 for cb.
#   - alpha=100 + sc=20 is the best tested combo, but this is the first sweep
#     to systematically vary alpha, sc, bs, lr, and epochs *together* for cb.
#   - cb_lat best (Score 0.1177) used bs=8 + alpha=50, hinting cb might also
#     benefit from higher bs. However cb has no LAT, so the scaling dynamics differ.
#
# This sweep explores:
#   1. Batch size scaling: does bs=8 help cb the way it helped cb_lat?
#   2. Alpha scaling: does lowering alpha (50, 200) at bs=4 or bs=8 push WMDP lower?
#   3. Steering coeff scaling: sc=10, 30, 40 — is sc=20 optimal?
#   4. Higher LR (5e-05, 7e-05): can cb tolerate more aggressive updates?
#   5. More epochs (4, 5): does cb benefit from longer training at ly5-6-7?
#   6. Alternative tight spans: ly4-5-6, ly6-7-8 — is layer 5-6-7 the sweet spot?

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting NINTH hyperparameter sweep"
echo "Optimizing cb: alpha, steering_coeff, bs, lr, epochs, layers"
echo "Model: $BASE"
echo "=========================================================="

METHOD="cb"

# ---------------------------------------------------------------
# 1. Batch Size Scaling at Optimal Config
# Best: bs=4, alpha=100, sc=20, lr=3e-05 -> 0.0921
# Question: Does bs=8 smooth gradients the way it did for cb_lat?
# If pattern from cb_lat holds (bs=8 + alpha/2), test alpha=50 at bs=8.
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Batch size scaling at alpha=100 and alpha=50 <<<" 
for alpha in "100.0" "50.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=8 ALPHA=$alpha STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 2. Alpha Sweep at Baseline Config (bs=4, lr=3e-05)
# Best tested: alpha=100. Does higher alpha push WMDP lower?
# Does lower alpha preserve MMLU while still unlearning?
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Alpha sweep at baseline bs=4, lr=3e-05 <<<"
for alpha in "50.0" "150.0" "200.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=4 ALPHA=$alpha STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 3. Steering Coefficient Sweep
# sc=20 is the only tested value at ly5-6-7. Higher sc -> stronger
# steering toward the retaining direction. Lower sc -> weaker regularization.
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Steering coefficient sweep at best alpha=100 <<<"
for sc in "10.0" "30.0" "40.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=4 ALPHA=100.0 STEERING_COEFF=$sc LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 4. Higher Learning Rate
# lr=3e-05 is the best tested. Can cb tolerate 5e-05 or 7e-05 without MMLU collapse?
# cb_lat survived 7e-05 at bs=8 (Score 0.0924), so cb may too.
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Higher LR sweep at optimal alpha=100, bs=4 <<<"
for lr in "5e-05" "7e-05"; do
    EPOCHS=3 LR=$lr BATCH_SIZE=4 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh $METHOD
done

# Also try higher LR at bs=8 — the combination may work where bs=4 collapses
echo -e "\n>>> [cb] Higher LR at bs=8 (gradient smoothing may protect MMLU) <<<"
EPOCHS=3 LR=5e-05 BATCH_SIZE=8 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
    ./unlearn/run_unlearn.sh $METHOD

# ---------------------------------------------------------------
# 5. Epoch Sweep at Best Config
# All prior cb runs used ep=3. Does more training help at ly5-6-7?
# (Unlike ly5-10-15-20-25-30, this span directly hits factual layers.)
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Epoch sweep at best config (bs=4, alpha=100, sc=20) <<<"
for ep in "4" "5"; do
    EPOCHS=$ep LR=3e-05 BATCH_SIZE=4 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 6. Adjacent Layer Span Variants
# ly5-6-7 works best, but maybe shifting one layer helps.
# Test ly4-5-6 (earlier) and ly6-7-8 (later) at the best config.
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Adjacent layer spans at best config <<<"
for layers in "4,5,6" "6,7,8"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=4 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="$layers" \
        ./unlearn/run_unlearn.sh $METHOD
done

echo "=========================================================="
echo "Sweep 9 completed successfully!"
echo "=========================================================="
