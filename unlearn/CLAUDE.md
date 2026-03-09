# Experiment Logs and Unlearning Hyperparameters

## Logging:

When you run a training experiment or hyperparameter tune, save one markdown file per algorithm in an `experiment_logs/` directory

Minimize tables — append rows to existing tables rather than creating new ones

First row is always the baseline model evaluation results

Log rows for the settings you are about to test first, then fill in results as they come

Columns: number of training steps, batch size, final training losses (each available as separately logged loss terms), MMLU accuracy, WMDP-Bio Robust accuracy, experiment date

## Hyperparameter Tuning Steps:

First, find the boundary zone between where accuracy drops on both MMLU and WMDP-Bio Robust, and where it drops on neither

Second, find a good point within that boundary zone — either where both evaluation accuracies drop partway, or where WMDP-Bio Robust reduces to random while MMLU is preserved

Once you find a set of hyperparameters that produces a point within the boundary zone, you may be able to improve performance by reducing the learning rate and increasing the remove coefficient

## Four Subsequent Evaluation States and Actions for Each State

### 1. Both MMLU and WMDP scores drop to random (~25%)

In this state, reduce your learning rate and/or increase your retain coefficient and/or reduce your remove coefficient

### 2. Both MMLU and WMDP scores stay high (~43%-45%)

In this state, increase your learning rate and/or reduce your retain coefficient and/or increase your remove coefficient

### 3. Both drop to between high performance and random (both around 30% to 40%)

In this state, try: 1) reduce your learning rate a small amount and increase your remove coefficient, and 2) increase your retain coefficient

### 4. WMDP drops more than MMLU (27% vs. 43% — this is a decent result)

Success!

## Constraints:
- Don't change training steps unless told to
- Don't write analysis/commentary in the log file such as "Key Findings" or "Conclusions" — just raw results


## Training mode
Default to SFT (full parameter training) unless LoRA is specifically requested.

Tuned lens unlearning requires FSDP (Fully Sharded Data Parallel) when running on GPUs with 95GB of VRAM or less using torchrun, because it holds a reference model and several tuned lenses in memory alongside the training model.

Checkpoint transfer unlearning supports FSDP and DDP via torchrun, and gradient accumulation steps. It holds a frozen checkpoint model copy on each GPU for source activations. SFT requires pdbs=2 on 95GB GPUs (pdbs=4 OOMs).

Sequential SFT uses FSDP (full_shard auto_wrap) via torchrun with a frozen ref model per GPU for retain KL loss.

Orth circuit breakers and simple NPO use DDP with gradient accumulation via torchrun. No reference models.

## Epochs and data budget
Always use 1 epoch unless explicitly told otherwise. Control training length via num_train_examples (or dataset size) and batch size, not epochs. If there is insufficient data, report this.

Before launching any training run, compute and report to the user:

- Total unique training examples
- Total training steps (= examples / (batch_size × grad_accumulation × world_size))
- Effective number of epochs (= steps × batch_size × grad_accumulation / unique_examples)

If the effective epoch count exceeds 1, flag it.

## Learning rates
When training a LoRA the most common successful value is lr=1e-3 or below. When doing SFT it's around 2e-4. Don't push SFT higher than 5e-4 without permission - if you're failing to get learning with an lr above this you likely have a bug.