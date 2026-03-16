"""
UnlearningTrainer — HF Trainer subclass for mixed-precision unlearning.

Why this exists:
  PyTorch's default bf16 training puts EVERYTHING (model, optimizer states,
  Adam m/v buffers) in bf16, which loses precision in the gradient accumulator.
  The HF Trainer's bf16=True instead enables *mixed precision*:
    - Forward pass + loss: bf16  (fast)
    - Master weight copy + Adam m/v states: fp32  (precise)
  This gives us the speed of bf16 with the stability of fp32 optimizers.

Optimizer options (controlled by unlearn_args.optimizer):
  'adamw' (default) — standard HF Trainer AdamW, no extra deps.
  'muon'            — MuonAdamW: hidden 2D weights use torch.optim.Muon
                      (Newton-Schulz orthogonalised updates), all other params
                      (embeddings, biases, norms, lm_head) use AdamW.
                      Requires PyTorch ≥2.6 (no external deps).

Usage (from unlearn.py main()):
    from unlearn.trainer import UnlearningTrainer, UnlearningDataset, UnlearningCollator
    dataset  = UnlearningDataset(forget_batches, retain_batches)
    collator = UnlearningCollator()
    trainer  = UnlearningTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, unlearn_args=args, ref_model=ref_model,
        random_targets=random_targets, retain_act_cache=retain_act_cache,
        layer_ids=layer_ids,
    )
    trainer.train()
"""

import torch
from torch.utils.data import Dataset
from transformers import Trainer


# ============================================================
# Dataset / Collator
# ============================================================

class UnlearningDataset(Dataset):
    """Paired forget/retain batch dataset.

    Each item is a (forget_batch, retain_batch) pair of pre-tokenized
    {input_ids, attention_mask} dicts.  The batches are already stacked
    to batch_size by unlearn.py before being passed here, so the Trainer's
    DataLoader just fetches one pair per step.

    We set per_device_train_batch_size=1 in TrainingArguments because the
    "batch" dimension was already handled by make_batches() in unlearn.py.
    """
    def __init__(self, forget_batches: list[dict], retain_batches: list[dict]):
        n = min(len(forget_batches), len(retain_batches))
        self.forget = forget_batches[:n]
        self.retain = retain_batches[:n]

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # Return CPU tensors; the Trainer moves them to the correct device.
        return {
            "forget": {k: v.cpu() for k, v in self.forget[idx].items()},
            "retain": {k: v.cpu() for k, v in self.retain[idx].items()},
        }


class UnlearningCollator:
    """Collate a list of (forget, retain) pairs into a single batch dict.

    Since each item already has the full batch dimension from make_batches(),
    we just stack the outer list dimension (which will be size 1 because
    per_device_train_batch_size=1) and squeeze it back out.
    """
    def __call__(self, features: list[dict]) -> dict:
        # features is a list of {"forget": {...}, "retain": {...}}
        # We concatenate along dim 0 to combine items from the DataLoader batch.
        def _cat(tensors):
            return torch.cat(tensors, dim=0)

        forget = {
            k: _cat([f["forget"][k] for f in features])
            for k in features[0]["forget"]
        }
        retain = {
            k: _cat([f["retain"][k] for f in features])
            for k in features[0]["retain"]
        }
        return {"forget": forget, "retain": retain}


# ============================================================
# MuonAdamW — torch.optim.Muon + AdamW composite optimizer
# ============================================================

# Non-hidden parameter name fragments — these always go to AdamW.
_NON_HIDDEN = ("embed", "norm", "lm_head", "bias")


class MuonAdamW(torch.optim.Optimizer):
    """Composite optimizer: torch.optim.Muon for 2D hidden matrices,
    torch.optim.AdamW for everything else (embeddings, biases, norms).

    Presents a single optimizer interface so HF Trainer's LR scheduler
    can see and update all param_groups.

    Requires PyTorch ≥2.6 (torch.optim.Muon).
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        muon_momentum=0.95,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        weight_decay=0.01,
    ):
        muon_params, adam_params = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and not any(k in name for k in _NON_HIDDEN):
                muon_params.append(p)
            else:
                adam_params.append(p)

        self.optimizers = []

        if muon_params:
            self.muon = torch.optim.Muon(
                muon_params,
                lr=lr,
                momentum=muon_momentum,
                weight_decay=weight_decay,
            )
            self.optimizers.append(self.muon)

        if adam_params:
            self.adam = torch.optim.AdamW(
                adam_params,
                lr=lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=weight_decay,
            )
            self.optimizers.append(self.adam)

        # Combine param_groups so HF Scheduler can see/update all LRs.
        combined_groups = []
        for opt in self.optimizers:
            combined_groups.extend(opt.param_groups)
        super().__init__(combined_groups, {})

        print(
            f"[MuonAdamW] {len(muon_params)} params → Muon, "
            f"{len(adam_params)} params → AdamW"
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)


# ============================================================
# Trainer subclass
# ============================================================

class UnlearningTrainer(Trainer):
    """HF Trainer subclass that uses our custom unlearning losses.

    The Trainer handles:
      - Mixed-precision bf16 (fp32 master weights + fp32 Adam states)
      - Gradient accumulation  (TrainingArguments.gradient_accumulation_steps)
      - Gradient clipping      (TrainingArguments.max_grad_norm)
      - Cosine LR scheduler    (TrainingArguments.lr_scheduler_type="cosine")
      - WandB logging          (TrainingArguments.report_to="wandb")
      - Multi-GPU via Accelerate (transparent)

    We only override compute_loss() to plug in our unlearning objectives.
    """

    def __init__(
        self,
        *args,
        unlearn_args,        # argparse Namespace from unlearn.py
        ref_model=None,      # frozen reference model (DPO / NPO only)
        random_targets=None, # {layer_id: (D,) unit-norm tensor} for RMU/CB/CB-LAT
        retain_act_cache=None,  # list of {layer_id: (B,T,D)} cached activations
        layer_ids=None,      # list[int] target layer indices
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unlearn_args = unlearn_args
        self.ref_model = ref_model
        self.random_targets = random_targets or {}
        self.retain_act_cache = retain_act_cache or []
        self.layer_ids = layer_ids or []
        self._step_idx = 0  # tracks which retain_act_cache entry to use
        # Per-step component metrics to be flushed into W&B via log().
        # compute_loss() populates this dict; log() drains it.
        self._custom_metrics: dict = {}

        # Load loss functions once from the sibling unlearn.py file.
        # We can't do `from unlearn import ...` because unlearn.py is run as a
        # standalone uv --script, not an installed package. importlib lets us
        # load it by absolute path regardless.
        # We cache the result so compute_loss doesn't re-execute unlearn.py
        # top-level code (argparse, etc.) on every training step.
        import importlib.util, os
        _unlearn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlearn.py")
        _spec = importlib.util.spec_from_file_location("_unlearn_module", _unlearn_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        self._loss_fns = {
            "ga_simple_loss":   _mod.ga_simple_loss,
            "ga_loss":          _mod.ga_loss,
            "grad_diff_loss":   _mod.grad_diff_loss,
            "dpo_loss":         _mod.dpo_loss,
            "npo_loss":         _mod.npo_loss,
            "simnpo_loss":      _mod.simnpo_loss,
            "rmu_loss":         _mod.rmu_loss,
            "cb_loss":          _mod.cb_loss,
            "lat_loss":         _mod.lat_loss,
            "cb_lat_loss":      _mod.cb_lat_loss,
            "wt_dist_loss":     _mod.wt_dist_loss,
            "wt_dist_reg_loss": _mod.wt_dist_reg_loss,
            # primitives needed to decompose losses for W&B logging
            "nll_loss":         _mod.nll_loss,
            "avg_log_prob":     _mod.avg_log_prob,
        }

    # ------------------------------------------------------------------
    # W&B metric helpers
    # ------------------------------------------------------------------

    def log(self, logs: dict, *args, **kwargs) -> None:
        """Merge any per-step component metrics into every Trainer log call.

        compute_loss() stashes per-component scalars in self._custom_metrics
        (e.g. forget_loss, retain_loss).  The Trainer calls log() once per
        logging step, so we drain that dict here and let the super() call
        forward everything to W&B / console as usual.
        """
        if self._custom_metrics:
            logs.update(self._custom_metrics)
            self._custom_metrics = {}
        super().log(logs, *args, **kwargs)

    def _record(self, **metrics):
        """Store scalar metric values for the current step.

        Converts tensors to Python floats so they are JSON-serialisable for
        W&B.  Existing keys are overwritten (last write per step wins, which
        is fine since compute_loss is called once per step).
        """
        for k, v in metrics.items():
            self._custom_metrics[k] = v.item() if hasattr(v, "item") else float(v)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Dispatch to the appropriate unlearning loss function.

        `inputs` comes from UnlearningCollator: {"forget": {...}, "retain": {...}}.
        The Trainer has already moved tensors to the right device.

        Note on LAT / CB-LAT inner loop:
          The adversarial PGD inner loop calls loss.backward(inputs=[delta]) —
          this computes grad only w.r.t. delta, NOT model params.  This is safe
          inside compute_loss because the Trainer calls .backward() on the *return
          value* of this function (the outer loss), not on the inner-loop losses.
          The two backward passes are independent.
        """
        a = self.unlearn_args
        fb = inputs["forget"]
        rb = inputs["retain"]

        # Move to the model's device (the Trainer may not do this for nested dicts)
        device = next(model.parameters()).device
        fb = {k: v.to(device) for k, v in fb.items()}
        rb = {k: v.to(device) for k, v in rb.items()}

        # Unpack loss functions from the cached dict
        ga_simple_loss   = self._loss_fns["ga_simple_loss"]
        ga_loss          = self._loss_fns["ga_loss"]
        grad_diff_loss   = self._loss_fns["grad_diff_loss"]
        dpo_loss         = self._loss_fns["dpo_loss"]
        npo_loss         = self._loss_fns["npo_loss"]
        simnpo_loss      = self._loss_fns["simnpo_loss"]
        rmu_loss         = self._loss_fns["rmu_loss"]
        cb_loss          = self._loss_fns["cb_loss"]
        lat_loss         = self._loss_fns["lat_loss"]
        cb_lat_loss      = self._loss_fns["cb_lat_loss"]
        wt_dist_loss     = self._loss_fns["wt_dist_loss"]
        wt_dist_reg_loss = self._loss_fns["wt_dist_reg_loss"]
        nll_loss         = self._loss_fns.get("nll_loss")
        avg_log_prob     = self._loss_fns.get("avg_log_prob")

        method = a.method

        if method == "ga_simple":
            forget_nll = nll_loss(model, fb)
            loss = -forget_nll
            self._record(forget_loss=forget_nll)

        elif method == "ga":
            forget_nll = nll_loss(model, fb)
            retain_nll = nll_loss(model, rb)
            loss = -forget_nll + a.retain_weight * retain_nll
            self._record(forget_loss=forget_nll, retain_loss=retain_nll)

        elif method == "grad_diff":
            forget_nll = nll_loss(model, fb)
            retain_nll = nll_loss(model, rb)
            loss = retain_nll - a.forget_weight * forget_nll
            self._record(forget_loss=forget_nll, retain_loss=retain_nll)

        elif method == "dpo":
            loss = dpo_loss(model, self.ref_model, fb, rb, a.beta)

        elif method == "npo":
            # Recompute components for logging without an extra forward pass
            lp_forget = avg_log_prob(model, fb)
            with torch.no_grad():
                import torch.nn.functional as _F
                ref_lp = avg_log_prob(self.ref_model, fb)
            npo_term = -(2.0 / a.beta) * _F.logsigmoid(
                -a.beta * (lp_forget - ref_lp)
            ).mean()
            retain_nll = nll_loss(model, rb)
            loss = npo_term + a.retain_weight * retain_nll
            self._record(npo_term=npo_term, retain_loss=retain_nll)

        elif method == "simnpo":
            lp_forget = avg_log_prob(model, fb)
            import torch.nn.functional as _F
            simnpo_term = -(2.0 / a.beta) * _F.logsigmoid(-a.beta * lp_forget).mean()
            retain_nll = nll_loss(model, rb)
            loss = simnpo_term + a.retain_weight * retain_nll
            self._record(simnpo_term=simnpo_term, retain_loss=retain_nll)

        elif method == "rmu":
            cache_entry = self.retain_act_cache[self._step_idx % len(self.retain_act_cache)]
            loss = rmu_loss(
                model, fb, rb, self.layer_ids,
                self.random_targets, cache_entry,
                a.steering_coeff, a.alpha,
            )

        elif method == "cb":
            cache_entry = self.retain_act_cache[self._step_idx % len(self.retain_act_cache)]
            loss = cb_loss(
                model, fb, rb, self.layer_ids,
                self.random_targets, cache_entry,
                a.steering_coeff, a.alpha,
            )

        elif method == "lat":
            loss = lat_loss(
                model, fb, rb, self.layer_ids,
                a.lat_eps, a.lat_steps, a.retain_weight,
            )

        elif method == "cb_lat":
            cache_entry = self.retain_act_cache[self._step_idx % len(self.retain_act_cache)]
            loss = cb_lat_loss(
                model, fb, rb, self.layer_ids,
                self.random_targets, cache_entry,
                a.steering_coeff, a.alpha,
                a.lat_eps, a.lat_steps,
            )

        elif method == "wt_dist":
            loss = wt_dist_loss(model, rb)

        elif method == "wt_dist_reg":
            # pretrained_params is stored on the trainer via set_pretrained_params()
            loss = wt_dist_reg_loss(model, rb, self.pretrained_params, a.wt_reg_lambda)

        else:
            raise ValueError(f"[UnlearningTrainer] Unknown method: {method}")

        self._step_idx += 1

        return (loss, None) if return_outputs else loss

    def set_pretrained_params(self, pretrained_params: dict):
        """Store frozen pretrained params for wt_dist_reg loss."""
        self.pretrained_params = pretrained_params

    def create_optimizer(self):
        """Build the optimizer; use MuonAdamW when --optimizer muon is set.

        Muon is designed for hidden 2D weight matrices only.  Embeddings,
        biases, layer-norm scales, and the lm_head must still use AdamW.
        MuonAdamW composes torch.optim.Muon + torch.optim.AdamW behind a
        single optimizer interface that HF Trainer's LR scheduler can drive.

        For 'adamw' (the default) we fall through to the HF Trainer's own
        create_optimizer(), which handles weight-decay groups, etc.
        """
        opt_name = getattr(self.unlearn_args, "optimizer", "adamw")
        if opt_name != "muon":
            return super().create_optimizer()

        self.optimizer = MuonAdamW(
            self.model.named_parameters(),
            lr=self.args.learning_rate,
        )
        return self.optimizer
