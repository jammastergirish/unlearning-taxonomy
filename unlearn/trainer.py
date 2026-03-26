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
  'muon'            — MuonWithAuxAdam: hidden 2D weights use Muon
                      (Newton-Schulz orthogonalised updates), all other params
                      (embeddings, biases, norms, lm_head) use AdamW internally.
                      Requires the muon package (declared as a uv inline dep).

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
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import Trainer


# ============================================================
# Muon RMS-matched optimizer wrapper
# ============================================================

class _RMSMatchedMuonWithAuxAdam(torch.optim.Optimizer):
    """SingleDeviceMuonWithAuxAdam with adjust_lr_fn='match_rms_adamw'.

    Muon's Newton-Schulz orthogonalised update has per-element RMS ~1/sqrt(n_cols),
    while AdamW's element-wise normalised update has RMS ~1.  This wrapper scales
    each Muon parameter's effective step by sqrt(n_cols) so the user-facing LR has
    the same meaning for both optimizer groups — no separate Muon LR sweep needed.

    We reimplement step() rather than subclassing the upstream class because it
    enforces strict assert checks on param-group keys that prevent adding metadata.
    """

    def __init__(self, param_groups):
        from muon import muon_update, adam_update          # noqa: F811
        self._muon_update = muon_update
        self._adam_update = adam_update

        # Compute per-parameter RMS scale factors before registering groups.
        self._rms_scales: dict[int, float] = {}
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                for p in group["params"]:
                    self._rms_scales[id(p)] = p.shape[-1] ** 0.5
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = self._muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=group["momentum"],
                    )
                    # match_rms_adamw: Muon's NS-orthogonalised update has
                    # per-element RMS ~ 1/sqrt(n_cols), vs AdamW's ~ 1.
                    # Scaling by sqrt(n_cols) equalises the two so one LR
                    # works for both groups without a separate Muon sweep.
                    scale = self._rms_scales.get(id(p), 1.0)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape),
                           alpha=-group["lr"] * scale)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = self._adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


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

    We override compute_loss() to plug in our unlearning objectives,
    and get_train_dataloader() to disable shuffling (activation caches
    are indexed by step, so iteration order must match cache order).
    """

    def __init__(
        self,
        *args,
        unlearn_args,        # argparse Namespace from unlearn.py
        ref_model=None,      # frozen reference model (DPO / NPO only)
        random_targets=None, # {layer_id: (D,) unit-norm tensor} for RMU
        forget_act_cache=None,  # list of {layer_id: (B,T,D)} cached frozen forget activations
        retain_act_cache=None,  # list of {layer_id: (B,T,D)} cached activations
        layer_ids=None,      # list[int] target layer indices
        norm_reg_target_norms=None,  # list[float] per-layer L2 norm targets
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unlearn_args = unlearn_args
        self.ref_model = ref_model
        self.random_targets = random_targets or {}
        self.forget_act_cache = forget_act_cache or []
        self.retain_act_cache = retain_act_cache or []
        self.layer_ids = layer_ids or []
        self.norm_reg_target_norms = norm_reg_target_norms
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
            "norm_reg_loss":    _mod.norm_reg_loss,
            # primitives needed to decompose losses for W&B logging
            "nll_loss":         _mod.nll_loss,
            "avg_log_prob":     _mod.avg_log_prob,
        }

    def _get_train_sampler(self, dataset=None):
        """Override to disable shuffling — activation caches are indexed by
        step count, so dataset iteration order must match cache order."""
        return SequentialSampler(dataset if dataset is not None else self.train_dataset)

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
            total_steps = len(self.train_dataset)
            scheduled_coeff = min(1.0, self._step_idx / total_steps)
            forget_cache_entry = self.forget_act_cache[self._step_idx % len(self.forget_act_cache)]
            retain_cache_entry = self.retain_act_cache[self._step_idx % len(self.retain_act_cache)]
            loss, orthogonal_raw, retain_raw = cb_loss(
                model, fb, rb, self.layer_ids,
                forget_cache_entry, retain_cache_entry,
                remove_coef=a.steering_coeff, retain_coef=a.alpha,
                scheduled_coeff=scheduled_coeff,
            )
            cb_coeff = a.steering_coeff * (1.0 - 0.25 * scheduled_coeff)
            ret_coeff = a.alpha * scheduled_coeff
            self._record(
                orthogonal_loss=orthogonal_raw,
                retain_loss=retain_raw,
                weighted_forget=cb_coeff * orthogonal_raw,
                weighted_retain=ret_coeff * retain_raw,
                scheduled_coeff=scheduled_coeff,
                forget_coeff=cb_coeff,
                retain_coeff=ret_coeff,
            )

        elif method == "lat":
            total_steps = len(self.train_dataset)
            scheduled_coeff = min(1.0, self._step_idx / total_steps)
            loss = lat_loss(
                model, fb, rb, self.layer_ids,
                a.lat_eps, a.lat_steps,
                retain_coef=a.retain_weight,
                scheduled_coeff=scheduled_coeff,
            )

        elif method == "cb_lat":
            total_steps = len(self.train_dataset)
            scheduled_coeff = min(1.0, self._step_idx / total_steps)
            forget_cache_entry = self.forget_act_cache[self._step_idx % len(self.forget_act_cache)]
            retain_cache_entry = self.retain_act_cache[self._step_idx % len(self.retain_act_cache)]
            loss, orthogonal_raw, retain_raw = cb_lat_loss(
                model, fb, rb, self.layer_ids,
                forget_cache_entry, retain_cache_entry,
                remove_coef=a.steering_coeff, retain_coef=a.alpha,
                lat_eps=a.lat_eps, lat_steps=a.lat_steps,
                scheduled_coeff=scheduled_coeff,
            )
            cb_coeff = a.steering_coeff * (1.0 - 0.25 * scheduled_coeff)
            ret_coeff = a.alpha * scheduled_coeff
            self._record(
                orthogonal_loss=orthogonal_raw,
                retain_loss=retain_raw,
                weighted_forget=cb_coeff * orthogonal_raw,
                weighted_retain=ret_coeff * retain_raw,
                scheduled_coeff=scheduled_coeff,
                forget_coeff=cb_coeff,
                retain_coeff=ret_coeff,
            )

        elif method == "wt_dist":
            loss = wt_dist_loss(model, rb)

        elif method == "wt_dist_reg":
            # pretrained_params is stored on the trainer via set_pretrained_params()
            loss = wt_dist_reg_loss(model, rb, self.pretrained_params, a.wt_reg_lambda)

        else:
            raise ValueError(f"[UnlearningTrainer] Unknown method: {method}")

        # ---- Optional activation-norm regularisation (cross-cutting) ----
        nrl = getattr(a, "norm_reg_lambda", 0.0)
        if nrl > 0 and self.norm_reg_target_norms is not None:
            norm_reg_loss = self._loss_fns["norm_reg_loss"]
            l_norm = norm_reg_loss(model, fb, self.norm_reg_target_norms)
            loss = loss + nrl * l_norm
            self._record(norm_reg_loss=l_norm)

        self._step_idx += 1

        return (loss, None) if return_outputs else loss

    def set_pretrained_params(self, pretrained_params: dict):
        """Store frozen pretrained params for wt_dist_reg loss."""
        self.pretrained_params = pretrained_params

    def create_optimizer(self):
        """Build the optimizer; use MuonWithAuxAdam when --optimizer muon is set.

        Muon is designed for hidden 2D weight matrices only.  Embeddings,
        biases, layer-norm scales, and the lm_head must still use AdamW —
        MuonWithAuxAdam handles this with a single unified optimizer object
        (one param group with use_muon=True, one with use_muon=False).

        The Muon group uses match_rms_adamw LR adjustment: each parameter's
        effective step is scaled by sqrt(n_cols) so that the per-element RMS
        of the Muon update matches AdamW's.  This means a single --learning-rate
        value works for both groups without separate tuning.

        For 'adamw' (the default) we fall through to the HF Trainer's own
        create_optimizer(), which handles weight-decay groups, etc.
        """
        opt_name = getattr(self.unlearn_args, "optimizer", "adamw")
        if opt_name != "muon":
            return super().create_optimizer()

        # Non-hidden parameters: embeddings, layer norms, biases, lm_head.
        # These go to the auxiliary AdamW inside _RMSMatchedMuonWithAuxAdam.
        _NON_HIDDEN = ("embed", "norm", "lm_head", "bias")

        model = self.model
        hidden_weights, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_hidden_2d = (
                param.ndim >= 2
                and not any(k in name for k in _NON_HIDDEN)
            )
            if is_hidden_2d:
                hidden_weights.append(param)
            else:
                other_params.append(param)

        lr = self.args.learning_rate
        param_groups = [
            # Muon group: hidden weight matrices
            dict(params=hidden_weights, use_muon=True,
                 lr=lr, weight_decay=0.01),
            # AdamW group: everything else
            dict(params=other_params, use_muon=False,
                 lr=lr, betas=(0.9, 0.95), weight_decay=0.01),
        ]

        print(
            f"[UnlearningTrainer] Using MuonWithAuxAdam (match_rms_adamw): "
            f"{len(hidden_weights)} hidden-weight tensors → Muon, "
            f"{len(other_params)} other tensors → AdamW"
        )

        self.optimizer = _RMSMatchedMuonWithAuxAdam(param_groups)

        # Log the per-parameter RMS scale factors for transparency
        scales = self.optimizer._rms_scales
        unique_scales = sorted(set(scales.values()))
        print(f"[UnlearningTrainer] match_rms_adamw scale factors: {unique_scales}")

        return self.optimizer
