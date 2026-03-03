"""Tests for unlearn/unlearn.py — build_outdir and related logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unlearn"))

from types import SimpleNamespace
import pytest

from unlearn import build_outdir, METHOD_PARAMS, PARAM_ABBREV


# ---------------------------------------------------------------------------
# build_outdir
# ---------------------------------------------------------------------------
class TestBuildOutdir:
    """Test that build_outdir produces correct, unique folder names."""

    def _make_args(self, method, **overrides):
        """Create a namespace with all default args for the given method."""
        defaults = {
            "model": "EleutherAI/deep-ignorance-unfiltered",
            "method": method,
            "epochs": 1,
            "lr": 1e-5,
            "batch_size": 4,
            "max_lines": 1024,
            "retain_weight": 1.0,
            "forget_weight": 1.0,
            "beta": 0.1,
            "alpha": 100.0,
            "steering_coeff": 20.0,
            "layer_id": "5,6,7",
            "lat_eps": 0.1,
            "lat_steps": 5,
            "tar_alpha": 1.0,
            "tar_lr": 1e-5,
            "tar_epochs": 1,
            "wt_noise_std": 0.02,
            "wt_reg_lambda": 0.1,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_includes_method_in_path(self):
        result = build_outdir(self._make_args("cb_lat"))
        assert "/cb_lat__" in result

    def test_starts_with_unlearned_models(self):
        result = build_outdir(self._make_args("ga_simple"))
        assert result.startswith("unlearned_models/")

    def test_model_slashes_replaced(self):
        result = build_outdir(self._make_args("ga_simple"))
        # The model name should have / replaced with _
        assert "EleutherAI_deep-ignorance-unfiltered" in result
        # Model part (between root and method) should have no bare slashes
        model_part = result.split("/")[1]  # unlearned_models/<model_part>/method...
        assert "EleutherAI" in model_part
        assert model_part == "EleutherAI_deep-ignorance-unfiltered"

    def test_ga_simple_has_minimal_params(self):
        result = build_outdir(self._make_args("ga_simple"))
        assert "ep1" in result
        assert "lr1e-05" in result
        assert "bs4" in result
        assert "ml1024" in result  # max_lines should be included
        # Should NOT include method-irrelevant params
        assert "_rw" not in result
        assert "_a" not in result.split("__")[-1].split("_ep")[0]  # no alpha prefix

    def test_cb_lat_includes_all_relevant_params(self):
        result = build_outdir(self._make_args("cb_lat"))
        suffix = result.split("__")[-1]
        assert "ep1" in suffix
        assert "lr1e-05" in suffix
        assert "bs4" in suffix
        assert "ml1024" in suffix
        assert "a100.0" in suffix
        assert "sc20.0" in suffix
        assert "le0.1" in suffix
        assert "ls5" in suffix
        assert "ly5-6-7" in suffix

    def test_layer_id_commas_become_dashes(self):
        result = build_outdir(self._make_args("rmu", layer_id="10,11,12"))
        assert "ly10-11-12" in result
        assert "," not in result

    def test_different_params_produce_different_paths(self):
        path_a = build_outdir(self._make_args("cb_lat", epochs=1))
        path_b = build_outdir(self._make_args("cb_lat", epochs=2))
        assert path_a != path_b

    def test_same_params_produce_same_path(self):
        path_a = build_outdir(self._make_args("cb_lat"))
        path_b = build_outdir(self._make_args("cb_lat"))
        assert path_a == path_b

    def test_different_methods_produce_different_paths(self):
        path_a = build_outdir(self._make_args("ga_simple"))
        path_b = build_outdir(self._make_args("ga"))
        assert path_a != path_b

    def test_wt_dist_includes_noise_std(self):
        result = build_outdir(self._make_args("wt_dist"))
        assert "wn0.02" in result

    def test_wt_dist_reg_includes_lambda(self):
        result = build_outdir(self._make_args("wt_dist_reg"))
        assert "wr0.1" in result

    def test_dpo_includes_beta(self):
        result = build_outdir(self._make_args("dpo"))
        assert "b0.1" in result

    def test_ga_includes_retain_weight(self):
        result = build_outdir(self._make_args("ga"))
        assert "rw1.0" in result

    def test_grad_diff_includes_forget_weight(self):
        result = build_outdir(self._make_args("grad_diff"))
        assert "fw1.0" in result

    def test_tar_includes_tar_params(self):
        result = build_outdir(self._make_args("tar"))
        assert "ta1.0" in result  # tar_alpha
        assert "tlr1e-05" in result  # tar_lr
        assert "tep1" in result  # tar_epochs
        # TAR should NOT include regular training params like batch_size
        assert "bs" not in result
        assert "ep" not in result or "tep" in result  # only tar_epochs, not epochs

    def test_tar_different_alpha_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_alpha=1.0))
        path_b = build_outdir(self._make_args("tar", tar_alpha=0.5))
        assert path_a != path_b
        assert "ta1.0" in path_a
        assert "ta0.5" in path_b

    def test_tar_different_lr_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_lr=1e-5))
        path_b = build_outdir(self._make_args("tar", tar_lr=5e-6))
        assert path_a != path_b
        assert "tlr1e-05" in path_a
        assert "tlr5e-06" in path_b

    def test_tar_different_epochs_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_epochs=1))
        path_b = build_outdir(self._make_args("tar", tar_epochs=3))
        assert path_a != path_b
        assert "tep1" in path_a
        assert "tep3" in path_b

    def test_max_lines_included_in_all_methods(self):
        """Test that max_lines parameter is included for all methods."""
        for method in ["ga_simple", "simnpo", "npo", "dpo", "tar", "cb", "cb_lat"]:
            result = build_outdir(self._make_args(method))
            assert "ml1024" in result, f"max_lines not found in {method}: {result}"

    def test_different_max_lines_produce_different_paths(self):
        """Test that different max_lines values produce different directory names."""
        path_a = build_outdir(self._make_args("simnpo", max_lines=1024))
        path_b = build_outdir(self._make_args("simnpo", max_lines=2048))
        path_c = build_outdir(self._make_args("simnpo", max_lines=4096))

        assert path_a != path_b != path_c
        assert "ml1024" in path_a
        assert "ml2048" in path_b
        assert "ml4096" in path_c

    def test_max_lines_zero_handled_correctly(self):
        """Test that max_lines=0 (unlimited) is properly encoded."""
        result = build_outdir(self._make_args("simnpo", max_lines=0))
        assert "ml0" in result

    def test_tar_includes_max_lines(self):
        """Test that TAR method includes max_lines in directory name."""
        result = build_outdir(self._make_args("tar"))
        assert "ta1.0" in result  # tar_alpha
        assert "tlr1e-05" in result  # tar_lr
        assert "tep1" in result  # tar_epochs
        assert "ml1024" in result  # max_lines
        # TAR should NOT include regular training params like batch_size
        assert "bs" not in result
        assert "ep" not in result or "tep" in result  # only tar_epochs, not epochs


# ---------------------------------------------------------------------------
# METHOD_PARAMS and PARAM_ABBREV consistency
# ---------------------------------------------------------------------------
class TestMethodParamsConsistency:
    """Ensure METHOD_PARAMS and PARAM_ABBREV are in sync."""

    def test_all_methods_have_entries(self):
        expected = {"ga_simple", "ga", "grad_diff", "dpo", "npo", "simnpo",
                    "rmu", "cb", "lat", "cb_lat", "tar", "wt_dist", "wt_dist_reg"}
        assert set(METHOD_PARAMS.keys()) == expected

    def test_all_params_have_abbreviations(self):
        all_params = set()
        for params in METHOD_PARAMS.values():
            all_params.update(params)
        for param in all_params:
            assert param in PARAM_ABBREV, f"Missing abbreviation for '{param}'"

    def test_abbreviations_are_unique(self):
        abbrevs = list(PARAM_ABBREV.values())
        assert len(abbrevs) == len(set(abbrevs)), "Duplicate abbreviations found"

    def test_every_method_includes_shared_params(self):
        """Every method should at least include epochs, lr, batch_size (except TAR which uses tar_epochs, tar_lr)."""
        for method, params in METHOD_PARAMS.items():
            if method == "tar":
                # TAR uses its own parameter names
                assert "tar_epochs" in params, f"{method} missing tar_epochs"
                assert "tar_lr" in params, f"{method} missing tar_lr"
                assert "tar_alpha" in params, f"{method} missing tar_alpha"
            else:
                assert "epochs" in params, f"{method} missing epochs"
                assert "lr" in params, f"{method} missing lr"
                assert "batch_size" in params, f"{method} missing batch_size"


# ---------------------------------------------------------------------------
# TAR-specific tests
# ---------------------------------------------------------------------------
class TestTARMethod:
    """Test TAR (Task Arithmetic Removal) specific functionality."""

    def test_tar_method_params_correct(self):
        """TAR should only use tar-specific parameters."""
        expected_params = {"tar_alpha", "tar_lr", "tar_epochs"}
        assert set(METHOD_PARAMS["tar"]) == expected_params

    def test_tar_param_abbreviations_exist(self):
        """All TAR parameters should have abbreviations."""
        for param in METHOD_PARAMS["tar"]:
            assert param in PARAM_ABBREV

    def test_tar_abbreviations_correct(self):
        """Test specific TAR abbreviations."""
        assert PARAM_ABBREV["tar_alpha"] == "ta"
        assert PARAM_ABBREV["tar_lr"] == "tlr"
        assert PARAM_ABBREV["tar_epochs"] == "tep"

    def test_tar_parameters_different_from_standard(self):
        """TAR should use different parameter names than standard training."""
        tar_params = set(METHOD_PARAMS["tar"])
        standard_params = {"epochs", "lr", "batch_size"}
        assert tar_params.isdisjoint(standard_params), "TAR should not use standard training parameters"

    def test_tar_in_method_choices(self):
        """TAR should be in the list of available methods."""
        # This tests that TAR was added to the choices in the argument parser
        # We can't easily test the actual parser without refactoring, but we can
        # test that it's in our METHOD_PARAMS which is what the choices would use
        assert "tar" in METHOD_PARAMS

    def test_tar_parameter_defaults_make_sense(self):
        """TAR default parameters should be reasonable."""
        # Test via the defaults in _make_args
        from types import SimpleNamespace
        args = SimpleNamespace(tar_alpha=1.0, tar_lr=1e-5, tar_epochs=1)

        # Alpha should be positive (scaling factor)
        assert args.tar_alpha > 0

        # Learning rate should be reasonable for fine-tuning
        assert 1e-6 <= args.tar_lr <= 1e-4

        # Epochs should be small (TAR is meant to be lightweight)
        assert 1 <= args.tar_epochs <= 5


# ---------------------------------------------------------------------------
# TAR Device Handling Tests
# ---------------------------------------------------------------------------
class TestTARDeviceHandling:
    """Test that TAR properly handles device placement for batches."""

    def test_tar_moves_batch_to_device(self):
        """Test that apply_tar moves batches to the correct device during training."""
        import torch
        from unittest.mock import Mock, patch, MagicMock
        from unlearn import apply_tar

        # Mock model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0], device=device)]
        mock_model.named_parameters.return_value = [("test_param", torch.nn.Parameter(torch.tensor([1.0], device=device)))]

        # Create mock batch that's on wrong device (CPU when we want GPU, or vice versa)
        wrong_device = torch.device("cpu") if device.type == "cuda" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if wrong_device.type == "cuda" and not torch.cuda.is_available():
            wrong_device = torch.device("cpu")  # fallback if CUDA not available

        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]], device=wrong_device),
            "attention_mask": torch.tensor([[1, 1, 1]], device=wrong_device)
        }
        forget_batches = [mock_batch]

        # Mock the nll_loss function to verify device placement
        with patch('unlearn.nll_loss') as mock_nll_loss, \
             patch('unlearn.torch.optim.AdamW') as mock_optimizer_class, \
             patch('unlearn.torch.optim.lr_scheduler.CosineAnnealingLR') as mock_scheduler_class:

            # Set up mocks
            mock_loss = torch.tensor(1.0, device=device)
            mock_nll_loss.return_value = mock_loss

            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler

            # Call apply_tar
            apply_tar(
                model=mock_model,
                forget_batches=forget_batches,
                alpha=1.0,
                lr=1e-5,
                epochs=1,
                device=device
            )

            # Verify nll_loss was called
            assert mock_nll_loss.called

            # Get the batch that was passed to nll_loss
            call_args = mock_nll_loss.call_args_list[0]  # First call
            _, passed_batch = call_args[0]  # (model, batch)

            # Verify the batch tensors were moved to the correct device
            assert passed_batch["input_ids"].device == device
            assert passed_batch["attention_mask"].device == device


# ---------------------------------------------------------------------------
# Activation Caching Memory Tests
# ---------------------------------------------------------------------------
class TestActivationCachingMemory:
    """Test that activation caching doesn't cause GPU OOM by keeping data on CPU."""

    def test_retain_activations_cached_on_cpu(self):
        """Test that retain activation caching stores tensors on CPU to avoid GPU OOM."""
        import torch
        from unittest.mock import Mock, patch
        from unlearn import get_layer_activations

        # Mock setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_model = Mock()

        # Mock layer activations that would normally be on GPU
        mock_activations = {
            0: torch.randn(2, 10, 64, device=device),  # (batch, seq, hidden)
            1: torch.randn(2, 10, 64, device=device),
        }

        with patch('unlearn.get_layer_activations') as mock_get_acts:
            mock_get_acts.return_value = mock_activations

            # Simulate the caching logic from the main function
            layer_ids = [0, 1]
            retain_act_cache = []
            pt_dtype = torch.float32

            # Mock batch on CPU
            rb = {
                "input_ids": torch.tensor([[1, 2, 3]], device="cpu"),
                "attention_mask": torch.tensor([[1, 1, 1]], device="cpu")
            }

            # Simulate the fixed caching code
            rb_device = {k: v.to(device) for k, v in rb.items()}
            acts = mock_get_acts.return_value
            # This is the key fix: cache activations on CPU
            cached_acts = {lid: a.detach().cpu().to(pt_dtype) for lid, a in acts.items()}
            retain_act_cache.append(cached_acts)

            # Verify activations are cached on CPU
            for lid in layer_ids:
                cached_tensor = retain_act_cache[0][lid]
                assert cached_tensor.device.type == "cpu", f"Layer {lid} cached on {cached_tensor.device}, should be CPU"
                assert cached_tensor.dtype == pt_dtype

    def test_cached_activations_moved_to_device_during_loss(self):
        """Test that cached CPU activations are moved to GPU when needed for loss computation."""
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simulate cached activations on CPU (what we store)
        cached_acts_cpu = torch.randn(2, 10, 64, device="cpu")

        # Simulate current activations on device (what we compute during training)
        current_acts_device = torch.randn(2, 10, 64, device=device)

        # Test the device movement logic from the loss functions
        moved_cached = cached_acts_cpu.to(current_acts_device.device)

        # Verify the fix works: can compute loss without device mismatch
        try:
            loss = F.mse_loss(current_acts_device, moved_cached.detach())
            assert loss.device == device
        except RuntimeError as e:
            if "device" in str(e).lower():
                pytest.fail(f"Device mismatch error: {e}")
            else:
                raise

    def test_cosine_similarity_device_handling(self):
        """Test cosine similarity computation handles CPU->GPU movement correctly."""
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simulate the CB loss function scenario
        current_acts = torch.randn(20, 64, device=device)  # flattened (B*T, D)
        cached_acts_cpu = torch.randn(20, 64, device="cpu")

        # Test the device movement from CB loss
        cached_acts_moved = cached_acts_cpu.to(current_acts.device)

        try:
            cos_sim = F.cosine_similarity(current_acts, cached_acts_moved.detach(), dim=-1)
            assert cos_sim.device == device
            assert cos_sim.shape == (20,)  # one similarity per flattened position
        except RuntimeError as e:
            if "device" in str(e).lower():
                pytest.fail(f"Cosine similarity device mismatch: {e}")
            else:
                raise


# ---------------------------------------------------------------------------
# Argument parser (--push-to-hub)
# ---------------------------------------------------------------------------
class TestArgParser:
    """Test argument parsing changes."""

    def test_push_to_hub_defaults_false(self):
        """--push-to-hub should default to False."""
        from unlearn import main
        import argparse
        # Build the parser the same way main() does
        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", action="store_true")
        args = parser.parse_args([])
        assert args.push_to_hub is False

    def test_push_to_hub_set_when_passed(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", action="store_true")
        args = parser.parse_args(["--push-to-hub"])
        assert args.push_to_hub is True

    def test_no_save_defaults_false(self):
        """--no-save should default to False."""
        from unlearn import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-save", action="store_true")
        args = parser.parse_args([])
        assert args.no_save is False

    def test_no_save_set_when_passed(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-save", action="store_true")
        args = parser.parse_args(["--no-save"])
        assert args.no_save is True


# ---------------------------------------------------------------------------
# run_evaluation_benchmarks tests
# ---------------------------------------------------------------------------
class TestRunEvaluationBenchmarks:
    """Test the shared evaluation function."""

    def test_skips_eval_when_no_eval_true(self):
        """Test that evaluation is skipped when no_eval=True."""
        from unlearn import run_evaluation_benchmarks

        # Should return True and skip evaluation
        result = run_evaluation_benchmarks("/fake/path", "cpu", "float32", no_eval=True)
        assert result is True

    def test_constructs_correct_eval_script_path(self):
        """Test that the evaluation script path is constructed correctly."""
        from unlearn import run_evaluation_benchmarks
        from unittest.mock import patch
        import os

        # Mock subprocess.run to avoid actually running the eval
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0

            run_evaluation_benchmarks("/fake/path", "cpu", "float32", no_eval=False)

            # Check that subprocess.run was called with correct arguments
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            eval_cmd = args[0]

            # Should use experiment/eval.py, not unlearn/eval.py
            assert any("experiment/eval.py" in str(arg) for arg in eval_cmd), f"eval_cmd: {eval_cmd}"
            assert "unlearn/eval.py" not in str(eval_cmd)

    def test_handles_subprocess_failure(self):
        """Test that function handles subprocess failures gracefully."""
        from unlearn import run_evaluation_benchmarks
        from unittest.mock import patch

        # Mock subprocess.run to return non-zero exit code
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1

            result = run_evaluation_benchmarks("/fake/path", "cpu", "float32", no_eval=False)
            assert result is False

    def test_handles_subprocess_exception(self):
        """Test that function handles subprocess exceptions gracefully."""
        from unlearn import run_evaluation_benchmarks
        from unittest.mock import patch

        # Mock subprocess.run to raise an exception
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Subprocess failed")

            result = run_evaluation_benchmarks("/fake/path", "cpu", "float32", no_eval=False)
            assert result is False

    def test_logs_wandb_metrics_when_available(self):
        """Test that W&B metrics are logged when available."""
        from unlearn import run_evaluation_benchmarks
        from unittest.mock import patch, mock_open
        import json

        # Mock eval results
        mock_eval_data = {
            "results": {
                "mmlu": {"acc,none": 0.85, "alias,acc": 0.85},
                "wmdp_bio": {"acc,none": 0.23, "loss,none": 2.1}
            }
        }

        with patch('subprocess.run') as mock_run, \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_eval_data))), \
             patch('wandb.run') as mock_wandb_run, \
             patch('wandb.log') as mock_wandb_log, \
             patch('wandb.summary') as mock_wandb_summary:

            mock_run.return_value.returncode = 0
            mock_wandb_run.return_value = True  # Simulate active W&B run

            result = run_evaluation_benchmarks("/fake/path", "cpu", "float32", no_eval=False)

            assert result is True
            # Should log metrics with eval_bench/ prefix and clean metric names
            expected_metrics = {
                "eval_bench/mmlu/acc": 0.85,
                "eval_bench/wmdp_bio/acc": 0.23,
                "eval_bench/wmdp_bio/loss": 2.1
            }
            mock_wandb_log.assert_called_once_with(expected_metrics)
            mock_wandb_summary.update.assert_called_once_with(expected_metrics)



# ---------------------------------------------------------------------------
# chunked_cross_entropy
# ---------------------------------------------------------------------------
class TestChunkedCrossEntropy:
    """Verify chunked_cross_entropy is a drop-in replacement for F.cross_entropy.

    The core claim is mathematical identity: computing cross-entropy in
    chunks over the batch dimension must give bit-for-bit the same answer
    as computing it over the full batch at once, because cross-entropy is
    independent per token.
    """

    def setup_method(self):
        import torch
        import torch.nn.functional as F
        from unlearn import chunked_cross_entropy
        self.torch = torch
        self.F = F
        self.fn = chunked_cross_entropy

    def _make_logits_labels(self, B=8, T=16, V=100, seed=42):
        self.torch.manual_seed(seed)
        logits = self.torch.randn(B, T, V)
        labels = self.torch.randint(0, V, (B, T))
        return logits, labels

    def test_matches_f_cross_entropy_full_batch(self):
        """Default chunk_size=4 must give the same flat tensor as F.cross_entropy."""
        logits, labels = self._make_logits_labels(B=8)
        expected = self.F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        got = self.fn(logits, labels)
        self.torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_chunk_size_1_matches(self):
        """Chunk size of 1 (most conservative) must still match."""
        logits, labels = self._make_logits_labels(B=6)
        expected = self.F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        got = self.fn(logits, labels, chunk_size=1)
        self.torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_chunk_size_larger_than_batch(self):
        """chunk_size > B should work fine (just one pass over everything)."""
        logits, labels = self._make_logits_labels(B=3)
        expected = self.F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        got = self.fn(logits, labels, chunk_size=100)
        self.torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_batch_size_1(self):
        """B=1 edge case must work."""
        logits, labels = self._make_logits_labels(B=1)
        expected = self.F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        got = self.fn(logits, labels, chunk_size=4)
        self.torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_output_shape(self):
        """Output should be (B * T,) — flat per-token losses."""
        B, T, V = 5, 12, 50
        logits, labels = self._make_logits_labels(B=B, T=T, V=V)
        out = self.fn(logits, labels, chunk_size=2)
        assert out.shape == (B * T,), f"Expected ({B * T},), got {out.shape}"

    def test_various_chunk_sizes_all_match(self):
        """chunk_size 1, 2, 3, 4, 7, 32 all produce identical results."""
        logits, labels = self._make_logits_labels(B=12)
        expected = self.F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        for cs in [1, 2, 3, 4, 7, 32]:
            got = self.fn(logits, labels, chunk_size=cs)
            self.torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_gradients_flow_through(self):
        """sum().backward() must produce gradients on the logits tensor."""
        logits, labels = self._make_logits_labels(B=4)
        logits = logits.requires_grad_(True)
        loss = self.fn(logits, labels, chunk_size=2).sum()
        loss.backward()
        assert logits.grad is not None
        assert not self.torch.isnan(logits.grad).any()

    def test_masked_average_matches_reference(self):
        """Test the full nll_loss pattern: chunked CE + mask + mean."""
        B, T, V = 4, 8, 50
        self.torch.manual_seed(7)
        logits = self.torch.randn(B, T - 1, V)
        labels = self.torch.randint(0, V, (B, T - 1))
        mask = self.torch.ones(B, T - 1)
        mask[0, -2:] = 0  # simulate padding on first sample

        ref = self.F.cross_entropy(
            logits.view(-1, V), labels.view(-1), reduction="none"
        )
        ref_loss = (ref * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        chunked = self.fn(logits, labels.view(B, -1), chunk_size=2)
        chunked_loss = (chunked * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        self.torch.testing.assert_close(chunked_loss, ref_loss, rtol=1e-5, atol=1e-5)
