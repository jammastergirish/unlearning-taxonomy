"""Tests for experiment/collect_weight_comparison.py — SmartLoader and stats computation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import torch
import numpy as np
import pytest

from utils import stable_rank_and_spectral, extract_layer, classify_granular


# ---------------------------------------------------------------------------
# SmartLoader — single-file safetensors
# ---------------------------------------------------------------------------
class TestSmartLoaderSingleFile:
    def test_loads_all_param_names(self, safetensors_single_model):
        from utils import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)
        names = loader.get_all_param_names()

        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.embed_tokens.weight" in names
        assert len(names) == 5

    def test_get_param_returns_correct_tensor(self, safetensors_single_model, sample_weights):
        from utils import SmartLoader

        weight_a, _ = sample_weights
        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)

        name = "model.layers.0.self_attn.q_proj.weight"
        tensor = loader.get_param(name, "cpu", torch.float32)

        assert tensor is not None
        assert tensor.shape == weight_a[name].shape
        assert torch.allclose(tensor, weight_a[name], atol=1e-6)

    def test_get_param_nonexistent_returns_none(self, safetensors_single_model):
        from utils import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)

        tensor = loader.get_param("nonexistent.weight", "cpu", torch.float32)
        assert tensor is None

    def test_is_safetensors_flag(self, safetensors_single_model):
        from utils import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)
        assert loader.is_safetensors is True


# ---------------------------------------------------------------------------
# SmartLoader — sharded safetensors
# ---------------------------------------------------------------------------
class TestSmartLoaderSharded:
    def test_loads_all_param_names(self, safetensors_sharded_model):
        from utils import SmartLoader

        loader = SmartLoader(safetensors_sharded_model)
        names = loader.get_all_param_names()

        assert len(names) == 5
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.1.self_attn.q_proj.weight" in names

    def test_loads_params_across_shards(self, safetensors_sharded_model, sample_weights):
        from utils import SmartLoader

        weight_a, _ = sample_weights
        loader = SmartLoader(safetensors_sharded_model)

        # Load a param from each shard and verify correctness
        for name, expected_tensor in weight_a.items():
            tensor = loader.get_param(name, "cpu", torch.float32)
            assert tensor is not None, f"Failed to load {name}"
            assert torch.allclose(tensor, expected_tensor, atol=1e-6), f"Mismatch for {name}"


# ---------------------------------------------------------------------------
# SmartLoader — error handling
# ---------------------------------------------------------------------------
class TestSmartLoaderErrors:
    def test_nonexistent_path_raises(self, temp_dir):
        from utils import SmartLoader

        fake_path = os.path.join(temp_dir, "does_not_exist")
        os.makedirs(fake_path)  # exists but has no weights
        with pytest.raises(FileNotFoundError):
            SmartLoader(fake_path)


# ---------------------------------------------------------------------------
# Stats computation — verify math on known tensors
# ---------------------------------------------------------------------------
class TestStatsComputation:
    """Verify the core stats computation that happens in the main loop."""

    def test_frobenius_norm_of_difference(self, safetensors_single_model, sample_weights):
        from utils import SmartLoader

        weight_a, weight_b = sample_weights
        dir_a, dir_b = safetensors_single_model
        loader_a = SmartLoader(dir_a)
        loader_b = SmartLoader(dir_b)

        name = "model.layers.0.self_attn.q_proj.weight"
        Wa = loader_a.get_param(name, "cpu", torch.float32)
        Wb = loader_b.get_param(name, "cpu", torch.float32)
        dW = Wb - Wa

        # The perturbation was 0.01 * randn for an 8x8 matrix
        # Frobenius norm should be small (roughly 0.01 * sqrt(64) ≈ 0.08)
        dW_fro = float(dW.float().norm().item())
        assert 0 < dW_fro < 0.5  # sanity bound

        # Relative norm should also be small
        W_fro = float(Wa.float().norm().item())
        assert W_fro > 0
        assert dW_fro / W_fro < 0.2

    def test_stable_rank_of_identity_perturbation(self, safetensors_single_model):
        from utils import SmartLoader

        dir_a, dir_b = safetensors_single_model
        loader_a = SmartLoader(dir_a)
        loader_b = SmartLoader(dir_b)

        # The q_proj weight in model A is identity (8x8)
        name = "model.layers.0.self_attn.q_proj.weight"
        Wa = loader_a.get_param(name, "cpu", torch.float32)
        Wb = loader_b.get_param(name, "cpu", torch.float32)
        dW = Wb - Wa

        # Stable rank of the identity matrix should be close to its dimension
        sr_W, spec_W = stable_rank_and_spectral(Wa, use_svd=True)
        assert abs(sr_W - 8.0) < 0.5  # identity 8x8 -> stable rank = 8

        # dW is small random perturbation — stable rank should be reasonable
        sr_dW, spec_dW = stable_rank_and_spectral(dW, use_svd=True)
        assert sr_dW > 0
        assert spec_dW > 0

    def test_utility_functions_on_param_names(self):
        """Verify extract_layer and classify_granular work for the param names we use."""
        assert extract_layer("model.layers.0.self_attn.q_proj.weight") == 0
        assert extract_layer("model.layers.1.mlp.down_proj.weight") == 1
        assert extract_layer("model.embed_tokens.weight") is None

        assert classify_granular("model.layers.0.self_attn.q_proj.weight") == "qkv"
        assert classify_granular("model.layers.0.mlp.gate_proj.weight") == "mlp_expand"
        assert classify_granular("model.layers.1.mlp.down_proj.weight") == "mlp_contract"
        assert classify_granular("model.embed_tokens.weight") == "other"


# ---------------------------------------------------------------------------
# _compute_metrics — verify new Wb_* fields
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))
from collect_weight_comparison import _compute_metrics


class TestComputeMetrics:
    """Verify _compute_metrics returns the absolute model-B norms added recently."""

    def _pair(self, shape=(8, 8), seed=0):
        torch.manual_seed(seed)
        Wa = torch.randn(*shape)
        Wb = Wa + 0.01 * torch.randn(*shape)
        return Wa, Wb

    def test_returns_wb_fro(self):
        Wa, Wb = self._pair()
        m = _compute_metrics(Wa, Wb)
        assert "Wb_fro" in m
        expected = float(Wb.float().norm().item())
        assert abs(m["Wb_fro"] - expected) < 1e-5

    def test_returns_wb_spectral(self):
        Wa, Wb = self._pair()
        m = _compute_metrics(Wa, Wb)
        assert "Wb_spectral" in m
        assert m["Wb_spectral"] > 0

    def test_returns_wb_stable_rank(self):
        Wa, Wb = self._pair()
        m = _compute_metrics(Wa, Wb)
        assert "Wb_stable_rank" in m
        assert m["Wb_stable_rank"] > 0

    def test_self_comparison_wb_equals_wa(self):
        """When Wa == Wb, absolute norms of both models must be identical.

        Note: spectral norm uses power iteration with a random start, so two
        independent calls on the same matrix may differ slightly (~1%).
        """
        torch.manual_seed(0)
        W = torch.randn(8, 8)
        m = _compute_metrics(W, W)
        assert abs(m["W_fro"] - m["Wb_fro"]) < 1e-5
        assert abs(m["W_spectral"] - m["Wb_spectral"]) / m["W_spectral"] < 0.02  # within 2%
        assert abs(m["W_stable_rank"] - m["Wb_stable_rank"]) < 0.5


    def test_rel_frobenius_still_correct(self):
        Wa, Wb = self._pair()
        m = _compute_metrics(Wa, Wb)
        dW_fro = float((Wb - Wa).float().norm().item())
        W_fro = float(Wa.float().norm().item())
        expected_rel = dW_fro / W_fro
        assert abs(m["rel_frobenius"] - expected_rel) < 1e-5


# ---------------------------------------------------------------------------
# plot_weight_comparison — verify all 6 PNG files are created
# ---------------------------------------------------------------------------
import tempfile
import pandas as pd


class TestPlotWeightComparison:
    """Verify plot_weight_comparison writes all 6 expected PNG files."""

    def _make_df(self, n_layers=4):
        """Minimal per_matrix DataFrame with all columns the plotter reads."""
        rows = []
        for layer in range(n_layers):
            for comp in ["qkv", "proj", "mlp_expand", "mlp_contract"]:
                rows.append({
                    "layer": layer, "component": comp,
                    "rel_frobenius": 0.05 + 0.01 * layer,
                    "W_fro": 10.0 + layer,
                    "Wb_fro": 9.5 + layer,
                    "dW_stable_rank": 3.0,
                    "W_stable_rank": 8.0,
                    "Wb_stable_rank": 7.5,
                    "dW_spectral_rel": 0.1,
                    "W_spectral": 2.0,
                    "Wb_spectral": 1.9,
                })
        return pd.DataFrame(rows)

    def test_all_six_pngs_created(self, temp_dir):
        from collect_weight_comparison import plot_weight_comparison

        df = self._make_df()
        csv_path = os.path.join(temp_dir, "per_matrix.csv")
        df.to_csv(csv_path, index=False)

        plot_weight_comparison(
            csv_path, temp_dir,
            model_a="org/baseline", model_b="org/unlearned",
        )

        expected = [
            "layer_locality.png",
            "stable_rank.png",
            "spectral_norm.png",
            "absolute_frobenius.png",
            "absolute_stable_rank.png",
            "absolute_spectral_norm.png",
        ]
        for fname in expected:
            assert os.path.exists(os.path.join(temp_dir, fname)), f"Missing: {fname}"

