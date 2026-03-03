"""Tests for utils.py — pure utility functions."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils import (
    model_outdir,
    comparison_outdir,
    extract_layer,
    classify_coarse,
    classify_granular,
    frobenius_norm,
    nuclear_norm,
    spectral_norm_power,
    stable_rank_and_spectral,
    stable_rank,
    empirical_rank,
    condition_number,
    compute_rank_deficiency,
    compute_spectral_norm,
    resolve_dtype,
    write_csv,
    filter_gpus_by_free_vram,
    compute_training_max_memory,
)


# ---------------------------------------------------------------------------
# model_outdir
# ---------------------------------------------------------------------------
class TestModelOutdir:
    def test_basic(self):
        assert model_outdir("org/model") == os.path.join("outputs", "org_model")

    def test_with_suffix(self):
        assert model_outdir("org/model", suffix="evals") == os.path.join("outputs", "org_model", "evals")

    def test_custom_root(self):
        assert model_outdir("org/model", root="unlearned_models", suffix="cb") == os.path.join("unlearned_models", "org_model", "cb")

    def test_no_slash(self):
        assert model_outdir("local-model") == os.path.join("outputs", "local-model")

    def test_no_suffix(self):
        result = model_outdir("org/model", suffix="")
        assert result == os.path.join("outputs", "org_model")


# ---------------------------------------------------------------------------
# comparison_outdir
# ---------------------------------------------------------------------------
class TestComparisonOutdir:
    def test_basic(self):
        result = comparison_outdir("org/base", "org/filtered", suffix="weight_comparison")
        assert result == os.path.join("outputs", "org_base__to__org_filtered", "weight_comparison")

    def test_no_suffix(self):
        result = comparison_outdir("org/base", "org/filtered")
        assert result == os.path.join("outputs", "org_base__to__org_filtered")

    def test_custom_root(self):
        result = comparison_outdir("a/b", "c/d", root="results", suffix="x")
        assert result == os.path.join("results", "a_b__to__c_d", "x")

    def test_no_slashes(self):
        result = comparison_outdir("local-a", "local-b", suffix="s")
        assert result == os.path.join("outputs", "local-a__to__local-b", "s")


# ---------------------------------------------------------------------------
# extract_layer
# ---------------------------------------------------------------------------
class TestExtractLayer:
    def test_layers_pattern(self):
        assert extract_layer("model.layers.15.self_attn.q_proj.weight") == 15

    def test_h_pattern(self):
        assert extract_layer("transformer.h.3.attn.weight") == 3

    def test_blocks_pattern(self):
        assert extract_layer("encoder.blocks.7.mlp.weight") == 7

    def test_no_layer(self):
        assert extract_layer("model.embed_tokens.weight") is None

    def test_layer_zero(self):
        assert extract_layer("model.layers.0.mlp.weight") == 0


# ---------------------------------------------------------------------------
# classify_coarse
# ---------------------------------------------------------------------------
class TestClassifyCoarse:
    def test_attention(self):
        assert classify_coarse("model.layers.0.self_attn.q_proj.weight") == "attn"

    def test_mlp(self):
        assert classify_coarse("model.layers.0.mlp.gate_proj.weight") == "mlp"

    def test_other(self):
        assert classify_coarse("model.embed_tokens.weight") == "other"

    def test_ffn_counts_as_mlp(self):
        assert classify_coarse("model.layers.0.ffn.dense.weight") == "mlp"


# ---------------------------------------------------------------------------
# classify_granular
# ---------------------------------------------------------------------------
class TestClassifyGranular:
    def test_qkv_fused(self):
        assert classify_granular("model.layers.0.attention.query_key_value.weight") == "qkv"

    def test_qkv_separate(self):
        assert classify_granular("model.layers.0.self_attn.q_proj.weight") == "qkv"
        assert classify_granular("model.layers.0.self_attn.k_proj.weight") == "qkv"
        assert classify_granular("model.layers.0.self_attn.v_proj.weight") == "qkv"

    def test_output_projection(self):
        assert classify_granular("model.layers.0.self_attn.o_proj.weight") == "proj"
        assert classify_granular("model.layers.0.attention.dense.weight") == "proj"

    def test_mlp_expand(self):
        assert classify_granular("model.layers.0.mlp.gate_proj.weight") == "mlp_expand"
        assert classify_granular("model.layers.0.mlp.up_proj.weight") == "mlp_expand"
        assert classify_granular("model.layers.0.mlp.fc1.weight") == "mlp_expand"

    def test_mlp_contract(self):
        assert classify_granular("model.layers.0.mlp.down_proj.weight") == "mlp_contract"
        assert classify_granular("model.layers.0.mlp.fc2.weight") == "mlp_contract"

    def test_other(self):
        assert classify_granular("model.embed_tokens.weight") == "other"
        assert classify_granular("lm_head.weight") == "other"


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------
class TestFrobeniusNorm:
    def test_identity(self):
        identity = torch.eye(4)
        # Frobenius norm of 4x4 identity = sqrt(4) = 2.0
        assert abs(frobenius_norm(identity) - 2.0) < 1e-5

    def test_zero_matrix(self):
        assert frobenius_norm(torch.zeros(3, 3)) == 0.0

    def test_known_value(self):
        # [[1, 2], [3, 4]] -> sqrt(1 + 4 + 9 + 16) = sqrt(30)
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected = (30.0) ** 0.5
        assert abs(frobenius_norm(matrix) - expected) < 1e-5


class TestSpectralNorm:
    def test_identity(self):
        identity = torch.eye(8)
        # Spectral norm of identity = 1.0
        assert abs(compute_spectral_norm(identity) - 1.0) < 1e-4

    def test_scaled_identity(self):
        scaled = 5.0 * torch.eye(8)
        assert abs(compute_spectral_norm(scaled) - 5.0) < 1e-4

    def test_power_iteration_approximates_svd(self):
        torch.manual_seed(42)
        matrix = torch.randn(16, 8)
        svd_norm = compute_spectral_norm(matrix)
        power_norm = spectral_norm_power(matrix, iters=20)
        # Power iteration with enough iters should be close to SVD
        assert abs(svd_norm - power_norm) / svd_norm < 0.05


class TestStableRank:
    def test_identity_has_full_rank(self):
        identity = torch.eye(8)
        rank, spectral = stable_rank_and_spectral(identity)
        # Identity: Frobenius^2 = 8, spectral = 1, stable_rank = 8
        assert abs(rank - 8.0) < 0.5
        assert abs(spectral - 1.0) < 0.1

    def test_rank_one_matrix(self):
        # Outer product creates a rank-1 matrix
        vector = torch.randn(8, 1)
        rank_one = vector @ vector.T
        rank, _ = stable_rank_and_spectral(rank_one, use_svd=True)
        assert abs(rank - 1.0) < 0.1

    def test_zero_matrix(self):
        rank, spectral = stable_rank_and_spectral(torch.zeros(4, 4))
        assert rank == 0.0
        assert spectral == 0.0

    def test_shorthand_matches(self):
        torch.manual_seed(0)
        matrix = torch.randn(8, 8)
        # Use SVD for both to avoid power iteration randomness
        rank_short = stable_rank(matrix, use_svd=True)
        rank_full, _ = stable_rank_and_spectral(matrix, use_svd=True)
        assert abs(rank_short - rank_full) < 1e-6


class TestEmpiricalRank:
    def test_identity(self):
        # Identity has all singular values = 1, so empirical rank = full dimension
        identity = torch.eye(8)
        assert empirical_rank(identity, threshold=0.99) == 8

    def test_low_rank(self):
        # Rank-2 matrix: only 2 singular values are nonzero
        torch.manual_seed(0)
        u = torch.randn(8, 2)
        low_rank = u @ u.T
        rank = empirical_rank(low_rank, threshold=0.99)
        assert rank == 2

    def test_empty_matrix(self):
        assert empirical_rank(torch.empty(0, 0)) == 0


class TestConditionNumber:
    def test_identity(self):
        # Condition number of identity = 1.0
        assert abs(condition_number(torch.eye(8)) - 1.0) < 1e-4

    def test_ill_conditioned(self):
        # Near-singular matrix should have large condition number
        matrix = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1e-8]))
        cond = condition_number(matrix)
        assert cond > 1e7


class TestRankDeficiency:
    def test_full_rank(self):
        assert compute_rank_deficiency(torch.eye(8)) == 0

    def test_rank_deficient(self):
        # Rank-1 matrix in 4x4 space: deficiency = 3
        vector = torch.ones(4, 1)
        rank_one = vector @ vector.T
        deficiency = compute_rank_deficiency(rank_one)
        assert deficiency == 3


# ---------------------------------------------------------------------------
# resolve_dtype
# ---------------------------------------------------------------------------
class TestResolveDtype:
    def test_auto_cuda(self):
        assert resolve_dtype("auto", "cuda") == torch.bfloat16

    def test_auto_mps(self):
        assert resolve_dtype("auto", "mps") == torch.float16

    def test_auto_cpu(self):
        assert resolve_dtype("auto", "cpu") == torch.float32

    def test_explicit_fp32(self):
        assert resolve_dtype("fp32", "cpu") == torch.float32

    def test_explicit_bf16(self):
        assert resolve_dtype("bf16", "cuda") == torch.bfloat16

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            resolve_dtype("int8", "cpu")


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------
class TestWriteCsv:
    def test_round_trip(self, temp_dir):
        path = os.path.join(temp_dir, "test.csv")
        rows = [
            {"name": "alpha", "value": 1.0},
            {"name": "beta", "value": 2.5},
        ]
        write_csv(path, rows, ["name", "value"])

        # Read back
        with open(path) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert len(read_rows) == 2
        assert read_rows[0]["name"] == "alpha"
        assert float(read_rows[1]["value"]) == 2.5

    def test_creates_directories(self, temp_dir):
        deep_path = os.path.join(temp_dir, "a", "b", "c", "out.csv")
        write_csv(deep_path, [{"x": 1}], ["x"])
        assert os.path.exists(deep_path)


# ---------------------------------------------------------------------------
# filter_gpus_by_free_vram
# ---------------------------------------------------------------------------
class TestFilterGpusByFreeVram:
    def _mem(self, free_gib, total_gib=50):
        """Return (free_bytes, total_bytes) for a given free_gib."""
        return (int(free_gib * 1024 ** 3), int(total_gib * 1024 ** 3))

    def test_all_gpus_qualify(self):
        """When every GPU has enough free VRAM all indices are returned."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 3)
            mem_map = {0: self._mem(20), 1: self._mem(20), 2: self._mem(20)}
            mp.setattr("torch.cuda.mem_get_info", lambda i: mem_map[i])
            result = filter_gpus_by_free_vram(min_free_gib=10.0)
        assert result == [0, 1, 2]

    def test_some_gpus_excluded(self):
        """GPUs with < min_free_gib are excluded."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 4)
            mem_map = {0: self._mem(2), 1: self._mem(20), 2: self._mem(3), 3: self._mem(15)}
            mp.setattr("torch.cuda.mem_get_info", lambda i: mem_map[i])
            result = filter_gpus_by_free_vram(min_free_gib=10.0)
        assert result == [1, 3]

    def test_no_gpu_qualifies_falls_back_to_best(self):
        """When no GPU meets the threshold the best GPU is returned as fallback."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 2)
            mem_map = {0: self._mem(1), 1: self._mem(3)}
            mp.setattr("torch.cuda.mem_get_info", lambda i: mem_map[i])
            result = filter_gpus_by_free_vram(min_free_gib=10.0)
        # Falls back to GPU 1 (most free)
        assert result == [1]

    def test_no_cuda_returns_empty(self):
        """With no CUDA available an empty list is returned."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: False)
            result = filter_gpus_by_free_vram(min_free_gib=10.0)
        assert result == []


# ---------------------------------------------------------------------------
# compute_training_max_memory
# ---------------------------------------------------------------------------
class TestComputeTrainingMaxMemory:
    def _mem(self, free_gib, total_gib=140):
        return (int(free_gib * 1024 ** 3), int(total_gib * 1024 ** 3))

    def test_no_cuda_returns_none(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: False)
            assert compute_training_max_memory() is None

    def test_returns_dict_with_gpu_keys(self):
        """Result maps GPU index → string like '18GiB'."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 2)
            mp.setattr("torch.cuda.mem_get_info", lambda i: self._mem(140))
            result = compute_training_max_memory(optimizer_state_multiplier=6.0,
                                                 activation_buffer_gib=10.0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1}
        assert all(v.endswith("GiB") for v in result.values())

    def test_budget_formula(self):
        """weight_budget = (free - activation_buf) / (1 + optimizer_mult)."""
        free_gib = 140.0
        activation_buf = 10.0
        multiplier = 6.0
        expected_gib = int((free_gib - activation_buf) / (1 + multiplier))

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 1)
            mp.setattr("torch.cuda.mem_get_info", lambda i: self._mem(free_gib))
            result = compute_training_max_memory(optimizer_state_multiplier=multiplier,
                                                 activation_buffer_gib=activation_buf)
        assert result[0] == f"{expected_gib}GiB"

    def test_both_gpus_get_same_budget_when_equal_free(self):
        """With equal free VRAM both GPUs get the same budget string."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 2)
            mp.setattr("torch.cuda.mem_get_info", lambda i: self._mem(80))
            result = compute_training_max_memory()
        assert result[0] == result[1]

    def test_asymmetric_gpus_get_different_budgets(self):
        """GPUs with different free VRAM get proportionally different budgets."""
        mem_map = {0: self._mem(140), 1: self._mem(40)}
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 2)
            mp.setattr("torch.cuda.mem_get_info", lambda i: mem_map[i])
            result = compute_training_max_memory(optimizer_state_multiplier=6.0,
                                                 activation_buffer_gib=10.0)
        budget_0 = int(result[0].replace("GiB", ""))
        budget_1 = int(result[1].replace("GiB", ""))
        assert budget_0 > budget_1  # GPU 0 has more free, gets larger budget

    def test_minimum_budget_is_1gib(self):
        """Even if the computed budget is tiny, it's clamped to at least 1 GiB."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("torch.cuda.is_available", lambda: True)
            mp.setattr("torch.cuda.device_count", lambda: 1)
            # Only 0.5 GiB free — well below activation buffer
            mp.setattr("torch.cuda.mem_get_info", lambda i: self._mem(0.5))
            result = compute_training_max_memory(activation_buffer_gib=10.0)
        assert result[0] == "1GiB"
