"""Tests for experiment/collect_activation_comparison.py — read_lines, caching, and diff logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import tempfile
import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from collect_activation_comparison import read_lines, cache_hidden_states, compute_activation_diffs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def text_file(temp_dir):
    """Create a simple text file with numbered lines."""
    path = os.path.join(temp_dir, "sample.txt")
    with open(path, "w") as f:
        for i in range(20):
            f.write(f"This is line number {i}\n")
    return path


@pytest.fixture
def text_file_with_blanks(temp_dir):
    """Create a text file containing blank and whitespace-only lines."""
    path = os.path.join(temp_dir, "blanks.txt")
    with open(path, "w") as f:
        f.write("Line one\n")
        f.write("\n")
        f.write("   \n")
        f.write("Line two\n")
        f.write("\n")
        f.write("Line three\n")
    return path


class FakeHiddenStates:
    """Simulate model output with hidden states."""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


# ---------------------------------------------------------------------------
# read_lines
# ---------------------------------------------------------------------------
class TestReadLines:
    def test_reads_all_lines(self, text_file):
        lines = read_lines(text_file, max_samples=100)
        assert len(lines) == 20
        assert lines[0] == "This is line number 0"
        assert lines[19] == "This is line number 19"

    def test_respects_max_samples(self, text_file):
        lines = read_lines(text_file, max_samples=5)
        assert len(lines) == 5
        assert lines[-1] == "This is line number 4"

    def test_skips_blank_lines(self, text_file_with_blanks):
        lines = read_lines(text_file_with_blanks, max_samples=100)
        assert len(lines) == 3
        assert lines == ["Line one", "Line two", "Line three"]

    def test_strips_whitespace(self, text_file_with_blanks):
        lines = read_lines(text_file_with_blanks, max_samples=100)
        for line in lines:
            assert line == line.strip()

    def test_empty_file(self, temp_dir):
        path = os.path.join(temp_dir, "empty.txt")
        with open(path, "w") as f:
            pass
        lines = read_lines(path, max_samples=100)
        assert lines == []

    def test_max_samples_zero(self, text_file):
        lines = read_lines(text_file, max_samples=0)
        assert lines == []


# ---------------------------------------------------------------------------
# cache_hidden_states  (using mocked model)
# ---------------------------------------------------------------------------
class TestCacheHiddenStates:
    """Test the caching pipeline with a mocked transformer model."""

    def _make_mock_model_and_tokenizer(self, num_layers=3, hidden_dim=8, vocab_size=100):
        """Create mock model and tokenizer that produce deterministic hidden states."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"

        def fake_tokenize(texts, **kwargs):
            batch_size = len(texts)
            seq_len = 4  # fixed length for simplicity
            return {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            }

        tokenizer.side_effect = fake_tokenize

        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def fake_forward(**kwargs):
            input_ids = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            # Create deterministic hidden states: layer i has values filled with (i+1)
            hidden_states = tuple(
                torch.full((batch_size, seq_len, hidden_dim), float(i + 1))
                for i in range(num_layers)
            )
            return FakeHiddenStates(hidden_states)

        model.side_effect = fake_forward
        return model, tokenizer

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_returns_correct_keys(self, mock_model_cls, mock_tok_cls, temp_dir):
        model, tokenizer = self._make_mock_model_and_tokenizer()
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        texts = ["Hello world", "Test text"]
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        result = cache_hidden_states(
            "fake-model", texts, cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=2
        )

        assert "mean_l1_norm" in result
        assert "mean_l2_norm" in result
        assert "num_layers" in result

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_correct_number_of_layers(self, mock_model_cls, mock_tok_cls, temp_dir):
        num_layers = 5
        model, tokenizer = self._make_mock_model_and_tokenizer(num_layers=num_layers)
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        result = cache_hidden_states(
            "fake-model", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1
        )

        assert result["num_layers"] == num_layers
        assert len(result["mean_l1_norm"]) == num_layers
        assert len(result["mean_l2_norm"]) == num_layers

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_norms_are_positive(self, mock_model_cls, mock_tok_cls, temp_dir):
        model, tokenizer = self._make_mock_model_and_tokenizer()
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        result = cache_hidden_states(
            "fake-model", ["Hello"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1
        )

        for norm in result["mean_l1_norm"]:
            assert norm > 0
        for norm in result["mean_l2_norm"]:
            assert norm > 0

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_l1_norm_increases_with_layer(self, mock_model_cls, mock_tok_cls, temp_dir):
        """Since layer i is filled with (i+1), L1 norms should increase with layer."""
        model, tokenizer = self._make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8)
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        result = cache_hidden_states(
            "fake-model", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1
        )

        l1_norms = result["mean_l1_norm"]
        for i in range(1, len(l1_norms)):
            assert l1_norms[i] > l1_norms[i - 1], f"Layer {i} L1 norm should be > layer {i-1}"

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_cache_files_created(self, mock_model_cls, mock_tok_cls, temp_dir):
        model, tokenizer = self._make_mock_model_and_tokenizer()
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        texts = ["A", "B", "C", "D"]
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        cache_hidden_states(
            "fake-model", texts, cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=2
        )

        # 4 texts with batch_size=2 → 2 batch files
        assert os.path.isfile(os.path.join(cache_dir, "batch_0.pt"))
        assert os.path.isfile(os.path.join(cache_dir, "batch_1.pt"))

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_cached_data_has_correct_structure(self, mock_model_cls, mock_tok_cls, temp_dir):
        num_layers = 3
        model, tokenizer = self._make_mock_model_and_tokenizer(num_layers=num_layers)
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        cache_hidden_states(
            "fake-model", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1
        )

        cached = torch.load(os.path.join(cache_dir, "batch_0.pt"), weights_only=True)
        assert "hidden_states" in cached
        assert "attention_mask" in cached
        assert len(cached["hidden_states"]) == num_layers

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_half_precision_cache(self, mock_model_cls, mock_tok_cls, temp_dir):
        model, tokenizer = self._make_mock_model_and_tokenizer()
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)

        cache_hidden_states(
            "fake-model", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1, use_half_precision_cache=True
        )

        cached = torch.load(os.path.join(cache_dir, "batch_0.pt"), weights_only=True)
        for hidden_tensor in cached["hidden_states"]:
            assert hidden_tensor.dtype == torch.float16


# ---------------------------------------------------------------------------
# compute_activation_diffs  (using mocked model + real cached files)
# ---------------------------------------------------------------------------
class TestComputeActivationDiffs:
    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_returns_correct_keys(self, mock_model_cls, mock_tok_cls, temp_dir):
        """Diff computation should return absolute norms + diff norms."""
        num_layers = 3
        hidden_dim = 8
        batch_size = 1
        seq_len = 4

        # Create fake cached data (model A hidden states)
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)
        cached_data = {
            "hidden_states": [
                torch.ones(batch_size, seq_len, hidden_dim) * (i + 1)
                for i in range(num_layers)
            ],
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }
        torch.save(cached_data, os.path.join(cache_dir, "batch_0.pt"))

        # Mock model B (produces hidden states filled with (i+2))
        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"

        def fake_tokenize(texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (len(texts), seq_len)),
                "attention_mask": torch.ones(len(texts), seq_len, dtype=torch.long),
            }
        tokenizer.side_effect = fake_tokenize

        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def fake_forward(**kwargs):
            input_ids = kwargs["input_ids"]
            bs, sl = input_ids.shape
            hidden_states = tuple(
                torch.full((bs, sl, hidden_dim), float(i + 2))
                for i in range(num_layers)
            )
            return FakeHiddenStates(hidden_states)

        model.side_effect = fake_forward
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        result = compute_activation_diffs(
            "fake-model-b", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1, num_layers=num_layers
        )

        assert "mean_l1_norm" in result
        assert "mean_l2_norm" in result
        assert "mean_diff_l1" in result
        assert "mean_diff_l2" in result
        assert len(result["mean_diff_l1"]) == num_layers

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_diff_norms_are_positive_when_models_differ(self, mock_model_cls, mock_tok_cls, temp_dir):
        """If model A and B have different hidden states, diff norms should be > 0."""
        num_layers = 2
        hidden_dim = 4
        seq_len = 3

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)
        cached_data = {
            "hidden_states": [
                torch.zeros(1, seq_len, hidden_dim) for _ in range(num_layers)
            ],
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        torch.save(cached_data, os.path.join(cache_dir, "batch_0.pt"))

        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"

        def fake_tokenize(texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (len(texts), seq_len)),
                "attention_mask": torch.ones(len(texts), seq_len, dtype=torch.long),
            }
        tokenizer.side_effect = fake_tokenize

        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def fake_forward(**kwargs):
            bs = kwargs["input_ids"].shape[0]
            return FakeHiddenStates(tuple(
                torch.ones(bs, seq_len, hidden_dim) * 5.0 for _ in range(num_layers)
            ))

        model.side_effect = fake_forward
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        result = compute_activation_diffs(
            "fake-model-b", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1, num_layers=num_layers
        )

        for norm in result["mean_diff_l1"]:
            assert norm > 0
        for norm in result["mean_diff_l2"]:
            assert norm > 0

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_diff_norms_zero_when_identical(self, mock_model_cls, mock_tok_cls, temp_dir):
        """If model B produces same hidden states as cached model A, diff norms should be 0."""
        num_layers = 2
        hidden_dim = 4
        seq_len = 3
        fill_value = 3.0

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)
        cached_data = {
            "hidden_states": [
                torch.full((1, seq_len, hidden_dim), fill_value)
                for _ in range(num_layers)
            ],
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        torch.save(cached_data, os.path.join(cache_dir, "batch_0.pt"))

        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"

        def fake_tokenize(texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (len(texts), seq_len)),
                "attention_mask": torch.ones(len(texts), seq_len, dtype=torch.long),
            }
        tokenizer.side_effect = fake_tokenize

        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def fake_forward(**kwargs):
            bs = kwargs["input_ids"].shape[0]
            return FakeHiddenStates(tuple(
                torch.full((bs, seq_len, hidden_dim), fill_value)
                for _ in range(num_layers)
            ))

        model.side_effect = fake_forward
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        result = compute_activation_diffs(
            "fake-model-b", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1, num_layers=num_layers
        )

        for norm in result["mean_diff_l1"]:
            assert abs(norm) < 1e-6
        for norm in result["mean_diff_l2"]:
            assert abs(norm) < 1e-6

    @patch("collect_activation_comparison.AutoTokenizer")
    @patch("collect_activation_comparison.AutoModelForCausalLM")
    def test_known_diff_values(self, mock_model_cls, mock_tok_cls, temp_dir):
        """Verify diff norm math: model A = zeros, model B = ones → known norms."""
        num_layers = 1
        hidden_dim = 4
        seq_len = 2  # 2 tokens, all unmasked

        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir)
        cached_data = {
            "hidden_states": [torch.zeros(1, seq_len, hidden_dim)],
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        torch.save(cached_data, os.path.join(cache_dir, "batch_0.pt"))

        tokenizer = MagicMock()
        tokenizer.pad_token = "<pad>"

        def fake_tokenize(texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 100, (len(texts), seq_len)),
                "attention_mask": torch.ones(len(texts), seq_len, dtype=torch.long),
            }
        tokenizer.side_effect = fake_tokenize

        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def fake_forward(**kwargs):
            bs = kwargs["input_ids"].shape[0]
            # Hidden states = all ones
            return FakeHiddenStates((torch.ones(bs, seq_len, hidden_dim),))

        model.side_effect = fake_forward
        mock_tok_cls.from_pretrained.return_value = tokenizer
        mock_model_cls.from_pretrained.return_value = model

        result = compute_activation_diffs(
            "fake-model-b", ["text"], cache_dir, "cpu", torch.float32,
            max_length=32, batch_size=1, num_layers=num_layers
        )

        # diff = ones - zeros = ones
        # L1 norm of [1,1,1,1] = 4, per token, 2 tokens → mean = 4.0
        assert abs(result["mean_diff_l1"][0] - 4.0) < 1e-5
        # L2 norm of [1,1,1,1] = sqrt(4) = 2.0, per token → mean = 2.0
        assert abs(result["mean_diff_l2"][0] - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# plot_activation_comparison  (smoke test — just verify it runs and produces files)
# ---------------------------------------------------------------------------
class TestPlotActivationNorms:
    def test_smoke_produces_png_files(self, temp_dir):
        """Smoke test: plot_activation_comparison creates expected PNGs from valid CSV."""
        from collect_activation_comparison import plot_activation_comparison
        from utils import write_csv

        csv_path = os.path.join(temp_dir, "activation_comparison.csv")
        fieldnames = [
            "layer", "split",
            "model_a_l1_norm", "model_a_l2_norm",
            "model_b_l1_norm", "model_b_l2_norm",
            "mean_diff_l1", "mean_diff_l2",
        ]
        rows = []
        for layer in range(4):
            for split in ["forget", "retain"]:
                rows.append({
                    "layer": layer,
                    "split": split,
                    "model_a_l1_norm": 1.0 + layer,
                    "model_a_l2_norm": 0.5 + layer,
                    "model_b_l1_norm": 1.2 + layer,
                    "model_b_l2_norm": 0.6 + layer,
                    "mean_diff_l1": 0.2,
                    "mean_diff_l2": 0.1,
                })
        write_csv(csv_path, rows, fieldnames)

        plot_outdir = os.path.join(temp_dir, "plots")
        plot_activation_comparison(csv_path, plot_outdir, title="Test Title")

        # Check that all 4 expected PNGs were created (2 splits × 2 plot types)
        assert os.path.isfile(os.path.join(plot_outdir, "activation_norms_forget.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_norms_retain.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_diffs_forget.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_diffs_retain.png"))

    def test_error_band_produces_png_files(self, temp_dir):
        """Smoke test: plot_activation_comparison renders error bands when _std columns present."""
        from collect_activation_comparison import plot_activation_comparison
        from utils import write_csv

        # Build aggregated CSV that includes _std columns (as produced by the aggregator)
        csv_path = os.path.join(temp_dir, "activation_comparison_agg.csv")
        fieldnames = [
            "layer", "split",
            "model_a_l1_norm", "model_a_l1_norm_std",
            "model_a_l2_norm", "model_a_l2_norm_std",
            "model_b_l1_norm", "model_b_l1_norm_std",
            "model_b_l2_norm", "model_b_l2_norm_std",
            "mean_diff_l1", "mean_diff_l1_std",
            "mean_diff_l2", "mean_diff_l2_std",
        ]
        rows = []
        for layer in range(4):
            for split in ["forget", "retain"]:
                rows.append({
                    "layer": layer, "split": split,
                    "model_a_l1_norm": 1.0 + layer, "model_a_l1_norm_std": 0.05,
                    "model_a_l2_norm": 0.5 + layer, "model_a_l2_norm_std": 0.02,
                    "model_b_l1_norm": 1.2 + layer, "model_b_l1_norm_std": 0.07,
                    "model_b_l2_norm": 0.6 + layer, "model_b_l2_norm_std": 0.03,
                    "mean_diff_l1": 0.2, "mean_diff_l1_std": 0.01,
                    "mean_diff_l2": 0.1, "mean_diff_l2_std": 0.005,
                })
        write_csv(csv_path, rows, fieldnames)

        plot_outdir = os.path.join(temp_dir, "plots_agg")
        plot_activation_comparison(csv_path, plot_outdir, title="Test (3 seeds)")

        # All 4 PNGs should still be produced even with _std columns present
        assert os.path.isfile(os.path.join(plot_outdir, "activation_norms_forget.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_norms_retain.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_diffs_forget.png"))
        assert os.path.isfile(os.path.join(plot_outdir, "activation_diffs_retain.png"))


