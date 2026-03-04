"""Shared fixtures for model_diffs test suite."""

import os
import sys
import json
import tempfile

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def _disable_wandb_in_tests(monkeypatch):
    """Prevent any test from accidentally creating real W&B runs.

    Sets WANDB_MODE=disabled for every test so that even if WANDB_API_KEY
    is present in .env, no runs are logged to the live cambridge_era project.
    W&B unit tests are unaffected because they mock wandb via
    patch.dict(sys.modules) before any real wandb.init call.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")




@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_weights():
    """Return two small weight dicts simulating model A and model B."""
    torch.manual_seed(0)
    weight_a = {
        "model.layers.0.self_attn.q_proj.weight": torch.eye(8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.1.self_attn.q_proj.weight": torch.eye(8),
        "model.layers.1.mlp.down_proj.weight": torch.randn(8, 16),
        "model.embed_tokens.weight": torch.randn(32, 8),
    }
    # Model B = Model A + small perturbation
    weight_b = {
        name: tensor + 0.01 * torch.randn_like(tensor)
        for name, tensor in weight_a.items()
    }
    return weight_a, weight_b


@pytest.fixture
def safetensors_single_model(temp_dir, sample_weights):
    """Create a single-file safetensors model directory and return its path."""
    weight_a, weight_b = sample_weights
    dir_a = os.path.join(temp_dir, "model_a")
    dir_b = os.path.join(temp_dir, "model_b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    save_safetensors(weight_a, os.path.join(dir_a, "model.safetensors"))
    save_safetensors(weight_b, os.path.join(dir_b, "model.safetensors"))
    return dir_a, dir_b


@pytest.fixture
def safetensors_sharded_model(temp_dir, sample_weights):
    """Create a sharded safetensors model directory (2 shards + index) and return its path."""
    weight_a, _ = sample_weights
    model_dir = os.path.join(temp_dir, "model_sharded")
    os.makedirs(model_dir)

    names = list(weight_a.keys())
    # Split into two shards
    shard_1_names = names[:2]
    shard_2_names = names[2:]

    shard_1 = {name: weight_a[name] for name in shard_1_names}
    shard_2 = {name: weight_a[name] for name in shard_2_names}

    save_safetensors(shard_1, os.path.join(model_dir, "model-00001-of-00002.safetensors"))
    save_safetensors(shard_2, os.path.join(model_dir, "model-00002-of-00002.safetensors"))

    # Write the index file
    weight_map = {}
    for name in shard_1_names:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name in shard_2_names:
        weight_map[name] = "model-00002-of-00002.safetensors"

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    return model_dir


@pytest.fixture
def sample_mmlu_items():
    """Return a small list of synthetic MMLU-like items."""
    return [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
            "subject": "geography",
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
            "subject": "math",
        },
        {
            "question": "Which planet is closest to the sun?",
            "choices": ["Venus", "Earth", "Mercury", "Mars"],
            "answer": 2,
            "subject": "astronomy",
        },
        {
            "question": "What gas do plants absorb?",
            "choices": ["Oxygen", "Nitrogen", "CO2", "Hydrogen"],
            "answer": 2,
            "subject": "biology",
        },
    ]
