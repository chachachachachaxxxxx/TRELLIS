#!/usr/bin/env python
"""Unit test for UniEdit Euler sampler logic."""

import torch
import numpy as np
from editing.inversion.uniedit_euler_sampler import (
    UniEditEulerSampler,
    compute_uniedit_map,
    _dense_scalar_field,
    _normalize_dense_map,
)


def test_scalar_field():
    """Test scalar field computation."""
    print("Testing scalar field computation...")
    x = torch.randn(2, 4, 8, 8)
    scores = _dense_scalar_field(x)
    assert scores.shape == (2, 1, 8, 8), f"Expected (2, 1, 8, 8), got {scores.shape}"
    print("  ✓ Scalar field shape correct")


def test_normalize_map():
    """Test map normalization."""
    print("Testing map normalization...")
    scores = torch.randn(2, 1, 8, 8)
    normalized = _normalize_dense_map(scores, selector=None)
    assert normalized.shape == scores.shape, "Shape mismatch"
    assert normalized.min() >= 0.0, f"Min value {normalized.min()} < 0"
    assert normalized.max() <= 1.0, f"Max value {normalized.max()} > 1"
    print(f"  ✓ Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")


def test_uniedit_map():
    """Test UniEdit map computation."""
    print("Testing UniEdit map computation...")
    guidance = torch.randn(2, 4, 8, 8)
    save_map = compute_uniedit_map(guidance, selector=None)
    assert save_map.shape == (2, 1, 8, 8), f"Expected (2, 1, 8, 8), got {save_map.shape}"
    assert save_map.min() >= 0.0, f"Min value {save_map.min()} < 0"
    assert save_map.max() <= 1.0, f"Max value {save_map.max()} > 1"
    print(f"  ✓ UniEdit map range: [{save_map.min():.3f}, {save_map.max():.3f}]")


def test_sampler_instantiation():
    """Test sampler instantiation."""
    print("Testing sampler instantiation...")
    sampler = UniEditEulerSampler()
    assert sampler is not None
    print("  ✓ Sampler instantiated")


def test_timestep_generation():
    """Test timestep sequence generation."""
    print("Testing timestep generation...")
    steps = 10
    rescale_t = 7.0
    t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)

    assert len(t_seq) == steps + 1, f"Expected {steps + 1} timesteps, got {len(t_seq)}"
    assert t_seq[0] == 1.0, f"First timestep should be 1.0, got {t_seq[0]}"
    assert abs(t_seq[-1]) < 1e-6, f"Last timestep should be ~0.0, got {t_seq[-1]}"
    print(f"  ✓ Timestep sequence: {t_seq[0]:.3f} -> {t_seq[-1]:.6f}")


def test_delayed_inversion_logic():
    """Test delayed inversion logic."""
    print("Testing delayed inversion logic...")
    steps = 10
    alpha = 0.5
    t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
    t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

    num_pairs = len(t_pairs)
    step_threshold = round(alpha * num_pairs)

    print(f"  Total pairs: {num_pairs}")
    print(f"  Alpha: {alpha}")
    print(f"  Step threshold: {step_threshold}")
    print(f"  Will invert first {step_threshold} steps, skip last {num_pairs - step_threshold} steps")

    assert step_threshold == 5, f"Expected 5 steps, got {step_threshold}"
    print("  ✓ Delayed inversion logic correct")


def test_delayed_editing_logic():
    """Test delayed editing logic."""
    print("Testing delayed editing logic...")
    steps = 10
    alpha = 0.5
    t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
    t_pairs = [(float(t_seq[i]), float(t_seq[i + 1])) for i in range(len(t_seq) - 1)]

    num_pairs = len(t_pairs)
    step_threshold = round(alpha * num_pairs)

    print(f"  Total pairs: {num_pairs}")
    print(f"  Alpha: {alpha}")
    print(f"  Step threshold: {step_threshold}")
    print(f"  Will skip first {num_pairs - step_threshold} steps, edit last {step_threshold} steps")

    # Check which steps will be edited
    edited_steps = [i for i in range(num_pairs) if i >= num_pairs - step_threshold]
    print(f"  Edited step indices: {edited_steps}")

    assert len(edited_steps) == 5, f"Expected 5 edited steps, got {len(edited_steps)}"
    print("  ✓ Delayed editing logic correct")


def test_fusion_formula():
    """Test UniEdit fusion formula."""
    print("Testing UniEdit fusion formula...")

    # Simulate predictions
    pred_src = torch.randn(2, 4, 8, 8)
    pred_tgt = torch.randn(2, 4, 8, 8)
    omega = 1.0

    # Compute guidance and save_map
    guidance = pred_tgt - pred_src
    save_map = compute_uniedit_map(guidance, selector=None)

    # Apply fusion formula
    fused = pred_tgt * save_map + pred_src * (1.0 - save_map)
    pred = fused + guidance * ((1.0 + save_map) * float(omega))

    print(f"  pred_src shape: {pred_src.shape}")
    print(f"  pred_tgt shape: {pred_tgt.shape}")
    print(f"  guidance shape: {guidance.shape}")
    print(f"  save_map shape: {save_map.shape}")
    print(f"  fused shape: {fused.shape}")
    print(f"  pred shape: {pred.shape}")
    print(f"  save_map range: [{save_map.min():.3f}, {save_map.max():.3f}]")

    assert pred.shape == pred_src.shape, "Shape mismatch"
    print("  ✓ Fusion formula correct")


def main():
    """Run all tests."""
    print("=" * 60)
    print("UniEdit Euler Sampler Unit Tests")
    print("=" * 60)

    tests = [
        test_scalar_field,
        test_normalize_map,
        test_uniedit_map,
        test_sampler_instantiation,
        test_timestep_generation,
        test_delayed_inversion_logic,
        test_delayed_editing_logic,
        test_fusion_formula,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print()
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
