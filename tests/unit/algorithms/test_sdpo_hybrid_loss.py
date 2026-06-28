# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SDPOHybridLossFn (SDPO+GRPO hybrid, paper §4.5).

The hybrid blends a clipped policy-gradient term with the SDPO logit-level KL
term at the loss level:

    L = grpo_weight · L_GRPO  +  (1 − grpo_weight) · L_SDPO

Following ``test_loss_functions.py``, the loss is driven through its real
``__call__`` (raw student logits + a data dict) on GPU. These tests verify the
blend reduces to each component at the extremes and is the exact weighted sum
in between.
"""

import pytest
import torch

from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossFn,
    SDPOHybridLossFn,
    SDPOLossFn,
)


def _grpo_cfg(reference_policy_kl_penalty: float = 0.0) -> dict:
    return {
        "reference_policy_kl_penalty": reference_policy_kl_penalty,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": None,
        "kl_output_clamp_value": None,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }


def _sdpo_cfg() -> dict:
    return {
        "kl_type": "reverse",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,  # top-k-only KL keeps the math simple
        "success_reward_threshold": 1.0,
    }


def setup_hybrid_test_data(batch_size=4, seq_len=6, vocab_size=32, topk=5):
    """Build self-consistent GPU inputs for both the SDPO and GRPO components.

    Returns ``(student_logits, data, global_valid_seqs, global_valid_toks)``.
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"

    student_logits = torch.randn((batch_size, seq_len, vocab_size), device=device)
    teacher_logits = torch.randn((batch_size, seq_len, vocab_size), device=device)
    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)

    token_mask = torch.ones((batch_size, seq_len), device=device)
    sample_mask = torch.ones(batch_size, device=device)

    data = {
        "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "sdpo_mask": torch.ones(batch_size, device=device),
        "teacher_topk_logits": teacher_topk_logits,
        "teacher_topk_indices": teacher_topk_indices,
        "advantages": torch.randn((batch_size, seq_len), device=device),
        "prev_logprobs": torch.randn((batch_size, seq_len), device=device),
        "generation_logprobs": torch.randn((batch_size, seq_len), device=device),
    }

    global_valid_seqs = torch.sum(sample_mask)
    global_valid_toks = torch.sum(sample_mask.unsqueeze(-1) * token_mask)
    return student_logits, data, global_valid_seqs, global_valid_toks


def _call_hybrid(grpo_weight, student_logits, data, gvs, gvt):
    hybrid = SDPOHybridLossFn(
        {"grpo_weight": grpo_weight, "sdpo": _sdpo_cfg(), "grpo": _grpo_cfg()}
    )
    return hybrid(
        student_logits, data, global_valid_seqs=gvs, global_valid_toks=gvt
    )


def _call_sdpo(student_logits, data, gvs, gvt):
    return SDPOLossFn(_sdpo_cfg())(
        student_logits, data, global_valid_seqs=gvs, global_valid_toks=gvt
    )


def _call_grpo(student_logits, data, gvs, gvt):
    return ClippedPGLossFn(_grpo_cfg())(
        student_logits, data, global_valid_seqs=gvs, global_valid_toks=gvt
    )


def test_hybrid_lambda_zero_equals_pure_sdpo():
    args = setup_hybrid_test_data()
    hybrid_loss, _ = _call_hybrid(0.0, *args)
    sdpo_loss, _ = _call_sdpo(*args)
    assert torch.allclose(hybrid_loss, sdpo_loss, atol=1e-6), (
        hybrid_loss.item(),
        sdpo_loss.item(),
    )


def test_hybrid_lambda_one_equals_pure_grpo():
    args = setup_hybrid_test_data()
    hybrid_loss, _ = _call_hybrid(1.0, *args)
    grpo_loss, _ = _call_grpo(*args)
    assert torch.allclose(hybrid_loss, grpo_loss, atol=1e-6), (
        hybrid_loss.item(),
        grpo_loss.item(),
    )


@pytest.mark.parametrize("grpo_weight", [0.25, 0.5, 0.75])
def test_hybrid_intermediate_is_weighted_sum(grpo_weight):
    args = setup_hybrid_test_data()
    hybrid_loss, _ = _call_hybrid(grpo_weight, *args)
    sdpo_loss, _ = _call_sdpo(*args)
    grpo_loss, _ = _call_grpo(*args)
    expected = grpo_weight * grpo_loss + (1.0 - grpo_weight) * sdpo_loss
    assert torch.allclose(hybrid_loss, expected, atol=1e-6), (
        hybrid_loss.item(),
        expected.item(),
    )


def test_hybrid_metrics_structure():
    args = setup_hybrid_test_data()
    _, metrics = _call_hybrid(0.5, *args)
    assert "loss" in metrics
    assert "num_valid_samples" in metrics
    assert "hybrid/loss_grpo" in metrics
    assert "hybrid/loss_sdpo" in metrics
    # GRPO component metrics are namespaced; SDPO's keep their sdpo/ prefix.
    assert any(k.startswith("grpo/") for k in metrics)
    assert "sdpo/per_pos_kl" in metrics
    # grpo_weight is logged from sdpo_train (not the loss fn) to avoid the
    # worker's per-microbatch metric scaling corrupting a constant.
    assert "hybrid/grpo_weight" not in metrics


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5])
def test_hybrid_rejects_out_of_range_weight(bad_weight):
    with pytest.raises(ValueError, match="grpo_weight"):
        SDPOHybridLossFn(
            {"grpo_weight": bad_weight, "sdpo": _sdpo_cfg(), "grpo": _grpo_cfg()}
        )


def test_hybrid_advantage_prompt_ids_grouping():
    """Regression: sdpo_train builds a 2-D [B, 1] prompt-id index for the GRPO
    advantage estimator. A 1-D index crashes calculate_baseline_and_std_per_prompt
    (it groups via torch.unique(dim=0) and reduces with .all(1)).

    Verifies the prompt-major repeat_interleave layout yields correct
    leave-one-out group baselines and the expected [B, S] advantage shape.
    """
    from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator

    num_prompts, num_generations, seq_len = 2, 2, 5
    batch_size = num_prompts * num_generations  # 4

    # Same construction as sdpo_train (the line that was buggy).
    prompt_ids = (
        torch.arange(num_prompts).repeat_interleave(num_generations).unsqueeze(-1)
    )
    assert prompt_ids.shape == (batch_size, 1)

    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])  # group0: [1,0], group1: [1,1]
    mask = torch.ones(batch_size, seq_len)

    est = GRPOAdvantageEstimator(
        {"name": "grpo", "use_leave_one_out_baseline": True, "normalize_rewards": False},
        {},
    )
    advantages = est.compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask)

    assert advantages.shape == (batch_size, seq_len)
    # Advantages are constant across the sequence dimension (expanded from [B, 1]).
    assert torch.allclose(advantages, advantages[:, :1].expand_as(advantages))
    # Leave-one-out baselines: group0 sample0 baseline=reward of sample1 (0.0) -> adv 1.0;
    # sample1 baseline=1.0 -> adv -1.0. group1 both baselines=1.0 -> adv 0.0.
    expected_col = torch.tensor([1.0, -1.0, 0.0, 0.0])
    assert torch.allclose(advantages[:, 0], expected_col), advantages[:, 0]
