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

"""Unit tests for SDPOLossFn.

Following the convention of ``test_loss_functions.py``, these tests drive the
loss through its real ``__call__`` (raw student logits + a data dict carrying
the teacher top-k) on GPU, skipping when no GPU is available.
"""

import math

import pytest
import torch

from nemo_rl.algorithms.loss_functions import SDPOLossFn


def setup_sdpo_test_data(
    batch_size=2,
    seq_len=6,
    vocab_size=16,
    topk=8,
    student_logits=None,
    teacher_topk_logits=None,
    teacher_topk_indices=None,
):
    """Setup test data for SDPO loss function tests.

    Returns ``(data, student_logits)`` with everything on GPU. When the teacher
    top-k tensors are not supplied they are drawn from an independent random
    teacher, so student and teacher disagree by construction.
    """
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"

    if student_logits is None:
        student_logits = torch.randn((batch_size, seq_len, vocab_size), device=device)
    student_logits = student_logits.to(device)

    if teacher_topk_logits is None or teacher_topk_indices is None:
        teacher_logits = torch.randn((batch_size, seq_len, vocab_size), device=device)
        teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)

    data = {
        "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),
        "token_mask": torch.ones((batch_size, seq_len), device=device),
        "sample_mask": torch.ones(batch_size, device=device),
        "sdpo_mask": torch.ones(batch_size, device=device),
        "teacher_topk_logits": teacher_topk_logits.to(device),
        "teacher_topk_indices": teacher_topk_indices.to(device),
    }
    return data, student_logits


def _global_valid_toks(data):
    return torch.sum(data["sample_mask"].unsqueeze(-1) * data["token_mask"])


@pytest.mark.parametrize("kl_type", ["forward", "reverse", "mixed", "js"])
def test_sdpo_loss_runs_for_all_kl_types(kl_type):
    """The loss returns a finite scalar and the expected metrics for each KL."""
    data, student_logits = setup_sdpo_test_data()
    loss_fn = SDPOLossFn(
        {
            "kl_type": kl_type,
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": True,
            "success_reward_threshold": 1.0,
        }
    )
    loss, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert "loss" in metrics
    assert "sdpo/per_pos_kl" in metrics


@pytest.mark.parametrize("kl_type", ["forward", "reverse", "mixed", "js"])
def test_sdpo_loss_zero_when_teacher_equals_student(kl_type):
    """When the teacher top-k is the student's own top-k, the KL is ~0."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    student_logits = torch.randn((2, 6, 16), device="cuda")
    topk = 8
    teacher_topk_logits, teacher_topk_indices = student_logits.topk(topk, dim=-1)
    data, student_logits = setup_sdpo_test_data(
        student_logits=student_logits,
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
    )
    loss_fn = SDPOLossFn(
        {
            "kl_type": kl_type,
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # top-k-only KL; no tail correction
            "success_reward_threshold": 1.0,
        }
    )
    loss, _ = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), loss.item()


def test_sdpo_loss_positive_when_distributions_differ():
    """Reverse KL is strictly positive when teacher != student."""
    data, student_logits = setup_sdpo_test_data()
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    loss, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert loss.item() > 0
    assert metrics["sdpo/per_pos_kl"] > 0


def test_sdpo_loss_zero_when_no_demos():
    """Samples without a demonstration (sdpo_mask=0) contribute zero."""
    data, student_logits = setup_sdpo_test_data()
    data["sdpo_mask"] = torch.zeros_like(data["sdpo_mask"])
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    # global_valid_toks=1 to avoid division by zero; mask reduces to all-zeros.
    loss, _ = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=torch.tensor(1.0, device=student_logits.device),
    )
    assert loss.item() == 0.0


def test_sdpo_loss_token_mask_excludes_prompt_positions():
    """token_mask=0 at a position removes it from the loss average."""
    data, student_logits = setup_sdpo_test_data(seq_len=6)
    bsz, seq_len = data["token_mask"].shape
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": True,  # exercise the tail-correction path
            "success_reward_threshold": 1.0,
        }
    )

    tm = torch.ones((bsz, seq_len), device=student_logits.device)
    tm[:, : seq_len // 2] = 0.0
    data["token_mask"] = tm
    loss_masked, _ = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )

    data["token_mask"] = torch.ones((bsz, seq_len), device=student_logits.device)
    loss_full, _ = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert loss_masked.item() != loss_full.item()


def test_sdpo_loss_js_bounded_by_log2():
    """JS divergence per position is bounded above by log 2."""
    data, student_logits = setup_sdpo_test_data()
    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    _, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert metrics["sdpo/per_pos_kl"] <= math.log(2) + 1e-3


def test_sdpo_loss_invalid_config_raises():
    with pytest.raises(ValueError, match="kl_type"):
        SDPOLossFn(
            {
                "kl_type": "bogus",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
            }
        )
    with pytest.raises(ValueError, match="mixed_kl_weight"):
        SDPOLossFn(
            {
                "kl_type": "mixed",
                "mixed_kl_weight": 1.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
            }
        )


def test_sdpo_ref_kl_zero_when_student_equals_ref():
    """Trust-region penalty is exactly 0 when prev_logprobs == reference."""
    data, student_logits = setup_sdpo_test_data()
    bsz, seq_len = data["token_mask"].shape
    student_lp = torch.randn((bsz, seq_len), device=student_logits.device)
    data["prev_logprobs"] = student_lp
    data["reference_policy_logprobs"] = student_lp.clone()

    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
            "reference_policy_kl_penalty": 1.0,
            "reference_policy_kl_type": "k3",
        }
    )
    _, metrics = loss_fn(
        student_logits,
        data,
        global_valid_seqs=torch.sum(data["sample_mask"]),
        global_valid_toks=_global_valid_toks(data),
    )
    assert metrics["sdpo/ref_kl"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("kl_estimator", ["k1", "k2", "k3"])
def test_sdpo_ref_kl_changes_loss_when_drifted(kl_estimator):
    """Ref-KL shifts the loss between beta=0 and beta=1 when the student has
    drifted from the reference; k2/k3 are non-negative by construction."""
    data, student_logits = setup_sdpo_test_data()
    bsz, seq_len = data["token_mask"].shape
    student_lp = torch.randn((bsz, seq_len), device=student_logits.device)
    ref_lp = student_lp + torch.randn_like(student_lp) * 0.5  # drifted
    data["prev_logprobs"] = student_lp
    data["reference_policy_logprobs"] = ref_lp

    cfg_base = {
        "kl_type": "js",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,
        "success_reward_threshold": 1.0,
        "reference_policy_kl_type": kl_estimator,
    }
    gvs = torch.sum(data["sample_mask"])
    gvt = _global_valid_toks(data)
    loss_low, _ = SDPOLossFn({**cfg_base, "reference_policy_kl_penalty": 0.0})(
        student_logits, data, global_valid_seqs=gvs, global_valid_toks=gvt
    )
    loss_high, metrics_high = SDPOLossFn(
        {**cfg_base, "reference_policy_kl_penalty": 1.0}
    )(student_logits, data, global_valid_seqs=gvs, global_valid_toks=gvt)

    assert abs(loss_high.item() - loss_low.item()) > 1e-4
    assert "sdpo/ref_kl" in metrics_high
    if kl_estimator in {"k2", "k3"}:
        assert metrics_high["sdpo/ref_kl"] >= -1e-7


def test_sdpo_ref_kl_invalid_config_raises():
    with pytest.raises(ValueError, match="reference_policy_kl_penalty"):
        SDPOLossFn(
            {
                "kl_type": "js",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
                "reference_policy_kl_penalty": -0.1,
            }
        )
    with pytest.raises(ValueError, match="reference_policy_kl_type"):
        SDPOLossFn(
            {
                "kl_type": "js",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
                "reference_policy_kl_penalty": 0.1,
                "reference_policy_kl_type": "k4",
            }
        )
