import torch
from torch import nn


class MOELinear(nn.Module):
    r"""
    Mixture-of-experts linear layer.
    """

    def __init__(self, experts: list[nn.Module], gate: nn.Module, topk_experts: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = len(self.experts)
        self.topk_experts = topk_experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.topk_experts)
        weights = nn.functional.softmax(weights, dim=1, dtype=torch.float).to(
            inputs.dtype
        )
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
