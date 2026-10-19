"""Recent-event preservation and candidate-conditioned summaries of older events."""
import math

import torch
from torch import nn


def _assemble(tokens, mask, keep, summaries, pool):
    batch, length, width = tokens.shape
    prefix = min(keep, length)
    parts, masks = [tokens[:, :prefix]], [mask[:, :prefix]]
    for index in range(summaries):
        start = prefix + (length - prefix) * index // summaries
        end = prefix + (length - prefix) * (index + 1) // summaries
        if start == end:
            parts.append(tokens.new_zeros(batch, 1, width))
            masks.append(torch.ones(batch, 1, dtype=torch.bool, device=tokens.device))
            continue
        padding = mask[:, start:end]
        parts.append(pool(tokens[:, start:end], padding).unsqueeze(1))
        masks.append(padding.all(1, keepdim=True))
    return torch.cat(parts, 1), torch.cat(masks, 1)


def masked_mean(tokens, mask):
    valid = (~mask).unsqueeze(-1)
    return tokens.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)


def compress_history(tokens, mask, keep, summary_count):
    """Mean-pooling control with the same bins, ordering and output token budget."""
    return _assemble(tokens, mask, keep, summary_count, masked_mean)


class TargetTailPool(nn.Module):
    """Score tail events against the candidate and blend with their masked mean.

    Every domain owns a scorer and a gate. The score output starts at zero,
    so training starts with uniform attention over the valid events in each bin.
    """
    def __init__(self, width):
        super().__init__()
        self.event_norm = nn.LayerNorm(width)
        self.target_norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(4 * width, max(8, width // 2)),
                                   nn.SiLU(), nn.Linear(max(8, width // 2), 1))
        self.gate = nn.Linear(2 * width, 1)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(0.1 / 0.9))

    def pool(self, events, padding, target):
        events = events.masked_fill(padding.unsqueeze(-1), 0)
        mean = masked_mean(events, padding)
        event_view = self.event_norm(events)
        target_view = self.target_norm(target)
        query = target_view.unsqueeze(1).expand_as(event_view)
        features = torch.cat([event_view, query, event_view * query, event_view - query], -1)
        scores = self.score(features).squeeze(-1).float()
        # Float32 softmax and an explicit empty-bin normalization also support BF16.
        weights = scores.masked_fill(padding, torch.finfo(scores.dtype).min).softmax(1)
        weights = weights.masked_fill(padding, 0)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        attended = (weights.unsqueeze(-1) * events.float()).sum(1).to(events.dtype)
        gate = self.gate(torch.cat([target_view, self.event_norm(mean)], -1)).sigmoid()
        summary = mean + gate * (attended - mean)
        return summary.masked_fill(padding.all(1, keepdim=True), 0)

    def forward(self, tokens, mask, target, keep, summary_count):
        return _assemble(tokens, mask, keep, summary_count,
                         lambda events, padding: self.pool(events, padding, target))
