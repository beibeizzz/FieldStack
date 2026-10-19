"""TailScope: candidate-conditioned history summaries and routed attention."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from blocks import (RoutingBackbone, BidirectionalQueries,
                  ContextEncoder, TS_STAT_DIM, TOKEN_TYPE_USER, TOKEN_TYPE_ITEM,
                  TOKEN_TYPE_DUAL_Q, _SEQ_DOMAIN_TO_TYPE,
                  _SEQ_DOMAIN_TO_DUAL_Q_TOPOLOGY)
from dataset import NUM_TIME_BUCKETS
from history import TargetTailPool, compress_history


@dataclass
class ModelConfig:
    d_model: int = 64
    emb_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    hidden_mult: int = 4
    dropout: float = 0.02
    user_tokens: int = 5
    item_tokens: int = 4
    user_queries: int = 4
    item_queries: int = 2
    global_memory: int = 8
    domain_memory: int = 2
    adapter_rank: int = 8
    local_window: int = 8
    recent_window: int = 32
    tail_summaries: int = 4
    tail_pooling: str = "target"
    max_embedding_vocab: int = 1000000
    query_mode: str = "dualq"
    topology: str = "semilocal"
    seq_max_lens: dict = field(default_factory=lambda: {
        "seq_a": 512, "seq_b": 256, "seq_c": 512, "seq_d": 768})
    recent_budgets: dict = field(default_factory=lambda: {
        "seq_a": 96, "seq_b": 64, "seq_c": 96, "seq_d": 128})

    def __post_init__(self):
        positive = ("d_model", "emb_dim", "num_heads", "num_layers", "hidden_mult",
                    "user_tokens", "item_tokens", "user_queries", "item_queries",
                    "global_memory", "adapter_rank", "max_embedding_vocab")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("Model dimensions, query counts and global memory must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if min(self.domain_memory, self.local_window, self.recent_window, self.tail_summaries) < 0:
            raise ValueError("Memory/window/summary counts must be nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.query_mode not in ("dualq", "mean") or self.topology not in ("semilocal", "full"):
            raise ValueError("Unknown query mode or topology")
        if self.tail_pooling not in ("target", "mean"):
            raise ValueError("tail_pooling must be target or mean")
        if any(v < 1 for v in self.seq_max_lens.values()) or any(v < 1 for v in self.recent_budgets.values()):
            raise ValueError("Sequence lengths and recent budgets must be positive")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def projection(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU())


class SparseTokens(nn.Module):
    """Embed sparse fields and project contiguous groups into tokens."""
    def __init__(self, specs, config, token_count):
        super().__init__()
        self.specs, self.emb_dim = specs, config.emb_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab + 1, config.emb_dim, padding_idx=0)
            if 0 < vocab <= config.max_embedding_vocab else None
            for _, vocab, _ in specs])
        self.chunk = max(1, math.ceil(len(specs) * config.emb_dim / token_count))
        self.projections = nn.ModuleList([
            projection(self.chunk, config.d_model) for _ in range(token_count)])

    def forward(self, values):
        fields, offset = [], 0
        for (_, _, width), embedding in zip(self.specs, self.embeddings):
            ids = values[:, offset:offset + width]
            offset += width
            if embedding is None:
                fields.append(torch.zeros(values.shape[0], self.emb_dim, device=values.device))
            else:
                valid = (ids != 0).unsqueeze(-1)
                fields.append((embedding(ids) * valid).sum(1) / valid.sum(1).clamp_min(1))
        flat = torch.cat(fields, -1) if fields else torch.zeros(values.shape[0], 0, device=values.device)
        flat = F.pad(flat, (0, self.chunk * len(self.projections) - flat.shape[1]))
        return torch.stack([proj(part) for proj, part in
                            zip(self.projections, flat.split(self.chunk, dim=1))], dim=1)


class DenseTokens(nn.Module):
    """One schema-defined dense field per token; no feature-ID-specific rules."""
    def __init__(self, specs, d_model):
        super().__init__()
        self.widths, self.d_model = [width for _, width in specs], d_model
        self.projections = nn.ModuleList([projection(width, d_model) for width in self.widths])

    def forward(self, values):
        if not self.widths:
            return values.new_zeros(values.shape[0], 0, self.d_model)
        return torch.stack([proj(part) for proj, part in
                            zip(self.projections, values.split(self.widths, dim=1))], dim=1)


class SequenceEmbedding(nn.Module):
    def __init__(self, specs, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab + 1, config.emb_dim, padding_idx=0)
            if 0 < vocab <= config.max_embedding_vocab else None for _, vocab in specs])
        self.proj = nn.Sequential(nn.Linear(len(specs) * config.emb_dim, config.d_model),
                                  nn.LayerNorm(config.d_model), nn.GELU())

    def forward(self, ids):
        fields = [embedding(ids[..., i]) if embedding is not None else
                  torch.zeros(*ids.shape[:2], self.emb_dim, device=ids.device)
                  for i, embedding in enumerate(self.embeddings)]
        return self.proj(torch.cat(fields, dim=-1))


class MeanQuery(nn.Module):
    """Generate domain queries from attributes and mean history."""
    def __init__(self, ns_count, domain_count, query_count, config):
        super().__init__()
        in_dim = (ns_count + 1) * config.d_model
        self.norm = nn.LayerNorm(in_dim)
        self.heads = nn.ModuleList([nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, config.d_model * config.hidden_mult), nn.SiLU(),
                          nn.Linear(config.d_model * config.hidden_mult, config.d_model),
                          nn.LayerNorm(config.d_model)) for _ in range(query_count)])
            for _ in range(domain_count)])

    def forward(self, ns, sequences, masks):
        outputs = []
        for heads, seq, mask in zip(self.heads, sequences, masks):
            valid = (~mask).unsqueeze(-1)
            pooled = (seq * valid).sum(1) / valid.sum(1).clamp_min(1)
            context = self.norm(torch.cat([ns.flatten(1), pooled], dim=-1))
            outputs.append(torch.stack([head(context) for head in heads], dim=1))
        return outputs


class FullAttentionBackbone(RoutingBackbone):
    """Same parameters and token budgets; all valid token pairs are visible."""
    def _build_domain_memory_mask(self, topology_ids):
        return torch.zeros(topology_ids.numel(), topology_ids.numel(), device=topology_ids.device)


class TailScope(nn.Module):
    def __init__(self, schema, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.domains = sorted(schema["seq"])
        if not self.domains or any(domain not in _SEQ_DOMAIN_TO_TYPE for domain in self.domains):
            raise ValueError("Expected one or more of seq_a, seq_b, seq_c, seq_d")
        self.user_sparse = SparseTokens(schema["user_int"], cfg, cfg.user_tokens)
        self.item_sparse = SparseTokens(schema["item_int"], cfg, cfg.item_tokens)
        self.user_dense = DenseTokens(schema.get("user_dense", []), cfg.d_model)
        self.item_dense = DenseTokens(schema.get("item_dense", []), cfg.d_model)
        self.sequence_embeddings = nn.ModuleDict({domain: SequenceEmbedding([
            (fid, vocab) for fid, vocab in schema["seq"][domain]["features"]
            if fid != schema["seq"][domain].get("ts_fid")], cfg) for domain in self.domains})
        self.time_embedding = nn.Embedding(NUM_TIME_BUCKETS, cfg.d_model, padding_idx=0)
        self.time_context = ContextEncoder(len(self.domains), cfg.d_model, hod_dim=1)
        ns_count = cfg.user_tokens + cfg.item_tokens + len(schema.get("user_dense", [])) + len(schema.get("item_dense", []))
        if cfg.query_mode == "dualq":
            self.query = BidirectionalQueries(cfg.d_model, cfg.num_heads, len(self.domains),
                                        dropout=cfg.dropout, user_q_tokens=cfg.user_queries,
                                        item_q_tokens=cfg.item_queries)
        else:
            self.query = MeanQuery(ns_count, len(self.domains), cfg.user_queries + cfg.item_queries, cfg)
        backbone = RoutingBackbone if cfg.topology == "semilocal" else FullAttentionBackbone
        self.backbone = backbone(cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.hidden_mult,
                                 cfg.dropout, cfg.global_memory, cfg.domain_memory,
                                 adapter_rank=cfg.adapter_rank, local_window=cfg.local_window,
                                 recent_window=cfg.recent_window, seq_domains=self.domains)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Sequential(
            nn.Linear((3 + len(self.domains)) * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, 1))
        # Construct after the common model so both pooling modes share initialization.
        self.tail_compressors = nn.ModuleDict({domain: TargetTailPool(cfg.d_model)
            for domain in self.domains} if cfg.tail_pooling == "target" and cfg.tail_summaries else {})

    def forward(self, batch, return_state=False):
        cfg = self.config
        user = torch.cat([self.user_sparse(batch["user_int"]), self.user_dense(batch["user_dense"])], 1)
        item = torch.cat([self.item_sparse(batch["item_int"]), self.item_dense(batch["item_dense"])], 1)
        sequences, masks = [], []
        for domain in self.domains:
            seq = self.sequence_embeddings[domain](batch["seq"][domain])
            seq = seq + self.time_embedding(batch["time_buckets"][domain])
            mask = batch["seq_mask"][domain]
            sequences.append(seq.masked_fill(mask.unsqueeze(-1), 0))
            masks.append(mask)
        if cfg.query_mode == "dualq":
            # Context uses request hour and domain activity; optional statistics are zero.
            stats = [user.new_zeros(user.shape[0], TS_STAT_DIM) for _ in self.domains]
            context = self.time_context(batch["hour"], stats, [(~m).sum(1) for m in masks])
            queries = self.query(user, item, context, stats)
        else:
            queries = self.query(torch.cat([user, item], 1), sequences, masks)
        query = torch.cat(queries, 1)
        batch_size, device = user.shape[0], user.device

        def ids(count, value):
            return torch.full((batch_size, count), value, dtype=torch.long, device=device)

        def valid_mask(count):
            return torch.zeros(batch_size, count, dtype=torch.bool, device=device)

        parts = [query, user, item]
        types = [ids(query.shape[1], TOKEN_TYPE_DUAL_Q), ids(user.shape[1], TOKEN_TYPE_USER), ids(item.shape[1], TOKEN_TYPE_ITEM)]
        topology = [torch.cat([ids(q.shape[1], _SEQ_DOMAIN_TO_DUAL_Q_TOPOLOGY[d])
                               for d, q in zip(self.domains, queries)], 1), types[1], types[2]]
        token_masks = [valid_mask(query.shape[1]), valid_mask(user.shape[1]), valid_mask(item.shape[1])]
        target = item.mean(1)
        for domain, seq, mask in zip(self.domains, sequences, masks):
            keep = cfg.recent_budgets.get(domain, 96)
            if domain in self.tail_compressors:
                seq, mask = self.tail_compressors[domain](seq, mask, target, keep, cfg.tail_summaries)
            else:
                seq, mask = compress_history(seq, mask, keep, cfg.tail_summaries)
            parts.append(seq)
            domain_ids = ids(seq.shape[1], _SEQ_DOMAIN_TO_TYPE[domain])
            types.append(domain_ids)
            topology.append(domain_ids)
            token_masks.append(mask)
        tokens = self.dropout(torch.cat(parts, 1))
        final, final_types = self.backbone(tokens, torch.cat(types, 1),
                                          torch.cat(token_masks, 1), torch.cat(topology, 1))
        offset = self.backbone.total_memory_tokens
        query_out = final[:, offset:offset + query.shape[1]].mean(1)
        item_start = offset + query.shape[1] + user.shape[1]
        item_out = final[:, item_start:item_start + item.shape[1]].mean(1)
        readout = [query_out, item_out, final[:, :cfg.global_memory].mean(1)]
        for i in range(len(self.domains)):
            start = cfg.global_memory + i * cfg.domain_memory
            readout.append(final[:, start:start + cfg.domain_memory].mean(1)
                           if cfg.domain_memory else final.new_zeros(batch_size, cfg.d_model))
        logits = self.classifier(torch.cat(readout, -1)).squeeze(-1)
        if return_state:
            return logits, {"tokens": final, "type_ids": final_types,
                            "prepared_sequence_tokens": sum(p.shape[1] for p in parts[3:])}
        return logits
