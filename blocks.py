"""Attention, conditional queries and routing for heterogeneous event tokens."""
import torch
from torch import nn
from torch.nn import functional as F

TS_STAT_DIM = 6
TOKEN_TYPE_USER, TOKEN_TYPE_ITEM = 0, 1
TOKEN_TYPE_MEMORY, TOKEN_TYPE_DUAL_Q = 7, 8
NUM_TOKEN_TYPES = 13
DOMAINS = ("seq_a", "seq_b", "seq_c", "seq_d")
_SEQ_DOMAIN_TO_TYPE = {domain: i + 3 for i, domain in enumerate(DOMAINS)}
_SEQ_DOMAIN_TO_MEMORY_TYPE = {domain: i + 9 for i, domain in enumerate(DOMAINS)}
_SEQ_DOMAIN_TO_DUAL_Q_TOPOLOGY = {domain: i + 13 for i, domain in enumerate(DOMAINS)}


class GatedAttention(nn.Module):
    """Scaled dot-product attention with a query-dependent output gate."""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(self, query, context, allowed=None, padding=None):
        batch, length, width = query.shape

        def heads(x):
            return x.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        visible = None if allowed is None else allowed[None, None]
        if padding is not None:
            valid = ~padding[:, None, None, :]
            visible = valid if visible is None else visible & valid
        out = F.scaled_dot_product_attention(
            heads(self.W_q(query)), heads(self.W_k(context)), heads(self.W_v(context)),
            attn_mask=visible, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(batch, length, width)
        return self.W_o(out * torch.sigmoid(self.W_g(query)))


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = GatedAttention(d_model, num_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, query, context):
        return query + self.attn(self.norm_q(query), self.norm_kv(context))


class ContextEncoder(nn.Module):
    def __init__(self, num_sequences, d_model, hod_dim=1):
        super().__init__()
        width = hod_dim + num_sequences * (TS_STAT_DIM + 1)
        self.proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * d_model),
                                  nn.SiLU(), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, hour, statistics, lengths):
        active = torch.stack([(n > 0).to(hour.dtype) for n in lengths], 1)
        features = torch.cat([hour, torch.cat(statistics, 1).to(hour), active], 1)
        return self.proj(features).unsqueeze(1)


class BidirectionalQueries(nn.Module):
    """Cross-condition attribute tokens and pool user/item query slots per domain."""
    def __init__(self, d_model, num_heads, num_sequences, dropout=0.0,
                 user_q_tokens=4, item_q_tokens=2):
        super().__init__()
        self.user_pool_score = nn.Linear(d_model, user_q_tokens)
        self.item_pool_score = nn.Linear(d_model, item_q_tokens)
        self.cross_user_view = CrossAttention(d_model, num_heads, dropout)
        self.cross_item_view = CrossAttention(d_model, num_heads, dropout)
        self.film_per_domain = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(TS_STAT_DIM), nn.Linear(TS_STAT_DIM, d_model),
                          nn.SiLU(), nn.Linear(d_model, 2 * d_model))
            for _ in range(num_sequences)])
        for film in self.film_per_domain:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    @staticmethod
    def pool(tokens, score):
        return torch.einsum("blk,bld->bkd", score(tokens).softmax(1), tokens)

    def forward(self, user, item, context, statistics):
        user_view = self.cross_user_view(user, torch.cat([item, context], 1))
        item_view = self.cross_item_view(item, torch.cat([user, context], 1))
        shared = torch.cat([self.pool(user_view, self.user_pool_score),
                            self.pool(item_view, self.item_pool_score)], 1)
        queries = []
        for film, stats in zip(self.film_per_domain, statistics):
            scale, shift = film(stats.to(shared.device)).chunk(2, -1)
            queries.append(shared * (1 + scale[:, None]) + shift[:, None])
        return queries


class SwiGLU(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.fc = nn.Linear(width, 2 * width * expansion)
        self.fc_out = nn.Linear(width * expansion, width)

    def forward(self, x):
        value, gate = self.fc(x).chunk(2, -1)
        return self.fc_out(value * F.silu(gate))


class LowRankAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.silu(self.down(x)))


class TypedFeedForward(nn.Module):
    def __init__(self, width, expansion, num_types, rank, dropout):
        super().__init__()
        self.shared = SwiGLU(width, expansion)
        self.adapters = nn.ModuleList([LowRankAdapter(width, rank) for _ in range(num_types)])
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, types):
        hidden = self.norm(x)
        update = self.shared(hidden)
        for index, adapter in enumerate(self.adapters):
            selected = types == index
            if selected.any():
                update = update + adapter(hidden) * selected.unsqueeze(-1).float()
        return x + self.dropout(update)


class RoutedBlock(nn.Module):
    def __init__(self, width, heads, expansion, dropout, num_types, rank):
        super().__init__()
        self.norm_attn = nn.LayerNorm(width)
        self.attn = GatedAttention(width, heads, dropout)
        self.ffn = TypedFeedForward(width, expansion, num_types, rank, dropout)

    def forward(self, x, types, allowed, padding):
        hidden = self.norm_attn(x)
        return self.ffn(x + self.attn(hidden, hidden, allowed, padding), types)


class RoutingBackbone(nn.Module):
    """Joint token updates with local history edges and hierarchical memory."""
    def __init__(self, d_model, num_heads, num_layers, hidden_mult=4, dropout=0.0,
                 num_memory_tokens=8, num_domain_memory_tokens=2, num_token_types=NUM_TOKEN_TYPES,
                 adapter_rank=8, local_window=8, recent_window=32, seq_domains=DOMAINS):
        super().__init__()
        self.seq_domains = list(seq_domains)
        self.num_memory_tokens = num_memory_tokens
        self.num_domain_memory_tokens = num_domain_memory_tokens
        self.local_window, self.recent_window = local_window, recent_window
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, d_model) * 0.02)
        self.domain_memory_tokens = nn.ParameterDict({
            domain: nn.Parameter(torch.randn(num_domain_memory_tokens, d_model) * 0.02)
            for domain in self.seq_domains})
        self.type_embedding = nn.Embedding(num_token_types, d_model)
        self.layers = nn.ModuleList([RoutedBlock(d_model, num_heads, hidden_mult, dropout,
                                                num_token_types, adapter_rank) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def total_memory_tokens(self):
        return self.num_memory_tokens + len(self.seq_domains) * self.num_domain_memory_tokens

    def _build_domain_memory_mask(self, topology_ids):
        """Rows read columns. Zero allows an edge; negative infinity blocks it."""
        n = topology_ids.numel()
        query, key = topology_ids[:, None], topology_ids[None, :]
        global_query, global_key = query == TOKEN_TYPE_MEMORY, key == TOKEN_TYPE_MEMORY
        sequence = (topology_ids >= 3) & (topology_ids <= 6)
        domain_memory = (topology_ids >= 9) & (topology_ids <= 12)
        memory = (topology_ids == TOKEN_TYPE_MEMORY) | domain_memory
        position = torch.arange(n, device=topology_ids.device)
        distance = (position[:, None] - position[None, :]).abs()
        allow = global_query.expand(n, n).clone()
        allow |= sequence[:, None] & sequence[None, :] & (query == key) & (distance <= self.local_window)
        recent = torch.zeros(n, dtype=torch.bool, device=topology_ids.device)
        for domain in self.seq_domains:
            seq_id = _SEQ_DOMAIN_TO_TYPE[domain]
            mem_id = _SEQ_DOMAIN_TO_MEMORY_TYPE[domain]
            q_id = _SEQ_DOMAIN_TO_DUAL_Q_TOPOLOGY[domain]
            allow |= (query == mem_id) & ((key == seq_id) | (key == q_id) | memory[None, :])
            allow |= (query == seq_id) & ((key == mem_id) | (key == q_id) | global_key)
            allow |= (query == q_id) & (key == seq_id)
            indices = (topology_ids == seq_id).nonzero().flatten()
            recent[indices[:self.recent_window]] = True
        ordinary = ~sequence & ~memory
        allow |= ordinary[:, None] & ((~sequence)[None, :] | recent[None, :])
        return torch.zeros(n, n, device=topology_ids.device).masked_fill(~allow, float("-inf"))

    def forward(self, tokens, type_ids, token_mask=None, topology_ids=None):
        batch, _, _ = tokens.shape
        topology_ids = type_ids if topology_ids is None else topology_ids
        tokens = tokens + self.type_embedding(type_ids)
        memories = [self.memory_tokens] + [self.domain_memory_tokens[d] for d in self.seq_domains]
        memory_types = [TOKEN_TYPE_MEMORY] + [_SEQ_DOMAIN_TO_MEMORY_TYPE[d] for d in self.seq_domains]
        memory_tokens = torch.cat(memories, 0).unsqueeze(0).expand(batch, -1, -1)
        memory_ids = torch.cat([torch.full((batch, m.shape[0]), t, dtype=torch.long, device=tokens.device)
                                for m, t in zip(memories, memory_types)], 1)
        hidden = torch.cat([memory_tokens, tokens], 1)
        full_types = torch.cat([memory_ids, type_ids], 1)
        full_topology = torch.cat([memory_ids, topology_ids], 1)
        padding = None if token_mask is None else torch.cat([
            torch.zeros(batch, self.total_memory_tokens, dtype=torch.bool, device=tokens.device), token_mask], 1)
        allowed = self._build_domain_memory_mask(full_topology[0]) == 0
        for layer in self.layers:
            hidden = layer(hidden, full_types, allowed, padding)
        return self.output_norm(hidden), full_types
