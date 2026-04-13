from __future__ import annotations

import torch
from torch import nn

from .relation_graph_critic_data import RelationGraphBatch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


class RelationMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_type_count: int) -> None:
        super().__init__()
        self.edge_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(edge_type_count)]
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_resolved: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, hidden_dim = node_states.shape
        if node_count == 0:
            return node_states

        messages = torch.zeros_like(node_states)
        local_stats = torch.zeros((batch_size, node_count, 3), dtype=node_states.dtype, device=node_states.device)

        source_index = edge_index[..., 0].clamp(min=0, max=max(node_count - 1, 0))
        target_index = edge_index[..., 1].clamp(min=0, max=max(node_count - 1, 0))
        gather_index = source_index.unsqueeze(-1).expand(-1, -1, hidden_dim)
        source_states = torch.gather(node_states, dim=1, index=gather_index)
        edge_messages = torch.zeros_like(source_states)

        for relation_id, relation_linear in enumerate(self.edge_linears):
            relation_mask = edge_mask & (edge_type_ids == relation_id)
            if relation_mask.any():
                edge_messages[relation_mask] = relation_linear(source_states[relation_mask])

        edge_weights = edge_mask.unsqueeze(-1).float() * (1.0 + 0.1 * edge_resolved.unsqueeze(-1))
        edge_messages = edge_messages * edge_weights
        messages.scatter_add_(
            dim=1,
            index=target_index.unsqueeze(-1).expand(-1, -1, hidden_dim),
            src=edge_messages,
        )
        valid_edge_weight = edge_mask.float()
        local_stats[..., 0].scatter_add_(dim=1, index=target_index, src=valid_edge_weight)
        local_stats[..., 1].scatter_add_(
            dim=1,
            index=target_index,
            src=edge_resolved * valid_edge_weight,
        )
        local_stats[..., 2].scatter_add_(dim=1, index=source_index, src=valid_edge_weight)

        updated = self.update(torch.cat([node_states, messages, local_stats], dim=-1))
        updated = self.norm(node_states + updated)
        return updated * node_mask.unsqueeze(-1).float()


class RelationGraphCritic(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        hidden_dim: int,
        node_type_count: int,
        role_count: int,
        edge_type_count: int,
        candidate_kind_count: int,
    ) -> None:
        super().__init__()
        self.node_type_embed = nn.Embedding(node_type_count, 16)
        self.role_embed = nn.Embedding(role_count, 16)
        self.candidate_kind_embed = nn.Embedding(candidate_kind_count, 16)
        self.node_project = nn.Sequential(
            nn.Linear(text_dim + 16 + 16 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [RelationMessageLayer(hidden_dim, edge_type_count) for _ in range(2)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + text_dim * 2 + 16 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: RelationGraphBatch) -> torch.Tensor:
        node_inputs = torch.cat(
            [
                batch.node_text_embeddings,
                self.node_type_embed(batch.node_type_ids),
                self.role_embed(batch.node_role_ids),
                batch.node_scalars,
            ],
            dim=-1,
        )
        node_states = self.node_project(node_inputs) * batch.graph_mask.unsqueeze(-1).float()
        for layer in self.layers:
            node_states = layer(
                node_states=node_states,
                edge_index=batch.edge_index,
                edge_type_ids=batch.edge_type_ids,
                edge_resolved=batch.edge_resolved,
                edge_mask=batch.edge_mask,
                node_mask=batch.graph_mask,
            )

        graph_summary = _masked_mean(node_states, batch.graph_mask)
        target_summary = _masked_mean(node_states, batch.target_mask)
        neighbor_summary = _masked_mean(node_states, batch.neighbor_mask)
        action_summary = torch.cat(
            [
                batch.candidate_text_embeddings,
                self.candidate_kind_embed(batch.candidate_kind_ids),
                batch.is_commit.unsqueeze(-1),
            ],
            dim=-1,
        )
        score_inputs = torch.cat(
            [
                graph_summary,
                target_summary,
                neighbor_summary,
                action_summary,
                batch.state_text_embeddings,
            ],
            dim=-1,
        )
        return self.scorer(score_inputs).squeeze(-1)
