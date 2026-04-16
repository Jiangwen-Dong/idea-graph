from __future__ import annotations

import torch
from torch import nn

from .relation_graph_critic_data import RelationGraphBatch
from .relation_graph_critic_model import RelationMessageLayer, _masked_mean
from .relation_graph_two_head_data import RelationGraphCommitBatch


class RelationGraphSharedEncoder(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        hidden_dim: int,
        node_type_count: int,
        role_count: int,
        edge_type_count: int,
    ) -> None:
        super().__init__()
        self.node_type_embed = nn.Embedding(node_type_count, 16)
        self.role_embed = nn.Embedding(role_count, 16)
        self.node_project = nn.Sequential(
            nn.Linear(text_dim + 16 + 16 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([RelationMessageLayer(hidden_dim, edge_type_count) for _ in range(2)])

    def encode(
        self,
        *,
        node_text_embeddings: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_role_ids: torch.Tensor,
        node_scalars: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_resolved: torch.Tensor,
        edge_mask: torch.Tensor,
        graph_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_inputs = torch.cat(
            [
                node_text_embeddings,
                self.node_type_embed(node_type_ids),
                self.role_embed(node_role_ids),
                node_scalars,
            ],
            dim=-1,
        )
        node_states = self.node_project(node_inputs) * graph_mask.unsqueeze(-1).float()
        for layer in self.layers:
            node_states = layer(
                node_states=node_states,
                edge_index=edge_index,
                edge_type_ids=edge_type_ids,
                edge_resolved=edge_resolved,
                edge_mask=edge_mask,
                node_mask=graph_mask,
            )
        return node_states


class RelationGraphTwoHeadCritic(nn.Module):
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
        self.encoder = RelationGraphSharedEncoder(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            node_type_count=node_type_count,
            role_count=role_count,
            edge_type_count=edge_type_count,
        )
        self.candidate_kind_embed = nn.Embedding(candidate_kind_count, 16)
        self.edit_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + text_dim * 2 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.commit_head = nn.Sequential(
            nn.Linear(hidden_dim + text_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_edit_batch(self, batch: RelationGraphBatch) -> torch.Tensor:
        node_states = self.encoder.encode(
            node_text_embeddings=batch.node_text_embeddings,
            node_type_ids=batch.node_type_ids,
            node_role_ids=batch.node_role_ids,
            node_scalars=batch.node_scalars,
            edge_index=batch.edge_index,
            edge_type_ids=batch.edge_type_ids,
            edge_resolved=batch.edge_resolved,
            edge_mask=batch.edge_mask,
            graph_mask=batch.graph_mask,
        )
        graph_summary = _masked_mean(node_states, batch.graph_mask)
        target_summary = _masked_mean(node_states, batch.target_mask)
        neighbor_summary = _masked_mean(node_states, batch.neighbor_mask)
        action_summary = torch.cat(
            [
                batch.candidate_text_embeddings,
                self.candidate_kind_embed(batch.candidate_kind_ids),
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
        return self.edit_head(score_inputs).squeeze(-1)

    def score_commit_batch(self, batch: RelationGraphCommitBatch) -> torch.Tensor:
        node_states = self.encoder.encode(
            node_text_embeddings=batch.node_text_embeddings,
            node_type_ids=batch.node_type_ids,
            node_role_ids=batch.node_role_ids,
            node_scalars=batch.node_scalars,
            edge_index=batch.edge_index,
            edge_type_ids=batch.edge_type_ids,
            edge_resolved=batch.edge_resolved,
            edge_mask=batch.edge_mask,
            graph_mask=batch.graph_mask,
        )
        graph_summary = _masked_mean(node_states, batch.graph_mask)
        commit_inputs = torch.cat(
            [
                graph_summary,
                batch.state_text_embeddings,
                batch.graph_features,
            ],
            dim=-1,
        )
        return self.commit_head(commit_inputs).squeeze(-1)
