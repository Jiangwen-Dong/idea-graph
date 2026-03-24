from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Provenance:
    role: str
    branch_id: str
    source: str


@dataclass
class Node:
    id: str
    type: str
    text: str
    role: str
    branch_id: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(default_factory=utc_now)
    provenance: list[Provenance] = field(default_factory=list)


@dataclass
class Edge:
    id: str
    source_id: str
    relation: str
    target_id: str
    role: str
    branch_id: str
    evidence_id: str | None = None
    note: str = ""
    resolved: bool = False
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Branch:
    id: str
    role: str
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    frozen: bool = False
    rejected: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class GraphAction:
    id: str
    round_name: str
    role: str
    kind: str
    target_ids: list[str]
    payload: dict[str, object] = field(default_factory=dict)
    rationale: str = ""
    timestamp: datetime = field(default_factory=utc_now)


@dataclass
class MaturitySnapshot:
    support_coverage: float
    unresolved_contradiction_ratio: float
    utility: float
    utility_stable: bool
    completeness: bool
    is_mature: bool


@dataclass
class FinalProposal:
    title: str = ""
    abstract: str = ""
    problem: str = ""
    existing_methods: str = ""
    motivation: str = ""
    hypothesis: str = ""
    method: str = ""
    evaluation: str = ""
    significance: str = ""
    caveats: str = ""


@dataclass
class IdeaGraph:
    topic: str
    literature: list[str]
    metadata: dict[str, object] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    branches: dict[str, Branch] = field(default_factory=dict)
    actions: list[GraphAction] = field(default_factory=list)
    round_summaries: list[tuple[str, MaturitySnapshot]] = field(default_factory=list)
    utility_history: list[float] = field(default_factory=list)
    matured_at_round: str | None = None
    final_subgraph: dict[str, object] | None = None
    final_proposal: FinalProposal | None = None
    _node_counter: int = 0
    _edge_counter: int = 0
    _branch_counter: int = 0
    _action_counter: int = 0

    def next_node_id(self) -> str:
        self._node_counter += 1
        return f"N{self._node_counter:03d}"

    def next_edge_id(self) -> str:
        self._edge_counter += 1
        return f"E{self._edge_counter:03d}"

    def next_branch_id(self) -> str:
        self._branch_counter += 1
        return f"B{self._branch_counter:03d}"

    def next_action_id(self) -> str:
        self._action_counter += 1
        return f"A{self._action_counter:03d}"

    def active_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.status == "active"]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target_id == node_id]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source_id == node_id]
