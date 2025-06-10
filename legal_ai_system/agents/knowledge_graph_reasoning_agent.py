"""Advanced reasoning over the legal knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import json

try:  # pragma: no cover - optional dependency
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional networkx
    nx = None  # type: ignore

# Optional graph neural network libraries
GNN_LIB_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import torch_geometric  # type: ignore
    GNN_LIB_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when torch_geometric not installed
    try:
        import dgl  # type: ignore
        GNN_LIB_AVAILABLE = True
    except Exception:
        GNN_LIB_AVAILABLE = False

from ..core.base_agent import BaseAgent
from ..services.knowledge_graph_manager import (
    KnowledgeGraphManager,
    EntityType,
    RelationshipType,
)


@dataclass
class InferredRelationship:
    """Represents an inferred connection between two entities."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float = 1.0
    reasoning: Optional[str] = None


@dataclass
class InferredRelationships:
    """Container for inferred relationships for an entity."""

    entity_id: str
    relationships: List[InferredRelationship] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConflictFinding:
    """Represents a detected conflict of interest."""

    entity_a: str
    entity_b: str
    conflict_type: str
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictAnalysis:
    """Results from conflict of interest detection."""

    conflicts: List[ConflictFinding] = field(default_factory=list)
    summary: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class JurisdictionAnalysisResult:
    """Jurisdictional information for an entity."""

    entity_id: str
    jurisdictions: List[str] = field(default_factory=list)
    reasoning: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PrecedentMatch:
    """Match found during precedent search."""

    case_id: str
    similarity: float
    path: List[str] = field(default_factory=list)


@dataclass
class PrecedentGraphSearchResult:
    """Structured result of precedent graph search."""

    matches: List[PrecedentMatch] = field(default_factory=list)
    search_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphReasoningAgent(BaseAgent):
    """Agent performing higher-level reasoning over the knowledge graph."""

    def __init__(self, services: Any, **config: Any) -> None:
        super().__init__(services, name="KnowledgeGraphReasoningAgent")
        self.graph_manager: Optional[KnowledgeGraphManager] = self.get_knowledge_graph_manager()
        if not self.graph_manager:
            self.logger.warning(
                "KnowledgeGraphManager not available. Reasoning capabilities will be limited.")
        self.gnn_available = GNN_LIB_AVAILABLE
        if self.gnn_available:
            self.logger.info("GNN libraries detected - advanced reasoning enabled.")
        else:
            self.logger.debug("GNN libraries not available - falling back to heuristics.")

    # ------------------------------------------------------------------
    def _build_networkx_graph(self):
        """Build a NetworkX graph from the manager's data."""
        if not self.graph_manager or not nx:
            return None

        G = nx.DiGraph()
        for ent in self.graph_manager.entities.values():
            G.add_node(ent.id, type=ent.type.value)
        for rel in self.graph_manager.relationships.values():
            G.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                type=rel.type.value,
                confidence=rel.confidence,
            )
        return G

    # ------------------------------------------------------------------
    async def entity_relationship_inference(self, entity_id: str, depth: int = 2) -> InferredRelationships:
        """Infer relationships around the provided entity."""
        if not self.graph_manager:
            return InferredRelationships(entity_id=entity_id)

        graph = self._build_networkx_graph()
        if not graph or entity_id not in graph:
            return InferredRelationships(entity_id=entity_id)

        neighbors = nx.single_source_shortest_path_length(graph, entity_id, cutoff=depth).keys()
        inferred: List[InferredRelationship] = []
        for target in neighbors:
            if target == entity_id:
                continue
            if graph.has_edge(entity_id, target):
                edge_data = graph[entity_id][target]
            elif graph.has_edge(target, entity_id):
                edge_data = graph[target][entity_id]
            else:
                edge_data = {"type": "related", "confidence": 0.5}
            inferred.append(
                InferredRelationship(
                    source_entity_id=entity_id,
                    target_entity_id=target,
                    relationship_type=edge_data.get("type", "related"),
                    confidence=float(edge_data.get("confidence", 1.0)),
                )
            )

        return InferredRelationships(entity_id=entity_id, relationships=inferred)

    # ------------------------------------------------------------------
    async def conflict_of_interest_detection(self, party_entity_ids: List[str]) -> ConflictAnalysis:
        """Detect conflicts of interest among a set of parties."""
        if not self.graph_manager:
            return ConflictAnalysis()

        conflicts: List[ConflictFinding] = []
        # Map of representative -> list of represented parties
        rep_map: Dict[str, List[str]] = {}
        for rel in self.graph_manager.relationships.values():
            if rel.type == RelationshipType.REPRESENTS and rel.target_entity_id in party_entity_ids:
                rep_map.setdefault(rel.source_entity_id, []).append(rel.target_entity_id)

        for lawyer_id, clients in rep_map.items():
            if len(clients) > 1:
                for i in range(len(clients)):
                    for j in range(i + 1, len(clients)):
                        conflicts.append(
                            ConflictFinding(
                                entity_a=clients[i],
                                entity_b=clients[j],
                                conflict_type="shared_representative",
                                confidence=1.0,
                                details={"lawyer_id": lawyer_id},
                            )
                        )

        summary = (
            f"{len(conflicts)} potential conflicts detected" if conflicts else "No conflicts detected"
        )
        return ConflictAnalysis(conflicts=conflicts, summary=summary)

    # ------------------------------------------------------------------
    async def jurisdiction_analysis(self, entity_id: str) -> JurisdictionAnalysisResult:
        """Analyze jurisdiction information for an entity."""
        if not self.graph_manager:
            return JurisdictionAnalysisResult(entity_id=entity_id)

        jurisdictions: List[str] = []
        for rel in self.graph_manager.relationships.values():
            if rel.source_entity_id == entity_id and rel.type == RelationshipType.LOCATED_IN:
                jurisdictions.append(rel.target_entity_id)

        reasoning = (
            "Collected LOCATED_IN relationships" if jurisdictions else "No jurisdiction information found"
        )
        return JurisdictionAnalysisResult(
            entity_id=entity_id, jurisdictions=jurisdictions, reasoning=reasoning
        )

    # ------------------------------------------------------------------
    async def precedent_graph_search(self, issue: str, limit: int = 5) -> PrecedentGraphSearchResult:
        """Search for precedent cases related to a specific issue."""
        if not self.graph_manager:
            return PrecedentGraphSearchResult()

        start = datetime.utcnow()
        matches: List[PrecedentMatch] = []

        for entity in self.graph_manager.entities.values():
            if entity.type == EntityType.CASE:
                text = json.dumps(entity.properties)
                if issue.lower() in text.lower():
                    matches.append(PrecedentMatch(case_id=entity.id, similarity=1.0))
                    if len(matches) >= limit:
                        break

        duration = (datetime.utcnow() - start).total_seconds()
        return PrecedentGraphSearchResult(matches=matches, search_time_sec=duration)

