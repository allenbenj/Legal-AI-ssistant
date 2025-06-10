# -*- coding: utf-8 -*-
"""Dataclass models for advanced knowledge graph analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_graph_manager import Relationship


@dataclass
class InferredRelationships:
    """Container for relationships inferred from analysis."""

    relationships: List[Relationship] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "unspecified"


@dataclass
class ConflictAnalysis:
    """Details about detected conflicts in the graph."""

    conflicting_entities: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JurisdictionalAnalysis:
    """Summary of jurisdictional scope discovered during analysis."""

    jurisdictions: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class PrecedentNetwork:
    """Representation of cited precedents and their connections."""

    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class GraphInference:
    """Combined results from knowledge graph inference."""

    document_id: str
    inferred_relationships: InferredRelationships = field(
        default_factory=InferredRelationships
    )
    conflict_analysis: ConflictAnalysis = field(default_factory=ConflictAnalysis)
    jurisdictional_analysis: JurisdictionalAnalysis = field(
        default_factory=JurisdictionalAnalysis
    )
    precedent_network: PrecedentNetwork = field(default_factory=PrecedentNetwork)
    processing_time: float = 0.0
    generated_at: datetime = field(default_factory=datetime.utcnow)
