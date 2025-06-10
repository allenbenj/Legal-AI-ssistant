from __future__ import annotations




@dataclass
class ConnectedEntities:


    entity_id: str
    connected: List[Entity]


@dataclass
class CaseEntities:

    case_id: str
    entities: List[Entity]


@dataclass
class PathResult:

    path: List[Entity]


class KnowledgeGraphReasoningAgent:

