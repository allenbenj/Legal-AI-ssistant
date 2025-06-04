# legal_ai_system/agents/knowledge_management/knowledge_base_agent.py
"""
Knowledge Base Agent - Design.txt Compliant
Handles entity resolution, disambiguation, and ensures data is properly structured
and persisted in the knowledge graph and/or other persistent stores.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, before_sleep_log
from pybreaker import CircuitBreaker, CircuitBreakerError

# Core imports
from ...core.base_agent import BaseAgent, ProcessingResult
from ...core.shared_components import measure_performance, ProcessingCache
from ...utils.error_recovery import ErrorRecovery # Assuming error_recovery.py is in utils/
from ...core.unified_exceptions import AgentExecutionError, KnowledgeGraphError, DatabaseError, AgentInitializationError
from ...core.detailed_logging import LogCategory, get_detailed_logger
# For interacting with persistent stores (these would be services)
from ...knowledge.knowledge_graph_manager import KnowledgeGraphManager, KGEntity, KGRelationship # Added KGRelationship
# from ...persistence.enhanced_persistence import EntityRepository # Conceptual for relational DB

# Ontology imports for type checking if available
try:
    from ...utils.ontology import LegalEntityType, LegalRelationshipType, get_entity_type_by_label, get_relationship_type_by_label
    ONTOLOGY_AVAILABLE_KB = True
except ImportError:
    ONTOLOGY_AVAILABLE_KB = False
    # Minimal fallbacks for type hinting if ontology is not found
    class LegalEntityType: pass # type: ignore
    class LegalRelationshipType: pass # type: ignore
    def get_entity_type_by_label(label: str) -> Optional[Any]: return None
    def get_relationship_type_by_label(label: str) -> Optional[Any]: return None
    self.logger.warning("Ontology module not fully available for KnowledgeBaseAgent. Type validation might be limited.")


# Circuit breaker for critical database/KG operations
kb_persistence_breaker = CircuitBreaker(fail_max=3, reset_timeout=90)

@dataclass
class ResolvedEntity:
    entity_id: str # This is the canonical, persistent ID (e.g., UUID from KG)
    canonical_name: str
    entity_type: str # Should map to ontology.LegalEntityType.label
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict) # All properties including those from KG
    confidence_score: float = 0.0 # Confidence in this *resolved representation*
    source_document_ids: Set[str] = field(default_factory=set)
    # Direct relationships are now primarily managed IN the KG.
    # This object might store a summary or count if needed for quick reference.
    # linked_entity_ids: Dict[str, List[str]] = field(default_factory=dict) # e.g. {"RELATED_TO": ["id1", "id2"]}
    created_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1 # For optimistic locking or tracking updates
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., source of this resolution, resolution method

    def add_alias(self, alias: str):
        norm_alias = alias.lower().strip()
        if norm_alias and norm_alias != self.canonical_name.lower().strip():
            if norm_alias not in self.aliases:
                self.aliases.add(norm_alias)
                self.updated_at_iso = datetime.now(timezone.utc).isoformat()

    def merge_attributes(self, new_attributes: Dict[str, Any], strategy: str = "overwrite_if_empty_or_higher_quality"):
        # Example strategy: overwrite if new value is present and old is empty, or if new implies higher quality
        # (e.g. non-empty string overwrites empty, or specific logic based on attribute key)
        changed = False
        for key, value in new_attributes.items():
            if value is not None: # Only consider non-null new values
                if key not in self.attributes or not self.attributes[key] or strategy == "overwrite":
                    self.attributes[key] = value
                    changed = True
                # Add more sophisticated merge logic if needed, e.g., for list attributes
        if changed: self.updated_at_iso = datetime.now(timezone.utc).isoformat()

    def add_source_document(self, doc_id: str):
        if doc_id not in self.source_document_ids:
            self.source_document_ids.add(doc_id)
            self.updated_at_iso = datetime.now(timezone.utc).isoformat()

    def to_kg_entity(self) -> KGEntity: # Helper to convert to KGManager's expected type
        return KGEntity(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            name=self.canonical_name, # KGM uses 'name'
            attributes={**self.attributes, 'aliases': sorted(list(self.aliases))}, # Store aliases as attribute
            confidence=self.confidence_score,
            # source_document needs to be singular for KGEntity or KGM needs to handle list
            # For now, let's assume KGM can handle a list or we pick one primary source for the KGEntity object
            source_document=next(iter(self.source_document_ids), None) if self.source_document_ids else None,
            created_at=datetime.fromisoformat(self.created_at_iso),
            updated_at=datetime.fromisoformat(self.updated_at_iso)
        )

    @classmethod
    def from_kg_entity(cls, kg_entity: KGEntity) -> 'ResolvedEntity':
        aliases = set(kg_entity.attributes.pop('aliases', [])) # Extract and remove aliases from attributes
        return cls(
            entity_id=kg_entity.entity_id,
            canonical_name=kg_entity.name,
            entity_type=kg_entity.entity_type,
            aliases=aliases,
            attributes=kg_entity.attributes,
            confidence_score=kg_entity.confidence or 0.5, # Default if None
            source_document_ids={kg_entity.source_document} if kg_entity.source_document else set(),
            created_at_iso=kg_entity.created_at.isoformat() if kg_entity.created_at else datetime.now(timezone.utc).isoformat(),
            updated_at_iso=kg_entity.updated_at.isoformat() if kg_entity.updated_at else datetime.now(timezone.utc).isoformat(),
            # Version would typically be managed by the persistence layer itself
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['aliases'] = sorted(list(self.aliases))
        data['source_document_ids'] = sorted(list(self.source_document_ids))
        # created_at_iso and updated_at_iso are already strings
        return data

@dataclass
class KnowledgeBaseOutput: # Renamed from EntityResolutionOutput
    document_id_context: str # Document context for this resolution batch
    resolved_entities_updated_or_created: List[ResolvedEntity] = field(default_factory=list)
    relationships_updated_or_created: List[Dict[str, Any]] = field(default_factory=list) # Summary of relationships processed
    resolution_summary_metrics: Dict[str, Any] = field(default_factory=dict) # Renamed
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    processed_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat()) # Renamed
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['resolved_entities_updated_or_created'] = [e.to_dict() for e in self.resolved_entities_updated_or_created]
        return data


class KnowledgeBaseAgent(BaseAgent):
    
    def __init__(self, service_container: Any, **config: Any):
        super().__init__(service_container, name="KnowledgeBaseAgent", agent_type="knowledge_management")
        
        self.config = config
        self.kg_manager: Optional[KnowledgeGraphManager] = self._get_service("knowledge_graph_manager")
        if not self.kg_manager:
            self.logger.error("KnowledgeGraphManager service not available. KnowledgeBaseAgent persistence will be severely impacted.")
            # This agent heavily relies on KGM for its core function. Consider AgentInitializationError.

        self.cache = ProcessingCache(cache_dir_str=config.get('cache_dir_kb_agent', './storage/cache/kb_agent')) # Specific key
        self.error_recovery = ErrorRecovery( # For KG operations
            max_retries=int(config.get('kg_max_retries', 2)),
            base_delay=float(config.get('kg_base_delay_sec', 1.5))
        )

        # Entity resolution configuration
        self.name_similarity_threshold = float(config.get('kb_name_similarity_threshold', 0.88)) # Higher for more precision
        self.attribute_similarity_threshold = float(config.get('kb_attribute_similarity_threshold', 0.75))
        self.min_confidence_for_new_entity_creation = float(config.get('kb_min_confidence_for_new_entity', 0.65))
        self.min_confidence_for_relationship_creation = float(config.get('kb_min_confidence_for_relationship', 0.7))

        # This local registry is for short-term caching within a single processing run or session.
        # It does NOT replace the persistent KG.
        self._local_entity_cache_by_id: Dict[str, ResolvedEntity] = {}
        self._local_entity_cache_by_name_type: Dict[Tuple[str, str], ResolvedEntity] = {} # (type.upper(), name.lower()) -> ResolvedEntity
        
        self.stats.update({
            'raw_entities_processed': 0, 'raw_relationships_processed': 0,
            'new_entities_persisted': 0, 'entities_merged_in_kg': 0,
            'new_relationships_persisted': 0, 'relationships_updated_in_kg':0,
            'avg_entity_resolution_confidence': 0.0,
            'kg_ops_failed_count':0
        })
        
        self.logger.info(f"{self.name} initialized.",
                       parameters={'kg_manager_available': bool(self.kg_manager),
                                   'name_sim_thresh': self.name_similarity_threshold})
    
    @measure_performance("kb_agent_task")
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1.5, min=1, max=6),
        reraise=True,
        before_sleep=before_sleep_log(get_detailed_logger("KBAgentRetry", LogCategory.AGENT_LIFECYCLE), logging.WARNING)
    )
    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        document_id = metadata.get("document_id", f"kb_batch_{uuid.uuid4().hex[:8]}")
        # user_id can be used for auditing who triggered the update
        # user_id = metadata.get("user_id", "system_kb_agent")
        self.logger.info(f"Knowledge base processing task started for doc context '{document_id}'.")
        start_time = datetime.now(timezone.utc)
        
        output = KnowledgeBaseOutput(document_id_context=document_id)

        try:
            # Extract raw entities and relationships from task_data.
            # Assumes task_data might come from OntologyExtractionAgent or similar,
            # which would have 'entities': List[ExtractedEntityDict], 'relationships': List[ExtractedRelationshipDict]
            raw_entities_input = task_data.get('entities', [])
            raw_relationships_input = task_data.get('relationships', [])

            if not raw_entities_input and not raw_relationships_input:
                self.logger.warning(f"No raw entities or relationships provided for KB processing (doc context '{document_id}').")
                output.errors.append("No raw entities or relationships provided.")
                # Fall through to finally block for timing
                raise AgentExecutionError("Empty input for KnowledgeBaseAgent.", details={'doc_id': document_id})

            self.stats['raw_entities_processed'] += len(raw_entities_input)
            self.stats['raw_relationships_processed'] += len(raw_relationships_input)

            # 1. Resolve and Persist Entities
            # This step needs to handle finding existing entities or creating new ones in the KG.
            # It returns a list of ResolvedEntity objects that are now canonical.
            # It also needs a mapping from the input raw entity IDs/keys to the canonical ResolvedEntity IDs.
            resolved_entities_list, raw_to_canonical_id_map = await self._resolve_and_persist_entities_batch(
                raw_entities_input, document_id
            )
            output.resolved_entities_updated_or_created = resolved_entities_list

            # 2. Process and Persist Relationships using canonical entity IDs
            if raw_relationships_input and self.kg_manager:
                processed_rels_summary = await self._process_and_persist_relationships_batch(
                    raw_relationships_input, raw_to_canonical_id_map, document_id
                )
                output.relationships_updated_or_created = processed_rels_summary
            
            output.resolution_summary_metrics = self._calculate_task_summary_metrics(
                raw_entities_input, output.resolved_entities_updated_or_created, raw_relationships_input, output.relationships_updated_or_created
            )
            
            self._update_cumulative_agent_metrics(output)
            self.logger.info(f"Knowledge base processing completed for doc context '{document_id}'.",
                           parameters={'entities_final': len(output.resolved_entities_updated_or_created),
                                       'rels_final': len(output.relationships_updated_or_created)})
            
        except RetryError as re_err:
             self.logger.error(f"KB processing failed after multiple retries for doc context '{document_id}'.", exception=re_err)
             output.errors.append(f"KB processing failed after retries: {str(re_err)}")
        except (KnowledgeGraphError, DatabaseError) as persist_err:
            self.logger.error(f"Persistence error during KB processing for doc context '{document_id}'.", exception=persist_err)
            output.errors.append(f"KB persistence error: {str(persist_err)}")
            self.stats['kg_ops_failed_count'] += 1
        except AgentExecutionError as ae_err: # Catch errors from this agent's logic
            self.logger.error(f"AgentExecutionError in KB processing for doc context '{document_id}'.", exception=ae_err)
            output.errors.append(f"Agent logic error: {str(ae_err)}")
        except Exception as e: # Catch-all for unexpected errors
            self.logger.error(f"Unexpected critical error during KB processing for doc context '{document_id}'.", exception=e, exc_info=True)
            output.errors.append(f"Unexpected critical error: {type(e).__name__} - {str(e)}")
        
        finally:
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
            # Clear local caches for this run to avoid staleness if agent is long-lived.
            # Or, implement a more sophisticated TTL cache.
            self._local_entity_cache_by_id.clear()
            self._local_entity_cache_by_name_type.clear()

        return output.to_dict()

    async def _resolve_and_persist_entities_batch(self, raw_entity_dicts: List[Dict[str, Any]],
                                               document_id: str) -> Tuple[List[ResolvedEntity], Dict[str, str]]:
        """Resolves a batch of raw entities, persists them, and returns canonical versions + mapping."""
        if not self.kg_manager: # Critical dependency for persistence
            self.logger.error("KnowledgeGraphManager not available. Cannot resolve/persist entities.")
            raise AgentInitializationError("KnowledgeGraphManager is required for entity persistence.")

        canonical_entities_list: List[ResolvedEntity] = []
        # Map raw entity identifier (e.g. its temp ID or a hash of its name/type) to canonical KG ID
        raw_input_to_canonical_id: Dict[str, str] = {}

        # Sort raw entities to process potentially more important ones first (e.g. higher confidence)
        raw_entity_dicts.sort(key=lambda x: float(x.get('confidence', x.get('confidence_score', 0.0))), reverse=True)

        for raw_entity_data in raw_entity_dicts:
            # Extract key fields from raw_entity_data (which is a dict from another agent's output)
            # This assumes keys like 'name', 'entity_type', 'attributes', 'confidence_score', 'entity_id' (temp)
            raw_name = str(raw_entity_data.get("name", raw_entity_data.get("source_text_snippet", ""))).strip()
            raw_type = str(raw_entity_data.get("entity_type", "UNKNOWN_TYPE")).upper()
            raw_confidence = float(raw_entity_data.get("confidence_score", raw_entity_data.get("confidence", 0.0)))
            raw_attributes = raw_entity_data.get("attributes", {})
            raw_entity_temp_id = str(raw_entity_data.get("entity_id", self._generate_temp_key(raw_name, raw_type))) # Use temp ID if provided

            if not raw_name or raw_confidence < self.min_confidence_for_new_entity_creation:
                self.logger.debug(f"Skipping raw entity: insufficient name or confidence.", parameters={'name': raw_name, 'conf': raw_confidence})
                continue

            # Attempt to find an existing entity in KG that matches this raw entity
            # This is the core resolution step.
            existing_resolved_entity = await self._find_or_create_canonical_entity(
                raw_name, raw_type, raw_attributes, raw_confidence, document_id
            )

            # `existing_resolved_entity` is the canonical representation from KG (new or updated)
            raw_input_to_canonical_id[raw_entity_temp_id] = existing_resolved_entity.entity_id

            # Check if this canonical entity (by ID) is already in our results list for this batch
            already_in_batch = next((e for e in canonical_entities_list if e.entity_id == existing_resolved_entity.entity_id), None)
            if already_in_batch:
                # If already processed in this batch, merge new info (e.g., another source_doc_id or alias)
                already_in_batch.add_alias(raw_name)
                already_in_batch.add_source_document(document_id)
                already_in_batch.merge_attributes(raw_attributes)
                already_in_batch.confidence_score = max(already_in_batch.confidence_score, raw_confidence) # Keep highest confidence
                # Persist these updates to the KG for the already_in_batch entity
                await self.kg_manager.add_entity(already_in_batch.to_kg_entity()) # add_entity should handle updates
                self.stats['entities_merged_in_kg'] +=1 # Count as merge/update
            else:
                canonical_entities_list.append(existing_resolved_entity)
                # New entity creation count is handled within _find_or_create_canonical_entity

        self.logger.info(f"Entity resolution batch complete for doc '{document_id}'. Processed {len(raw_entity_dicts)} raw, resulting in {len(canonical_entities_list)} canonical entities.")
        return canonical_entities_list, raw_input_to_canonical_id

    async def _find_or_create_canonical_entity(self, name: str, entity_type: str,
                                             attributes: Dict[str, Any], confidence: float,
                                             document_id: str) -> ResolvedEntity:
        """
        Finds an existing canonical entity in the KG or creates a new one.
        Returns the canonical ResolvedEntity object.
        This method encapsulates the core "fuzzy matching" and "create if not exists" logic.
        """
        if not self.kg_manager: raise AgentInitializationError("KGManager unavailable for _find_or_create_canonical_entity")

        # 1. Check local short-term cache first (for entities resolved earlier in *this same agent run*)
        cache_key_name_type = (entity_type.upper(), name.lower().strip())
        if cache_key_name_type in self._local_entity_cache_by_name_type:
            cached_entity = self._local_entity_cache_by_name_type[cache_key_name_type]
            # Update this cached entity with new info if current raw entity is a better match or adds info
            cached_entity.add_alias(name)
            cached_entity.add_source_document(document_id)
            cached_entity.merge_attributes(attributes)
            cached_entity.confidence_score = max(cached_entity.confidence_score, confidence)
            await self.kg_manager.add_entity(cached_entity.to_kg_entity()) # Persist update
            self.stats['entities_merged_in_kg'] +=1
            return cached_entity

        # 2. Query KG for potential matches (fuzzy search on name and attributes, filtered by type)
        # KGManager.find_entities should be robust enough to handle this.
        # Example: name_pattern could support wildcards or similarity options.
        potential_kg_matches_raw = await self.kg_manager.find_entities(
            entity_type_filter=entity_type,
            name_pattern=name, # KGManager might do fuzzy matching here
            properties_filter=attributes, # KGManager might use some attributes for filtering
            limit=5
        )

        best_match_kg_entity: Optional[KGEntity] = None
        highest_similarity_score = -1.0

        if potential_kg_matches_raw:
            for kg_candidate in potential_kg_matches_raw:
                # Calculate similarity (name + attributes)
                current_sim_score = self._calculate_entities_similarity(
                    {'name': name, 'type': entity_type, 'attributes': attributes},
                    {'name': kg_candidate.name, 'type': kg_candidate.entity_type, 'attributes': kg_candidate.attributes}
                )
                if current_sim_score > highest_similarity_score:
                    highest_similarity_score = current_sim_score
                    best_match_kg_entity = kg_candidate

        # 3. Decide to merge or create new
        if best_match_kg_entity and highest_similarity_score >= self.name_similarity_threshold: # Name similarity is primary
            # Found a strong match in KG: MERGE
            self.logger.debug(f"Strong match found in KG for '{name}' ({entity_type}). Merging.",
                             parameters={'kg_id': best_match_kg_entity.entity_id, 'similarity': highest_similarity_score})
            
            resolved_entity = ResolvedEntity.from_kg_entity(best_match_kg_entity)
            resolved_entity.add_alias(name)
            resolved_entity.add_source_document(document_id)
            resolved_entity.merge_attributes(attributes)
            resolved_entity.confidence_score = max(resolved_entity.confidence_score, confidence, highest_similarity_score) # Blend confidences
            resolved_entity.version += 1 # Assuming version is loaded by from_kg_entity or defaults to 1
            
            await self.kg_manager.add_entity(resolved_entity.to_kg_entity()) # Persist merged/updated entity
            self.stats['entities_merged_in_kg'] +=1
        else:
            # No strong match found: CREATE NEW canonical entity
            self.logger.debug(f"No strong KG match for '{name}' ({entity_type}). Creating new canonical entity.",
                             parameters={'highest_sim_found': highest_similarity_score})

            new_entity_id = f"CE_{entity_type.upper()}_{uuid.uuid4().hex[:10]}" # Canonical Entity ID
            resolved_entity = ResolvedEntity(
                entity_id=new_entity_id,
                canonical_name=name, # First instance becomes canonical name
                entity_type=entity_type,
                attributes=attributes,
                confidence_score=confidence,
                source_document_ids={document_id},
                metadata={'resolution_method': 'created_new_canonical'}
            )
            await self.kg_manager.add_entity(resolved_entity.to_kg_entity()) # Persist new entity
            self.stats['new_entities_persisted'] +=1
        
        # Update local short-term cache
        self._local_entity_cache_by_id[resolved_entity.entity_id] = resolved_entity
        self._local_entity_cache_by_name_type[(resolved_entity.entity_type.upper(), resolved_entity.canonical_name.lower())] = resolved_entity

        return resolved_entity

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculates similarity between two names (e.g., Jaccard, Levenshtein)."""
        # Simple Jaccard for now
        set1 = set(name1.lower().split())
        set2 = set(name2.lower().split())
        if not set1 or not set2: return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_entities_similarity(self, entity1_data: Dict, entity2_data: Dict) -> float:
        """Calculates similarity between two entity representations (name, type, attributes)."""
        # Name similarity (e.g., Jaro-Winkler, Levenshtein, or even embedding-based if available)
        name_sim = self._calculate_name_similarity(entity1_data['name'], entity2_data['name'])

        # Attribute similarity (e.g., Jaccard index on attribute keys and/or values)
        attrs1 = entity1_data.get('attributes', {})
        attrs2 = entity2_data.get('attributes', {})
        common_attrs = 0
        # A simple count of common key-value pairs for shared keys
        # This could be much more sophisticated
        shared_keys = set(attrs1.keys()).intersection(set(attrs2.keys()))
        for key in shared_keys:
            if str(attrs1[key]).lower() == str(attrs2[key]).lower(): # Simple value equality
                common_attrs +=1
        
        attr_sim = 0.0
        total_unique_keys = len(set(attrs1.keys()).union(set(attrs2.keys())))
        if total_unique_keys > 0:
            attr_sim = common_attrs / total_unique_keys

        # Weighted average: Name similarity is often more important
        # Type must match for this function to be called in the first place usually.
        return round(0.7 * name_sim + 0.3 * attr_sim, 3)

    async def _process_and_persist_relationships_batch(self, raw_relationship_dicts: List[Dict[str, Any]],
                                                   raw_to_canonical_id_map: Dict[str, str],
                                                   document_id: str) -> List[Dict[str, Any]]:
        """Processes raw relationships, maps to canonical entity IDs, and persists them."""
        if not self.kg_manager: return []
        
        processed_rels_summary: List[Dict[str, Any]] = []
        created_count = 0
        updated_count = 0

        for raw_rel_data in raw_relationship_dicts:
            # Extract fields from raw_rel_data (dict from e.g. OntologyExtractionAgent)
            raw_source_id = str(raw_rel_data.get('source_entity_id', raw_rel_data.get('source_entity','')))
            raw_target_id = str(raw_rel_data.get('target_entity_id', raw_rel_data.get('target_entity','')))
            rel_type_label = str(raw_rel_data.get('relationship_type', 'RELATED_TO')).upper()
            properties = raw_rel_data.get('properties', {})
            confidence = float(raw_rel_data.get('confidence', 0.0))
            evidence_text = raw_rel_data.get('evidence_text_snippet', raw_rel_data.get('source_text', ''))

            if confidence < self.min_confidence_for_relationship_creation:
                continue

            canonical_source_id = raw_to_canonical_id_map.get(raw_source_id)
            canonical_target_id = raw_to_canonical_id_map.get(raw_target_id)

            if not canonical_source_id or not canonical_target_id or canonical_source_id == canonical_target_id:
                self.logger.debug("Skipping relationship due to missing/invalid canonical IDs or self-loop.",
                                 parameters={'raw_src': raw_source_id, 'raw_tgt': raw_target_id, 'type': rel_type_label})
                continue

            # Add document_id and evidence_text to properties if not already there
            properties['source_document_id'] = document_id
            if 'evidence_text' not in properties and evidence_text:
                properties['evidence_text'] = evidence_text[:500] # Limit length

            kg_relationship = KGRelationship(
                source_id=canonical_source_id,
                target_id=canonical_target_id,
                rel_type=rel_type_label,
                properties=properties,
                confidence=confidence
            )

            try:
                # add_relationship should handle CREATE or MERGE (update properties if exists with lower confidence)
                # It should return a status or the persisted relationship ID.
                # For now, assuming it handles creation/update.
                persisted_rel_id_or_status = await self.kg_manager.add_relationship(kg_relationship)

                # This needs KGM to tell us if it was new or merged
                # Placeholder logic:
                if persisted_rel_id_or_status: # Assuming KGM returns ID or truthy on success
                    # Heuristic: if properties are identical to an existing one, it might be an update.
                    # This is hard to track without more info from KGM.
                    # For now, let's assume most are new unless KGM explicitly says MERGED.
                    # Let's assume KGM.add_relationship returns a tuple (id, created_boolean)
                    # if isinstance(persisted_rel_id_or_status, tuple) and len(persisted_rel_id_or_status) == 2:
                    #    rel_id, created = persisted_rel_id_or_status
                    #    if created: created_count +=1
                    #    else: updated_count +=1
                    # else: # Fallback
                    created_count += 1 # Assume new for now

                    processed_rels_summary.append({
                        'source_id': canonical_source_id, 'target_id': canonical_target_id,
                        'type': rel_type_label, 'status': 'persisted' # or 'updated'
                    })
            except KnowledgeGraphError as kge_rel:
                self.logger.error(f"Failed to persist relationship ({rel_type_label}) between {canonical_source_id} and {canonical_target_id}.",
                                 exception=kge_rel)
                self.stats['kg_ops_failed_count'] += 1

        self.stats['new_relationships_persisted'] += created_count
        self.stats['relationships_updated_in_kg'] += updated_count
        self.logger.info(f"Relationship processing batch complete for doc '{document_id}'. "
                         f"Persisted {created_count} new, updated {updated_count} relationships.")
        return processed_rels_summary

    def _calculate_task_summary_metrics(self, raw_entities: List[Dict[str,Any]],
                                     resolved_entities: List[ResolvedEntity],
                                     raw_relationships: List[Dict[str,Any]],
                                     processed_relationships_summary: List[Dict[str,Any]]) -> Dict[str, Any]:
        num_raw_e = len(raw_entities)
        num_resolved_e = len(resolved_entities)

        # Get counts from agent_metrics which are updated during the process
        new_e_this_run = self.stats.get('_temp_new_entities_this_run', 0) # Need to track this per run
        merged_e_this_run = self.stats.get('_temp_merged_entities_this_run', 0)

        # These would ideally be reset per _process_task or use the delta from agent_metrics
        # For simplicity, this uses the output list sizes.
        return {
            "input_raw_entities_count": num_raw_e,
            "output_canonical_entities_count": num_resolved_e,
            "input_raw_relationships_count": len(raw_relationships),
            "output_processed_relationships_count": len(processed_relationships_summary),
            # "new_entities_created_this_run": new_e_this_run, # More accurate counts from agent_metrics
            # "entities_merged_this_run": merged_e_this_run,
            "entity_resolution_ratio": (num_raw_e - num_resolved_e) / num_raw_e if num_raw_e > 0 else 0.0,
        }

    def _update_cumulative_agent_metrics(self, output: KnowledgeBaseOutput):
        # Cumulative metrics are already updated within _resolve_and_persist_entities_batch and _process_and_persist_relationships_batch
        # This method can be used to calculate averages or more complex aggregate metrics if needed at the end of a task.
        if output.resolved_entities_updated_or_created:
            current_total_ops = self.stats['new_entities_persisted'] + self.stats['entities_merged_in_kg']
            if current_total_ops > 0:
                new_avg_conf = sum(e.confidence_score for e in output.resolved_entities_updated_or_created) / len(output.resolved_entities_updated_or_created)
                # Update running average of entity resolution confidence
                # This logic needs to be careful not to double count if metrics are updated elsewhere
                # Let's assume self.stats['avg_entity_resolution_confidence'] stores the sum of confidences
                # and we calculate average when needed.
                # For simplicity, the current _update_agent_metrics_summary in BaseAgent or a custom one can handle this.
        pass # Metrics updated incrementally

    def _generate_temp_key(self, name: str, entity_type: str) -> str:
        """Generates a temporary key for a raw entity if it doesn't have an ID."""
        return hashlib.md5(f"{entity_type.upper()}:{name.lower()}".encode()).hexdigest()[:16]

    async def health_check(self) -> Dict[str, Any]:
        base_health = await super().health_check()
        kgm_healthy = False
        kgm_status_details = "KGManager not configured"
        if self.kg_manager:
            try:
                kgm_health = await self.kg_manager.health_check() # Assuming KGM has health_check
                kgm_healthy = kgm_health.get("status") == "healthy"
                kgm_status_details = kgm_health
            except Exception as e:
                kgm_status_details = f"Error checking KGM health: {str(e)}"
                self.logger.warning("Failed to get KGM health status.", exception=e)

        base_health.update({
            "knowledge_graph_manager_available": bool(self.kg_manager),
            "knowledge_graph_manager_healthy": kgm_healthy,
            "knowledge_graph_manager_details": kgm_status_details,
            "local_cache_entities_by_id_size": len(self._local_entity_cache_by_id),
            "local_cache_entities_by_name_type_size": len(self._local_entity_cache_by_name_type),
            "persistence_circuit_breaker_state": str(kb_persistence_breaker.current_state),
            "agent_operational_metrics": self.stats.copy(),
            "configuration_summary": {
                "name_similarity_threshold": self.name_similarity_threshold,
                "attribute_similarity_threshold": self.attribute_similarity_threshold,
                "min_confidence_for_new_entity": self.min_confidence_for_new_entity_creation,
            }
        })
        if not self.kg_manager or not kgm_healthy:
            base_health['status'] = 'degraded'
            base_health['reason'] = 'KnowledgeGraphManager is unavailable or unhealthy, critical for persistence.'
        
        self.logger.info(f"{self.name} health check.", parameters={'status': base_health.get('status')})
        return base_health