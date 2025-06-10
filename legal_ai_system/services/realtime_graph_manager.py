# legal_ai_system/knowledge/realtime_graph_manager.py

# RealTimeGraphManager - Orchestrates real-time updates to knowledge representations.

# This manager coordinates between the semantic Knowledge Graph (Neo4j/NetworkX)
# and the Vector Store, ensuring consistency and enabling real-time analysis
# capabilities.


from typing import Dict, List, Any, Optional, Callable
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone

# Use detailed_logging
from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)

# Import other necessary components
from .knowledge_graph_manager import (
    KnowledgeGraphManager,
    EntityType,
    RelationshipType,
)
from ..core.vector_store import VectorStore

rtgm_logger = get_detailed_logger("RealTimeGraphManager", LogCategory.KNOWLEDGE_GRAPH)


class RealTimeGraphManager:
    """
    Manages real-time interactions with both semantic graph and vector store.
    """

    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    def __init__(
        self,
        knowledge_graph_manager: KnowledgeGraphManager,
        vector_store: VectorStore,
        service_config: Optional[Dict[str, Any]] = None,
    ):
        rtgm_logger.info("Initializing RealTimeGraphManager.")
        self.config = service_config or {}
        self.kg_manager = knowledge_graph_manager
        self.vector_store = vector_store

        self.enable_hybrid_updates = self.config.get("enable_hybrid_updates", True)
        self.update_callbacks: List[Callable] = (
            []
        )  # For notifying other systems of graph changes

        rtgm_logger.info(
            "RealTimeGraphManager initialized.",
            parameters={"hybrid_updates_enabled": self.enable_hybrid_updates},
        )

    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def process_entity_realtime(
        self,
        entity_data: Any,  # Could be ExtractedEntity, ValidationResult, etc.
        document_id: str,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Processes an entity in real-time: adds to KG and Vector Store.
        Returns the entity_id if successful.
        """
        rtgm_logger.info(
            "Processing entity in real-time.",
            parameters={
                "doc_id": document_id,
                "entity_text_preview": str(
                    getattr(entity_data, "entity_text", entity_data)
                )[:50],
            },
        )
        source_metadata = source_metadata or {}
        entity_id = None

        try:
            # Extract relevant fields from the incoming entity data
            name = getattr(
                entity_data,
                "entity_text",
                getattr(entity_data, "text", str(entity_data)),
            )
            entity_type_str = getattr(
                entity_data,
                "consensus_type",
                getattr(entity_data, "entity_type", "Unknown"),
            )
            confidence = getattr(entity_data, "confidence", 0.8)
            attributes = getattr(entity_data, "attributes", {})

            if isinstance(attributes, list):  # Handle list of attribute dicts
                merged_attrs: Dict[str, Any] = {}
                for attr_dict in attributes:
                    merged_attrs.update(attr_dict)
                attributes = merged_attrs

            attributes["extraction_source"] = source_metadata.get(
                "extraction_method", "unknown_rtgm"
            )
            attributes["original_confidence"] = getattr(
                entity_data, "raw_confidence", confidence
            )

            try:
                entity_type = EntityType(entity_type_str.lower())
            except ValueError:
                entity_type = EntityType.CONCEPT

            # Create entity via KnowledgeGraphManager
            kg_entity = await self.kg_manager.create_entity(
                entity_type=entity_type,
                name=name,
                properties=attributes,
                confidence=float(confidence),
                source_document=document_id,
            )
            entity_id = kg_entity.id
            rtgm_logger.debug(
                "Entity processed by KnowledgeGraphManager.",
                parameters={"kg_entity_id": entity_id},
            )

            # 3. Add to VectorStore (if hybrid updates enabled)
            if self.enable_hybrid_updates and self.vector_store and entity_id:
                vector_content = f"{kg_entity.name} ({kg_entity.type.value})"
                if kg_entity.properties:
                    vector_content += (
                        " "
                        + json.dumps(kg_entity.properties, sort_keys=True, default=str)[
                            :200
                        ]
                    )

                await self.vector_store.add_vector_async(
                    vector_id_override=f"kg_entity_{entity_id}",
                    content_to_embed=vector_content,
                    document_id_ref=document_id,
                    index_target="entity",
                    confidence_score=kg_entity.confidence,
                    source_file=document_id,
                    custom_metadata={"kg_entity_id": entity_id, **attributes},
                )
                rtgm_logger.debug(
                    "Entity vector processed by VectorStore.",
                    parameters={"kg_entity_id": entity_id},
                )

            await self._notify_callbacks(
                "entity_processed", {"entity_id": entity_id, "document_id": document_id}
            )
            return entity_id

        except Exception as e:
            rtgm_logger.error(
                "Failed to process entity in real-time.",
                parameters={
                    "doc_id": document_id,
                    "entity_preview": str(entity_data)[:100],
                },
                exception=e,
            )
            # Raise a more specific error if needed, or handle gracefully
            # raise KnowledgeGraphError(f"Real-time entity processing failed for doc {document_id}", cause=e)
            return None

    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def process_relationship_realtime(
        self,
        relationship_data: Any,  # e.g., ontology_extraction.ExtractedRelationship
        document_id: str,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Processes a relationship in real-time: adds to KG.
        Returns the relationship_id if successful.
        """
        rtgm_logger.info(
            "Processing relationship in real-time.", parameters={"doc_id": document_id}
        )
        source_metadata = source_metadata or {}
        relationship_id = None

        try:
            # Extract relationship fields
            from_id = getattr(relationship_data, "source_entity", None)
            to_id = getattr(relationship_data, "target_entity", None)
            rel_type_str = getattr(relationship_data, "relationship_type", "RELATED_TO")
            confidence = getattr(relationship_data, "confidence", 0.8)
            properties = getattr(relationship_data, "properties", {})

            if not (from_id and to_id and rel_type_str):
                rtgm_logger.warning(
                    "Skipping relationship due to missing from_id, to_id, or type.",
                    parameters=asdict(relationship_data),
                )
                return None

            # Add source method to properties
            properties["extraction_source"] = source_metadata.get(
                "extraction_method", "unknown_rtgm"
            )

            try:
                rel_type = RelationshipType(rel_type_str.lower())
            except ValueError:
                rel_type = RelationshipType.RELATES_TO

            kg_relationship = await self.kg_manager.create_relationship(
                source_entity_id=str(from_id),
                target_entity_id=str(to_id),
                relationship_type=rel_type,
                properties=properties,
                confidence=float(confidence),
                source_document=document_id,
            )
            relationship_id = kg_relationship.id
            rtgm_logger.debug(
                "Relationship processed by KnowledgeGraphManager.",
                parameters={"kg_rel_id": relationship_id},
            )

            # Relationships are typically not directly vectorized unless they represent complex events.
            # If needed, a textual representation of the relationship could be added to the vector store.

            await self._notify_callbacks(
                "relationship_processed",
                {"relationship_id": relationship_id, "document_id": document_id},
            )
            return relationship_id

        except Exception as e:
            rtgm_logger.error(
                "Failed to process relationship in real-time.",
                parameters={
                    "doc_id": document_id,
                    "rel_preview": str(relationship_data)[:100],
                },
                exception=e,
            )
            # raise KnowledgeGraphError(f"Real-time relationship processing failed for doc {document_id}", cause=e)
            return None

    def register_update_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback function for graph updates."""
        self.update_callbacks.append(callback)
        rtgm_logger.info(f"Registered new update callback: {callback.__name__}")

    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks of an update."""
        rtgm_logger.debug(
            f"Notifying callbacks for event.",
            parameters={
                "event_type": event_type,
                "num_callbacks": len(self.update_callbacks),
            },
        )
        for callback in self.update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(
                        event_type, data
                    )  # Assuming sync callback is okay or wrapped by caller
            except Exception as e:
                rtgm_logger.error(
                    f"Error in update callback '{callback.__name__}'.",
                    parameters={"event_type": event_type},
                    exception=e,
                )

    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time statistics about the graph and vector store."""
        kg_stats = (
            self.kg_manager.get_statistics()
        )  # Assuming this is fast enough or KGM caches it
        vs_stats: Dict[str, Any] = {}
        total_vector_count = 0
        if self.vector_store:
            vs_stats = (
                await self.vector_store.get_service_status()
            )  # Assuming this returns relevant stats
            total_vector_count = vs_stats.get("total_vectors", 0)

        stats = {
            "knowledge_graph_entities": kg_stats.get("total_entities", 0),
            "knowledge_graph_relationships": kg_stats.get("total_relationships", 0),
            "vector_store_total_vectors": total_vector_count,  # Corrected
            "hybrid_updates_enabled": self.enable_hybrid_updates,
            "last_activity_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),  # Placeholder
        }
        rtgm_logger.info("Real-time graph stats retrieved.", parameters=stats)
        return stats

    async def initialize_service(self):  # For service container
        rtgm_logger.info("RealTimeGraphManager (async) initialize called.")
        # Initialization logic here if KGM or VS need async init not handled by their own init
        # For now, assume KGM and VS are initialized before RTGM
        return self

    async def get_service_status(self) -> Dict[str, Any]:  # For service container
        rtgm_logger.debug("Performing RealTimeGraphManager health check.")
        kgm_status = self.kg_manager.health_check()
        vs_status = (
            await self.vector_store.get_service_status()
            if self.vector_store
            else {"status": "unavailable"}
        )

        overall_healthy = kgm_status.get("healthy", False) and (
            vs_status.get("status") == "healthy" if self.vector_store else True
        )

        status_report = {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "knowledge_graph_manager": kgm_status,
                "vector_store": vs_status,
            },
            "hybrid_updates_enabled": self.enable_hybrid_updates,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rtgm_logger.info(
            "RealTimeGraphManager health check complete.", parameters=status_report
        )
        return status_report

    async def close(self):  # For service container
        rtgm_logger.info("Closing RealTimeGraphManager.")
        # KGM and VS should be closed by the service container if they are separate services
        # If RTGM owns them, close them here.
        # await self.kg_manager.close()
        # await self.vector_store.close()
        rtgm_logger.info("RealTimeGraphManager closed.")


# Factory for service container
def create_realtime_graph_manager(
    kg_manager: KnowledgeGraphManager,
    vector_store: VectorStore,
    service_config: Optional[Dict[str, Any]] = None,
) -> RealTimeGraphManager:
    return RealTimeGraphManager(kg_manager, vector_store, service_config=service_config)
