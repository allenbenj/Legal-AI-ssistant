# legal_ai_system/agents/ontology_extraction/ontology_extraction_agent.py
"""
Ontology-driven legal entity and relationship extraction agent.
Enhanced with externalized patterns, advanced deduplication, and coreference resolution.
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy
import yaml
from sklearn.metrics.pairwise import (
    cosine_similarity,  # Optional, if scikit-learn is available
)
from spacy.language import Language
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Span, SpanGroup

from ..core.agent_unified_config import create_agent_memory_mixin
from ..core.base_agent import BaseAgent
from ..core.models import LegalDocument, ProcessingResult
from ..core.unified_exceptions import AgentError, ConfigurationError

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()
from ..utils.ontology import (
    get_entity_type_by_label,
    get_extraction_prompt,
    get_relationship_type_by_label,
)


class DSU:
    """Lightweight Disjoint Set Union for clustering duplicates."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


@dataclass
class ExtractedEntity:
    entity_id: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_text_snippet: str = ""
    span: Tuple[int, int] = (0, 0)
    canonical_entity_id: Optional[str] = None
    is_canonical_mention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedRelationship:
    relationship_id: str
    relationship_type: str
    source_entity_id: str  # This will be the canonical_entity_id of the source
    target_entity_id: str  # This will be the canonical_entity_id of the target
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_text_snippet: str = ""
    span: Tuple[int, int] = (0, 0)  # Span of the text evidencing the relationship

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyExtractionOutput:
    document_id: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_sec: float = 0.0
    overall_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        # Ensure nested dataclasses are also converted if they have custom to_dict for some reason
        # asdict usually handles this well for simple dataclasses.
        data = asdict(self)
        data["entities"] = [e.to_dict() for e in self.entities]
        data["relationships"] = [r.to_dict() for r in self.relationships]
        return data


class OntologyExtractionAgent(BaseAgent, MemoryMixin):

    def __init__(self, services: Any, **config: Any):
        super().__init__(services, **config)

        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(
            f"OntologyExtractionAgent configured with model: {self.llm_config.get('llm_model', 'default')}"
        )
        self.version = config.get("agent_version", "1.0.0")
        self.name = "OntologyExtractionAgent"
        self.description = "Performs ontology-driven legal entity and relationship extraction with advanced features."

        self.config = config
        self.llm_manager = self.get_llm_manager()
        self.embedding_manager = self.get_embedding_manager()

        # Configuration values
        self._load_agent_config()

        # Load spaCy NER and Legal-BERT models if enabled
        self.nlp_ner: Optional[Language] = None
        self.legal_bert_pipeline = None
        self._initialize_ner_models()

        # Load spaCy model for coreference resolution
        self.nlp_coref: Optional[Language] = None
        if self.enable_coreference_resolution:
            self._initialize_coref_model()

        # Load regex patterns from external files
        self.entity_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.relationship_patterns: Dict[str, List[Dict[str, Any]]] = {}
        if self.enable_regex_extraction:
            self.entity_patterns = self._load_patterns_from_file(
                self.entity_patterns_path, "entity"
            )
            self.relationship_patterns = self._load_patterns_from_file(
                self.relationship_patterns_path, "relationship"
            )

        self.logger.info(
            f"{self.name} initialized. Config: {self.get_agent_config_summary()}"
        )

    def _load_agent_config(self):
        """Loads various configuration parameters for the agent."""
        self.confidence_threshold_final = float(
            self.config.get("confidence_threshold_final", 0.6)
        )
        self.confidence_threshold_regex = float(
            self.config.get("confidence_threshold_regex", 0.4)
        )
        self.confidence_threshold_llm = float(
            self.config.get("confidence_threshold_llm", 0.7)
        )
        self.max_entities_per_type = int(self.config.get("max_entities_per_type", 30))
        self.max_text_chunk_size_for_llm = int(
            self.config.get("max_text_chunk_size_for_llm", 4000)
        )  # Characters for LLM input

        self.enable_regex_extraction = bool(
            self.config.get("enable_regex_extraction", True)
        )
        self.enable_llm_extraction = bool(
            self.config.get("enable_llm_extraction", True)
        )
        self.enable_spacy_ner = bool(self.config.get("enable_spacy_ner", False))
        self.spacy_ner_model = self.config.get("spacy_ner_model", "en_core_web_sm")
        self.enable_legal_bert = bool(self.config.get("enable_legal_bert", False))
        self.legal_bert_model_name = self.config.get(
            "legal_bert_model_name", "nlpaueb/legal-bert-base-uncased"
        )

        self.regex_confidence_weight = float(
            self.config.get("regex_confidence_weight", 0.6)
        )
        self.spacy_confidence_weight = float(
            self.config.get("spacy_confidence_weight", 0.8)
        )
        self.legal_bert_confidence_weight = float(
            self.config.get("legal_bert_confidence_weight", 0.9)
        )

        # Deduplication config
        self.enable_advanced_deduplication = bool(
            self.config.get("enable_advanced_deduplication", True)
        )
        self.deduplication_span_overlap_threshold = float(
            self.config.get("deduplication_span_overlap_threshold", 0.7)
        )
        self.deduplication_semantic_similarity_threshold = float(
            self.config.get("deduplication_semantic_similarity_threshold", 0.80)
        )
        if self.enable_advanced_deduplication and not self.embedding_manager:
            self.logger.warning(
                "EmbeddingManager service not found, but semantic deduplication is enabled. "
                "Semantic part of deduplication will be skipped."
            )

        # Coreference config
        self.enable_coreference_resolution = bool(
            self.config.get("enable_coreference_resolution", True)
        )
        self.coref_model_name = self.config.get(
            "coref_model_name", "en_coreference_web_trf"
        )

        # Pattern file paths
        # Assuming a 'config_base_path' might be in self.config or resolved globally
        base_path = Path(
            self.config.get("config_base_path", ".")
        )  # Default to current dir if not set
        self.entity_patterns_path = base_path / self.config.get(
            "entity_patterns_file", "config/ontology_extraction/entity_patterns.yaml"
        )
        self.relationship_patterns_path = base_path / self.config.get(
            "relationship_patterns_file",
            "config/ontology_extraction/relationship_patterns.yaml",
        )

    def get_agent_config_summary(self) -> Dict[str, Any]:
        return {
            "regex_extraction": self.enable_regex_extraction,
            "llm_extraction": self.enable_llm_extraction,
            "coref_resolution": self.enable_coreference_resolution,
            "advanced_deduplication": self.enable_advanced_deduplication,
            "final_confidence_threshold": self.confidence_threshold_final,
            "spacy_ner": self.enable_spacy_ner,
            "legal_bert": self.enable_legal_bert,
        }

    def _initialize_coref_model(self):
        """Loads and initializes the spaCy coreference model."""
        try:
            self.nlp_coref = spacy.load(
                self.coref_model_name,
                disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
            )
            # Verify coref component exists
            pipe_names = self.nlp_coref.pipe_names
            if not (
                "coref" in pipe_names
                or "experimental_coref" in pipe_names
                or "neuralcoref" in pipe_names
            ):  # Check for variations
                raise ConfigurationError(
                    f"Coreference component (e.g., 'coref', 'experimental_coref') not found in spaCy model '{self.coref_model_name}'. Model pipes: {pipe_names}"
                )
            self.logger.info(
                f"Coreference resolution model '{self.coref_model_name}' loaded successfully."
            )
        except OSError as e:
            self.logger.error(
                f"Could not load spaCy coreference model '{self.coref_model_name}'. "
                "Ensure it's downloaded (e.g., python -m spacy download en_coreference_web_trf). Disabling coreference.",
                exc_info=False,
            )
            self.logger.debug(
                "Detailed spaCy load error:", exc_info=True
            )  # Debug level for full trace
            self.enable_coreference_resolution = False
            self.nlp_coref = None
        except Exception as e:  # Catch other potential errors during model loading
            self.logger.error(
                f"Unexpected error loading coreference model '{self.coref_model_name}': {e}. Disabling coreference.",
                exc_info=True,
            )
            self.enable_coreference_resolution = False
            self.nlp_coref = None

    def _initialize_ner_models(self) -> None:
        """Initialize spaCy NER and Legal-BERT pipelines if enabled."""
        if self.enable_spacy_ner:
            try:
                self.nlp_ner = spacy.load(self.spacy_ner_model)
                self.logger.info(
                    f"spaCy NER model '{self.spacy_ner_model}' loaded successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load spaCy NER model '{self.spacy_ner_model}': {e}"
                )
                self.enable_spacy_ner = False

        if self.enable_legal_bert:
            try:
                from transformers import pipeline

                self.legal_bert_pipeline = pipeline(
                    "ner",
                    model=self.legal_bert_model_name,
                    aggregation_strategy="simple",
                )
                self.logger.info(
                    f"Legal-BERT model '{self.legal_bert_model_name}' loaded for NER."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load Legal-BERT model '{self.legal_bert_model_name}': {e}"
                )
                self.enable_legal_bert = False
                self.legal_bert_pipeline = None

    def _load_patterns_from_file(
        self, file_path: Path, pattern_type: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Loads regex patterns from a YAML file."""
        patterns: Dict[str, List[Dict[str, Any]]] = {}
        if not file_path.is_file():
            self.logger.warning(
                f"{pattern_type.capitalize()} patterns file not found at {file_path}. No regex patterns of this type will be used."
            )
            return patterns
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_patterns = yaml.safe_load(f)
            if not isinstance(loaded_patterns, dict):
                self.logger.error(
                    f"Invalid format in {pattern_type} patterns file {file_path}. Expected a root dictionary. Loaded: {type(loaded_patterns)}"
                )
                return patterns

            for type_label, pattern_config_list in loaded_patterns.items():
                if not isinstance(pattern_config_list, list):
                    self.logger.warning(
                        f"Patterns for '{type_label}' in {file_path} is not a list. Skipping."
                    )
                    continue
                valid_configs_for_type = []
                for p_config in pattern_config_list:
                    if isinstance(p_config, dict) and "pattern" in p_config:
                        valid_configs_for_type.append(p_config)
                    else:
                        self.logger.warning(
                            f"Invalid pattern config under '{type_label}' in {file_path}: {p_config}. Missing 'pattern' key or not a dict. Skipping this config."
                        )
                if valid_configs_for_type:
                    patterns[type_label] = valid_configs_for_type

            self.logger.info(
                f"Successfully loaded {sum(len(p_list) for p_list in patterns.values())} "
                f"{pattern_type} pattern configurations for {len(patterns)} types from {file_path}"
            )
        except yaml.YAMLError as e:
            self.logger.error(
                f"Error parsing YAML {pattern_type} patterns file {file_path}: {e}",
                exc_info=True,
            )
        except IOError as e:
            self.logger.error(
                f"Error reading {pattern_type} patterns file {file_path}: {e}",
                exc_info=True,
            )
        except Exception as e:  # Catch-all for other unexpected errors
            self.logger.error(
                f"Unexpected error loading {pattern_type} patterns from {file_path}: {e}",
                exc_info=True,
            )
        return patterns

    def _generate_unique_id(self, prefix: str) -> str:
        """Generates a unique ID with a given prefix."""
        return f"{prefix.upper()}_{uuid.uuid4().hex[:8]}"

    def _get_text_content(self, document: LegalDocument) -> str:
        """Extracts text content from the LegalDocument, preferring processed_content."""
        if hasattr(document, "processed_content") and isinstance(
            document.processed_content, dict
        ):
            text = document.processed_content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        if (
            hasattr(document, "content")
            and isinstance(document.content, str)
            and document.content.strip()
        ):
            return document.content.strip()
        self.logger.warning(f"Document {document.id} has no usable text content.")
        return ""

    def _calculate_context_relevance(
        self, text: str, span: Tuple[int, int], keywords: List[str]
    ) -> float:
        """Calculates relevance score based on keywords in a window around the span."""
        if not keywords:
            return 0.5
        window_size = 100
        context_start = max(0, span[0] - window_size)
        context_end = min(len(text), span[1] + window_size)
        context_text = text[context_start:context_end].lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in context_text)
        relevance = (matches / len(keywords)) * 0.6 + 0.4  # Base score of 0.4, max 1.0
        return min(1.0, round(relevance, 3))

    # --- Regex Extraction ---
    def _extract_entities_by_patterns(
        self, text: str, doc_id: str
    ) -> List[ExtractedEntity]:
        """Extracts entities using pre-loaded regex patterns."""
        extracted_entities: List[ExtractedEntity] = []
        if not self.entity_patterns:
            return extracted_entities

        for entity_type_label, patterns_for_type in self.entity_patterns.items():
            entity_type_enum = get_entity_type_by_label(entity_type_label)
            if not entity_type_enum:
                self.logger.warning(
                    f"Unknown entity type label '{entity_type_label}' from pattern config. Skipping."
                )
                continue

            for pattern_config in patterns_for_type:
                regex_pattern = pattern_config["pattern"]
                default_attribute_names = pattern_config.get("attributes", [])
                attributes_map = pattern_config.get("attributes_map", {})

                try:
                    for match in re.finditer(
                        regex_pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL
                    ):
                        context_score = self._calculate_context_relevance(
                            text,
                            match.span(),
                            pattern_config.get("context_keywords", []),
                        )
                        if context_score < self.confidence_threshold_regex:
                            continue

                        attributes = {}
                        if attributes_map:
                            for group_name, attr_name in attributes_map.items():
                                try:
                                    attributes[attr_name] = match.group(
                                        group_name
                                    ).strip()
                                except (
                                    IndexError
                                ):  # Should not happen with named groups if pattern is correct
                                    self.logger.warning(
                                        f"Named capture group '{group_name}' not found in match for pattern '{regex_pattern}' of type '{entity_type_label}'."
                                    )
                        elif default_attribute_names:
                            if match.groups():
                                for i, attr_name in enumerate(default_attribute_names):
                                    if i < len(match.groups()):
                                        attributes[attr_name] = match.group(
                                            i + 1
                                        ).strip()
                                    else:
                                        break
                            elif default_attribute_names:
                                attributes[default_attribute_names[0]] = match.group(
                                    0
                                ).strip()
                        else:
                            attributes["matched_text"] = match.group(0).strip()

                        if not attributes:  # Skip if no attributes were extracted
                            continue

                        entity_id = self._generate_unique_id(
                            entity_type_enum.value.label
                        )
                        extracted_entities.append(
                            ExtractedEntity(
                                entity_id=entity_id,
                                entity_type=entity_type_enum.value.label,
                                attributes=attributes,
                                confidence=context_score,
                                source_text_snippet=match.group(0)[
                                    :150
                                ].strip(),  # Increased snippet size
                                span=match.span(),
                            )
                        )
                except re.error as ree:
                    self.logger.error(
                        f"Regex error with pattern '{regex_pattern}' for type '{entity_type_label}': {ree}"
                    )

        self.logger.info(
            f"Regex entity extraction for doc {doc_id} yielded {len(extracted_entities)} entities."
        )
        return extracted_entities

    # --- Coreference Resolution ---
    async def _resolve_coreferences(
        self, text: str, doc_id: str
    ) -> Dict[Tuple[int, int], str]:
        """
        Performs coreference resolution. Returns map: mention_span -> canonical_cluster_id.
        """
        if not self.enable_coreference_resolution or not self.nlp_coref:
            return {}

        self.logger.debug(
            f"Starting coreference resolution for doc {doc_id} (text length: {len(text)} chars)"
        )
        try:
            # spaCy processing can be time-consuming for very long texts. Consider chunking if performance is an issue.
            spacy_doc: SpacyDoc = self.nlp_coref(text)
        except Exception as e:
            self.logger.error(
                f"Error during spaCy processing for coreference on doc {doc_id}: {e}",
                exc_info=True,
            )
            return {}

        mention_to_canonical_id: Dict[Tuple[int, int], str] = {}

        # Determine which key holds the coreference clusters based on spaCy version/model
        # For 'en_coreference_web_trf' (from spacy-experimental or spacy v3.x with compatible models)
        # it's typically in doc.spans with a key like 'coref' or 'experimental_coref'.
        # For older 'neuralcoref', it's doc._.coref_clusters.

        clusters_data = None
        potential_coref_keys = [
            "coref",
            "experimental_coref",
            "coref_clusters",
        ]  # Add others if known

        for key in potential_coref_keys:
            if key in spacy_doc.spans:
                clusters_data = spacy_doc.spans[
                    key
                ]  # This is a list of SpanGroup objects, each SpanGroup a cluster
                self.logger.debug(f"Using coref clusters from doc.spans['{key}']")
                break

        if clusters_data is None and hasattr(
            spacy_doc._, "coref_clusters"
        ):  # Fallback for neuralcoref
            clusters_data = (
                spacy_doc._.coref_clusters
            )  # This is a list of neuralcoref.Cluster objects
            self.logger.debug(
                "Using coref clusters from doc._.coref_clusters (neuralcoref)"
            )

        if clusters_data is None:
            self.logger.warning(
                f"No coreference cluster data found in spaCy doc for {doc_id} after checking common locations."
            )
            return {}

        num_clusters = 0
        num_mentions = 0
        for i, cluster_group_or_obj in enumerate(clusters_data):
            canonical_cluster_id = f"CORECL_{doc_id}_{i}"
            mentions_in_cluster: List[Span] = []

            if isinstance(cluster_group_or_obj, SpanGroup):  # New spaCy SpanGroup
                mentions_in_cluster = list(
                    cluster_group_or_obj
                )  # Iterate over the SpanGroup
            elif hasattr(
                cluster_group_or_obj, "mentions"
            ):  # neuralcoref.Cluster object
                mentions_in_cluster = cluster_group_or_obj.mentions
            elif isinstance(
                cluster_group_or_obj, list
            ):  # Some models might return List[List[Span]]
                mentions_in_cluster = cluster_group_or_obj
            else:
                self.logger.warning(
                    f"Unexpected coref cluster data type: {type(cluster_group_or_obj)}. Skipping cluster {i}."
                )
                continue

            if not mentions_in_cluster:
                continue
            num_clusters += 1

            # Log the main mention (often the first or longest)
            # main_mention_text = mentions_in_cluster[0].text
            # self.logger.trace(f"Coref Cluster {canonical_cluster_id} (Main: '{main_mention_text}'): "
            #                  f"{[m.text for m in mentions_in_cluster]}")

            for mention_span_obj in mentions_in_cluster:
                mention_span_tuple = (
                    mention_span_obj.start_char,
                    mention_span_obj.end_char,
                )
                mention_to_canonical_id[mention_span_tuple] = canonical_cluster_id
                num_mentions += 1

        self.logger.info(
            f"Coreference resolution for doc {doc_id} identified {num_clusters} clusters, {num_mentions} total mentions."
        )
        return mention_to_canonical_id

    async def _extract_with_ner(self, text: str, doc_id: str) -> List[ExtractedEntity]:
        """Combine regex, spaCy NER and Legal-BERT outputs."""
        combined: List[ExtractedEntity] = []

        if self.enable_regex_extraction:
            regex_entities = self._extract_entities_by_patterns(text, doc_id)
            for ent in regex_entities:
                ent.confidence = min(1.0, ent.confidence * self.regex_confidence_weight)
            combined.extend(regex_entities)

        if self.enable_spacy_ner and self.nlp_ner:
            try:
                doc = self.nlp_ner(text)
                for ent in doc.ents:
                    confidence = self.spacy_confidence_weight
                    combined.append(
                        ExtractedEntity(
                            entity_id=self._generate_unique_id(f"{ent.label_}_SPACY"),
                            entity_type=ent.label_,
                            attributes={"text": ent.text},
                            confidence=confidence,
                            source_text_snippet=ent.text[:150],
                            span=(ent.start_char, ent.end_char),
                        )
                    )
            except Exception as e:
                self.logger.error(f"spaCy NER failed on doc {doc_id}: {e}")

        if self.enable_legal_bert and self.legal_bert_pipeline:
            try:
                results = self.legal_bert_pipeline(text)
                for r in results:
                    snippet = text[r["start"] : r["end"]]
                    confidence = min(
                        1.0,
                        float(r.get("score", 0.0)) * self.legal_bert_confidence_weight,
                    )
                    combined.append(
                        ExtractedEntity(
                            entity_id=self._generate_unique_id(
                                f"{r['entity_group']}_BERT"
                            ),
                            entity_type=r["entity_group"],
                            attributes={"text": snippet},
                            confidence=confidence,
                            source_text_snippet=snippet[:150],
                            span=(int(r["start"]), int(r["end"])),
                        )
                    )
            except Exception as e:
                self.logger.error(f"Legal-BERT NER failed on doc {doc_id}: {e}")

        # Deduplicate by span and entity type, keep highest confidence
        unique_map: Dict[Tuple[int, int, str], ExtractedEntity] = {}
        for ent in combined:
            key = (ent.span[0], ent.span[1], ent.entity_type)
            existing = unique_map.get(key)
            if not existing or ent.confidence > existing.confidence:
                unique_map[key] = ent
        final_entities = list(unique_map.values())
        self.logger.info(
            f"NER extraction for doc {doc_id} yielded {len(final_entities)} entities"
        )
        return final_entities

    # --- LLM Extraction ---
    async def _extract_with_llm(
        self,
        text: str,
        doc_id: str,
        mention_to_canonical_id_map: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Dict[str, List[Any]]:
        """Uses LLM for comprehensive entity and relationship extraction."""
        if not self.llm_manager:
            self.logger.warning(
                f"LLM provider not available for doc {doc_id}. Skipping LLM extraction."
            )
            return {"entities": [], "relationships": []}

        self.logger.debug(
            f"Initiating LLM extraction for document {doc_id} (text length: {len(text)} chars)"
        )

        # Basic chunking if text is too long (a more sophisticated strategy is needed for production)
        chunks = [
            text[i : i + self.max_text_chunk_size_for_llm]
            for i in range(0, len(text), self.max_text_chunk_size_for_llm)
        ]
        self.logger.info(
            f"Text for doc {doc_id} split into {len(chunks)} chunk(s) for LLM processing."
        )

        all_llm_entities: List[ExtractedEntity] = []
        all_llm_relationships: List[ExtractedRelationship] = []

        base_prompt_template = (
            get_extraction_prompt()
        )  # Contains structure, entity/rel types

        for chunk_idx, text_chunk in enumerate(chunks):
            self.logger.debug(
                f"Processing LLM extraction for doc {doc_id}, chunk {chunk_idx+1}/{len(chunks)}"
            )

            # Construct prompt for the current chunk
            # The prompt from get_extraction_prompt() should already be very detailed.
            # We just append the text to analyze.
            full_prompt = f"{base_prompt_template}\n\nTEXT TO ANALYZE (Chunk {chunk_idx+1}/{len(chunks)}):\n```text\n{text_chunk}\n```\n\nJSON_OUTPUT_ONLY:"

            try:
                llm_response_str = await self.llm_manager.complete(
                    prompt=full_prompt,
                    model_params={
                        "temperature": 0.05,
                        "max_tokens": 3000,
                    },  # Adjust as needed
                )

                parsed_data = self._parse_llm_json_response(llm_response_str)
                if not parsed_data or not isinstance(parsed_data, dict):
                    self.logger.warning(
                        f"LLM response for {doc_id}, chunk {chunk_idx+1} was not a valid JSON object after parsing."
                    )
                    continue  # Skip this chunk's results

                chunk_offset = chunk_idx * self.max_text_chunk_size_for_llm

                for entity_data in parsed_data.get("entities", []):
                    entity_type_label = entity_data.get("entity_type")
                    entity_type_enum = get_entity_type_by_label(entity_type_label)
                    if not entity_type_enum:
                        self.logger.warning(
                            f"LLM returned unknown entity type '{entity_type_label}' (doc {doc_id}, chunk {chunk_idx+1}). Skipping."
                        )
                        continue

                    confidence = float(
                        entity_data.get(
                            "confidence_score", self.confidence_threshold_llm
                        )
                    )
                    if confidence < self.confidence_threshold_llm:
                        continue

                    # Adjust span relative to the original document
                    chunk_span = entity_data.get("span", (0, 0))
                    original_doc_span = (
                        chunk_span[0] + chunk_offset,
                        chunk_span[1] + chunk_offset,
                    )

                    entity_id = entity_data.get(
                        "entity_id",
                        self._generate_unique_id(f"{entity_type_enum.value.label}_LLM"),
                    )

                    # Check coreference map for this span
                    canonical_id_from_coref = None
                    if mention_to_canonical_id_map:
                        canonical_id_from_coref = mention_to_canonical_id_map.get(
                            original_doc_span
                        )

                    all_llm_entities.append(
                        ExtractedEntity(
                            entity_id=entity_id,
                            entity_type=entity_type_enum.value.label,
                            attributes=entity_data.get("attributes", {}),
                            confidence=confidence,
                            source_text_snippet=entity_data.get(
                                "source_text_snippet", ""
                            )[:150].strip(),
                            span=original_doc_span,
                            canonical_entity_id=canonical_id_from_coref,  # Assign if found
                        )
                    )

                for rel_data in parsed_data.get("relationships", []):
                    rel_type_label = rel_data.get("relationship_type")
                    rel_type_enum = get_relationship_type_by_label(rel_type_label)
                    if not rel_type_enum:
                        self.logger.warning(
                            f"LLM returned unknown relationship type '{rel_type_label}' (doc {doc_id}, chunk {chunk_idx+1}). Skipping."
                        )
                        continue

                    confidence = float(
                        rel_data.get("confidence_score", self.confidence_threshold_llm)
                    )
                    if confidence < self.confidence_threshold_llm:
                        continue

                    # Adjust span relative to the original document
                    chunk_rel_span = rel_data.get("span", (0, 0))
                    original_doc_rel_span = (
                        chunk_rel_span[0] + chunk_offset,
                        chunk_rel_span[1] + chunk_offset,
                    )

                    # LLM provides source/target IDs. These might be chunk-local.
                    # We'll need to resolve them to global canonical IDs later.
                    all_llm_relationships.append(
                        ExtractedRelationship(
                            relationship_id=rel_data.get(
                                "relationship_id",
                                self._generate_unique_id(
                                    f"{rel_type_enum.value.label}_LLM"
                                ),
                            ),
                            relationship_type=rel_type_enum.value.label,
                            source_entity_id=rel_data.get(
                                "source_entity_id", ""
                            ),  # Store as is, resolve later
                            target_entity_id=rel_data.get(
                                "target_entity_id", ""
                            ),  # Store as is, resolve later
                            properties=rel_data.get("properties", {}),
                            confidence=confidence,
                            evidence_text_snippet=rel_data.get(
                                "evidence_text_snippet", ""
                            )[:150].strip(),
                            span=original_doc_rel_span,
                        )
                    )
            except Exception as e:
                self.logger.error(
                    f"LLM extraction for doc {doc_id}, chunk {chunk_idx+1} failed: {e}",
                    exc_info=True,
                )
                # Optionally, re-raise as AgentProcessingError or continue with next chunk

        self.logger.info(
            f"LLM extraction for doc {doc_id} (all chunks) yielded {len(all_llm_entities)} entities and {len(all_llm_relationships)} relationships."
        )
        return {"entities": all_llm_entities, "relationships": all_llm_relationships}

    def _parse_llm_json_response(self, response_str: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse JSON from LLM response, handling markdown code blocks and leading/trailing text."""
        try:
            # Find the JSON part: from the first '{' or '[' to the last '}' or ']'
            # This is a common heuristic for LLM JSON outputs that might have surrounding text.
            json_match = None
            # Try to find ```json ... ``` block first
            code_block_match = re.search(
                r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```",
                response_str,
                re.DOTALL | re.IGNORECASE,
            )
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Fallback: find first opening brace/bracket and last closing brace/bracket
                start_char_options = ["{", "["]
                end_char_map = {"{": "}", "[": "]"}

                first_idx = -1
                last_idx = -1
                start_char_found = ""

                for sc in start_char_options:
                    idx = response_str.find(sc)
                    if idx != -1:
                        if first_idx == -1 or idx < first_idx:
                            first_idx = idx
                            start_char_found = sc

                if first_idx != -1:
                    end_char_needed = end_char_map[start_char_found]
                    # Find the corresponding last closing character, considering nesting.
                    # This is a simplified approach; a proper parser for balanced braces/brackets is more robust.
                    # For now, just finding the last occurrence.
                    open_count = 0
                    temp_last_idx = -1
                    for i in range(first_idx, len(response_str)):
                        if response_str[i] == start_char_found:
                            open_count += 1
                        elif response_str[i] == end_char_needed:
                            open_count -= 1
                            if (
                                open_count == 0
                            ):  # Found the matching end for the first opening char
                                temp_last_idx = i
                                # We want the *outermost* structure, so continue if more content
                                # This logic is tricky. For now, take the segment from first open to its corresponding close.
                                # Or, more simply, find last occurrence of the needed end_char *after* first_idx.
                                last_idx = temp_last_idx  # Tentative
                                # To get the *outermost* structure, we might need to iterate multiple times or use rfind on a substring.
                                # Let's stick to a simpler "find first '{' and last '}'" type of heuristic if code block fails.

                    # Simpler fallback: find first '{' and last '}'
                    if start_char_found == "{":
                        last_idx = response_str.rfind("}")
                    elif start_char_found == "[":
                        last_idx = response_str.rfind("]")

                    if last_idx > first_idx:
                        json_str = response_str[first_idx : last_idx + 1]
                    else:  # Could not determine a valid JSON substring
                        self.logger.warning(
                            f"Could not isolate JSON substring reliably. Full response: {response_str[:200]}..."
                        )
                        return None
                else:  # No opening brace/bracket found
                    self.logger.warning(
                        f"No JSON start character found in LLM response: {response_str[:200]}..."
                    )
                    return None

            return json.loads(json_str)
        except json.JSONDecodeError as jde:
            self.logger.warning(
                f"Failed to decode JSON from LLM response. Error: {jde}. Response snippet: {json_str[:200] if 'json_str' in locals() else response_str[:200]}"
            )
            return None
        except Exception as e:  # Catch any other error during parsing
            self.logger.error(
                f"Unexpected error parsing LLM JSON response: {e}", exc_info=True
            )
            return None

    # --- Deduplication ---
    def _get_jaccard_index(
        self, span1: Tuple[int, int], span2: Tuple[int, int]
    ) -> float:
        """Calculates Jaccard index for two spans (char-based)."""
        start1, end1 = span1
        start2, end2 = span2
        set1 = set(range(start1, end1))
        set2 = set(range(start2, end2))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _merge_entities(
        self, primary: ExtractedEntity, secondary: ExtractedEntity
    ) -> ExtractedEntity:
        """Merges attributes of secondary into primary, preferring primary's existing values. Updates confidence."""
        merged_attributes = primary.attributes.copy()
        for attr, value in secondary.attributes.items():
            if value and (attr not in merged_attributes or not merged_attributes[attr]):
                merged_attributes[attr] = value

        primary.attributes = merged_attributes
        primary.confidence = max(
            primary.confidence, secondary.confidence
        )  # Take max confidence
        # Snippet and span could also be merged (e.g., longest snippet, union of spans)
        if len(secondary.source_text_snippet) > len(primary.source_text_snippet):
            primary.source_text_snippet = secondary.source_text_snippet
        # Span merging: primary.span = (min(primary.span[0], secondary.span[0]), max(primary.span[1], secondary.span[1]))
        return primary  # Primary entity is modified in place and returned

    async def _post_process_entities_advanced_dedup(
        self, entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Advanced deduplication: span overlap then semantic similarity."""
        if not entities or not self.enable_advanced_deduplication:
            return entities

        self.logger.debug(
            f"Starting advanced deduplication for {len(entities)} entities."
        )

        # Group by entity type for type-specific deduplication
        entities_by_type: Dict[str, List[ExtractedEntity]] = {}
        for e in entities:
            entities_by_type.setdefault(e.entity_type, []).append(e)

        final_deduplicated_entities: List[ExtractedEntity] = []

        for entity_type, type_specific_entities in entities_by_type.items():
            if len(type_specific_entities) <= 1:
                final_deduplicated_entities.extend(type_specific_entities)
                continue

            # Sort by confidence (desc) to prioritize more confident entities during merging
            # And by entity_id (asc) for deterministic behavior on ties
            sorted_entities = sorted(
                type_specific_entities, key=lambda e: (-e.confidence, e.entity_id)
            )

            dsu = DSU(len(sorted_entities))

            # --- Span Overlap Union ---
            for i in range(len(sorted_entities)):
                span_i = sorted_entities[i].span
                if span_i == (0, 0):
                    continue
                for j in range(i + 1, len(sorted_entities)):
                    span_j = sorted_entities[j].span
                    if span_j == (0, 0):
                        continue
                    if (
                        self._get_jaccard_index(span_i, span_j)
                        >= self.deduplication_span_overlap_threshold
                    ):
                        dsu.union(i, j)

            # --- Semantic Similarity Union ---
            if self.embedding_manager and len(sorted_entities) > 1:
                texts_to_embed = [e.source_text_snippet for e in sorted_entities]
                try:
                    embeddings_list = await self.embedding_manager.embed_texts(
                        texts_to_embed
                    )
                    if embeddings_list and len(embeddings_list) == len(texts_to_embed):
                        embeddings = np.array(embeddings_list)
                        similarity_matrix = cosine_similarity(embeddings)
                        for i in range(len(sorted_entities)):
                            for j in range(i + 1, len(sorted_entities)):
                                if (
                                    similarity_matrix[i, j]
                                    >= self.deduplication_semantic_similarity_threshold
                                ):
                                    dsu.union(i, j)
                        self.logger.debug(
                            f"Type '{entity_type}': semantic similarity matrix processed"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to get valid embeddings for semantic deduplication (Type: {entity_type}). Skipping semantic step."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error during semantic deduplication (Type: {entity_type}): {e}",
                        exc_info=True,
                    )

            clusters: Dict[int, List[int]] = {}
            for idx in range(len(sorted_entities)):
                root = dsu.find(idx)
                clusters.setdefault(root, []).append(idx)

            self.logger.debug(
                f"Type '{entity_type}': formed {len(clusters)} clusters from {len(sorted_entities)} entities."
            )

            for indices in clusters.values():
                indices.sort(
                    key=lambda i: (
                        -sorted_entities[i].confidence,
                        sorted_entities[i].entity_id,
                    )
                )
                canonical = sorted_entities[indices[0]]
                for idx in indices[1:]:
                    canonical = self._merge_entities(canonical, sorted_entities[idx])
                final_deduplicated_entities.append(canonical)

        self.logger.info(
            f"Advanced deduplication complete. Total entities: {len(final_deduplicated_entities)}."
        )
        return final_deduplicated_entities

    # --- Relationship Processing ---
    def _validate_relationships(
        self,
        relationships: List[ExtractedRelationship],
        final_entities_map: Dict[str, ExtractedEntity],
    ) -> List[ExtractedRelationship]:
        """Validates relationships against the final list of canonical entities."""
        valid_relationships = []
        for rel in relationships:
            source_valid = rel.source_entity_id in final_entities_map
            target_valid = rel.target_entity_id in final_entities_map

            if (
                source_valid
                and target_valid
                and rel.source_entity_id != rel.target_entity_id
            ):
                # Further check: are source/target types compatible with relationship type? (Ontology rule)
                # This is advanced and requires ontology definition for allowed (source_type, rel_type, target_type)
                valid_relationships.append(rel)
            else:
                self.logger.debug(
                    f"Invalidating relationship {rel.relationship_id} ({rel.relationship_type}): "
                    f"Source '{rel.source_entity_id}' (Valid: {source_valid}) or "
                    f"Target '{rel.target_entity_id}' (Valid: {target_valid}) not found in final entities or self-loop."
                )
        return valid_relationships

    def _post_process_relationships(
        self, relationships: List[ExtractedRelationship]
    ) -> List[ExtractedRelationship]:
        """Post-processes relationships: filter by confidence, deduplicate."""
        if not relationships:
            return []

        confident_rels = [
            r for r in relationships if r.confidence >= self.confidence_threshold_final
        ]

        # Deduplicate (by type, source_id, target_id, and key properties for more uniqueness)
        unique_rels_dict: Dict[Tuple[str, str, str, str], ExtractedRelationship] = {}
        for rel in sorted(
            confident_rels, key=lambda r: (-r.confidence, r.relationship_id)
        ):
            # Create a more unique key, e.g., including a primary property if available
            primary_prop_key = ""
            if (
                rel.properties
            ):  # Example: use sorted string of key-values from properties
                primary_prop_key = "_".join(
                    sorted([f"{k}-{v}" for k, v in rel.properties.items()])
                )

            dedup_key = (
                rel.relationship_type,
                rel.source_entity_id,
                rel.target_entity_id,
                primary_prop_key,
            )
            if dedup_key not in unique_rels_dict:
                unique_rels_dict[dedup_key] = rel
            else:  # Merge properties if duplicate found (preferring higher confidence)
                existing_rel = unique_rels_dict[dedup_key]
                if (
                    rel.confidence > existing_rel.confidence
                ):  # Replace if new one is more confident
                    unique_rels_dict[dedup_key] = rel
                else:  # Merge properties into existing, higher-confidence rel
                    for p_key, p_val in rel.properties.items():
                        if p_key not in existing_rel.properties:
                            existing_rel.properties[p_key] = p_val

        final_rels = list(unique_rels_dict.values())
        self.logger.debug(
            f"Post-processing relationships: {len(relationships)} -> {len(confident_rels)} (confident) -> {len(final_rels)} (unique)"
        )
        return final_rels

    # --- Orchestration: `process` method ---
    async def process(
        self, document: LegalDocument, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        start_time_utc = datetime.now(timezone.utc)
        self.logger.info(f"Starting ontology extraction for document: {document.id}")

        output = OntologyExtractionOutput(document_id=document.id)
        output.extraction_metadata = {
            "agent_version": getattr(self, "version", "1.0.0-advanced"),
            "ontology_version": "1.0",  # Placeholder
            "extraction_methods_used": [],
            "timestamp_utc": start_time_utc.isoformat(),
        }

        try:
            text_content = self._get_text_content(document)
            if not text_content:
                output.extraction_metadata["error"] = "No text content provided"
                return ProcessingResult(
                    success=False,
                    error="No text content for extraction.",
                    data=output.to_dict(),
                )
            output.extraction_metadata["text_length_chars"] = len(text_content)

            # 1. Coreference Resolution
            mention_to_canonical_id_map: Dict[Tuple[int, int], str] = {}
            if self.enable_coreference_resolution:
                output.extraction_metadata["extraction_methods_used"].append(
                    "coreference_resolution"
                )
                mention_to_canonical_id_map = await self._resolve_coreferences(
                    text_content, document.id
                )
                output.extraction_metadata["coreference_clusters_found"] = len(
                    set(mention_to_canonical_id_map.values())
                )

            # 2. Initial Entity Extraction (NER + LLM)
            initial_entities: List[ExtractedEntity] = []
            initial_relationships: List[ExtractedRelationship] = (
                []
            )  # Primarily from LLM

            if (
                self.enable_regex_extraction
                or self.enable_spacy_ner
                or self.enable_legal_bert
            ):
                output.extraction_metadata["extraction_methods_used"].append(
                    "ner_extraction"
                )
                ner_entities = await self._extract_with_ner(text_content, document.id)
                initial_entities.extend(ner_entities)

            if self.enable_llm_extraction and self.llm_manager:
                output.extraction_metadata["extraction_methods_used"].append(
                    "llm_extraction"
                )
                # Pass coref map so LLM can potentially use it or be aware of it
                llm_extracted_data = await self._extract_with_llm(
                    text_content, document.id, mention_to_canonical_id_map
                )
                initial_entities.extend(llm_extracted_data.get("entities", []))
                initial_relationships.extend(
                    llm_extracted_data.get("relationships", [])
                )

            # 3. Consolidate Entities by Coreference
            # Assign canonical_entity_id based on coref map if not already assigned by LLM/Regex step.
            # Then, select one canonical ExtractedEntity object per coreference cluster.

            # First pass: assign canonical_entity_id to all initial entities based on their span
            for entity in initial_entities:
                if (
                    not entity.canonical_entity_id
                    and entity.span in mention_to_canonical_id_map
                ):
                    entity.canonical_entity_id = mention_to_canonical_id_map[
                        entity.span
                    ]
                elif (
                    not entity.canonical_entity_id
                ):  # Not in a coref cluster, it's its own canonical
                    entity.canonical_entity_id = entity.entity_id

            # Second pass: choose one canonical ExtractedEntity object per cluster_id
            # Group entities by their canonical_entity_id
            entities_grouped_by_canonical_id: Dict[str, List[ExtractedEntity]] = {}
            for entity in initial_entities:
                if entity.canonical_entity_id is None:
                    continue
                entities_grouped_by_canonical_id.setdefault(
                    entity.canonical_entity_id, []
                ).append(entity)

            consolidated_pre_dedup_entities: List[ExtractedEntity] = []
            for _canon_id, mention_list in entities_grouped_by_canonical_id.items():
                if not mention_list:
                    continue
                # Choose the "best" mention as the canonical representative (e.g., highest confidence, longest snippet)
                mention_list.sort(
                    key=lambda e: (
                        -e.confidence,
                        -len(e.source_text_snippet),
                        e.entity_id,
                    )
                )
                canonical_mention_obj = mention_list[0]
                canonical_mention_obj.is_canonical_mention = True  # Mark it

                # Merge attributes from other mentions in the same cluster into the canonical_mention_obj
                for other_mention in mention_list[1:]:
                    canonical_mention_obj = self._merge_entities(
                        canonical_mention_obj, other_mention
                    )  # Modifies canonical_mention_obj
                consolidated_pre_dedup_entities.append(canonical_mention_obj)

            output.extraction_metadata["initial_entity_mentions_count"] = len(
                initial_entities
            )
            output.extraction_metadata["consolidated_entities_before_dedup_count"] = (
                len(consolidated_pre_dedup_entities)
            )

            # 4. Advanced Deduplication on Consolidated Canonical Entities
            if self.enable_advanced_deduplication:
                output.extraction_metadata["extraction_methods_used"].append(
                    "advanced_entity_deduplication"
                )
            output.entities = await self._post_process_entities_advanced_dedup(
                consolidated_pre_dedup_entities
            )

            # Filter by confidence and max_entities_per_type AFTER deduplication
            # These are now part of _post_process_entities_advanced_dedup or a final filter step
            final_entities_after_limits: List[ExtractedEntity] = []
            type_counts: Dict[str, int] = {}
            for entity in sorted(
                output.entities, key=lambda e: (-e.confidence, e.entity_id)
            ):
                if entity.confidence < self.confidence_threshold_final:
                    continue
                count = type_counts.get(entity.entity_type, 0)
                if count < self.max_entities_per_type:
                    final_entities_after_limits.append(entity)
                    type_counts[entity.entity_type] = count + 1
            output.entities = final_entities_after_limits
            output.extraction_metadata["final_entities_count"] = len(output.entities)

            # 5. Update Relationships to use Final Canonical Entity IDs
            # Create a map of [original_entity_id or span-based_temp_id_from_llm] -> final_canonical_entity_id
            # This is complex because LLM relationship source/target IDs might be temporary or refer to text.
            # For this iteration, we assume LLM relationship extraction gives `source_entity_id` and `target_entity_id`
            # that match `entity_id` fields of entities also extracted by the LLM (or regex).

            # Build a map: original initial_entity.entity_id -> final canonical_entity_object.entity_id
            # This mapping should consider that `initial_entities` were grouped by `canonical_entity_id`,
            # and then `_post_process_entities_advanced_dedup` might have further merged them and changed IDs.
            # This requires careful ID tracking.

            # Simplified approach: Use the `canonical_entity_id` assigned to initial entities.
            # Then map these cluster `canonical_entity_id`s to the `entity_id` of the final chosen `ExtractedEntity`
            # object that represents that cluster (after advanced deduplication).

            # Map: cluster_canonical_id -> final_entity_object_representing_that_cluster
            final_entities_map_by_id: Dict[str, ExtractedEntity] = {
                e.entity_id: e for e in output.entities
            }  # These are the ultimate canonical entities

            # Map: any initial entity's original ID -> its cluster_canonical_id
            initial_id_to_cluster_id: Dict[str, str] = {
                e.entity_id: e.canonical_entity_id
                for e in initial_entities
                if e.canonical_entity_id
            }

            updated_relationships: List[ExtractedRelationship] = []
            for rel in initial_relationships:
                # Resolve original source/target IDs from LLM/Regex to their cluster_canonical_ids
                source_cluster_id = initial_id_to_cluster_id.get(rel.source_entity_id)
                target_cluster_id = initial_id_to_cluster_id.get(rel.target_entity_id)

                if not source_cluster_id or not target_cluster_id:
                    self.logger.debug(
                        f"Rel {rel.relationship_id}: could not map source/target original IDs to cluster IDs. Skipping."
                    )
                    continue

                # Now check if these cluster_ids are represented by a final entity object.
                # The `entity_id` of entities in `output.entities` *are* their cluster's canonical_id (or a deduped version).
                if (
                    source_cluster_id in final_entities_map_by_id
                    and target_cluster_id in final_entities_map_by_id
                ):
                    rel.source_entity_id = (
                        source_cluster_id  # Use the cluster ID as the link
                    )
                    rel.target_entity_id = target_cluster_id
                    updated_relationships.append(rel)
                else:
                    self.logger.debug(
                        f"Rel {rel.relationship_id}: source cluster '{source_cluster_id}' "
                        f"or target cluster '{target_cluster_id}' not found in final entities map. Skipping."
                    )

            output.relationships = self._post_process_relationships(
                self._validate_relationships(
                    updated_relationships, final_entities_map_by_id
                )
            )
            output.extraction_metadata["final_relationships_count"] = len(
                output.relationships
            )

            # 6. Final Calculations
            output.overall_confidence = self._calculate_overall_confidence(
                output.entities, output.relationships
            )
            output.processing_time_sec = round(
                (datetime.now(timezone.utc) - start_time_utc).total_seconds(), 3
            )

            self.logger.info(
                f"Ontology extraction completed for {document.id} in {output.processing_time_sec}s. "
                f"Final Entities: {len(output.entities)}, Final Relationships: {len(output.relationships)}."
            )
            return ProcessingResult(success=True, data=output.to_dict())

        except Exception as e:
            self.logger.error(
                f"Critical error during ontology extraction for document {document.id}",
                exception=e,
                exc_info=True,
            )
            output.processing_time_sec = round(
                (datetime.now(timezone.utc) - start_time_utc).total_seconds(), 3
            )
            output.extraction_metadata["error"] = (
                f"Critical workflow error: {type(e).__name__} - {str(e)}"
            )
            return ProcessingResult(success=False, error=str(e), data=output.to_dict())

    def _calculate_overall_confidence(
        self,
        entities: List[ExtractedEntity],
        relationships: List[ExtractedRelationship],
    ) -> float:
        """Calculates a simple average confidence score."""
        confidences = [e.confidence for e in entities] + [
            r.confidence for r in relationships
        ]
        return round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    async def health_check(self) -> Dict[str, Any]:
        status = (
            await super().health_check()
            if hasattr(super(), "health_check")
            else {"status": "healthy", "checks": []}
        )
        status["agent_name"] = self.name
        status["config_summary"] = self.get_agent_config_summary()
        status["dependencies_status"] = {
            "llm_manager": "available" if self.llm_manager else "unavailable",
            "embedding_manager": (
                "available" if self.embedding_manager else "unavailable"
            ),
            "coreference_model": (
                f"'{self.coref_model_name}' loaded"
                if self.nlp_coref
                else f"'{self.coref_model_name}' not loaded/disabled"
            ),
        }
        status["patterns_loaded"] = {
            "entity_pattern_files": str(self.entity_patterns_path),
            "entity_pattern_types_loaded": len(self.entity_patterns),
            "relationship_pattern_files": str(self.relationship_patterns_path),
            "relationship_pattern_types_loaded": len(self.relationship_patterns),
        }
        # Determine overall agent health
        if (
            (self.enable_llm_extraction and not self.llm_manager)
            or (self.enable_coreference_resolution and not self.nlp_coref)
            or (self.enable_advanced_deduplication and not self.embedding_manager)
        ):
            status["status"] = "degraded"
            status["reason"] = (
                "One or more enabled critical dependencies are unavailable."
            )

        return status

    async def _process_task(
        self, task_data: Any, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implementation of abstract method from :class:`BaseAgent`."""
        if isinstance(task_data, LegalDocument):
            result = await self.process(task_data, metadata)
            return asdict(result)
        raise AgentError("Invalid task data type for OntologyExtractionAgent")

    async def initialize(self):
        """Placeholder async initializer for interface compatibility."""
        self.logger.info("OntologyExtractionAgent initialize called")

    async def close(self):
        """Placeholder async close method."""
        self.logger.info("OntologyExtractionAgent close called")
