# legal_ai_system/agents/entity_extraction_streamlined/entity_extraction_agent.py
"""
Streamlined Entity Extraction Agent - Design.txt Compliant
Uses shared components, DDD principles, structured logging, and error resilience.
This agent provides a more modern approach to entity extraction.
"""

import asyncio
import hashlib
import json
import re
import uuid
import logging # For tenacity before_sleep_log
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple # Ensure List and Tuple are imported
from dataclasses import dataclass, field, asdict

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, before_sleep_log
from pybreaker import CircuitBreaker, CircuitBreakerError

# Core imports
from ....core.base_agent import BaseAgent, ProcessingResult
from ....core.shared_components import (
    DocumentChunker, measure_performance,
    LegalDocumentClassifier, ProcessingCache
)
from ....utils.error_recovery import ErrorRecovery
from ....core.llm_providers import LLMManager, LLMProviderError, LLMProviderEnum
from ....core.unified_exceptions import AgentExecutionError, AgentInitializationError
from ....core.detailed_logging import LogCategory, get_detailed_logger

# Ontology imports (handle potential ImportError gracefully)
try:
    from ....utils.ontology import (
        LegalEntityType as OntologyLegalEntityType, # Alias to avoid name clash if defined locally too
        get_entity_types_for_prompt, 
        validate_entity_attributes as ontology_validate_entity_attributes,
        ENTITY_TYPE_MAPPING, get_entity_type_by_label
    )
    ONTOLOGY_AVAILABLE = True
except ImportError as e:
    ont_logger = get_detailed_logger("OntologyImportCheck_SEE", LogCategory.CONFIGURATION)
    ont_logger.warning("Ontology module not fully available for StreamlinedEntityExtractionAgent. Using fallbacks.", error_details=str(e))
    ONTOLOGY_AVAILABLE = False
    class OntologyLegalEntityType: # type: ignore
        def __getattr__(self, name): return "UNKNOWN_ONTOLOGY_TYPE"
    def get_entity_types_for_prompt() -> str: return "- PERSON\n- ORGANIZATION\n- LOCATION\n- DATE\n- CASE_REFERENCE\n- STATUTE_REFERENCE"
    def ontology_validate_entity_attributes(entity_type_label: Any, attributes: Any) -> Tuple[bool, List[str]]: return True, []
    ENTITY_TYPE_MAPPING = {}
    def get_entity_type_by_label(label: str) -> Optional[Any]: return None


# Circuit breaker for external LLM service calls
entity_extraction_llm_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

# --- DATACLASS DEFINITIONS ---
# LegalEntity must be defined BEFORE StreamlinedEntityExtractionOutput uses it as a type hint.

@dataclass
class LegalEntity: # Domain-Driven Design Entity
    """Core business entity representing an extracted legal entity."""
    entity_id: str = field(default_factory=lambda: f"LE_{uuid.uuid4().hex[:12]}")
    entity_type: str # Should map to ontology.LegalEntityType.label
    name: str # Canonical name or primary text
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    source_text_snippet: str = "" # The exact text span or representative snippet
    document_id: Optional[str] = None
    span_in_document: Optional[Tuple[int, int]] = None # Character offsets in the original document
    metadata: Dict[str, Any] = field(default_factory=dict) # For additional info like chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StreamlinedEntityExtractionOutput: # Consistent output name
    """Results from entity extraction with validation metrics and DDD structure."""
    document_id: str
    entities: List[LegalEntity] = field(default_factory=list) # Now LegalEntity is defined
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    overall_confidence_score: float = 0.0
    processing_time_sec: float = 0.0
    model_used_for_llm: Optional[str] = None
    extraction_method: str = "streamlined_chunked_llm"
    document_type_classified: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['entities'] = [e.to_dict() for e in self.entities]
        return data


class StreamlinedEntityExtractionAgent(BaseAgent):
    # ... (rest of the StreamlinedEntityExtractionAgent class implementation as provided before) ...
    # Ensure all methods like __init__, _process_task, etc., are present here.
    
    def __init__(self, service_container: Any, **config: Any):
        super().__init__(service_container, name="StreamlinedEntityExtractionAgent", agent_type="extraction")
        
        self.config = config
        self.llm_manager: Optional[LLMManager] = self._get_service("llm_manager")
        if not self.llm_manager:
            self.logger.error("LLMManager service not available. StreamlinedEntityExtractionAgent functionality will be severely limited.")

        self.chunker = DocumentChunker(
            chunk_size=int(config.get('entity_extraction_chunk_size', 3500)),
            overlap=int(config.get('entity_extraction_chunk_overlap', 300))
        )
        self.classifier = LegalDocumentClassifier()
        self.cache = ProcessingCache(cache_dir_str=config.get('cache_dir_entity_extraction', './storage/cache/entity_extraction_streamlined'))
        self.error_recovery = ErrorRecovery(
            max_retries=int(config.get('llm_max_retries_entity_extraction', 2)),
            base_delay=float(config.get('llm_base_delay_sec_entity_extraction', 0.75))
        )
        
        self.min_entity_confidence = float(config.get('min_entity_confidence_threshold', 0.60))
        self.max_entities_per_chunk_limit = int(config.get('max_entities_per_chunk_from_llm', 25))
        self.enable_ontology_validation = bool(config.get('enable_ontology_validation_for_entities', True)) and ONTOLOGY_AVAILABLE
        
        self.entity_types_prompt_str = get_entity_types_for_prompt() if ONTOLOGY_AVAILABLE else "- PERSON\n- ORGANIZATION\n- LOCATION\n- DATE"
        
        self.agent_metrics.update({
            'total_extractions_run': 0, 'successful_extractions_count': 0, 
            'total_entities_extracted_final': 0, 'avg_extraction_confidence_final': 0.0,
            'cache_hits': 0, 'cache_misses': 0,
            'chunks_processed': 0, 'llm_calls_made': 0
        })
        
        self.logger.info(f"{self.name} initialized.", 
                       parameters={'ontology_available': ONTOLOGY_AVAILABLE, 
                                   'min_confidence': self.min_entity_confidence,
                                   'llm_manager': 'Available' if self.llm_manager else 'Unavailable'})
    
    @measure_performance("entity_extraction_task")
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10), 
        reraise=True,
        before_sleep=before_sleep_log(get_detailed_logger("EntityExtractionRetryHandler", LogCategory.AGENT_LIFECYCLE), logging.WARNING)
    )
    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        document_id = metadata.get("document_id", f"entity_doc_{uuid.uuid4().hex[:8]}")
        self.logger.info(f"Streamlined entity extraction task started for doc '{document_id}'.") # Uses
        start_time = datetime.now(timezone.utc)
        
        output = StreamlinedEntityExtractionOutput(document_id=document_id) # Uses document_id
        cached_data_loaded = False # Flag to see if we loaded from cache

        try:
            text_content = self._extract_text_from_task_data(task_data)
            if not text_content or len(text_content.strip()) < 20:
                output.errors.append("Insufficient text content for extraction.")
                raise AgentExecutionError("Insufficient text for entity extraction.", details={'doc_id': document_id, 'text_len': len(text_content or "")})

            output.document_type_classified = await self._classify_document_type(text_content, document_id)
            cache_key = self._generate_extraction_cache_key(text_content, output.document_type_classified)
            cached_dict_data = self.cache.get(document_id, cache_key) # Use document_id
            # Using Path(doc_id) for file_path argument to cache. If doc_id is not a path, this needs adjustment.
            # A more generic approach for cache path: self.cache.get(f"entities/{doc_id}", cache_key)
            # Assuming cache can handle non-Path first arg if doc_id is just an ID.
            # For now, let's use doc_id directly as a conceptual file identifier for cache.
                        
            if cached_dict_data and isinstance(cached_dict_data, dict):
                self.logger.info(f"Cache hit for entity extraction (doc '{document_id}').", parameters={'cache_key': cache_key})
                self.agent_metrics['cache_hits'] += 1
                cached_data_loaded = True
                # Reconstruct output from cached data
                output = StreamlinedEntityExtractionOutput(
                    document_id=cached_dict_data.get('document_id', document_id),
                    entities=[LegalEntity(**e_data) for e_data in cached_dict_data.get('entities', [])],
                    validation_summary=cached_dict_data.get('validation_summary', {}),
                    overall_confidence_score=float(cached_dict_data.get('overall_confidence_score', 0.0)),
                    processing_time_sec=float(cached_dict_data.get('processing_time_sec', 0.0)), # Original time
                    model_used_for_llm=cached_dict_data.get('model_used_for_llm'),
                    extraction_method=cached_dict_data.get('extraction_method', output.extraction_method),
                    document_type_classified=cached_dict_data.get('document_type_classified'),
                    errors=cached_dict_data.get('errors', []),
                    extracted_at=cached_dict_data.get('extracted_at', output.extracted_at)
                )
                # When loading from cache, we don't re-run the extraction.
                # The finally block will set the current processing_time_sec for *this run* of the agent,
                # but the output object will retain the original processing_time_sec from the cached run.
                # This is generally desired behavior.
            else:
                self.agent_metrics['cache_misses'] += 1
                if not self.llm_manager:
                    raise AgentInitializationError("LLMManager is not available, cannot proceed with LLM-based entity extraction.")

                try:
                    extracted_raw_entities_from_llm = await entity_extraction_llm_breaker.call_async(
                        self._extract_entities_via_llm_chunks, text_content, metadata, output.document_type_classified
                    )
                except CircuitBreakerError as cbe:
                    self.logger.error(f"LLM Circuit Breaker OPEN for entity extraction on doc '{document_id}'.", exception=cbe)
                    raise AgentExecutionError("LLM service for entity extraction is currently unavailable (Circuit Breaker).", cause=cbe, details={'doc_id': document_id}) from cbe

                output.entities, output.validation_summary = self._process_and_validate_raw_entities(
                    extracted_raw_entities_from_llm, document_id, text_content
                )
                
                if output.entities:
                    output.overall_confidence_score = round(sum(e.confidence_score for e in output.entities) / len(output.entities), 3)
                
                # Assuming primary_config exists and is populated correctly
                output.model_used_for_llm = f"{self.llm_manager.primary_config.provider.value}/{self.llm_manager.primary_config.model}" if self.llm_manager and self.llm_manager.primary_config else "LLM_details_unavailable"
                
                # Set processing time for this actual run before caching
                output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
                self.cache.set(document_id, cache_key, output.to_dict())
            
            self.logger.info(f"Streamlined entity extraction completed for doc '{document_id}'.", 
                           parameters={'entities_extracted': len(output.entities), 'overall_conf': output.overall_confidence_score, 'from_cache': cached_data_loaded})
            
        except RetryError as re_err:
            self.logger.error(f"Entity extraction failed after multiple retries for doc '{document_id}'.", exception=re_err)
            output.errors.append(f"Processing failed after {re_err.last_attempt.attempt_number if re_err.last_attempt else 'N/A'} retries: {str(re_err)}")
        except AgentExecutionError as ae_err:
            self.logger.error(f"AgentExecutionError during entity extraction for doc '{document_id}'.", exception=ae_err)
            output.errors.append(f"Agent execution error: {str(ae_err)}")
        except Exception as e:
            self.logger.error(f"Unexpected critical error during entity extraction for doc '{document_id}'.", exception=e, exc_info=True)
            output.errors.append(f"Unexpected critical error: {type(e).__name__} - {str(e)}")
        
        finally:
            # If not loaded from cache, set the current run's processing time.
            # If loaded from cache, output.processing_time_sec already holds the original time.
            # We might want a different field for "time_taken_this_run_including_cache_lookup".
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
            self.cache.set(document_id, cache_key, output.to_dict()) # Use document_id
            self._update_agent_metrics(output)

        return output.to_dict()

    def _extract_text_from_task_data(self, task_data: Dict[str, Any]) -> str:
        text = str(task_data.get('text_content', task_data.get('document_content', task_data.get('text', ''))))
        if not text.strip():
            self.logger.warning("Empty text content received in task_data for entity extraction.")
        return text.strip()

    async def _classify_document_type(self, text_content: str, doc_id: str) -> Optional[str]:
        self.logger.debug(f"Classifying document type for doc '{doc_id}'.")
        try:
            loop = asyncio.get_event_loop()
            classification_result = await loop.run_in_executor(None, self.classifier.classify, text_content, f"doc_{doc_id}_entities")
            doc_type = classification_result.get("primary_type", "unknown_type")
            self.logger.info(f"Document '{doc_id}' classified as type '{doc_type}' for entity extraction context.", 
                             parameters={'is_legal': classification_result.get('is_legal_document', False)})
            return doc_type
        except Exception as e:
            self.logger.warning(f"Document classification failed for doc '{doc_id}' during entity extraction pipeline.", exception=e)
            return "classification_failed"

    def _generate_extraction_cache_key(self, text_content_sample: str, document_type: Optional[str]) -> str:
        sample_len = min(len(text_content_sample), 2048)
        # Use sha256 for better collision resistance, though md5 is often faster for cache keys.
        text_sample_hash = hashlib.sha256(text_content_sample[:sample_len].encode('utf-8', 'ignore')).hexdigest()[:16]
        return f"streamlined_entities_v2.1_{document_type or 'unknown_type'}_{text_sample_hash}"

    async def _extract_entities_via_llm_chunks(self, text_content: str, metadata: Dict[str, Any], 
                                             doc_type: Optional[str]) -> List[Dict[str, Any]]:
        if not self.llm_manager: return []

        chunks_with_offsets = self.chunker.chunk_text_with_offsets(text_content)
        doc_id_for_log = metadata.get('document_id', 'unknown_doc')
        self.logger.info(f"Processing text for doc '{doc_id_for_log}' in {len(chunks_with_offsets)} chunks for LLM entity extraction.")
        self.agent_metrics['chunks_processed'] += len(chunks_with_offsets)
        
        all_raw_entities: List[Dict[str, Any]] = []
        
        tasks = [self._extract_from_single_llm_chunk(chunk_text, chunk_start_offset, metadata, doc_type, i)
                 for i, (chunk_text, chunk_start_offset) in enumerate(chunks_with_offsets)]
        
        chunk_results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res_or_exc in enumerate(chunk_results_or_exceptions):
            if isinstance(res_or_exc, Exception):
                self.logger.warning(f"LLM extraction for chunk {i} (doc '{doc_id_for_log}') failed.", exception=res_or_exc)
                # Optionally add error to main output: metadata.setdefault('chunk_errors', []).append(str(res_or_exc))
            elif isinstance(res_or_exc, list):
                all_raw_entities.extend(res_or_exc)
        
        return all_raw_entities

    # In StreamlinedEntityExtractionAgent class:
async def _extract_from_single_llm_chunk(self, chunk_text: str, chunk_start_offset: int, 
                                       metadata: Dict[str, Any], doc_type: Optional[str], 
                                       chunk_idx: int) -> List[Dict[str,Any]]: # chunk_idx is already a parameter here
    if not self.llm_manager: return []
    doc_id_for_log = metadata.get("document_id", "unknown_doc")
    self.logger.debug(f"Extracting entities from chunk {chunk_idx} for doc '{doc_id_for_log}'. Offset: {chunk_start_offset}")
    self.agent_metrics['llm_calls_made'] +=1

    prompt = self._build_llm_extraction_prompt(chunk_text, doc_type)
    
    try:
        model_config = self.llm_manager.primary_config
        llm_response = await self.llm_manager.complete(
            prompt, model=model_config.model, provider=model_config.provider,
            model_params={'temperature': 0.05, 'max_tokens': 1024}
        )
        
        # Pass chunk_idx to the parser
        raw_entities_in_chunk = self._parse_llm_entity_response(
            llm_response.content, 
            chunk_text, 
            chunk_start_offset, 
            chunk_idx # <--- PASS chunk_idx HERE
        )
        self.logger.debug(f"LLM chunk {chunk_idx} for doc '{doc_id_for_log}' yielded {len(raw_entities_in_chunk)} raw entities.")
        return raw_entities_in_chunk
    except LLMProviderError as e:
        self.logger.error(f"LLM API call failed for chunk {chunk_idx} (doc '{doc_id_for_log}').", exception=e)
        raise AgentExecutionError(f"LLM failed for chunk {chunk_idx}", cause=e, details={'doc_id': doc_id_for_log, 'chunk': chunk_idx}) from e# Reraise to be caught by gather

    def _build_llm_extraction_prompt(self, text_chunk: str, doc_type: Optional[str]) -> str:
        # self.entity_types_prompt_str is a pre-formatted string of "- TYPE_LABEL"
        return f"""
        TASK: Extract legal entities from the provided TEXT EXCERPT. The document is broadly classified as a '{doc_type or 'general document'}'.
        Focus on precision and ensure the "entity_type" matches one from the provided list or a standard common type if it's a better fit.

        ENTITY TYPES TO EXTRACT (prioritize these; use common types like PERSON, ORGANIZATION, LOCATION, DATE if more suitable):
        {self.entity_types_prompt_str}
        
        For each distinct entity found, provide the following information in a JSON object:
        - "name": (string) The exact, concise text of the entity as it appears.
        - "entity_type": (string) The most specific type from the list above, or a standard common type (e.g., PERSON, ORGANIZATION, DATE, LOCATION, MONEY, PERCENTAGE, etc.).
        - "attributes": (object, optional) Any relevant attributes as key-value pairs (e.g., {{"role": "Judge"}} for a PERSON, {{"currency": "USD"}} for MONEY). If no specific attributes, use an empty object {{}}.
        - "confidence_score": (float, 0.0-1.0) Your confidence in this extraction (accuracy of name, type, and attributes).
        - "source_text": (string) The exact text span from the EXCERPT from which this entity was extracted. This should typically match "name" or be a slightly longer context if "name" is an abbreviation or part of a larger phrase representing the entity.
        - "span_in_chunk": (array of two integers) [start_character_offset, end_character_offset] of "source_text" *within THIS TEXT EXCERPT*.

        TEXT EXCERPT:
        ---
        {text_chunk}
        ---
        
        RETURN FORMAT: A valid JSON array of entity objects. If no entities are found in this excerpt, return an empty array [].
        Example of a valid JSON array response:
        [
          {{"name": "Justice Roberts", "entity_type": "PERSON", "attributes": {{"role": "Chief Justice"}}, "confidence_score": 0.98, "source_text": "Justice Roberts", "span_in_chunk": [15, 30]}},
          {{"name": "$1.2 million", "entity_type": "MONEY", "attributes": {{"currency":"USD", "value": 1200000}}, "confidence_score": 0.95, "source_text": "$1.2 million", "span_in_chunk": [100, 112]}},
          {{"name": "The Supreme Court", "entity_type": "ORGANIZATION", "attributes": {{"subtype":"Court"}}, "confidence_score": 0.90, "source_text": "The Supreme Court", "span_in_chunk": [200, 218]}}
        ]
        
        IMPORTANT: Ensure your entire response is ONLY the JSON array. Do not include any introductory text or explanations outside the JSON structure.
        JSON Output:
        """

   # In StreamlinedEntityExtractionAgent class:
def _parse_llm_entity_response(self, llm_response_str: str, source_chunk_text: str, 
                             chunk_start_offset: int, chunk_idx: int) -> List[Dict[str, Any]]: # <--- ADD chunk_idx HERE
    raw_entities_found_in_chunk: List[Dict[str,Any]] = []
    try:
        # ... (JSON parsing logic as before) ...
        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', llm_response_str, re.DOTALL | re.IGNORECASE)
        # ... (rest of the robust JSON parsing) ...
        json_payload_str = "" # Initialize
        if json_match:
            json_payload_str = json_match.group(1)
        else: 
            first_bracket = llm_response_str.find('[')
            last_bracket = llm_response_str.rfind(']')
            if first_bracket != -1 and last_bracket > first_bracket:
                json_payload_str = llm_response_str[first_bracket : last_bracket+1]
            else:
                self.logger.warning(f"No JSON array structure found in LLM entity response. Snippet: {llm_response_str[:200]}")
                return []
        
        parsed_json_list = json.loads(json_payload_str)
        if not isinstance(parsed_json_list, list):
            self.logger.warning(f"Parsed LLM entity response is not a list. Type: {type(parsed_json_list)}. Content: {str(parsed_json_list)[:200]}")
            return []

        for entity_data_dict in parsed_json_list:
            # ... (rest of the entity data extraction and validation) ...
            if not (isinstance(entity_data_dict, dict) and 'name' in entity_data_dict and 'entity_type' in entity_data_dict):
                self.logger.trace(f"Skipping malformed entity data from LLM: {str(entity_data_dict)[:100]}")
                continue
            
            name = str(entity_data_dict['name'])
            entity_type_from_llm = str(entity_data_dict['entity_type']).upper()
            source_text_from_llm = str(entity_data_dict.get('source_text', name))
            span_in_chunk_data = entity_data_dict.get("span_in_chunk")
            doc_span_start, doc_span_end = None, None

            if isinstance(span_in_chunk_data, list) and len(span_in_chunk_data) == 2:
                try:
                    s, e = int(span_in_chunk_data[0]), int(span_in_chunk_data[1])
                    if 0 <= s < e <= len(source_chunk_text):
                        doc_span_start = chunk_start_offset + s
                        doc_span_end = chunk_start_offset + e
                    else: self.logger.warning(f"Invalid span_in_chunk from LLM: [{s},{e}] for chunk len {len(source_chunk_text)}. Entity: '{name}'.")
                except ValueError:
                    self.logger.warning(f"Non-integer span_in_chunk from LLM: {span_in_chunk_data}. Entity: '{name}'.")
            
            if doc_span_start is None and source_text_from_llm:
                try:
                    search_text = source_text_from_llm[:100]
                    for match_in_chunk in re.finditer(re.escape(search_text), source_chunk_text):
                        s, e = match_in_chunk.span()
                        doc_span_start = chunk_start_offset + s
                        doc_span_end = chunk_start_offset + (s + len(source_text_from_llm))
                        break 
                except re.error:
                    self.logger.warning(f"Regex error finding source_text '{source_text_from_llm[:50]}' in chunk.")

            raw_entities_found_in_chunk.append({
                "name": name, "entity_type": entity_type_from_llm,
                "attributes": entity_data_dict.get('attributes', {}),
                "confidence_score": float(entity_data_dict.get('confidence_score', 0.55)),
                "source_text_snippet": source_text_from_llm,
                "span_in_document": (doc_span_start, doc_span_end) if doc_span_start is not None and doc_span_end is not None else None,
                "metadata": { # Now chunk_idx is defined in this scope
                    "llm_source_chunk_idx": chunk_idx, 
                    "llm_provided_span_in_chunk": span_in_chunk_data # Store what LLM gave for span in chunk
                }
            })
    except json.JSONDecodeError as e:
        self.logger.error(f"Failed to parse JSON from LLM entity response. JSON string was: '{json_payload_str if 'json_payload_str' in locals() else llm_response_str[:200]}'", exception=e)
    except Exception as e:
        self.logger.error(f"Unexpected error parsing LLM entity response item. Item: {entity_data_dict if 'entity_data_dict' in locals() else 'N/A'}", exception=e, exc_info=True)
    return raw_entities_found_in_chunk

    def _process_and_validate_raw_entities(self, raw_entities_from_llm: List[Dict[str,Any]], 
                                         doc_id: str, original_full_text: str) -> Tuple[List[LegalEntity], Dict[str,Any]]:
        final_entities_for_output: List[LegalEntity] = []
        raw_input_count = len(raw_entities_from_llm)
        
        # 1. Deduplication: More robustly handle entities that might be essentially the same
        # Key by (normalized type, normalized name, approximate document quarter for very rough location)
        unique_raw_entities_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        text_len_quarter = len(original_full_text) // 4 if original_full_text else 1000 # Avoid division by zero

        for raw_entity in sorted(raw_entities_from_llm, key=lambda x: x.get('confidence_score', 0.0), reverse=True):
            name_norm = re.sub(r'\s+', ' ', raw_entity.get('name','').lower().strip())
            type_norm = raw_entity.get('entity_type','UNKNOWN').upper()
            span_tuple = raw_entity.get('span_in_document')
            # Group by which quarter of the document the entity starts in for disambiguation
            # If no span, group all such entities together (e.g., in group 0 or -1)
            approx_location_group = (span_tuple[0] // text_len_quarter) if span_tuple and span_tuple[0] is not None else -1
            
            dedup_key = (type_norm, name_norm, approx_location_group)
            if dedup_key not in unique_raw_entities_map:
                unique_raw_entities_map[dedup_key] = raw_entity
            else: # Merge if current entity has more/better info
                existing = unique_raw_entities_map[dedup_key]
                if raw_entity.get('confidence_score', 0.0) > existing.get('confidence_score', 0.0):
                    unique_raw_entities_map[dedup_key] = raw_entity # Replace with higher confidence
                # Could add more sophisticated attribute merging here
        
        deduplicated_raw_list = list(unique_raw_entities_map.values())
        count_after_dedup = len(deduplicated_raw_list)
        count_discarded_confidence = 0
        count_discarded_ontology = 0
        count_passed_ontology_validation = 0

        # 2. Convert to LegalEntity, filter by confidence, and validate against ontology
        for raw_data_dict in deduplicated_raw_list:
            confidence = float(raw_data_dict.get('confidence_score', 0.0))
            if confidence < self.min_entity_confidence:
                count_discarded_confidence += 1
                continue
            
            entity_type_label_from_llm = raw_data_dict.get('entity_type', 'UNKNOWN_TYPE').upper()
            attributes_from_llm = raw_data_dict.get('attributes', {})
            final_entity_type = entity_type_label_from_llm # Default to LLM's type

            passes_ontology_check = True # Assume passes unless proven otherwise
            if self.enable_ontology_validation and ONTOLOGY_AVAILABLE:
                ontology_enum_type = get_entity_type_by_label(entity_type_label_from_llm) # Check if LLM type is in our ontology
                if ontology_enum_type: # If LLM type is known to ontology
                    final_entity_type = ontology_enum_type.label # Use canonical ontology label
                    is_valid_attrs, unknown_attrs = ontology_validate_entity_attributes(ontology_enum_type, attributes_from_llm)
                    if not is_valid_attrs:
                        self.logger.debug(f"Entity '{raw_data_dict.get('name')}' ({final_entity_type}) failed ontology attr validation.", 
                                         parameters={'unknown': unknown_attrs, 'doc_id': doc_id})
                        attributes_from_llm['_ontology_validation_failed_attrs'] = unknown_attrs
                        confidence *= 0.85 # Penalize confidence slightly
                        passes_ontology_check = False
                else: # LLM type is not in our defined ontology, treat as custom or potentially misclassified
                    self.logger.trace(f"Entity type '{entity_type_label_from_llm}' (from LLM for '{raw_data_dict.get('name')}') not in defined ontology. Keeping as is.")
                    attributes_from_llm['_ontology_type_is_custom'] = True
                    # No attribute validation possible against a non-existent ontology type.
            
            if not passes_ontology_check : count_discarded_ontology +=1
            else: count_passed_ontology_validation+=1
            # Policy: keep entities even if they fail ontology validation for now, but with penalized confidence / flags.

            span_val = raw_data_dict.get('span_in_document')
            source_text_val = raw_data_dict.get('source_text_snippet', raw_data_dict.get('name', ''))

            final_entities_for_output.append(LegalEntity(
                entity_type=final_entity_type, name=str(raw_data_dict.get('name', '')),
                attributes=attributes_from_llm, confidence_score=round(confidence, 3),
                source_text_snippet=source_text_val[:250].strip(), document_id=doc_id,
                span_in_document=span_val if isinstance(span_val, tuple) and len(span_val)==2 else None,
                metadata=raw_data_dict.get('metadata', {})
            ))

        summary = {
            "input_raw_entities_from_llm": raw_input_count,
            "after_initial_deduplication": count_after_dedup,
            "discarded_low_confidence": count_discarded_confidence,
            "attempted_ontology_validation": count_passed_ontology_validation + count_discarded_ontology, # If enabled
            "passed_ontology_validation": count_passed_ontology_validation, # If enabled
            "final_entities_count": len(final_entities_for_output)
        }
        self.logger.debug(f"Entity validation and processing summary for doc '{doc_id}'.", parameters=summary)
        return final_entities_for_output, summary

    def _update_agent_metrics(self, result_output: StreamlinedEntityExtractionOutput) -> None:
        self.agent_metrics['total_extractions_run'] += 1
        if not result_output.errors and result_output.entities: # Consider an extraction successful if it produces entities without critical errors
            self.agent_metrics['successful_extractions_count'] += 1
            self.agent_metrics['total_entities_extracted_final'] += len(result_output.entities)
            
            # Cumulative moving average for confidence
            n = self.agent_metrics['successful_extractions_count']
            current_avg_conf = self.agent_metrics['avg_extraction_confidence_final']
            self.agent_metrics['avg_extraction_confidence_final'] = \
                round(((current_avg_conf * (n - 1)) + result_output.overall_confidence_score) / n if n > 0 else result_output.overall_confidence_score, 3)
        
        self.logger.trace(f"StreamlinedEntityExtractionAgent metrics updated for doc '{result_output.document_id}'.", 
                         parameters=self.agent_metrics) # Log subset of metrics if too verbose
    
    async def health_check(self) -> Dict[str, Any]:
        base_health = await super().health_check() if hasattr(super(), 'health_check') else {"status": "healthy", "checks": []}
        base_health['agent_name'] = self.name
        base_health.update({
            "ontology_module_status": "Available" if ONTOLOGY_AVAILABLE else "Unavailable_Using_Fallbacks",
            "llm_manager_status": 'Available' if self.llm_manager else 'Unavailable_Critical',
            "llm_circuit_breaker_state": str(entity_extraction_llm_breaker.current_state),
            "cache_info": self.cache.get_stats() if hasattr(self.cache, 'get_stats') else "No stats method",
            "agent_operational_metrics": self.agent_metrics.copy(),
            "configuration_summary": {
                "min_entity_confidence": self.min_entity_confidence,
                "chunk_size_chars": self.chunker.chunk_size,
                "chunk_overlap_chars": self.chunker.overlap,
                "ontology_validation_enabled": self.enable_ontology_validation,
            }
        })
        if not self.llm_manager: # LLM is critical for this agent's design
            base_health['status'] = 'degraded'
            base_health['reason'] = 'LLMManager is unavailable, which is critical for entity extraction.'
        
        self.logger.info(f"{self.name} health check status: {base_health.get('status')}")
        return base_health