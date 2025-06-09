# legal_ai_system/agents/auto_tagging/auto_tagging_agent.py
"""
Auto Tagging Agent - Learning-based Document Classification and Tagging
This agent automatically tags and classifies legal documents.
"""

import re
import uuid
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from ..core.base_agent import BaseAgent
from ..core.llm_providers import LLMManager, LLMProviderError, LLMProviderEnum
from ..core.unified_exceptions import AgentProcessingError
from ..core.detailed_logging import get_detailed_logger, LogCategory
from ..core.agent_unified_config import create_agent_memory_mixin
from ..core.unified_memory_manager import UnifiedMemoryManager

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()

@dataclass
class AutoTaggingOutput:
    document_id: str
    document_type_classification: Optional[Dict[str, Any]] = None
    legal_domain_tags: List[str] = field(default_factory=list)
    procedural_stage_tag: Optional[Dict[str, Any]] = None
    importance_level_tag: Optional[Dict[str, Any]] = None
    extracted_entity_tags: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict) # e.g. {"court": [{"name": "Supreme Court", "confidence": 0.8}]}
    generated_subject_tags: List[str] = field(default_factory=list) # e.g. ["subject:due_process"]
    all_generated_tags: List[str] = field(default_factory=list) # Combined, formatted tags
    suggested_new_tags: List[str] = field(default_factory=list) # Tags not in existing_tags metadata
    overall_confidence: float = 0.0
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    tagged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used_for_llm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoTaggingAgent(BaseAgent, MemoryMixin):
    """Intelligent auto-tagging agent that learns from patterns and user feedback,
    persisting learning via UnifiedMemoryManager if available."""
    
    def _get_service(self, name: str) -> Any:
        """Safely retrieve a service from the container if available."""
        if not self.service_container:
            return None

        getter = getattr(self.service_container, "get_service", None)
        if getter:
            try:
                svc = getter(name)
                if asyncio.iscoroutine(svc):
                    try:
                        return asyncio.get_event_loop().run_until_complete(svc)
                    except RuntimeError:
                        return asyncio.run(svc)
                return svc
            except Exception:
                return None
        return getattr(self.service_container, name, None)

    def __init__(self, service_container: Any, **config: Any):
        super().__init__(service_container, name="AutoTaggingAgent")

        # Initialize logger for this agent
        self.logger: Any = get_detailed_logger(self.__class__.__name__, LogCategory.AGENT)
        
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(f"AutoTaggingAgentAgent configured with model: {self.llm_config.get('llm_model', 'default')}")
        self.llm_manager: Optional[LLMManager] = self._get_service("llm_manager")
        self.unified_memory_manager: Optional[UnifiedMemoryManager] = self._get_service("unified_memory_manager")

        self.config = config
        self.min_pattern_confidence = float(config.get('min_pattern_confidence', 0.45))
        self.min_llm_tag_confidence = float(config.get('min_llm_tag_confidence', 0.6))
        self.enable_llm_tag_generation = bool(config.get('enable_llm_tag_generation', True))
        self.max_text_for_llm = int(config.get('max_text_for_llm_autotag', 3000)) # Chars

        self._init_tagging_frameworks_from_config() # Load patterns from config or use defaults
        self.tag_accuracy_scores_cache: Dict[str, Dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "incorrect": 0.0, "suggested": 0.0, "last_updated_ts": 0.0})
        self.feedback_session_log: List[Dict[str,Any]] = [] # In-memory log for current agent instance

        self.logger.info(
            f"{self.name} initialized.",
            parameters={
                'llm_tag_gen': self.enable_llm_tag_generation,
                'llm_available': self.llm_manager is not None,
                'umm_available': self.unified_memory_manager is not None,
            },
        )

    async def initialize_agent_async(self): # Optional: called by a system if agent needs async setup
        """Asynchronously initializes components, e.g., loading learning data."""
        await self._load_learning_data_from_umm_async()

    def _init_tagging_frameworks_from_config(self):
        """Initialize tagging patterns from agent configuration or use defaults."""
        default_doc_type_patterns = {
            'motion': [r'motion\s+to\s+\w+', r'motion\s+for\s+(?:summary\s+)?judgment', r'notice\s+of\s+motion'],
            'brief': [r'brief\s+in\s+(?:support|opposition)', r'memorandum\s+of\s+law', r'amicus\s+curiae\s+brief'],
            'order': [r'court\s+order', r'it\s+is\s+(?:hereby\s+)?ordered', r'judgment\s+and\s+order', r'minute\s+order'],
            'complaint': [r'complaint\s+for\s+\w+', r'plaintiff\s+\w+\s+alleges', r'first\s+amended\s+complaint'],
            'affidavit': [r'affidavit\s+of\s+\w+', r'declaration\s+of\s+\w+'],
            'discovery': [r'interrogatories', r'request\s+for\s+production', r'notice\s+of\s+deposition'],
            'agreement': [r'settlement\s+agreement', r'contract', r'stipulation'],
            'transcript': [r'hearing\s+transcript', r'deposition\s+transcript', r'(?:official\s+)?reporter\'s\s+transcript']
        }
        default_legal_domain_patterns = {
            'criminal_law': [r'defendant\s+charged', r'criminal\s+case', r'prosecution', r'indictment', r'sentencing'],
            'civil_litigation': [r'civil\s+action', r'damages', r'liability', r'plaintiff', r'summons'],
            'corporate_law': [r'merger', r'acquisition', r'shareholder', r'sec\s+filing', r'corporate\s+governance'],
            'ip_law': [r'patent', r'trademark', r'copyright', r'infringement', r'intellectual\s+property'],
            'family_law': [r'divorce', r'custody', r'child\s+support', r'marital\s+agreement'],
            'bankruptcy': [r'chapter\s+\d+', r'bankruptcy', r'creditor', r'debtor', r'discharge']
        }
        # ... (add defaults for procedural_stage_patterns, importance_indicators, entity_patterns_for_tags, subject_tags_patterns)
        
        self.document_type_patterns = self.config.get('document_type_patterns', default_doc_type_patterns)
        self.legal_domain_patterns = self.config.get('legal_domain_patterns', default_legal_domain_patterns)
        # For brevity, assuming other pattern sets are similarly loaded or default.
        self.procedural_stage_patterns = self.config.get('procedural_stage_patterns', {'pleading': [r'complaint', r'answer'], 'discovery': [r'interrogatories', r'deposition']})
        self.importance_indicators = self.config.get('importance_indicators', {'high': [r'constitutional', r'supreme\s+court'], 'medium': [r'significant'], 'low': [r'minor']})
        self.entity_patterns_for_tags = self.config.get('entity_patterns_for_tags', {'court': [r'supreme\s+court'], 'judge': [r'judge\s+\w+']})
        self.subject_tags_patterns = self.config.get('subject_tags_patterns', [r'breach\s+of\s+contract', r'due\s+process'])
        
        self.logger.debug("Tagging frameworks initialized (patterns loaded from config/defaults).")

    async def _load_learning_data_from_umm_async(self):
        """Loads initial learning statistics for tags from UnifiedMemoryManager."""
        if not self.unified_memory_manager:
            self.logger.info("UMM not available, skipping load of learning data for auto-tagging.")
            return
        try:
            # Example: Load stats for top N most frequently suggested tags or a predefined list
            # This is a conceptual placeholder. A real implementation would query UMM.
            # common_tags_stats = await self.unified_memory_manager.get_batch_tag_learning_stats_async(tags_to_load)
            # self.tag_accuracy_scores_cache.update(common_tags_stats)
            self.logger.info("Successfully loaded (placeholder) learning data from UMM for auto-tagging.")
        except Exception as e:
            self.logger.error("Failed to load learning data from UMM for auto-tagging.", exception=e)
    
    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        document_id = metadata.get('document_id', f"autotag_doc_{uuid.uuid4().hex[:8]}")
        text_content = task_data.get('text', '')
        existing_tags_list = metadata.get('existing_tags', []) # Tags already on the document
        user_feedback_data = metadata.get('user_feedback', {}) # e.g., {'doc_id': ..., 'correct_tags': [], 'incorrect_tags': []}

        self.logger.info(f"Starting auto-tagging for doc '{document_id}'.", 
                         parameters={'text_len': len(text_content), 'has_feedback': bool(user_feedback_data)})
        start_time_obj = datetime.now(timezone.utc)
        output = AutoTaggingOutput(document_id=document_id)

        if not text_content or len(text_content.strip()) < 20:
            output.errors.append("Insufficient text content for auto-tagging.")
            # ... (set processing_time_sec and return as in ViolationDetector)
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time_obj).total_seconds(), 3)
            return output.to_dict()

        try:
            if user_feedback_data and user_feedback_data.get('document_id') == document_id:
                await self._apply_user_feedback(document_id, user_feedback_data)
            
            tagging_result = await self._perform_comprehensive_tagging_async(text_content, document_id, existing_tags_list)
            
            output.document_type_classification = tagging_result.get('document_type_classification')
            output.legal_domain_tags = tagging_result.get('legal_domain_tags', [])
            output.procedural_stage_tag = tagging_result.get('procedural_stage_tag')
            output.importance_level_tag = tagging_result.get('importance_level_tag')
            output.extracted_entity_tags = tagging_result.get('extracted_entity_tags', {})
            output.generated_subject_tags = tagging_result.get('generated_subject_tags', [])
            output.all_generated_tags = tagging_result.get('all_generated_tags', [])
            # Ensure suggested_new_tags only contains tags not in the original existing_tags_list
            output.suggested_new_tags = [tag for tag in output.all_generated_tags if tag not in existing_tags_list]
            output.model_used_for_llm = tagging_result.get('model_used_for_llm')
            output.overall_confidence = self._calculate_overall_tagging_confidence(tagging_result)

            await self._update_learning_from_session_async(document_id, output.all_generated_tags)

            self.logger.info(f"Auto-tagging completed for doc '{document_id}'.", 
                            parameters={'tags_generated': len(output.all_generated_tags), 'overall_conf': output.overall_confidence})
        
        except AgentProcessingError as ape:
            self.logger.error(f"AgentProcessingError during auto-tagging for doc '{document_id}'.", exception=ape, exc_info=True)
            output.errors.append(f"Agent processing error: {str(ape)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during auto-tagging for doc '{document_id}'.", exception=e, exc_info=True)
            output.errors.append(f"Unexpected error: {str(e)}")
        
        finally:
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time_obj).total_seconds(), 3)
        
        return output.to_dict()
    
    async def _perform_comprehensive_tagging_async(self, text: str, doc_id: str, 
                                                 existing_tags: List[str]) -> Dict[str, Any]:
        # Synchronous pattern-based parts
        doc_type = self._classify_document_type_sync(text)
        legal_domains = self._classify_legal_domain_sync(text)
        proc_stage = self._identify_procedural_stage_sync(text)
        importance = self._assess_importance_level_sync(text)
        entity_tags = self._extract_entity_tags_sync(text)
        subject_tags = self._generate_subject_tags_sync(text) # Already formatted list
        
        pattern_based_tags: Set[str] = set()
        if doc_type and doc_type.get('type') != 'unknown': pattern_based_tags.add(f"doc_type:{doc_type['type']}")
        pattern_based_tags.update(f"domain:{domain}" for domain in legal_domains)
        if proc_stage and proc_stage.get('stage'): pattern_based_tags.add(f"stage:{proc_stage['stage']}")
        if importance and importance.get('level'): pattern_based_tags.add(f"importance:{importance['level']}")
        
        for entity_type, entities_info_list in entity_tags.items():
            for entity_info in entities_info_list[:3]: # Limit number of entity tags per type
                norm_name = re.sub(r'\s+', '_', entity_info['name'].lower().strip())
                pattern_based_tags.add(f"entity:{entity_type}:{norm_name}")
        pattern_based_tags.update(subject_tags)

        # LLM-based additional tagging (async)
        llm_generated_formatted_tags: List[str] = []
        model_used_llm = None
        if self.enable_llm_tag_generation and self.llm_manager:
            try:
                text_for_llm = text if len(text) <= self.max_text_for_llm else text[:self.max_text_for_llm]
                llm_generated_formatted_tags, model_used_llm = await self._llm_generate_additional_tags(text_for_llm, list(pattern_based_tags), doc_id)
            except Exception as llm_e:
                self.logger.error(f"LLM tag generation failed for doc '{doc_id}'.", exception=llm_e)
        
        all_tags_set = pattern_based_tags.union(set(llm_generated_formatted_tags))
        
        # Simple learning: if a tag was often marked incorrect, lower its chance of being included or remove.
        # This requires tag_accuracy_scores_cache to be populated.
        final_tags_after_learning_filter: List[str] = []
        for tag in all_tags_set:
            final_tags_after_learning_filter.append(tag)

        return {
            'document_type_classification': doc_type,
            'legal_domain_tags': legal_domains, # These are raw domains, not formatted tags
            'procedural_stage_tag': proc_stage,
            'importance_level_tag': importance,
            'extracted_entity_tags': entity_tags, # Raw extracted entities info
            'generated_subject_tags': subject_tags, # Formatted subject tags
            'all_generated_tags': sorted(list(final_tags_after_learning_filter)),
            'model_used_for_llm': model_used_llm
        }

    # --- Synchronous Pattern-Based Methods ---
    def _classify_document_type_sync(self, text: str) -> Optional[Dict[str, Any]]:
        type_scores: Dict[str, float] = defaultdict(float)
        text_lower_snippet = text.lower()[:3000] # Analyze start of text

        for doc_type, patterns in self.document_type_patterns.items():
            match_count = 0
            for p_str in patterns:
                if re.search(p_str, text_lower_snippet, re.IGNORECASE):
                    match_count += 1
            if match_count > 0:
                # Score based on proportion of type's patterns matched, plus a boost for multiple matches
                type_scores[doc_type] = (match_count / len(patterns)) + (0.1 * (match_count -1))
        
        if not type_scores: return {'type': 'unknown', 'confidence': 0.3}
        best_type, best_raw_score = max(type_scores.items(), key=lambda x: x[1])
        # Normalize confidence to be between 0 and 1 roughly
        confidence = min(1.0, self.min_pattern_confidence + best_raw_score * 0.5) # Adjust scaling factor
        return {'type': best_type, 'confidence': round(confidence, 3)} if confidence > self.min_pattern_confidence else {'type': 'unknown', 'confidence': 0.3}

    def _classify_legal_domain_sync(self, text: str) -> List[str]:
        domain_scores: Dict[str, float] = defaultdict(float)
        text_lower = text.lower() # Analyze full text for domains
        for domain, patterns in self.legal_domain_patterns.items():
            match_count = sum(1 for p_str in patterns if re.search(p_str, text_lower, re.IGNORECASE))
            if match_count > 0:
                domain_scores[domain] = (match_count / len(patterns)) + (0.05 * match_count) # Small boost for more matches
        
        # Return top N domains or those above a threshold
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, s_val in sorted_domains if s_val > 0.3][:3] # Top 3 domains with min score

    def _identify_procedural_stage_sync(self, text: str) -> Optional[Dict[str, Any]]:
        # Similar logic to document_type classification
        stage_scores: Dict[str, float] = defaultdict(float)
        text_lower_snippet = text.lower()[:4000]
        for stage, patterns in self.procedural_stage_patterns.items():
            match_count = sum(1 for p_str in patterns if re.search(p_str, text_lower_snippet, re.IGNORECASE))
            if match_count > 0:
                stage_scores[stage] = (match_count / len(patterns)) + (0.1 * (match_count -1))
        
        if not stage_scores: return None
        best_stage, best_raw_score = max(stage_scores.items(), key=lambda x: x[1])
        confidence = min(1.0, self.min_pattern_confidence + best_raw_score * 0.4)
        return {'stage': best_stage, 'confidence': round(confidence, 3)} if confidence > self.min_pattern_confidence else None

    def _assess_importance_level_sync(self, text: str) -> Optional[Dict[str, Any]]:
        text_lower = text.lower()
        # Iterate from high to low, return first match
        for level in ['high', 'medium', 'low']:
            if level in self.importance_indicators:
                if any(re.search(ind_pat, text_lower, re.IGNORECASE) for ind_pat in self.importance_indicators[level]):
                    confidence_map = {'high': 0.8, 'medium': 0.65, 'low': 0.5}
                    return {'level': level, 'confidence': confidence_map.get(level, 0.5)}
        return {'level': 'medium', 'confidence': 0.4} # Default if no specific indicators

    def _extract_entity_tags_sync(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        entity_tags_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        text_snippet_for_entities = text[:7000] 

        for entity_type_label, patterns in self.entity_patterns_for_tags.items():
            seen_for_type: Set[str] = set()
            current_type_tags = []
            for p_str in patterns:
                try:
                    for match in re.finditer(p_str, text_snippet_for_entities, re.IGNORECASE | re.DOTALL):
                        entity_name = match.group(0).strip()
                        # Normalize name slightly for deduplication within this type
                        norm_name_key = re.sub(r'\s+', ' ', entity_name.lower()) 
                        if norm_name_key not in seen_for_type:
                            current_type_tags.append({'name': entity_name, 'confidence': round(self.min_pattern_confidence + 0.2, 3)}) # Base confidence for entity tag
                            seen_for_type.add(norm_name_key)
                except re.error as re_err:
                    self.logger.warning(f"Regex error during entity tag pattern matching for type '{entity_type_label}'.",
                                       parameters={'pattern': p_str, 'error': str(re_err)})
            # Sort by length (longer names often more specific) then add to map
            current_type_tags.sort(key=lambda x: -len(x['name']))
            entity_tags_map[entity_type_label] = current_type_tags[:5] # Limit to top 5 distinct entities per type
        return dict(entity_tags_map)

    def _generate_subject_tags_sync(self, text: str) -> List[str]:
        subject_tags: Set[str] = set()
        text_lower = text.lower()
        # Use pre-defined subject tag patterns (could be loaded from config)
        for p_str in self.subject_tags_patterns: 
            try:
                if re.search(p_str, text_lower, re.IGNORECASE):
                    # Normalize pattern to create tag: replace spaces and regex chars
                    tag_key = re.sub(r'[^a-z0-9_]+', '_', p_str.lower().strip())
                    tag = f"subject:{tag_key.strip('_')}"
                    subject_tags.add(tag)
            except re.error:
                self.logger.warning(f"Invalid regex pattern for subject tag: {p_str}")
        return sorted(list(subject_tags))[:15] # Limit number of subject tags

    # --- LLM-Based Tag Generation ---
    async def _llm_generate_additional_tags(self, text_for_llm: str, existing_tags: List[str], doc_id: str) -> Tuple[List[str], Optional[str]]:
        if not self.llm_manager: return [], None
        self.logger.debug(f"Using LLM for additional tag generation on doc '{doc_id}'.")

        existing_tags_str = ', '.join(sorted(list(set(existing_tags)))[:20]) # Unique, sorted, limited
        prompt = f"""
        You are a legal document tagging expert. Analyze the following DOCUMENT TEXT EXCERPT and suggest 5 to 7 highly relevant keyword tags.
        These tags should capture specific legal concepts, procedures, key topics, or named entities not already covered by the EXISTING TAGS.
        Tags should be concise, in lowercase, with spaces replaced by underscores (e.g., "breach_of_contract", "fourth_amendment_issue").

        EXISTING TAGS (for context, avoid suggesting conceptually identical tags):
        {existing_tags_str if existing_tags else "None"}
        
        DOCUMENT TEXT EXCERPT:
        ---
        {text_for_llm}
        ---
        
        Output ONLY a comma-separated list of your new suggested tags. Do not add prefixes like "llm:".
        Example: specific_statute_123, key_procedural_event, important_legal_doctrine
        """
        model_to_use = self.config.get('llm_model_for_autotag', self.llm_manager.primary_config.model)
        provider_to_use = self.config.get('llm_provider_for_autotag', self.llm_manager.primary_config.provider.value)
        
        try:
            llm_response = await self.llm_manager.complete(
                prompt, model=model_to_use, provider=LLMProviderEnum(provider_to_use),
                model_params={'temperature': 0.4, 'max_tokens': 150} # Temperature slightly higher for creativity
            )
            response_text = llm_response.content.strip()
            model_used_str = f"{provider_to_use}/{model_to_use}"
            
            llm_formatted_tags = []
            if response_text:
                potential_tags = [t.strip().lower() for t in response_text.split(',')]
                for tag_candidate in potential_tags:
                    # Basic validation and normalization
                    tag_candidate = re.sub(r'\s+', '_', tag_candidate)
                    tag_candidate = re.sub(r'[^a-z0-9_:]', '', tag_candidate)  # Allow colons for existing prefixes
                    is_covered = any(
                        self._normalize_tag(tag_candidate) == self._normalize_tag(t)
                        for t in existing_tags
                    )
                    if tag_candidate and len(tag_candidate) > 2 and tag_candidate not in existing_tags:
                        if not is_covered:
                            llm_formatted_tags.append(f"llm:{tag_candidate}")
            
            self.logger.debug(f"LLM ({model_used_str}) generated {len(llm_formatted_tags)} additional tags for doc '{doc_id}'.")
            return llm_formatted_tags[:7], model_used_str # Limit number of LLM tags
        except LLMProviderError as e:
            self.logger.error(f"LLM API call for tag generation failed for doc '{doc_id}'.", exception=e)
            raise AgentProcessingError("LLM tag generation failed.", underlying_exception=e) from e
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM tag generation for doc '{doc_id}'.", exception=e)
            raise AgentProcessingError("Unexpected error during LLM tag generation.", underlying_exception=e) from e

    # --- Learning System Methods ---
    async def _apply_user_feedback(self, doc_id: str, feedback: Dict[str, Any]):
        self.logger.info(f"Applying user feedback to learning model (UMM) for doc '{doc_id}'.")
        self.feedback_session_log.append({'doc_id': doc_id, 'feedback': feedback, 'timestamp': datetime.now(timezone.utc).isoformat()})

        # feedback structure: {'correct_tags': ['tag1', 'tag2'], 'incorrect_tags': ['tag3'], 'added_by_user': ['new_tag']}
        correct_tags = [self._normalize_tag(t) for t in feedback.get('correct_tags', [])]
        incorrect_tags = [self._normalize_tag(t) for t in feedback.get('incorrect_tags', [])]
        added_by_user = [self._normalize_tag(t) for t in feedback.get('added_by_user', [])]

        if self.unified_memory_manager:
            tasks = []
            ts = time.time()
            for tag in correct_tags:
                tasks.append(self.unified_memory_manager.update_tag_learning_stats_async(tag, correct_increment=1.0, last_updated_ts=ts))
                self.tag_accuracy_scores_cache[tag]["correct"] += 1.0
                self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            for tag in incorrect_tags:
                tasks.append(self.unified_memory_manager.update_tag_learning_stats_async(tag, incorrect_increment=1.0, last_updated_ts=ts))
                self.tag_accuracy_scores_cache[tag]["incorrect"] += 1.0
                self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            for tag in added_by_user: # Tags user added are considered "correct" and "suggested" if new
                tasks.append(self.unified_memory_manager.update_tag_learning_stats_async(tag, correct_increment=1.0, suggested_increment=1.0, last_updated_ts=ts))
                self.tag_accuracy_scores_cache[tag]["correct"] += 1.0
                self.tag_accuracy_scores_cache[tag]["suggested"] += 1.0
                self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            
            if tasks: await asyncio.gather(*tasks)
            self.logger.debug(f"User feedback for doc '{doc_id}' persisted to UMM and local cache updated.")
        else: # Fallback to in-memory only if UMM is not available
            ts = time.time()
            for tag in correct_tags: self.tag_accuracy_scores_cache[tag]["correct"] += 1.0; self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            for tag in incorrect_tags: self.tag_accuracy_scores_cache[tag]["incorrect"] += 1.0; self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            for tag in added_by_user: self.tag_accuracy_scores_cache[tag]["correct"] += 1.0; self.tag_accuracy_scores_cache[tag]["suggested"] += 1.0; self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            self.logger.warning("UMM not available. Feedback for auto-tagging applied to in-memory cache only.")

    def _normalize_tag(self, tag: str) -> str:
        """Normalizes a tag string to a consistent format."""
        if not isinstance(tag, str): return ""
        return tag.lower().strip().replace(" ", "_")

    async def _update_learning_from_session_async(self, doc_id: str, generated_tags: List[str]):
        normalized_tags = [self._normalize_tag(t) for t in generated_tags]
        if not normalized_tags: return

        if self.unified_memory_manager:
            tasks = []
            ts = time.time()
            for tag in normalized_tags:
                tasks.append(self.unified_memory_manager.update_tag_learning_stats_async(tag, suggested_increment=1.0, last_updated_ts=ts))
                self.tag_accuracy_scores_cache[tag]["suggested"] += 1.0
                self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            if tasks: await asyncio.gather(*tasks)
            self.logger.trace(f"Learning from current auto-tagging session (suggestion counts) persisted to UMM for doc '{doc_id}'.")
        else: # Fallback to in-memory
            ts = time.time()
            for tag in normalized_tags: self.tag_accuracy_scores_cache[tag]["suggested"] += 1.0; self.tag_accuracy_scores_cache[tag]["last_updated_ts"] = ts
            self.logger.warning("UMM not available. Learning from auto-tagging session applied to in-memory cache only.")

    def _calculate_overall_tagging_confidence(self, tagging_result: Dict[str, Any]) -> float:
        confidences = []
        if tagging_result.get('document_type_classification') and isinstance(tagging_result['document_type_classification'], dict):
            confidences.append(tagging_result['document_type_classification'].get('confidence', 0.0))
        if tagging_result.get('procedural_stage_tag') and isinstance(tagging_result['procedural_stage_tag'], dict):
            confidences.append(tagging_result['procedural_stage_tag'].get('confidence', 0.0))
        if tagging_result.get('importance_level_tag') and isinstance(tagging_result['importance_level_tag'], dict):
            confidences.append(tagging_result['importance_level_tag'].get('confidence', 0.0))
        
        # Add base confidence for presence of other tag types
        if tagging_result.get('legal_domain_tags'): confidences.append(0.6) 
        if tagging_result.get('extracted_entity_tags'): confidences.append(0.65)
        if tagging_result.get('generated_subject_tags'): confidences.append(0.55)
        if any(tag.startswith("llm:") for tag in tagging_result.get('all_generated_tags',[])): confidences.append(self.min_llm_tag_confidence)

        if not confidences: return 0.25 # Low base if nothing classified
        return round(sum(confidences) / len(confidences), 3)

    async def get_learning_statistics_async(self) -> Dict[str, Any]: # Public method
        """Retrieves learning statistics, potentially combining cache and UMM."""
        # For simplicity, this example returns cached stats. A real version would query UMM.
        if self.unified_memory_manager:
            # Example: fetch global stats or stats for top N tags from UMM
            # global_stats = await self.unified_memory_manager.get_overall_tag_learning_summary_async()
            # For now, combine with cache:
            # This is complex. Simplest is to return cache, assuming UMM is the source of truth updated by this agent.
            pass

        cached_stats = self.tag_accuracy_scores_cache.copy() # Operate on a copy
        total_feedback_events = sum(stats.get('correct',0) + stats.get('incorrect',0) for stats in cached_stats.values())
        total_correct_feedback = sum(stats.get('correct',0) for stats in cached_stats.values())
        
        return {
            'feedback_sessions_logged_this_instance': len(self.feedback_session_log),
            'distinct_tags_in_cache': len(cached_stats),
            'overall_cached_tag_accuracy': (total_correct_feedback / total_feedback_events) if total_feedback_events > 0 else 0.0,
            'top_correct_tags_cache': sorted([ (tag, data) for tag, data in cached_stats.items() if data.get('correct',0) > 0], key=lambda x: x[1].get('correct',0), reverse=True)[:10],
            'top_incorrect_tags_cache': sorted([ (tag, data) for tag, data in cached_stats.items() if data.get('incorrect',0) > 0], key=lambda x: x[1].get('incorrect',0), reverse=True)[:10],
            'umm_status': 'available' if self.unified_memory_manager else 'unavailable'
        }

    async def health_check(self) -> Dict[str, Any]:
        status = await super().health_check() if hasattr(super(), 'health_check') else {"status": "healthy", "checks": []}
        status['agent_name'] = self.name
        status['dependencies_status'] = {
            'llm_manager': 'available' if self.llm_manager else 'unavailable',
            'unified_memory_manager': 'available' if self.unified_memory_manager else 'unavailable'
        }
        status['config_summary'] = {
            'enable_llm_tag_generation': self.enable_llm_tag_generation,
            'min_pattern_confidence': self.min_pattern_confidence,
            'min_llm_tag_confidence': self.min_llm_tag_confidence
        }
        if (self.enable_llm_tag_generation and not self.llm_manager):
            status['status'] = 'degraded'
            status['reason'] = 'LLM tag generation enabled but LLMManager is unavailable.'
        return status
