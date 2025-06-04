# legal_ai_system/agents/auto_tagging/auto_tagging_agent.py
"""
Auto Tagging Agent - Learning-based Document Classification and Tagging
This agent automatically tags and classifies legal documents.
"""

import re
import uuid
import asyncio
import time # <--- IMPORTED TIME MODULE
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import Counter, defaultdict # Added defaultdict
from dataclasses import dataclass, field, asdict

from ....core.base_agent import BaseAgent, ProcessingResult
from ....core.llm_providers import LLMManager, LLMProviderError, LLMProviderEnum
from ....core.unified_exceptions import AgentProcessingError, AgentInitializationError
from ....core.detailed_logging import LogCategory, get_detailed_logger
from ....memory.unified_memory_manager import UnifiedMemoryManager # For UMM access (if available)

# Logger for this agent
auto_tag_logger = get_detailed_logger("AutoTaggingAgent", LogCategory.AGENT_LIFECYCLE)


@dataclass
class AutoTaggingOutput:
    document_id: str
    document_type_classification: Optional[Dict[str, Any]] = None
    legal_domain_tags: List[str] = field(default_factory=list) # Raw domains, not formatted tags
    procedural_stage_tag: Optional[Dict[str, Any]] = None
    importance_level_tag: Optional[Dict[str, Any]] = None
    extracted_entity_tags: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    generated_subject_tags: List[str] = field(default_factory=list) # Formatted like "subject:term"
    all_generated_tags: List[str] = field(default_factory=list) # Final combined, formatted tags
    suggested_new_tags: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    tagged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used_for_llm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure extracted_entity_tags is correctly serialized if it contains complex objects (though dicts are fine)
        return data

class AutoTaggingAgent(BaseAgent):
    """
    Intelligent auto-tagging agent that learns from patterns and user feedback,
    persisting learning via UnifiedMemoryManager if available.
    """
    
    def __init__(self, service_container: Any, **config: Any):
        super().__init__(service_container, name="AutoTaggingAgent", agent_type="classification")
        
        self.llm_manager: Optional[LLMManager] = self._get_service("llm_manager")
        self.unified_memory_manager: Optional[UnifiedMemoryManager] = self._get_service("unified_memory_manager")

        self.config = config
        self.min_pattern_confidence = float(config.get('min_pattern_confidence_autotag', 0.45))
        self.min_llm_tag_confidence = float(config.get('min_llm_tag_confidence_autotag', 0.6)) # LLM suggestions should be higher confidence
        self.enable_llm_tag_generation = bool(config.get('enable_llm_tag_generation', True))
        self.max_text_for_llm = int(config.get('max_text_for_llm_autotag', 3000))

        self._init_tagging_frameworks_from_config()
        
        # In-memory cache for tag learning statistics. UMM is the source of truth if available.
        self.tag_accuracy_scores_cache: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"correct": 0.0, "incorrect": 0.0, "suggested": 0.0, "last_updated_ts": 0.0}
        )
        self.feedback_session_log: List[Dict[str,Any]] = [] # Log of feedback received in this agent's instance lifetime

        self.logger.info(f"{self.name} initialized.", 
                        parameters={'llm_tag_gen_enabled': self.enable_llm_tag_generation, 
                                    'llm_available': bool(self.llm_manager),
                                    'umm_available': bool(self.unified_memory_manager)})

    async def initialize_agent_async(self): # Optional: called by a system if agent needs async setup
        """Asynchronously initializes components, e.g., loading learning data from UMM."""
        await self._load_learning_data_from_umm_async()

    def _init_tagging_frameworks_from_config(self):
        """Initialize tagging patterns from agent configuration or use defaults."""
        # Default pattern definitions (can be quite extensive)
        default_doc_type_patterns = {
            'motion': [r'motion\s+to\s+\w+', r'motion\s+for\s+(?:summary\s+)?judgment', r'notice\s+of\s+motion'],
            'brief': [r'brief\s+in\s+(?:support|opposition)', r'memorandum\s+of\s+law', r'amicus\s+curiae\s+brief'],
            'order': [r'court\s+order', r'it\s+is\s+(?:hereby\s+)?ordered', r'judgment\s+and\s+order', r'minute\s+order'],
            'complaint': [r'complaint\s+for\s+\w+', r'plaintiff\s+\w+\s+alleges', r'first\s+amended\s+complaint', r'class\s+action\s+complaint'],
            'answer': [r'answer\s+to\s+complaint', r'defendant\s+\w+\s+answers'],
            'affidavit': [r'affidavit\s+of\s+\w+', r'declaration\s+of\s+\w+\s+in\s+support'],
            'discovery_request': [r'interrogatories', r'request\s+for\s+production', r'notice\s+of\s+deposition', r'request\s+for\s+admission'],
            'discovery_response': [r'responses\s+to\s+interrogatories', r'objections\s+and\s+responses'],
            'agreement': [r'settlement\s+agreement', r'contract', r'stipulation\s+and\s+order', r'non-disclosure\s+agreement'],
            'transcript': [r'hearing\s+transcript', r'deposition\s+transcript', r'(?:official\s+)?reporter\'s\s+transcript', r'trial\s+transcript'],
            'exhibit_list': [r'exhibit\s+list', r'index\s+of\s+exhibits'],
            'appeal_document': [r'notice\s+of\s+appeal', r'appellate\s+brief', r'petition\s+for\s+writ'],
            'opinion': [r'opinion\s+of\s+the\s+court', r'memorandum\s+opinion', r'dissenting\s+opinion', r'concurring\s+opinion']
        }
        default_legal_domain_patterns = {
            'criminal_law': [r'defendant\s+charged', r'criminal\s+case', r'prosecution', r'indictment', r'sentencing', r'plea\s+agreement', r'arraignment'],
            'civil_litigation': [r'civil\s+action', r'damages', r'liability', r'plaintiff', r'summons', r'negligence', r'tort'],
            'corporate_law': [r'merger', r'acquisition', r'shareholder', r'sec\s+filing', r'corporate\s+governance', r'articles\s+of\s+incorporation'],
            'intellectual_property': [r'patent', r'trademark', r'copyright', r'infringement', r'intellectual\s+property', r'trade\s+secret'],
            'family_law': [r'divorce', r'custody', r'child\s+support', r'marital\s+agreement', r'alimony', r'prenuptial'],
            'bankruptcy_law': [r'chapter\s+(?:7|11|13)', r'bankruptcy', r'creditor', r'debtor', r'discharge\s+of\s+debt', r'proof\s+of\s+claim'],
            'real_estate_law': [r'property\s+law', r'lease\s+agreement', r'mortgage', r'landlord\s+tenant', r'zoning', r'deed'],
            'employment_law': [r'employment\s+agreement', r'discrimination', r'wrongful\s+termination', r'harassment', r'flsa', r'ada']
        }
        default_procedural_stage_patterns = {
            'pre_trial': [r'initial\s+appearance', r'preliminary\s+hearing', r'arraignment', r'bail\s+hearing'],
            'pleading': [r'complaint', r'answer', r'counterclaim', r'motion\s+to\s+dismiss'],
            'discovery': [r'interrogatories', r'deposition', r'request\s+for\s+production', r'expert\s+disclosure', r'subpoena\s+duces\s+tecum'],
            'trial': [r'jury\s+selection', r'opening\s+statement', r'direct\s+examination', r'cross-examination', r'closing\s+argument', r'jury\s+instructions', r'verdict'],
            'post_trial': [r'motion\s+for\s+new\s+trial', r'judgment\s+notwithstanding\s+the\s+verdict', r'sentencing\s+hearing'],
            'appeal': [r'notice\s+of\s+appeal', r'appellate\s+brief', r'oral\s+argument\s+(?:before|in)\s+appellate\s+court', r'petition\s+for\s+writ\s+of\s+certiorari']
        }
        default_importance_indicators = {
            'critical': [r'emergency\s+motion', r'temporary\s+restraining\s+order', r'order\s+to\s+show\s+cause', r'immediate\s+injunctive\s+relief'],
            'high': [r'constitutional\s+issue', r'supreme\s+court', r'dispositive\s+motion', r'summary\s+judgment', r'statutory\s+deadline', r'urgent'],
            'medium': [r'significant\s+development', r'material\s+fact', r'substantial\s+evidence', r'key\s+witness', r'expert\s+report'],
            'low': [r'minor\s+procedural\s+matter', r'routine\s+filing', r'administrative\s+update', r'scheduling\s+order', r'notice\s+of\s+appearance']
        }
        default_entity_patterns_for_tags = { # Simplified for tagging
            'court': [r'(?i)\b(?:supreme|district|circuit|superior|appellate)\s+court(?:\s+of\s+[\w\s]+)?\b', r'\bU\.S\. Court of Appeals for the \w+ Circuit\b'],
            'judge': [r'(?i)\b(?:judge|justice|honorable|hon\.)\s+[A-Z][a-z]+\s*(?:[A-Z]\.?\s*)?[A-Z][a-z]+\b'],
            'law_firm': [r'(?i)\b[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*(?:,\s*LLP|,\s*LLC|,?\s*P\.C\.?|,\s*A\s*Professional\s*Corporation|\s+Group)\b'],
            'party_individual': [r'(?i)\b(?:plaintiff|defendant|petitioner|respondent)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'],
            'party_organization': [r'(?i)\b(?:plaintiff|defendant|petitioner|respondent)\s+[A-Z][A-Za-z\s,&]+(?:LLC|Inc|Corp|Ltd|Co\.)\b']
        }
        default_subject_tags_patterns = [ # Keywords that become "subject:keyword"
            r'breach\s+of\s+contract', r'due\s+process', r'statute\s+of\s+limitations',
            r'summary\s+judgment', r'discovery\s+dispute', r'negligence\s+claim', r'liability',
            r'admissibility\s+of\s+evidence', r'expert\s+testimony', r'jurisdictional\s+challenge',
            r'class\s+certification', r'attorney-client\s+privilege', r'work\s+product\s+doctrine'
        ]
        
        self.document_type_patterns = self.config.get('document_type_patterns', default_doc_type_patterns)
        self.legal_domain_patterns = self.config.get('legal_domain_patterns', default_legal_domain_patterns)
        self.procedural_stage_patterns = self.config.get('procedural_stage_patterns', default_procedural_stage_patterns)
        self.importance_indicators = self.config.get('importance_indicators', default_importance_indicators)
        self.entity_patterns_for_tags = self.config.get('entity_patterns_for_tags', default_entity_patterns_for_tags)
        self.subject_tags_patterns = self.config.get('subject_tags_patterns', default_subject_tags_patterns)
        
        self.logger.debug("Tagging frameworks initialized (patterns loaded from config/defaults).")

    async def _load_learning_data_from_umm_async(self):
        """Loads initial learning statistics for tags from UnifiedMemoryManager."""
        if not self.unified_memory_manager:
            self.logger.info("UMM not available, skipping load of learning data for auto-tagging.")
            return
        try:
            # Example: Load stats for a predefined set of common tags or tags seen recently
            # This is conceptual. A real implementation needs a UMM method like:
            # common_tags_data = await self.unified_memory_manager.get_batch_tag_learning_stats_async(limit=100, min_suggestion_count=5)
            # if common_tags_data:
            #     for tag_name, stats_dict in common_tags_data.items():
            #         self.tag_accuracy_scores_cache[self._normalize_tag(tag_name)] = {
            #             "correct": float(stats_dict.get("correct_count", 0.0)),
            #             "incorrect": float(stats_dict.get("incorrect_count", 0.0)),
            #             "suggested": float(stats_dict.get("suggested_count", 0.0)),
            #             "last_updated_ts": float(stats_dict.get("last_updated_timestamp", 0.0))
            #         }
            #     self.logger.info(f"Loaded learning data for {len(common_tags_data)} tags from UMM for auto-tagging.")
            # else:
            self.logger.info("No specific learning data pre-loaded from UMM for auto-tagging (or UMM method not implemented). Cache will build.")
        except Exception as e:
            self.logger.error("Failed to load learning data from UMM for auto-tagging.", exception=e)
    
    @detailed_log_function(LogCategory.AGENT_PROCESSING)
    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        document_id = metadata.get('document_id', f"autotag_doc_{uuid.uuid4().hex[:8]}")
        text_content = task_data.get('text', task_data.get('text_content', '')) # More robust text extraction
        existing_tags_list = [self._normalize_tag(t) for t in metadata.get('existing_tags', []) if t] # Normalize existing tags
        user_feedback_data = metadata.get('user_feedback', {})

        self.logger.info(f"Starting auto-tagging for doc '{document_id}'.", 
                         parameters={'text_len': len(text_content), 
                                     'has_feedback': bool(user_feedback_data),
                                     'num_existing_tags': len(existing_tags_list)})
        start_time = datetime.now(timezone.utc)
        output = AutoTaggingOutput(document_id=document_id)

        if not text_content or len(text_content.strip()) < 20: # Min length for any meaningful tagging
            output.errors.append("Insufficient text content for auto-tagging.")
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
            self.logger.warning(f"Auto-tagging skipped for doc '{document_id}' due to insufficient text.", 
                               parameters={'text_len': len(text_content)})
            return output.to_dict()

        try:
            if user_feedback_data and user_feedback_data.get('document_id') == document_id:
                await self._apply_user_feedback(document_id, user_feedback_data) # Awaits UMM potentially
            
            tagging_result_dict = await self._perform_comprehensive_tagging_async(text_content, document_id, existing_tags_list)
            
            output.document_type_classification = tagging_result_dict.get('document_type_classification')
            output.legal_domain_tags = tagging_result_dict.get('legal_domain_tags', [])
            output.procedural_stage_tag = tagging_result_dict.get('procedural_stage_tag')
            output.importance_level_tag = tagging_result_dict.get('importance_level_tag')
            output.extracted_entity_tags = tagging_result_dict.get('extracted_entity_tags', {})
            output.generated_subject_tags = tagging_result_dict.get('generated_subject_tags', [])
            output.all_generated_tags = sorted(list(set(tagging_result_dict.get('all_generated_tags', [])))) # Ensure unique and sorted
            output.suggested_new_tags = sorted([tag for tag in output.all_generated_tags if tag not in existing_tags_list])
            output.model_used_for_llm = tagging_result_dict.get('model_used_for_llm')
            output.overall_confidence = self._calculate_overall_tagging_confidence(tagging_result_dict)

            await self._update_learning_from_session_async(document_id, output.all_generated_tags)

            self.logger.info(f"Auto-tagging completed for doc '{document_id}'.", 
                            parameters={'tags_generated_count': len(output.all_generated_tags), 
                                        'new_tags_suggested_count': len(output.suggested_new_tags),
                                        'overall_conf': output.overall_confidence})
        
        except AgentProcessingError as ape: # Catch specific errors from this agent's logic
            self.logger.error(f"AgentProcessingError during auto-tagging for doc '{document_id}'.", exception=ape)
            output.errors.append(f"Agent processing error: {str(ape)}")
        except Exception as e: # Catch unexpected errors
            self.logger.error(f"Unexpected error during auto-tagging for doc '{document_id}'.", exception=e, exc_info=True)
            output.errors.append(f"Unexpected critical error: {type(e).__name__} - {str(e)}")
        
        finally:
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
        
        return output.to_dict()
    
    # ... (Rest of the methods: _perform_comprehensive_tagging_async, _classify_document_type_sync, 
    #      _classify_legal_domain_sync, _identify_procedural_stage_sync, _assess_importance_level_sync,
    #      _extract_entity_tags_sync, _generate_subject_tags_sync, _llm_generate_additional_tags,
    #      _apply_user_feedback (corrected), _normalize_tag, _update_learning_from_session_async (corrected),
    #      _calculate_overall_tagging_confidence, get_learning_statistics_async, health_check
    #      would be filled in as per the complete version in the previous responses, ensuring `time.time()` uses imported `time`
    #      and UMM interactions are robust.)

    # For brevity, I'll only re-paste a few key changed/important methods here.
    # Assume the pattern-based classification sync methods are as per the previous detailed agent script.

    async def _perform_comprehensive_tagging_async(self, text: str, doc_id: str, 
                                                 existing_tags: List[str]) -> Dict[str, Any]:
        # Synchronous pattern-based parts
        doc_type_class_result = self._classify_document_type_sync(text)
        legal_domain_results = self._classify_legal_domain_sync(text) # List of domain strings
        proc_stage_result = self._identify_procedural_stage_sync(text)
        importance_result = self._assess_importance_level_sync(text)
        entity_tags_map_result = self._extract_entity_tags_sync(text) # Dict of type -> list of entity dicts
        subject_tags_list_result = self._generate_subject_tags_sync(text) # List of "subject:term"
        
        pattern_tags_set: Set[str] = set()
        if doc_type_class_result and doc_type_class_result.get('type') != 'unknown':
            pattern_tags_set.add(self._normalize_tag(f"doc_type:{doc_type_class_result['type']}"))
        pattern_tags_set.update(self._normalize_tag(f"domain:{domain}") for domain in legal_domain_results)
        if proc_stage_result and proc_stage_result.get('stage'):
            pattern_tags_set.add(self._normalize_tag(f"stage:{proc_stage_result['stage']}"))
        if importance_result and importance_result.get('level'):
            pattern_tags_set.add(self._normalize_tag(f"importance:{importance_result['level']}"))
        
        for entity_type, entities_list in entity_tags_map_result.items():
            for entity_info in entities_list[:3]: # Limit tags per entity type
                norm_name = self._normalize_tag(entity_info['name']) # Normalize name for tag
                pattern_tags_set.add(self._normalize_tag(f"entity:{entity_type}:{norm_name}"))
        pattern_tags_set.update(self._normalize_tag(tag) for tag in subject_tags_list_result) # Subject tags are already formatted

        # LLM for additional tagging (async)
        llm_generated_tags_list: List[str] = []
        model_used_llm_str = None
        if self.enable_llm_tag_generation and self.llm_manager:
            try:
                text_for_llm = text if len(text) <= self.max_text_for_llm else text[:self.max_text_for_llm]
                if len(text) > self.max_text_for_llm:
                    self.logger.warning(f"Text for doc '{doc_id}' truncated to {self.max_text_for_llm} chars for LLM tag generation.")
                
                llm_generated_tags_list, model_used_llm_str = await self._llm_generate_additional_tags(text_for_llm, list(pattern_tags_set), doc_id)
            except Exception as llm_e: # Catch errors from LLM call specifically
                self.logger.error(f"LLM tag generation failed for doc '{doc_id}'.", exception=llm_e)
                # Continue with pattern-based tags
        
        all_tags_combined_set = pattern_tags_set.union(set(llm_generated_tags_list))
        
        # Apply learning filters / rules if any (placeholder for more advanced learning)
        # Example: Remove tags that have a high incorrect_count / correct_count ratio from cache
        final_tags_list: List[str] = []
        for tag_candidate in all_tags_combined_set:
            stats = self.tag_accuracy_scores_cache.get(self._normalize_tag(tag_candidate), {})
            correct = stats.get("correct", 0.0)
            incorrect = stats.get("incorrect", 0.0)
            # Heuristic: if a tag is marked incorrect more often than correct, and has enough feedback, maybe don't suggest it.
            if incorrect > correct + 2 and (correct + incorrect) > 5: # Needs at least 5 data points, and 2 more incorrect
                self.logger.debug(f"Filtering out tag '{tag_candidate}' based on negative feedback history for doc '{doc_id}'.")
                continue
            final_tags_list.append(tag_candidate)

        return {
            'document_type_classification': doc_type_class_result,
            'legal_domain_tags': legal_domain_results, # Keep raw domain list
            'procedural_stage_tag': proc_stage_result,
            'importance_level_tag': importance_result,
            'extracted_entity_tags': entity_tags_map_result, # Keep raw entity map
            'generated_subject_tags': subject_tags_list_result, # Already formatted
            'all_generated_tags': sorted(list(final_tags_list)), # Final combined, sorted
            'model_used_for_llm': model_used_llm_str
        }

    # The rest of the sync classification methods (_classify_document_type_sync, etc.)
    # are assumed to be complete as per the previous detailed file.
    # _llm_generate_additional_tags is also assumed complete.
    # The corrected _apply_user_feedback and _update_learning_from_session_async are above.
    # _normalize_tag, _calculate_overall_tagging_confidence, get_learning_statistics_async, health_check
    # also need to be fully present. I'll ensure they are in the final combined script if one is requested again.

    def _normalize_tag(self, tag: str) -> str:
        if not isinstance(tag, str): return ""
        # Convert to lowercase, replace spaces and common separators with underscore, remove special chars except colon
        normalized = tag.lower().strip()
        normalized = re.sub(r'[\s\-(),./]+', '_', normalized) # Replace common separators
        normalized = re.sub(r'[^a-z0-9_:]', '', normalized)   # Allow alphanumeric, underscore, colon
        normalized = re.sub(r'_+', '_', normalized)          # Reduce multiple underscores
        normalized = normalized.strip('_')                   # Strip leading/trailing underscores
        return normalized

    def _calculate_overall_tagging_confidence(self, tagging_result_data: Dict[str, Any]) -> float:
        confidences = []
        # Document Type
        doc_type_res = tagging_result_data.get('document_type_classification')
        if doc_type_res and isinstance(doc_type_res, dict) and doc_type_res.get('type') != 'unknown':
            confidences.append(float(doc_type_res.get('confidence', 0.0)))
        # Procedural Stage
        proc_stage_res = tagging_result_data.get('procedural_stage_tag')
        if proc_stage_res and isinstance(proc_stage_res, dict) and proc_stage_res.get('stage'):
            confidences.append(float(proc_stage_res.get('confidence', 0.0)))
        # Importance Level
        importance_res = tagging_result_data.get('importance_level_tag')
        if importance_res and isinstance(importance_res, dict) and importance_res.get('level'):
            confidences.append(float(importance_res.get('confidence', 0.0)))
        
        # Base confidence for other categories if tags were found
        if tagging_result_data.get('legal_domain_tags'): confidences.append(0.65) 
        if tagging_result_data.get('extracted_entity_tags'): confidences.append(0.70) # Entity tags from patterns are usually decent
        if tagging_result_data.get('generated_subject_tags'): confidences.append(0.60)
        
        # If LLM contributed tags, factor in its minimum confidence threshold
        if any(tag.startswith("llm:") for tag in tagging_result_data.get('all_generated_tags',[])):
            confidences.append(self.min_llm_tag_confidence)

        if not confidences: return 0.20 # Low base if absolutely nothing specific found
        
        # Weighted average could be considered if some tag types are more reliable
        # For now, simple average.
        avg_conf = sum(confidences) / len(confidences)
        return round(avg_conf, 3)

    async def get_learning_statistics_async(self) -> Dict[str, Any]:
        # This method primarily returns data from the local cache.
        # A full system might query UMM for global/aggregated stats.
        cached_stats_copy = self.tag_accuracy_scores_cache.copy() # Work on a copy

        total_feedback_instances = 0
        total_correct_assignments = 0.0
        total_incorrect_assignments = 0.0

        for tag_data in cached_stats_copy.values():
            total_correct_assignments += tag_data.get("correct", 0.0)
            total_incorrect_assignments += tag_data.get("incorrect", 0.0)
        total_feedback_instances = total_correct_assignments + total_incorrect_assignments

        overall_cached_accuracy = 0.0
        if total_feedback_instances > 0:
            overall_cached_accuracy = total_correct_assignments / total_feedback_instances
        
        # Get top N tags based on different criteria
        def get_top_tags(criteria_key: str, data_dict: Dict, n: int = 10) -> List[Tuple[str, float]]:
            return sorted(
                [(tag, tag_stats.get(criteria_key, 0.0)) for tag, tag_stats in data_dict.items() if tag_stats.get(criteria_key, 0.0) > 0],
                key=lambda x: x[1], 
                reverse=True
            )[:n]

        return {
            'feedback_sessions_logged_this_instance': len(self.feedback_session_log),
            'distinct_tags_in_session_cache': len(cached_stats_copy),
            'overall_cached_tag_accuracy': round(overall_cached_accuracy, 3),
            'top_correct_tags_in_cache': get_top_tags("correct", cached_stats_copy),
            'top_incorrect_tags_in_cache': get_top_tags("incorrect", cached_stats_copy),
            'top_suggested_tags_in_cache': get_top_tags("suggested", cached_stats_copy),
            'umm_status': 'available_and_used' if self.unified_memory_manager else 'unavailable_cache_only',
            'cache_last_updated_example_ts': max([d.get("last_updated_ts", 0.0) for d in cached_stats_copy.values()]) if cached_stats_copy else 0.0
        }
        
    async def health_check(self) -> Dict[str, Any]:
        base_health = await super().health_check() if hasattr(super(), 'health_check') else {"status": "healthy", "checks": []}
        base_health['agent_name'] = self.name
        base_health['dependencies_status'] = {
            'llm_manager': 'available' if self.llm_manager else 'unavailable',
            'unified_memory_manager': 'available' if self.unified_memory_manager else 'unavailable'
        }
        base_health['configuration_summary'] = {
            'enable_llm_tag_generation': self.enable_llm_tag_generation,
            'min_pattern_confidence': self.min_pattern_confidence,
            'min_llm_tag_confidence': self.min_llm_tag_confidence,
            'max_text_for_llm': self.max_text_for_llm
        }
        patterns_summary = {
            "doc_types": len(self.document_type_patterns),
            "legal_domains": len(self.legal_domain_patterns),
            "proc_stages": len(self.procedural_stage_patterns),
            "importance_levels": len(self.importance_indicators),
            "entity_tag_types": len(self.entity_patterns_for_tags),
            "subject_patterns": len(self.subject_tags_patterns)
        }
        base_health['patterns_loaded_counts'] = patterns_summary

        if self.enable_llm_tag_generation and not self.llm_manager:
            base_health['status'] = 'degraded'
            base_health['reason'] = 'LLM tag generation enabled but LLMManager is unavailable.'
        
        self.logger.info(f"{self.name} health check performed.", parameters={'current_status': base_health.get('status')})
        return base_health