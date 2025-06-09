# legal_ai_system/agents/citation_analysis/citation_analysis_agent.py
"""
CitationAnalysisAgent - Legal citation detection, extraction, and classification.
Provides comprehensive analysis of legal citations.
"""

import asyncio
import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict

from ...core.base_agent import BaseAgent, ProcessingResult
from ...utils.ontology import LegalEntityType # Example, not strictly used here but good for context
from ...core.llm_providers import LLMManager, LLMProviderError, LLMProviderEnum
from ...core.model_switcher import ModelSwitcher, TaskComplexity # Assuming ModelSwitcher service
from ...core.unified_exceptions import AgentProcessingError
from ...core.detailed_logging import LogCategory

from ...config.agent_unified_config import create_agent_memory_mixin
from ...memory.unified_memory_manager import MemoryType

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()


@dataclass
class CitationDetails: # Dataclass for a single analyzed citation
    citation_id: str = field(default_factory=lambda: f"CITE_{uuid.uuid4().hex[:8]}")
    raw_text: str
    pattern_name: Optional[str] = None # Regex pattern that matched
    start_pos: int
    end_pos: int
    parsed_components: Dict[str, Any] = field(default_factory=dict) # e.g., volume, reporter, page
    citation_type: str = "unknown" # e.g., "case", "statute", "regulation"
    role_in_document: str = "neutral" # e.g., "supporting", "rebutting", "distinguishing", "background"
    significance_score: float = 0.5 # 0.0 to 1.0, how important this citation seems
    is_valid_format: bool = True # Based on regex or LLM validation
    validation_notes: Optional[str] = None
    llm_confidence: Optional[float] = None # Confidence from LLM if used for classification/validation
    context_snippet: str = "" # Text surrounding the citation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CitationAnalysisOutput: # Renamed from CitationAnalysisResult for consistency
    document_id: str
    citations_found: List[CitationDetails] = field(default_factory=list)
    citation_statistics: Dict[str, Any] = field(default_factory=dict)
    # citation_network_analysis might be too complex for single agent output, consider separate service
    # For now, simple summary:
    key_cited_authorities: List[Dict[str, Any]] = field(default_factory=list) # e.g. [{"text": "Marbury v. Madison", "count": 5}]
    quality_assessment_summary: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0 # Renamed from confidence_score
    processing_time_sec: float = 0.0
    model_used_for_llm: Optional[str] = None # Renamed from model_used
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['citations_found'] = [c.to_dict() for c in self.citations_found]
        return data


class CitationAnalysisAgent(BaseAgent):
    """
    Advanced legal citation analysis agent. Detects, parses, classifies,
    and assesses quality of legal citations.
    """
    
    def __init__(self, service_container: Any, **config: Any):
        super().__init__(service_container, name="CitationAnalysisAgent", agent_type="analysis")
        
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(f"CitationAnalysisAgentAgent configured with model: {self.llm_config.get('llm_model', 'default')}")
        self.llm_manager: Optional[LLMManager] = self._get_service("llm_manager")
        self.model_switcher: Optional[ModelSwitcher] = self._get_service("model_switcher")

        self.config = config
        self.min_regex_confidence_for_llm = float(config.get('min_regex_confidence_for_llm', 0.4))
        self.min_llm_classification_confidence = float(config.get('min_llm_classification_confidence', 0.65))
        self.max_citations_for_llm_batch = int(config.get('max_citations_for_llm_batch', 30))
        self.enable_llm_classification = bool(config.get('enable_llm_citation_classification', True))
        self.max_text_for_llm = int(config.get('max_text_for_llm_citation', 6000))


        # Load citation patterns from config or use defaults
        self._initialize_citation_patterns()
        
        self.classification_prompt_template = self._build_classification_prompt_template()
        
        self.analysis_run_stats = defaultdict(int) # For simple in-memory stats
        self.logger.info(f"{self.name} initialized.", parameters=self.get_config_summary_params())

    def get_config_summary_params(self) -> Dict[str,Any]:
        return {
            'min_llm_conf': self.min_llm_classification_confidence, 
            'max_llm_batch': self.max_citations_for_llm_batch,
            'llm_enabled': self.enable_llm_classification,
            'num_patterns': len(self.citation_patterns)
        }

    def _initialize_citation_patterns(self):
        """Initializes citation patterns from config or defaults."""
        default_patterns = {
            'case_standard': r'(?P<case_name>[A-Z][A-Za-z\s&.,\'"-]+(?:\s+v\.?|\s+vs\.?|\s+ex\s+rel\.|\s+in\s+re)\s+[A-Z][A-Za-z\s&.,\'"-]+),?\s*(?P<volume>\d+)\s+(?P<reporter>[A-Z]{1,5}\.?(?:\s*\d*d\.?)?)\s+(?P<page>\d+)(?:,\s*(?P<specific_page>\d+))?(?:\s*\((?P<court_and_year>[A-Za-z\s.,]*\d{4})\))?',
            'statute_usc': r'(?P<title_usc>\d+)\s+U\.?S\.?C\.?\s*§\s*(?P<section_usc>\d+(?:[a-z])?(?:\([a-zA-Z0-9]+\))*(?:\s+et\s+seq\.?)?)',
            'statute_general': r'(?P<volume_statute>\d+)\s+(?P<reporter_statute>[A-Z]{2,6}\.?(?:\s*[A-Z]{2,6}\.?)*)\s*§\s*(?P<section_statute>\d+(?:\.\d+)*(?:\([a-z0-9]\))?(?:\s+et\s+seq\.?)?)',
            'cfr_regulation': r'(?P<title_cfr>\d+)\s+C\.?F\.?R\.?\s*§?\s*(?P<section_cfr>\d+(?:\.\d+)*(?:-\d+(?:\.\d+)*)?)',
            'constitution': r'U\.?S\.?\s*Const\.?\s*(?:art\.?\s*(?P<article_roman>[IVXLCDM]+)(?:,?\s*§\s*(?P<section_const>\d+))?|amend\.?\s*(?P<amendment_roman>[IVXLCDM]+))',
            'loose_case_name_year': r'(?P<case_name_loose>[A-Z][A-Za-z\s&.,\'"-]+(?:\s+v\.?|\s+vs\.?)\s+[A-Z][A-Za-z\s&.,\'"-]+)(?:,\s*(?:\d+\s+[A-Z]{1,5}\.?\s*\d+)?)?\s*\((?P<year_loose>\d{4})\)',
            'law_review': r'(?P<author_lr>[A-Za-z\s.,]+),\s*(?P<title_lr>“.+?”|‘.+?’|[A-Z][A-Za-z\s:]+),\s*(?P<volume_lr>\d+)\s+(?P<journal_lr>[A-Z][A-Za-z\s&.]+)\s+(?P<page_lr>\d+)\s*\((?P<year_lr>\d{4})\)'
        }
        self.citation_patterns: Dict[str, str] = self.config.get('citation_patterns', default_patterns)
        self.logger.debug(f"Initialized {len(self.citation_patterns)} citation regex patterns.")

    def _build_classification_prompt_template(self) -> str:
        """Builds the detailed prompt for LLM-based citation classification."""
        # This template is crucial for getting good structured output from the LLM.
        return f"""
        TASK: Analyze and classify the provided legal citations found in the DOCUMENT TEXT EXCERPT.

        CITATION TYPES TO IDENTIFY:
        - Case: Court decisions (e.g., "Marbury v. Madison, 5 U.S. 137 (1803)")
        - Statute: Laws enacted by legislatures (e.g., "42 U.S.C. § 1983", "Cal. Penal Code § 187")
        - Regulation: Rules by executive agencies (e.g., "29 C.F.R. § 1604.11")
        - Constitution: Constitutional articles or amendments (e.g., "U.S. Const. amend. IV")
        - LawReview: Articles in legal journals (e.g., "John Doe, The Law of X, 110 Harv. L. Rev. 1 (1997)")
        - Other: If none of the above, classify as "Other" and describe.

        CITATION ROLE IN DOCUMENT (assess based on surrounding text):
        - Supporting: Upholds or strengthens an argument made in the document.
        - Rebutting: Contradicts or weakens an argument.
        - Distinguishing: Explains why a cited authority is different or not applicable.
        - Background: Provides general legal context or history.
        - Neutral: Merely mentions or lists a citation without clear argumentative role.

        LEGAL CONTEXT FOR THIS DOCUMENT (if provided):
        {{legal_context}}

        DOCUMENT TEXT EXCERPT (citations are within this text):
        ---
        {{document_content_excerpt}}
        ---

        RAW CITATIONS EXTRACTED (your task is to analyze these):
        ---
        {{citations_json_list}}
        ---

        INSTRUCTIONS:
        For each raw citation text in the list above:
        1.  Determine its `citation_type` from the list provided.
        2.  Parse its key `parsed_components` (e.g., for a case: {{ "case_name": "...", "volume": "...", "reporter": "...", "page": "...", "year": "..." }}; for a statute: {{ "title_code": "...", "code_name": "...", "section": "..." }}). Be precise.
        3.  Determine its `role_in_document` based on how it's used in the DOCUMENT TEXT EXCERPT.
        4.  Estimate `significance_score` (0.0 to 1.0) for its importance to the arguments in the excerpt.
        5.  Assess `is_valid_format` (boolean) based on standard legal citation formats. If not, add `validation_notes`.
        6.  Provide your `llm_confidence` (float, 0.0-1.0) for this entire analysis of the citation.
        7.  Include the original `raw_text` of the citation.

        OUTPUT FORMAT: Return a JSON object with a single key "analyzed_citations", which is a list of JSON objects, each representing one analyzed citation.
        Example for one citation object:
        {{
            "raw_text": "Original citation string...",
            "citation_type": "Case",
            "parsed_components": {{ "case_name": "...", "volume": "...", ... }},
            "role_in_document": "Supporting",
            "significance_score": 0.8,
            "is_valid_format": true,
            "validation_notes": null,
            "llm_confidence": 0.9
        }}
        Only include citations where your `llm_confidence` is >= {self.min_llm_classification_confidence}.
        If a raw citation cannot be confidently analyzed or is malformed, you may omit it or set `is_valid_format` to false with notes.
        """

    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = metadata.get("document_id", f"cite_doc_{uuid.uuid4().hex[:8]}")
        self.logger.info(f"Starting citation analysis for doc '{doc_id}'.")
        start_time = datetime.now(timezone.utc)

        document_content = task_data.get('document_content', '')
        legal_context = task_data.get('legal_context', {}) # Can be a summary of the case or legal issue

        output = CitationAnalysisOutput(document_id=doc_id)

        if not document_content:
            self.logger.warning(f"No document content for citation analysis (doc '{doc_id}').")
            output.errors.append("No document content provided.")
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
            return output.to_dict()
        
        try:
            # 1. Extract raw citations using regex
            raw_citations_from_regex = self._extract_citations_with_patterns(document_content)
            self.logger.info(f"Regex extracted {len(raw_citations_from_regex)} raw citations from doc '{doc_id}'.")

            if not raw_citations_from_regex:
                output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
                output.overall_confidence = 0.95 # High confidence that no citations exist
                return output.to_dict()

            # Prepare citations for LLM (convert regex output to list of CitationDetails)
            citations_for_llm_input: List[CitationDetails] = []
            for raw_cite_dict in raw_citations_from_regex:
                if raw_cite_dict['confidence'] >= self.min_regex_confidence_for_llm: # Filter low-confidence regex matches
                    citations_for_llm_input.append(CitationDetails(
                        raw_text=raw_cite_dict['raw_text'],
                        pattern_name=raw_cite_dict['pattern_name'],
                        start_pos=raw_cite_dict['start_pos'],
                        end_pos=raw_cite_dict['end_pos'],
                        parsed_components=self._parse_components_from_regex_match(raw_cite_dict['pattern_name'], raw_cite_dict['match_obj']), # Pass full match_obj
                        citation_type=raw_cite_dict['preliminary_type'],
                        is_valid_format=True, # Assume valid if regex matched well
                        context_snippet=self._extract_context_snippet(document_content, raw_cite_dict['start_pos'], raw_cite_dict['end_pos'])
                    ))
            
            analyzed_citations: List[CitationDetails] = citations_for_llm_input # Default if LLM fails or disabled
            model_used = "regex_only"

            # 2. LLM Classification and Enrichment
            if self.enable_llm_classification and self.llm_manager and citations_for_llm_input:
                try:
                    text_excerpt_for_llm = document_content if len(document_content) <= self.max_text_for_llm else document_content[:self.max_text_for_llm]
                    if len(document_content) > self.max_text_for_llm:
                         self.logger.warning(f"Document content truncated to {self.max_text_for_llm} chars for LLM citation analysis on doc '{doc_id}'.")

                    # Batch citations for LLM if too many
                    batched_llm_input = [citations_for_llm_input[i:i + self.max_citations_for_llm_batch] 
                                         for i in range(0, len(citations_for_llm_input), self.max_citations_for_llm_batch)]
                    
                    llm_results_combined: List[CitationDetails] = []
                    for batch_idx, batch_of_citations in enumerate(batched_llm_input):
                        self.logger.debug(f"Processing LLM citation batch {batch_idx+1}/{len(batched_llm_input)} for doc '{doc_id}'.")
                        complexity = self._assess_citation_complexity(batch_of_citations, text_excerpt_for_llm)
                        
                        # Model selection
                        llm_model_to_use = self.llm_manager.primary_config.model
                        llm_provider_to_use = self.llm_manager.primary_config.provider
                        if self.model_switcher:
                            suggested_model_name = self.model_switcher.suggest_model_for_task("citation_analysis", complexity)
                            if suggested_model_name: 
                                # Assuming model_switcher can also suggest provider or it's inferred
                                llm_model_to_use = suggested_model_name 
                                # Find provider for this model if not primary (complex, needs model registry)

                        model_used = f"{llm_provider_to_use.value}/{llm_model_to_use}" # Store model used for this batch

                        llm_classified_batch = await self._classify_citations_batch_with_llm(
                            text_excerpt_for_llm, batch_of_citations, legal_context, llm_model_to_use, llm_provider_to_use
                        )
                        llm_results_combined.extend(llm_classified_batch)
                    
                    analyzed_citations = self._merge_llm_results_with_originals(citations_for_llm_input, llm_results_combined)

                except AgentProcessingError as ape: # Catch errors from _classify_citations_batch_with_llm
                    self.logger.error(f"AgentProcessingError during LLM citation classification for doc '{doc_id}'. Using regex-based results.", exception=ape)
                    output.errors.append(f"LLM classification error: {str(ape)}")
                except Exception as e: # Catch other unexpected errors
                    self.logger.error(f"Unexpected error during LLM citation classification for doc '{doc_id}'. Using regex-based results.", exception=e)
                    output.errors.append(f"Unexpected LLM error: {str(e)}")
            
            output.citations_found = analyzed_citations
            output.model_used_for_llm = model_used

            # 3. Generate final analysis outputs
            if output.citations_found:
                output.citation_statistics = self._generate_citation_statistics(output.citations_found)
                output.key_cited_authorities = self._summarize_key_authorities(output.citations_found)
                output.quality_assessment_summary = self._assess_overall_citation_quality(output.citations_found)
                output.overall_confidence = self._calculate_overall_analysis_confidence(output.citations_found)
            else: # No valid citations after all processing
                output.overall_confidence = 0.90 # Confident no processable citations

            self.analysis_run_stats['total_docs_processed'] += 1
            self.analysis_run_stats['total_citations_analyzed'] += len(output.citations_found)

            self.logger.info(f"Citation analysis completed for doc '{doc_id}'.", 
                            parameters={'citations_final': len(output.citations_found), 'overall_conf': output.overall_confidence})

        except Exception as e: # Catch-all for the main try block
            self.logger.error(f"Critical error during citation analysis task for doc '{doc_id}'.", exception=e, exc_info=True)
            output.errors.append(f"Critical task error: {str(e)}")
            output.overall_confidence = 0.05
        
        finally:
            output.processing_time_sec = round((datetime.now(timezone.utc) - start_time).total_seconds(), 3)
        
        return output.to_dict()

    def _extract_citations_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extracts citations using regex, returns list of dicts with full match object."""
        raw_citations: List[Dict[str, Any]] = []
        current_matches_on_iteration = 0
        for pattern_name, regex_str in self.citation_patterns.items():
            try:
                for match_obj in re.finditer(regex_str, text, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                    current_matches_on_iteration +=1
                    raw_citations.append({
                        'raw_text': match_obj.group(0).strip(),
                        'pattern_name': pattern_name,
                        'start_pos': match_obj.start(),
                        'end_pos': match_obj.end(),
                        'match_obj': match_obj, # Store the match object for later parsing
                        'preliminary_type': self._classify_type_by_pattern_name(pattern_name),
                        'confidence': self._calculate_regex_match_confidence(pattern_name, match_obj)
                    })
            except re.error as re_err:
                self.logger.warning(f"Regex error with pattern '{pattern_name}': {re_err}")
        
        self.logger.debug(f"Regex initially found {current_matches_on_iteration} matches. Deduplicating...")
        # Deduplicate based on start_pos and raw_text, prioritizing more specific patterns
        return self._deduplicate_raw_citations(raw_citations)

    def _deduplicate_raw_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not citations: return []
        # Sort by start position, then by confidence (desc), then pattern name length (desc, heuristic for specificity)
        citations.sort(key=lambda c: (c['start_pos'], -c['confidence'], -len(c['pattern_name'])))
        
        deduplicated: List[Dict[str, Any]] = []
        last_added_citation_end_pos = -1
        
        for current_cite in citations:
            # If current citation starts at or after the last one ended, it's definitely new
            if current_cite['start_pos'] >= last_added_citation_end_pos:
                deduplicated.append(current_cite)
                last_added_citation_end_pos = current_cite['end_pos']
            else: # Overlap with the last added citation
                previous_cite = deduplicated[-1]
                # If current cite is "better" (more confident or more specific pattern)
                # AND it covers more or equal ground than the previous one it overlaps with.
                # This is a common scenario where a general pattern matches, then a more specific one.
                is_current_better = (current_cite['confidence'] > previous_cite['confidence']) or \
                                    (current_cite['confidence'] == previous_cite['confidence'] and \
                                     len(current_cite['pattern_name']) > len(previous_cite['pattern_name']))
                
                if is_current_better and current_cite['end_pos'] >= previous_cite['end_pos']:
                    # Replace the previous less specific/confident match if current is better and covers its span
                    self.logger.trace(f"Deduplication: Replacing '{previous_cite['raw_text']}' with '{current_cite['raw_text']}' due to better match.")
                    deduplicated[-1] = current_cite
                    last_added_citation_end_pos = current_cite['end_pos']
                # Else, the previous match was kept, or the overlap is partial and both might be valid distinct (less common for citations)
        
        self.logger.debug(f"Deduplicated raw citations: {len(citations)} -> {len(deduplicated)}.")
        return deduplicated

    def _classify_type_by_pattern_name(self, pattern_name: str) -> str:
        """Infers citation type from the regex pattern name."""
        name_lower = pattern_name.lower()
        if 'case' in name_lower: return 'Case'
        if 'statute' in name_lower or 'usc' in name_lower: return 'Statute'
        if 'regulation' in name_lower or 'cfr' in name_lower: return 'Regulation'
        if 'constitution' in name_lower: return 'Constitution'
        if 'law_review' in name_lower or 'journal' in name_lower : return 'LawReview'
        return 'Other'

    def _calculate_regex_match_confidence(self, pattern_name: str, match_obj: re.Match) -> float:
        """Calculates initial confidence for a regex match."""
        base_confidence = 0.5
        specificity_boost = { # Boosts for more specific pattern types
            'case_standard': 0.3, 'statute_usc': 0.25, 'cfr_regulation': 0.25, 
            'constitution': 0.3, 'law_review': 0.2
        }
        base_confidence += specificity_boost.get(pattern_name, 0.0)
        
        # Boost if all expected groups in a pattern are filled (heuristic for a good match)
        # This requires knowing expected groups per pattern, complex to generalize here simply.
        # For now, let's assume if more groups are captured, it's better.
        if len(match_obj.groups()) > 2: base_confidence += 0.1
        if len(match_obj.group(0)) > 30 : base_confidence += 0.05 # Longer matches often more complete
        
        return round(min(0.95, base_confidence), 3)

    def _parse_components_from_regex_match(self, pattern_name: str, match_obj: re.Match) -> Dict[str, Any]:
        """Parses named groups from a regex match into structured components."""
        components = {}
        # Named groups are directly available in match_obj.groupdict()
        # This relies on patterns having named capture groups (e.g., ?P<case_name>...)
        try:
            components = {k: v.strip() if v else None for k, v in match_obj.groupdict().items() if v is not None}
        except AttributeError: # Should not happen if match_obj is a valid re.Match
             self.logger.warning(f"Match object for pattern '{pattern_name}' does not support groupdict. Raw groups: {match_obj.groups()}")
             # Fallback to numbered groups if no named groups and pattern is known
             # This part needs specific logic per pattern if not using named groups consistently.
             # For now, groupdict is preferred.
        return components

    def _extract_context_snippet(self, full_text: str, start: int, end: int, window: int = 150) -> str:
        """Extracts a window of text around the citation."""
        context_start = max(0, start - window)
        context_end = min(len(full_text), end + window)
        return full_text[context_start:context_end].strip().replace("\n", " ")
        
    async def _classify_citations_batch_with_llm(
        self, document_excerpt: str, batch_citations: List[CitationDetails],
        legal_context: Optional[Dict[str, Any]], model_name: str, provider: LLMProviderEnum
    ) -> List[CitationDetails]:
        """Classifies a batch of citations using LLM."""
        if not self.llm_manager: return batch_citations # Should not happen if enable_llm_classification is true
        
        # Prepare list of raw citation texts for the prompt
        raw_texts_for_prompt = [{"raw_text": c.raw_text, "initial_type": c.citation_type, "id_ref": c.citation_id} for c in batch_citations]
        citations_json_list = json.dumps(raw_texts_for_prompt, indent=2)
        
        legal_context_str = json.dumps(legal_context, indent=2) if legal_context else "No specific legal context provided for this document."

        prompt = self.classification_prompt_template.format(
            legal_context=legal_context_str,
            document_content_excerpt=document_excerpt, # Already excerpted
            citations_json_list=citations_json_list,
            # min_confidence is part of the prompt template string already
        )
        
        try:
            llm_response = await self.llm_manager.complete(
                prompt=prompt, model=model_name, provider=provider,
                temperature=0.1, max_tokens=max(1000, len(batch_citations) * 150) # Dynamic max_tokens
            )
            # Parse response to get list of dicts, each corresponding to an analyzed citation
            parsed_llm_results = self._parse_llm_classification_response(llm_response.content)
            
            # Map LLM results back to the original CitationDetails objects in the batch
            # This assumes LLM returns items in same order or includes 'raw_text' or an 'id_ref' to match.
            # The prompt asks for raw_text to be included, which helps.
            
            updated_batch_citations: List[CitationDetails] = []
            original_citations_map = {c.raw_text: c for c in batch_citations} # Simple map by raw_text

            for llm_result_dict in parsed_llm_results:
                raw_text_from_llm = llm_result_dict.get("raw_text")
                original_citation_obj = original_citations_map.get(raw_text_from_llm)

                if original_citation_obj:
                    original_citation_obj.citation_type = llm_result_dict.get("citation_type", original_citation_obj.citation_type)
                    original_citation_obj.parsed_components.update(llm_result_dict.get("parsed_components", {})) # Merge components
                    original_citation_obj.role_in_document = llm_result_dict.get("role_in_document", original_citation_obj.role_in_document)
                    original_citation_obj.significance_score = float(llm_result_dict.get("significance_score", original_citation_obj.significance_score))
                    original_citation_obj.is_valid_format = bool(llm_result_dict.get("is_valid_format", original_citation_obj.is_valid_format))
                    original_citation_obj.validation_notes = llm_result_dict.get("validation_notes")
                    original_citation_obj.llm_confidence = float(llm_result_dict.get("llm_confidence", 0.0))
                    updated_batch_citations.append(original_citation_obj)
                else: # LLM might have slightly altered raw_text or found a new one (less likely with this prompt)
                    self.logger.warning(f"LLM result for raw_text '{raw_text_from_llm}' not found in original batch. Creating new entry.")
                    # Create a new CitationDetails if LLM found something not in raw regex matches (unlikely with this prompt setup)
                    # For now, we focus on enriching existing regex matches.
            
            self.logger.debug(f"LLM classified/enriched {len(updated_batch_citations)} citations in batch.")
            return updated_batch_citations

        except LLMProviderError as e:
            self.logger.error(f"LLMProviderError during citation batch classification using {model_name}.", exception=e)
            raise AgentProcessingError("LLM batch classification failed.", underlying_exception=e) from e # Propagate
        except Exception as e:
            self.logger.error(f"Unexpected error during LLM citation batch classification using {model_name}.", exception=e)
            raise AgentProcessingError("Unexpected error in LLM batch processing.", underlying_exception=e) from e

    def _parse_llm_classification_response(self, response_content: str) -> List[Dict[str, Any]]:
        """Parses the LLM's JSON response containing a list of analyzed citations."""
        try:
            # Robust JSON extraction (as used in ViolationDetector)
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_content, re.DOTALL | re.IGNORECASE)
            json_str = ""
            if json_match:
                json_str = json_match.group(1)
            else: # Fallback: try to find the main JSON object part
                first_brace = response_content.find('{')
                last_brace = response_content.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    json_str = response_content[first_brace : last_brace+1]
                else:
                    self.logger.warning(f"Could not reliably find JSON object in LLM classification response. Snippet: {response_content[:200]}")
                    return []
            
            parsed_data = json.loads(json_str)
            # Expecting: {"analyzed_citations": [...]}
            if isinstance(parsed_data, dict) and "analyzed_citations" in parsed_data and isinstance(parsed_data["analyzed_citations"], list):
                return parsed_data["analyzed_citations"]
            else:
                self.logger.warning(f"LLM classification response JSON does not match expected structure. Expected 'analyzed_citations' list. Got: {type(parsed_data)}")
                return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM classification response. JSON string was: '{json_str if 'json_str' in locals() else response_content[:200]}'", exception=e)
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing LLM classification response.", exception=e)
            return []

    def _merge_llm_results_with_originals(self, original_citations: List[CitationDetails], llm_analyzed_citations: List[CitationDetails]) -> List[CitationDetails]:
        """Merges LLM analyzed data back into original list, preferring LLM for classified fields."""
        # This method assumes `llm_analyzed_citations` is already the enriched version of originals.
        # The enrichment happened within `_classify_citations_batch_with_llm`.
        # If `llm_analyzed_citations` could contain *new* citations not in originals (less likely with this prompt),
        # then a more complex merge would be needed.
        # For now, this method might just filter based on final confidence.
        
        final_list: List[CitationDetails] = []
        for cite in llm_analyzed_citations: # Iterate through what LLM returned (which should be enriched originals)
            if cite.llm_confidence is not None and cite.llm_confidence >= self.min_llm_classification_confidence:
                final_list.append(cite)
            elif cite.llm_confidence is None and cite.confidence >= self.min_regex_confidence_for_llm: # Regex only, no LLM result for it
                final_list.append(cite)
        
        self.logger.debug(f"Merged LLM results: {len(original_citations)} original -> {len(llm_analyzed_citations)} from LLM processing -> {len(final_list)} final after confidence filter.")
        return final_list

    def _generate_citation_statistics(self, citations: List[CitationDetails]) -> Dict[str, Any]:
        stats = {"total_citations": len(citations), "types_count": defaultdict(int), "roles_count": defaultdict(int)}
        if not citations: return stats
        for c in citations:
            stats["types_count"][c.citation_type] += 1
            stats["roles_count"][c.role_in_document] += 1
        stats["types_count"] = dict(stats["types_count"]) # Convert defaultdict
        stats["roles_count"] = dict(stats["roles_count"])
        return stats

    def _summarize_key_authorities(self, citations: List[CitationDetails]) -> List[Dict[str, Any]]:
        if not citations: return []
        authority_counts = defaultdict(int)
        authority_details: Dict[str, Dict] = {}

        for c in citations:
            # Create a canonical key for the authority (e.g., case name or statute number)
            key_text = ""
            if c.citation_type == "Case" and c.parsed_components.get("case_name"):
                key_text = c.parsed_components["case_name"]
            elif c.citation_type == "Statute" and c.parsed_components.get("section_usc"): # Example for USC
                key_text = f"{c.parsed_components.get('title_usc')} U.S.C. § {c.parsed_components['section_usc']}"
            elif c.raw_text: # Fallback
                key_text = c.raw_text.split(',')[0].strip() # First part of raw text
            
            if key_text:
                authority_counts[key_text] += 1
                if key_text not in authority_details:
                     authority_details[key_text] = {'text': key_text, 'type': c.citation_type, 'roles': set()}
                authority_details[key_text]['roles'].add(c.role_in_document)


        sorted_authorities = sorted(authority_counts.items(), key=lambda x: x[1], reverse=True)
        
        summary_list = []
        for text, count in sorted_authorities[:10]: # Top 10
            details = authority_details.get(text, {})
            summary_list.append({
                "authority_text": text,
                "type": details.get('type', 'unknown'),
                "count": count,
                "roles_observed": sorted(list(details.get('roles', set())))
            })
        return summary_list

    def _assess_overall_citation_quality(self, citations: List[CitationDetails]) -> Dict[str, Any]:
        if not citations: return {"overall_quality": "N/A - No citations", "issues": []}
        
        valid_format_count = sum(1 for c in citations if c.is_valid_format)
        high_significance_count = sum(1 for c in citations if c.significance_score > 0.7)
        
        quality_score = 0.0
        if citations:
            quality_score = (valid_format_count / len(citations)) * 0.6 + (high_significance_count / len(citations)) * 0.4
        
        quality_label = "poor"
        if quality_score > 0.8: quality_label = "good"
        elif quality_score > 0.6: quality_label = "fair"
        
        issues = []
        if valid_format_count < len(citations):
            issues.append(f"{len(citations) - valid_format_count} citations may have formatting issues.")
        
        return {
            "overall_quality_score": round(quality_score, 3),
            "quality_label": quality_label,
            "valid_format_percentage": round((valid_format_count / len(citations)) * 100, 1) if citations else 0,
            "high_significance_percentage": round((high_significance_count / len(citations)) * 100, 1) if citations else 0,
            "potential_issues": issues
        }

    def _assess_citation_complexity(self, citations: List[CitationDetails], document_content: str) -> TaskComplexity:
        """Assess complexity for model switching."""
        if len(citations) > 50 or len(document_content) > 10000: return TaskComplexity.HIGH
        if len(citations) > 15 or any(c.citation_type == 'Other' for c in citations): return TaskComplexity.MEDIUM
        return TaskComplexity.LOW

    def _calculate_overall_analysis_confidence(self, citations: List[CitationDetails]) -> float:
        if not citations: return 0.0 # Or higher if confident none exist
        
        # Average of LLM confidence if available, else regex confidence.
        confidences = []
        for c in citations:
            if c.llm_confidence is not None:
                confidences.append(c.llm_confidence)
            elif c.confidence: # Regex confidence (stored in CitationDetails.confidence from raw match)
                confidences.append(c.confidence) 
        
        return round(sum(confidences) / len(confidences), 3) if confidences else 0.3 # Low default if no confidences

    def _update_internal_analysis_stats(self, result: CitationAnalysisOutput): # Output, not Result
        self.analysis_run_stats["total_docs_analyzed"] += 1
        self.analysis_run_stats["total_citations_found_final"] += len(result.citations_found)
        # More detailed stats can be added here based on result fields.

    async def health_check(self) -> Dict[str, Any]:
        status = await super().health_check() if hasattr(super(), 'health_check') else {"status": "healthy", "checks": []}
        status['agent_name'] = self.name
        status['config_summary'] = self.get_config_summary_params()
        status['dependencies_status'] = {
            'llm_manager': 'available' if self.llm_manager else 'unavailable',
            'model_switcher': 'available' if self.model_switcher else 'unavailable'
        }
        status['patterns_loaded'] = len(self.citation_patterns)
        status['cumulative_stats_this_session'] = dict(self.analysis_run_stats)

        if self.enable_llm_classification and not self.llm_manager:
            status['status'] = 'degraded'
            status['reason'] = 'LLM classification enabled but LLMManager is unavailable.'
        return status