"""
CitationAnalysisAgent - Legal citation detection, extraction, and classification.

Provides comprehensive analysis of legal citations including detection using regex patterns,
classification by type and role (supporting/rebutting/neutral), validation, and integration
with legal reasoning analysis.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent
from ..utils.ontology import LegalEntityType
from ..core.llm_providers import LLMProviderManager
from ..core.model_switcher import ModelSwitcher, TaskComplexity


@dataclass
class CitationAnalysisResult:
    """Results from citation analysis of legal document."""
    citations_found: List[Dict[str, Any]]
    citation_statistics: Dict[str, Any]
    citation_network: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations_found": self.citations_found,
            "citation_statistics": self.citation_statistics,
            "citation_network": self.citation_network,
            "quality_assessment": self.quality_assessment,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "analyzed_at": datetime.now().isoformat()
        }


class CitationAnalysisAgent(BaseAgent):
    """
    Advanced legal citation analysis agent.
    
    Features:
    - Multi-pattern citation detection (cases, statutes, regulations)
    - Citation classification by type and argumentative role
    - Citation validation and completeness checking
    - Citation network analysis and relationship mapping
    - Integration with legal ontology for enhanced classification
    - Quality assessment of citation usage and accuracy
    """
    
    def __init__(self, llm_manager: LLMProviderManager, model_switcher: ModelSwitcher):
        super().__init__("CitationAnalysisAgent")
        self.llm_manager = llm_manager
        self.model_switcher = model_switcher
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_confidence_threshold = 0.7
        self.max_citations_per_analysis = 100
        self.validate_citations = True
        self.analyze_citation_context = True
        
        # Citation detection patterns
        self.citation_patterns = {
            # Case citations: Party v. Party, Volume Reporter Page (Year)
            'case_standard': r'([A-Z][a-zA-Z\s&]+(?:\s+v\.?\s+|\s+vs\.?\s+)[A-Z][a-zA-Z\s&]+),?\s*(\d+)\s+([A-Z]{1,4}\.?(?:\s*\d*d?)?)\s+(\d+)(?:\s*\((\d{4})\))?',
            
            # Statute citations: Title Code § Section
            'statute': r'(\d+)\s+([A-Z]{2,6}\.?(?:\s*[A-Z]{2,6}\.?)*)\s*§\s*(\d+(?:\.\d+)*(?:\([a-z]\))?)',
            
            # Regulation citations: Title CFR Part.Section
            'regulation': r'(\d+)\s+C\.?F\.?R\.?\s*§?\s*(\d+(?:\.\d+)*)',
            
            # Constitutional citations
            'constitution': r'U\.?S\.?\s*Const\.?\s*(?:art\.?\s*([IVX]+)(?:,?\s*§\s*(\d+))?|amend\.?\s*([IVX]+))',
            
            # Federal statute citations
            'usc': r'(\d+)\s+U\.?S\.?C\.?\s*§\s*(\d+(?:[a-z])?(?:\(\d+\))?)',
            
            # Loose case patterns for missed citations
            'case_loose': r'([A-Z][a-zA-Z\s&]+(?:\s+v\.?\s+|\s+vs\.?\s+)[A-Z][a-zA-Z\s&]+)(?:,\s*(?:\d+\s+[A-Z]{1,4}\.?\s*\d+)?(?:\s*\([^)]*\d{4}[^)]*\))?)?'
        }
        
        # Citation classification prompt
        self.classification_prompt_template = """Analyze and classify legal citations for their type, role, and legal significance.

CITATION ONTOLOGY CONCEPTS:
{citation_hints}

CITATION ANALYSIS REQUIREMENTS:

1. CITATION TYPE CLASSIFICATION:
   - Case law citations (federal, state, appellate levels)
   - Statutory citations (federal, state, local)
   - Regulatory citations (CFR, state regulations)
   - Constitutional provisions
   - Secondary authorities (law review, treatises)

2. ARGUMENTATIVE ROLE ANALYSIS:
   - Supporting: Citations that support the author's position
   - Rebutting: Citations that counter opposing arguments
   - Neutral: Citations providing background or context
   - Distinguishing: Citations showing factual/legal differences

3. CITATION CONTEXT EVALUATION:
   - How the citation is used in legal reasoning
   - Strength of reliance on the cited authority
   - Position in argument structure (main authority vs. supporting)
   - Currency and precedential value

4. CITATION QUALITY ASSESSMENT:
   - Completeness of citation format
   - Accuracy of legal references
   - Appropriateness for the legal argument
   - Compliance with citation standards

DOCUMENT TEXT WITH CITATIONS:
{document_content}

IDENTIFIED CITATIONS:
{citations_list}

LEGAL CONTEXT:
{legal_context}

ANALYSIS INSTRUCTIONS:
- Classify each citation by type and argumentative role
- Assess the legal significance and usage context
- Evaluate citation quality and completeness
- Identify relationships between citations
- Note any citation issues or improvements needed

Return analysis in structured JSON format:
{{
    "citations_analyzed": [
        {{
            "citation_text": "Full citation as it appears",
            "normalized_citation": "Standardized citation format",
            "type": "case|statute|regulation|constitution|secondary",
            "subtype": "federal_case|state_statute|cfr|etc",
            "argumentative_role": "supporting|rebutting|neutral|distinguishing",
            "context_usage": "How citation is used in the argument",
            "legal_significance": "high|medium|low",
            "precedential_value": "binding|persuasive|none",
            "citation_quality": {{
                "completeness": "complete|partial|minimal",
                "accuracy": "verified|likely|questionable",
                "format_compliance": "standard|non_standard|incorrect"
            }},
            "location_info": {{
                "paragraph": "Where citation appears",
                "argument_position": "primary|supporting|background"
            }},
            "confidence": 0.9
        }}
    ],
    "citation_network": {{
        "primary_authorities": ["List of main cases/statutes relied upon"],
        "supporting_authorities": ["List of supporting citations"],
        "counter_authorities": ["List of citations addressing opposing views"],
        "citation_clusters": [
            {{
                "topic": "Legal topic or issue",
                "citations": ["Related citations"],
                "relationship": "How citations relate to each other"
            }}
        ]
    }},
    "quality_assessment": {{
        "overall_citation_quality": "excellent|good|fair|poor",
        "citation_density": "high|medium|low",
        "authority_strength": "strong|moderate|weak",
        "format_consistency": "consistent|mostly_consistent|inconsistent",
        "completeness_issues": ["List of incomplete citations"],
        "recommendations": ["Suggestions for improvement"]
    }},
    "overall_confidence": 0.85,
    "analysis_notes": "Summary of citation analysis findings"
}}

Focus on accurate classification with confidence ≥{min_confidence}. Emphasize legal reasoning and citation quality."""
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "total_citations_found": 0,
            "avg_citations_per_document": 0.0,
            "avg_confidence": 0.0,
            "citation_types_seen": {},
            "processing_time_avg": 0.0,
            "pattern_match_rates": {}
        }

    async def analyze_citations(
        self,
        document_content: str,
        legal_context: Dict[str, Any] = None,
        document_metadata: Dict[str, Any] = None
    ) -> CitationAnalysisResult:
        """
        Main citation analysis method.
        
        Args:
            document_content: Document text to analyze for citations
            legal_context: Legal analysis context for enhanced classification
            document_metadata: Document metadata for context
            
        Returns:
            CitationAnalysisResult with detected and classified citations
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Extract citations using regex patterns
            extracted_citations = self._extract_citations_with_patterns(document_content)
            
            if not extracted_citations:
                self.logger.info("No citations found in document")
                return self._create_empty_result(document_content, start_time)
            
            # Step 2: Assess analysis complexity and select model
            complexity = self._assess_citation_complexity(extracted_citations, document_content)
            model_config = await self.model_switcher.get_optimal_model(complexity)
            
            self.logger.info(f"Starting citation analysis with {model_config['model']} for {len(extracted_citations)} citations")
            
            # Step 3: Classify citations using LLM
            classified_citations = await self._classify_citations_with_llm(
                document_content, extracted_citations, legal_context, model_config
            )
            
            # Step 4: Generate citation statistics and network analysis
            citation_stats = self._generate_citation_statistics(classified_citations)
            citation_network = self._analyze_citation_network(classified_citations)
            quality_assessment = self._assess_citation_quality(classified_citations)
            
            # Step 5: Calculate overall metrics
            confidence_score = self._calculate_overall_confidence(classified_citations)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = CitationAnalysisResult(
                citations_found=classified_citations,
                citation_statistics=citation_stats,
                citation_network=citation_network,
                quality_assessment=quality_assessment,
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_config['model']
            )
            
            # Update performance statistics
            self._update_analysis_stats(result)
            
            self.logger.info(f"Citation analysis completed: "
                           f"{len(classified_citations)} citations analyzed, "
                           f"confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Citation analysis failed: {str(e)}")
            await self._record_error("citation_analysis_failed", {"error": str(e)})
            raise

    def _extract_citations_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations using regex patterns."""
        citations = []
        
        for pattern_name, pattern in self.citation_patterns.items():
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    citation = {
                        'raw_text': match.group(0),
                        'pattern_type': pattern_name,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'groups': match.groups(),
                        'confidence': self._calculate_pattern_confidence(pattern_name, match.group(0))
                    }
                    
                    # Add citation type classification based on pattern
                    citation['preliminary_type'] = self._classify_by_pattern(pattern_name)
                    
                    citations.append(citation)
                    
            except re.error as e:
                self.logger.warning(f"Regex pattern '{pattern_name}' failed: {str(e)}")
                continue
        
        # Remove duplicates and sort by position
        citations = self._deduplicate_citations(citations)
        citations.sort(key=lambda x: x['start_pos'])
        
        # Limit citations if too many found
        if len(citations) > self.max_citations_per_analysis:
            self.logger.warning(f"Found {len(citations)} citations, limiting to {self.max_citations_per_analysis}")
            citations = citations[:self.max_citations_per_analysis]
        
        return citations

    def _classify_by_pattern(self, pattern_name: str) -> str:
        """Classify citation type based on regex pattern matched."""
        pattern_to_type = {
            'case_standard': 'case',
            'case_loose': 'case',
            'statute': 'statute',
            'regulation': 'regulation',
            'constitution': 'constitution',
            'usc': 'statute'
        }
        return pattern_to_type.get(pattern_name, 'unknown')

    def _calculate_pattern_confidence(self, pattern_name: str, citation_text: str) -> float:
        """Calculate confidence score based on pattern match quality."""
        base_confidence = {
            'case_standard': 0.9,
            'statute': 0.85,
            'regulation': 0.8,
            'constitution': 0.9,
            'usc': 0.85,
            'case_loose': 0.6
        }
        
        confidence = base_confidence.get(pattern_name, 0.5)
        
        # Adjust based on citation completeness
        if len(citation_text) < 10:
            confidence *= 0.7
        elif len(citation_text) > 100:
            confidence *= 0.8
        
        # Look for year information (increases confidence)
        if re.search(r'\b\d{4}\b', citation_text):
            confidence = min(1.0, confidence * 1.1)
        
        return confidence

    def _deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate citations based on text similarity."""
        deduplicated = []
        seen_texts = set()
        
        for citation in citations:
            # Normalize text for comparison
            normalized = re.sub(r'\s+', ' ', citation['raw_text'].strip().lower())
            
            # Check for duplicates
            is_duplicate = False
            for seen in seen_texts:
                if self._citations_similar(normalized, seen):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(normalized)
                deduplicated.append(citation)
        
        return deduplicated

    def _citations_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two citations are similar enough to be considered duplicates."""
        # Simple similarity check based on shared words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

    async def _classify_citations_with_llm(
        self,
        document_content: str,
        citations: List[Dict[str, Any]],
        legal_context: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Classify citations using LLM analysis."""
        
        # Prepare classification context
        citation_hints = self._build_citation_hints()
        citations_list = json.dumps([c['raw_text'] for c in citations], indent=2)
        context_json = json.dumps(legal_context, indent=2) if legal_context else "None available"
        
        # Build classification prompt
        prompt = self.classification_prompt_template.format(
            citation_hints=citation_hints,
            document_content=self._trim_content(document_content, 3000),
            citations_list=citations_list,
            legal_context=context_json,
            min_confidence=self.min_confidence_threshold
        )
        
        try:
            # Perform LLM classification
            response = await self.llm_manager.query(
                prompt=prompt,
                model=model_config['model'],
                provider=model_config['provider'],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=3000
            )
            
            # Parse LLM response
            classification_data = self._parse_classification_response(response.content)
            
            # Merge LLM classification with extracted citations
            classified_citations = self._merge_classification_results(citations, classification_data)
            
            return classified_citations
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {str(e)}")
            # Return citations with basic pattern-based classification
            return self._apply_basic_classification(citations)

    def _build_citation_hints(self) -> str:
        """Build citation hints using legal ontology."""
        citation_concepts = ["CITATION", "PRECEDENT", "RULING", "STATUTE", "REGULATION"]
        
        hints = []
        for concept in citation_concepts:
            try:
                entity_type = getattr(LegalEntityType, concept, None)
                if entity_type:
                    hints.append(f"- {concept}: {entity_type.prompt_hint}")
                else:
                    # Fallback descriptions
                    fallback_hints = {
                        "CITATION": "Reference to legal authority supporting arguments",
                        "PRECEDENT": "Prior judicial decision with binding or persuasive authority",
                        "RULING": "Judicial decision or court order",
                        "STATUTE": "Written law enacted by legislative body",
                        "REGULATION": "Administrative rule implementing statutory law"
                    }
                    if concept in fallback_hints:
                        hints.append(f"- {concept}: {fallback_hints[concept]}")
            except AttributeError:
                continue
        
        return '\n'.join(hints)

    def _parse_classification_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM classification response."""
        try:
            # Handle JSON markdown blocks
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content:
                json_content = response_content.split('```')[1].split('```')[0]
            else:
                json_content = response_content
            
            return json.loads(json_content.strip())
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse classification response: {str(e)}")
            return {"citations_analyzed": [], "citation_network": {}, "quality_assessment": {}}

    def _merge_classification_results(
        self, 
        extracted_citations: List[Dict[str, Any]], 
        classification_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Merge extracted citations with LLM classification results."""
        
        classified_citations = classification_data.get('citations_analyzed', [])
        merged_results = []
        
        # Match extracted citations with classified ones
        for extracted in extracted_citations:
            best_match = None
            best_similarity = 0.0
            
            for classified in classified_citations:
                similarity = self._calculate_citation_similarity(
                    extracted['raw_text'], 
                    classified.get('citation_text', '')
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = classified
            
            # Merge data
            merged_citation = {
                **extracted,  # Start with extracted data
                'classification_confidence': 0.0
            }
            
            if best_match and best_similarity > 0.7:
                # Update with LLM classification
                merged_citation.update({
                    'normalized_citation': best_match.get('normalized_citation', ''),
                    'type': best_match.get('type', merged_citation.get('preliminary_type', 'unknown')),
                    'subtype': best_match.get('subtype', ''),
                    'argumentative_role': best_match.get('argumentative_role', 'neutral'),
                    'context_usage': best_match.get('context_usage', ''),
                    'legal_significance': best_match.get('legal_significance', 'medium'),
                    'precedential_value': best_match.get('precedential_value', 'none'),
                    'citation_quality': best_match.get('citation_quality', {}),
                    'location_info': best_match.get('location_info', {}),
                    'classification_confidence': float(best_match.get('confidence', 0.0))
                })
            else:
                # Use basic pattern-based classification
                merged_citation.update({
                    'type': merged_citation.get('preliminary_type', 'unknown'),
                    'argumentative_role': 'neutral',
                    'legal_significance': 'medium',
                    'precedential_value': 'none'
                })
            
            merged_results.append(merged_citation)
        
        return merged_results

    def _calculate_citation_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two citation texts."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        norm1 = re.sub(r'\s+', ' ', text1.strip().lower())
        norm2 = re.sub(r'\s+', ' ', text2.strip().lower())
        
        # Calculate word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _apply_basic_classification(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply basic pattern-based classification when LLM fails."""
        for citation in citations:
            citation.update({
                'type': citation.get('preliminary_type', 'unknown'),
                'argumentative_role': 'neutral',
                'legal_significance': 'medium',
                'precedential_value': 'none',
                'classification_confidence': citation.get('confidence', 0.5)
            })
        return citations

    def _generate_citation_statistics(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical summary of citations."""
        if not citations:
            return {}
        
        # Count by type
        type_counts = {}
        role_counts = {}
        significance_counts = {}
        
        for citation in citations:
            # Type counts
            citation_type = citation.get('type', 'unknown')
            type_counts[citation_type] = type_counts.get(citation_type, 0) + 1
            
            # Role counts
            role = citation.get('argumentative_role', 'neutral')
            role_counts[role] = role_counts.get(role, 0) + 1
            
            # Significance counts
            significance = citation.get('legal_significance', 'medium')
            significance_counts[significance] = significance_counts.get(significance, 0) + 1
        
        return {
            'total_citations': len(citations),
            'citation_types': type_counts,
            'argumentative_roles': role_counts,
            'significance_levels': significance_counts,
            'avg_confidence': sum(c.get('classification_confidence', 0) for c in citations) / len(citations)
        }

    def _analyze_citation_network(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships and networks among citations."""
        primary_authorities = []
        supporting_authorities = []
        counter_authorities = []
        
        for citation in citations:
            significance = citation.get('legal_significance', 'medium')
            role = citation.get('argumentative_role', 'neutral')
            citation_text = citation.get('normalized_citation', citation.get('raw_text', ''))
            
            if significance == 'high' and role == 'supporting':
                primary_authorities.append(citation_text)
            elif role == 'supporting':
                supporting_authorities.append(citation_text)
            elif role == 'rebutting':
                counter_authorities.append(citation_text)
        
        return {
            'primary_authorities': primary_authorities[:10],  # Limit to top 10
            'supporting_authorities': supporting_authorities[:15],
            'counter_authorities': counter_authorities[:10],
            'citation_clusters': []  # Could be enhanced with topic modeling
        }

    def _assess_citation_quality(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall quality of citations in the document."""
        if not citations:
            return {'overall_citation_quality': 'poor', 'citation_density': 'low'}
        
        # Calculate quality metrics
        complete_citations = sum(1 for c in citations if c.get('citation_quality', {}).get('completeness') == 'complete')
        high_significance = sum(1 for c in citations if c.get('legal_significance') == 'high')
        
        completeness_rate = complete_citations / len(citations)
        significance_rate = high_significance / len(citations)
        
        # Determine overall quality
        if completeness_rate >= 0.8 and significance_rate >= 0.3:
            overall_quality = 'excellent'
        elif completeness_rate >= 0.6 and significance_rate >= 0.2:
            overall_quality = 'good'
        elif completeness_rate >= 0.4:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'
        
        # Determine citation density (citations per 1000 words)
        # This would need document word count - simplified here
        density = 'medium'  # Could be calculated with document length
        
        return {
            'overall_citation_quality': overall_quality,
            'citation_density': density,
            'authority_strength': 'strong' if significance_rate >= 0.3 else 'moderate' if significance_rate >= 0.1 else 'weak',
            'format_consistency': 'consistent' if completeness_rate >= 0.7 else 'inconsistent',
            'completeness_issues': [],
            'recommendations': []
        }

    def _assess_citation_complexity(self, citations: List[Dict[str, Any]], document_content: str) -> TaskComplexity:
        """Assess citation analysis complexity for model selection."""
        
        citation_count = len(citations)
        content_length = len(document_content)
        
        # Base complexity on citation count and content length
        if citation_count < 5 and content_length < 2000:
            complexity = TaskComplexity.SIMPLE
        elif citation_count > 20 or content_length > 6000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        # Look for complex citation types
        complex_patterns = sum(1 for c in citations if c['pattern_type'] in ['case_standard', 'regulation', 'constitution'])
        
        if complex_patterns >= citation_count * 0.7:  # 70% complex citations
            if complexity == TaskComplexity.SIMPLE:
                complexity = TaskComplexity.MODERATE
            elif complexity == TaskComplexity.MODERATE:
                complexity = TaskComplexity.COMPLEX
        
        return complexity

    def _calculate_overall_confidence(self, citations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for citation analysis."""
        if not citations:
            return 0.0
        
        confidences = []
        for citation in citations:
            pattern_conf = citation.get('confidence', 0.0)
            classification_conf = citation.get('classification_confidence', 0.0)
            # Weighted average: pattern matching 30%, classification 70%
            overall_conf = (pattern_conf * 0.3) + (classification_conf * 0.7)
            confidences.append(overall_conf)
        
        return sum(confidences) / len(confidences)

    def _create_empty_result(self, document_content: str, start_time: datetime) -> CitationAnalysisResult:
        """Create empty result when no citations are found."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CitationAnalysisResult(
            citations_found=[],
            citation_statistics={'total_citations': 0},
            citation_network={'primary_authorities': [], 'supporting_authorities': [], 'counter_authorities': []},
            quality_assessment={'overall_citation_quality': 'poor', 'citation_density': 'low'},
            confidence_score=1.0,  # High confidence in finding no citations
            processing_time=processing_time,
            model_used='pattern_only'
        )

    def _trim_content(self, content: str, max_length: int) -> str:
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... [TRUNCATED]"

    def _update_analysis_stats(self, result: CitationAnalysisResult):
        """Update performance statistics."""
        self.analysis_stats["total_analyses"] += 1
        
        citations_count = len(result.citations_found)
        self.analysis_stats["total_citations_found"] += citations_count
        
        # Update rolling averages
        total = self.analysis_stats["total_analyses"]
        
        # Citations per document
        old_avg_citations = self.analysis_stats["avg_citations_per_document"]
        self.analysis_stats["avg_citations_per_document"] = (
            old_avg_citations * (total - 1) + citations_count
        ) / total
        
        # Confidence
        old_conf = self.analysis_stats["avg_confidence"]
        self.analysis_stats["avg_confidence"] = (
            old_conf * (total - 1) + result.confidence_score
        ) / total
        
        # Processing time
        old_time = self.analysis_stats["processing_time_avg"]
        self.analysis_stats["processing_time_avg"] = (
            old_time * (total - 1) + result.processing_time
        ) / total
        
        # Citation types seen
        for citation in result.citations_found:
            citation_type = citation.get('type', 'unknown')
            if citation_type in self.analysis_stats["citation_types_seen"]:
                self.analysis_stats["citation_types_seen"][citation_type] += 1
            else:
                self.analysis_stats["citation_types_seen"][citation_type] = 1

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get current citation analysis performance statistics."""
        return {
            **self.analysis_stats,
            "agent_status": await self.get_health_status(),
            "configuration": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_citations_per_analysis": self.max_citations_per_analysis,
                "validate_citations": self.validate_citations,
                "analyze_citation_context": self.analyze_citation_context,
                "patterns_available": list(self.citation_patterns.keys())
            }
        }

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process citation analysis task (implementation of abstract method)."""
        try:
            document_content = task_data.get('document_content', '')
            legal_context = task_data.get('legal_context', {})
            document_metadata = task_data.get('document_metadata', {})
            
            if not document_content:
                raise ValueError("No document content provided for citation analysis")
            
            result = await self.analyze_citations(
                document_content=document_content,
                legal_context=legal_context,
                document_metadata=document_metadata
            )
            
            return {
                "status": "success",
                "result": result.to_dict(),
                "metadata": {
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "citations_found": len(result.citations_found),
                    "citation_types": result.citation_statistics.get('citation_types', {}),
                    "overall_quality": result.quality_assessment.get('overall_citation_quality', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"processing_time": 0.0}
            }

    def apply_feedback_adjustments(self, feedback: List[Dict[str, Any]]):
        """Apply feedback adjustments to improve citation analysis quality."""
        adjustments = 0
        
        for fb in feedback:
            if fb.get("agent") != "citation_analysis":
                continue
                
            feedback_type = fb.get("type")
            
            if feedback_type == "missed_citations":
                # Make citation patterns more permissive
                self.citation_patterns['case_loose'] = r'([A-Z][a-zA-Z\s&]+(?:\s+v\.?\s+|\s+vs\.?\s+)[A-Z][a-zA-Z\s&]+)(?:,\s*(?:\d+\s+[A-Z]{1,4}\.?\s*\d+)?(?:\s*\([^)]*\d{4}[^)]*\))?)?'
                self.logger.info("Feedback applied: Made citation patterns more permissive")
                adjustments += 1
                
            elif feedback_type == "incorrectly_classified_citations":
                self.classification_prompt_template = self.classification_prompt_template.replace(
                    "Classify each citation by type and argumentative role",
                    "Classify each citation *precisely* by type and argumentative role with detailed reasoning"
                )
                self.logger.info("Feedback applied: Enhanced classification precision")
                adjustments += 1
                
            elif feedback_type == "poor_quality_assessment":
                self.validate_citations = True
                self.analyze_citation_context = True
                self.logger.info("Feedback applied: Enhanced citation quality assessment")
                adjustments += 1
                
            elif feedback_type == "low_confidence_results":
                self.min_confidence_threshold = max(0.5, self.min_confidence_threshold - 0.1)
                self.logger.info(f"Feedback applied: Lowered confidence threshold to {self.min_confidence_threshold}")
                adjustments += 1
        
        if adjustments > 0:
            self.logger.info(f"Applied {adjustments} feedback adjustments to CitationAnalysisAgent")