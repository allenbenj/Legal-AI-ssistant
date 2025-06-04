"""
LegalAnalysisAgent - IRAC analysis with ontology alignment and contradiction detection.

Performs comprehensive legal analysis using the IRAC framework (Issue, Rule, Application, Conclusion)
enhanced with contradiction detection, causal chain analysis, and legal reasoning validation.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent
from ..utils.ontology import LegalEntityType, get_entity_types_for_prompt
from ..core.llm_providers import LLMProviderManager
from ..core.model_switcher import ModelSwitcher, TaskComplexity


@dataclass
class LegalAnalysisResult:
    """Results from comprehensive legal analysis."""
    irac_summary: Dict[str, str]
    contradictions: List[Dict[str, Any]]
    causal_chains: List[Dict[str, Any]]
    legal_concepts: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    model_used: str
    analysis_depth: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "irac_summary": self.irac_summary,
            "contradictions": self.contradictions,
            "causal_chains": self.causal_chains,
            "legal_concepts": self.legal_concepts,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "analysis_depth": self.analysis_depth,
            "analyzed_at": datetime.now().isoformat()
        }


class LegalAnalysisAgent(BaseAgent):
    """
    Advanced legal analysis agent using IRAC framework with ontology alignment.
    
    Features:
    - IRAC (Issue, Rule, Application, Conclusion) analysis
    - Contradiction detection (conflicting claims, rebuttals)
    - Causal chain analysis (incident → violation → sanction)
    - Legal concept identification using ontology
    - Multi-complexity analysis with appropriate model selection
    - Validation and quality scoring
    """
    
    def __init__(self, llm_manager: LLMProviderManager, model_switcher: ModelSwitcher):
        super().__init__("LegalAnalysisAgent")
        self.llm_manager = llm_manager
        self.model_switcher = model_switcher
        self.logger = logging.getLogger(__name__)
        
        # Analysis configuration
        self.min_confidence_threshold = 0.7
        self.max_analysis_length = 8000
        self.include_precedent_analysis = True
        self.detect_contradictions = True
        self.analyze_causal_chains = True
        
        # Enhanced IRAC analysis prompt
        self.analysis_prompt_template = """Perform comprehensive legal analysis using the IRAC framework and ontology-aligned concepts.

LEGAL ONTOLOGY CONCEPTS:
{ontology_hints}

ANALYSIS FRAMEWORK:
Apply IRAC analysis (Issue, Rule, Application, Conclusion) with the following enhancements:

1. ISSUE IDENTIFICATION:
   - Identify all legal issues present in the document
   - Classify issues by type (constitutional, statutory, procedural, etc.)
   - Assess issue complexity and jurisdiction

2. RULE ANALYSIS:
   - Identify applicable legal rules, statutes, and precedents
   - Analyze rule hierarchy and conflicts
   - Note any rule interpretations or exceptions

3. APPLICATION:
   - Apply rules to the specific facts of the case
   - Analyze how facts satisfy or fail to satisfy legal requirements
   - Consider alternative interpretations and arguments

4. CONCLUSION:
   - Provide legal conclusions based on the analysis
   - Assess strength of legal positions
   - Identify potential outcomes and risks

ENHANCED ANALYSIS REQUIREMENTS:

CONTRADICTION DETECTION:
- Identify conflicting claims or statements
- Find contradictory evidence or testimony
- Note rebuttals and counter-arguments
- Assess impact of contradictions on case strength

CAUSAL CHAIN ANALYSIS:
- Map incident → violation → consequence sequences
- Identify cause-and-effect relationships
- Analyze proximate and but-for causation
- Connect violations to potential sanctions

DOCUMENT CONTENT:
{document_content}

SEMANTIC CONTEXT:
{semantic_context}

STRUCTURAL ANALYSIS:
{structural_context}

EXTRACTED ENTITIES:
{entities_context}

Return analysis in structured JSON format:
{{
    "irac_summary": {{
        "issues": ["List of legal issues identified"],
        "rules": ["Applicable legal rules and authorities"],
        "application": "Analysis of how rules apply to facts",
        "conclusion": "Legal conclusions and likely outcomes"
    }},
    "contradictions": [
        {{
            "type": "conflicting_claims|contradictory_evidence|rebuttal",
            "description": "Description of the contradiction",
            "sources": ["Source 1", "Source 2"],
            "impact": "Assessment of impact on case",
            "confidence": 0.8
        }}
    ],
    "causal_chains": [
        {{
            "sequence": ["incident", "violation", "consequence"],
            "description": "Description of causal relationship",
            "strength": "strong|moderate|weak",
            "legal_basis": "Legal theory supporting causation",
            "confidence": 0.9
        }}
    ],
    "legal_concepts": [
        {{
            "concept": "Legal concept name",
            "type": "claim|rule|evidence|violation|sanction",
            "description": "Detailed description",
            "relevance": "high|medium|low",
            "confidence": 0.8
        }}
    ],
    "overall_confidence": 0.85,
    "analysis_notes": "Additional observations and recommendations"
}}

Ensure thorough analysis with high confidence scores (≥{min_confidence}). Focus on legal accuracy and practical implications."""
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "avg_confidence": 0.0,
            "avg_issues_found": 0.0,
            "avg_contradictions": 0.0,
            "avg_causal_chains": 0.0,
            "processing_time_avg": 0.0
        }

    async def analyze_legal_document(
        self,
        document_content: str,
        semantic_context: str = "",
        structural_context: str = "",
        entities_context: List[Dict[str, Any]] = None,
        document_metadata: Dict[str, Any] = None
    ) -> LegalAnalysisResult:
        """
        Main legal analysis method using IRAC framework.
        
        Args:
            document_content: Primary document text to analyze
            semantic_context: Semantic analysis results for context
            structural_context: Structural analysis results
            entities_context: Extracted entities for reference
            document_metadata: Document metadata for context
            
        Returns:
            LegalAnalysisResult with comprehensive legal analysis
        """
        start_time = datetime.now()
        
        try:
            # Assess analysis complexity and select appropriate model
            complexity = self._assess_analysis_complexity(
                document_content, semantic_context, structural_context
            )
            model_config = await self.model_switcher.get_optimal_model(complexity)
            
            self.logger.info(f"Starting legal analysis with {model_config['model']} for complexity {complexity}")
            
            # Prepare analysis context
            ontology_hints = self._build_ontology_hints()
            entities_json = json.dumps(entities_context[:10], indent=2) if entities_context else "None available"
            
            # Build comprehensive analysis prompt
            prompt = self.analysis_prompt_template.format(
                ontology_hints=ontology_hints,
                document_content=self._trim_content(document_content, 4000),
                semantic_context=self._trim_content(semantic_context, 1500),
                structural_context=self._trim_content(structural_context, 1500),
                entities_context=entities_json,
                min_confidence=self.min_confidence_threshold
            )
            
            # Perform analysis with enhanced model parameters
            response = await self.llm_manager.query(
                prompt=prompt,
                model=model_config['model'],
                provider=model_config['provider'],
                temperature=0.2,  # Slightly higher for creative legal reasoning
                max_tokens=4000
            )
            
            # Parse and validate analysis results
            analysis_data = self._parse_analysis_response(response.content)
            
            # Calculate metrics and create result
            confidence_score = analysis_data.get('overall_confidence', 0.0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = LegalAnalysisResult(
                irac_summary=analysis_data.get('irac_summary', {}),
                contradictions=analysis_data.get('contradictions', []),
                causal_chains=analysis_data.get('causal_chains', []),
                legal_concepts=analysis_data.get('legal_concepts', []),
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_config['model'],
                analysis_depth=complexity.value
            )
            
            # Update performance statistics
            self._update_analysis_stats(result)
            
            self.logger.info(f"Legal analysis completed: "
                           f"{len(result.irac_summary.get('issues', []))} issues, "
                           f"{len(result.contradictions)} contradictions, "
                           f"{len(result.causal_chains)} causal chains, "
                           f"confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Legal analysis failed: {str(e)}")
            await self._record_error("legal_analysis_failed", {"error": str(e)})
            raise

    def _build_ontology_hints(self) -> str:
        """Build ontology hints for legal analysis guidance."""
        
        # Core legal analysis concepts
        analysis_concepts = [
            "CLAIM", "RULE", "APPLICATION", "CONCLUSION", "EVIDENCE", 
            "VIOLATION", "MISCONDUCT_INCIDENT", "SANCTION", "LEGAL_ISSUE"
        ]
        
        hints = []
        for concept in analysis_concepts:
            try:
                entity_type = getattr(LegalEntityType, concept, None)
                if entity_type:
                    hints.append(f"- {concept}: {entity_type.prompt_hint}")
                else:
                    # Fallback descriptions for missing ontology entries
                    fallback_hints = {
                        "CLAIM": "Legal assertion or demand made by a party",
                        "RULE": "Legal principle, statute, or precedent that governs",
                        "APPLICATION": "How legal rules apply to specific facts",
                        "CONCLUSION": "Legal outcome or decision based on analysis",
                        "EVIDENCE": "Facts, documents, or testimony supporting claims",
                        "VIOLATION": "Breach of legal duty or standard",
                        "MISCONDUCT_INCIDENT": "Specific instance of wrongful behavior",
                        "SANCTION": "Penalty or consequence for violation",
                        "LEGAL_ISSUE": "Legal question or matter requiring resolution"
                    }
                    if concept in fallback_hints:
                        hints.append(f"- {concept}: {fallback_hints[concept]}")
            except AttributeError:
                continue
        
        return '\n'.join(hints)

    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis data."""
        
        try:
            # Handle JSON markdown blocks
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content:
                json_content = response_content.split('```')[1].split('```')[0]
            else:
                json_content = response_content
            
            analysis_data = json.loads(json_content.strip())
            
            # Validate and normalize structure
            validated_data = {
                'irac_summary': analysis_data.get('irac_summary', {}),
                'contradictions': analysis_data.get('contradictions', []),
                'causal_chains': analysis_data.get('causal_chains', []),
                'legal_concepts': analysis_data.get('legal_concepts', []),
                'overall_confidence': float(analysis_data.get('overall_confidence', 0.0)),
                'analysis_notes': analysis_data.get('analysis_notes', '')
            }
            
            # Ensure IRAC summary has required components
            irac = validated_data['irac_summary']
            if not isinstance(irac, dict):
                irac = {}
                validated_data['irac_summary'] = irac
            
            for component in ['issues', 'rules', 'application', 'conclusion']:
                if component not in irac:
                    irac[component] = [] if component in ['issues', 'rules'] else ""
            
            # Validate contradiction and causal chain structures
            validated_data['contradictions'] = self._validate_contradictions(
                validated_data['contradictions']
            )
            validated_data['causal_chains'] = self._validate_causal_chains(
                validated_data['causal_chains']
            )
            validated_data['legal_concepts'] = self._validate_legal_concepts(
                validated_data['legal_concepts']
            )
            
            return validated_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse analysis response: {str(e)}")
            
            # Return minimal structure on parsing failure
            return {
                'irac_summary': {
                    'issues': [],
                    'rules': [],
                    'application': "Analysis parsing failed",
                    'conclusion': "Unable to complete analysis"
                },
                'contradictions': [],
                'causal_chains': [],
                'legal_concepts': [],
                'overall_confidence': 0.0,
                'analysis_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_contradictions(self, contradictions: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize contradiction data."""
        validated = []
        
        for item in contradictions:
            if not isinstance(item, dict):
                continue
                
            contradiction = {
                'type': item.get('type', 'unknown'),
                'description': item.get('description', ''),
                'sources': item.get('sources', []),
                'impact': item.get('impact', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include contradictions with meaningful content
            if contradiction['description'] and contradiction['confidence'] >= self.min_confidence_threshold:
                validated.append(contradiction)
        
        return validated

    def _validate_causal_chains(self, causal_chains: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize causal chain data."""
        validated = []
        
        for item in causal_chains:
            if not isinstance(item, dict):
                continue
                
            chain = {
                'sequence': item.get('sequence', []),
                'description': item.get('description', ''),
                'strength': item.get('strength', 'weak'),
                'legal_basis': item.get('legal_basis', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include chains with meaningful sequence and confidence
            if (len(chain['sequence']) >= 2 and 
                chain['description'] and 
                chain['confidence'] >= self.min_confidence_threshold):
                validated.append(chain)
        
        return validated

    def _validate_legal_concepts(self, legal_concepts: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize legal concept data."""
        validated = []
        
        for item in legal_concepts:
            if not isinstance(item, dict):
                continue
                
            concept = {
                'concept': item.get('concept', ''),
                'type': item.get('type', 'unknown'),
                'description': item.get('description', ''),
                'relevance': item.get('relevance', 'low'),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include concepts with meaningful content
            if (concept['concept'] and 
                concept['description'] and 
                concept['confidence'] >= self.min_confidence_threshold):
                validated.append(concept)
        
        return validated

    def _assess_analysis_complexity(
        self, 
        document_content: str, 
        semantic_context: str, 
        structural_context: str
    ) -> TaskComplexity:
        """Assess legal analysis complexity for model selection."""
        
        # Base complexity on content length
        total_content = len(document_content) + len(semantic_context) + len(structural_context)
        
        if total_content < 2000:
            complexity = TaskComplexity.SIMPLE
        elif total_content > 8000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        # Upgrade complexity for legal document indicators
        complex_indicators = [
            'constitutional', 'statute', 'precedent', 'appellant', 'appellee',
            'motion', 'brief', 'memorandum', 'contract', 'agreement',
            'violation', 'misconduct', 'sanction', 'disciplinary'
        ]
        
        content_lower = document_content.lower()
        indicator_count = sum(1 for indicator in complex_indicators if indicator in content_lower)
        
        if indicator_count >= 5:
            complexity = TaskComplexity.COMPLEX
        elif indicator_count >= 2 and complexity == TaskComplexity.SIMPLE:
            complexity = TaskComplexity.MODERATE
        
        return complexity

    def _trim_content(self, content: str, max_length: int) -> str:
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... [TRUNCATED]"

    def _update_analysis_stats(self, result: LegalAnalysisResult):
        """Update performance statistics."""
        self.analysis_stats["total_analyses"] += 1
        
        # Update rolling averages
        total = self.analysis_stats["total_analyses"]
        
        # Confidence
        old_conf = self.analysis_stats["avg_confidence"]
        self.analysis_stats["avg_confidence"] = (
            old_conf * (total - 1) + result.confidence_score
        ) / total
        
        # Issues found
        issues_count = len(result.irac_summary.get('issues', []))
        old_issues = self.analysis_stats["avg_issues_found"]
        self.analysis_stats["avg_issues_found"] = (
            old_issues * (total - 1) + issues_count
        ) / total
        
        # Contradictions
        old_contra = self.analysis_stats["avg_contradictions"]
        self.analysis_stats["avg_contradictions"] = (
            old_contra * (total - 1) + len(result.contradictions)
        ) / total
        
        # Causal chains
        old_causal = self.analysis_stats["avg_causal_chains"]
        self.analysis_stats["avg_causal_chains"] = (
            old_causal * (total - 1) + len(result.causal_chains)
        ) / total
        
        # Processing time
        old_time = self.analysis_stats["processing_time_avg"]
        self.analysis_stats["processing_time_avg"] = (
            old_time * (total - 1) + result.processing_time
        ) / total

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get current analysis performance statistics."""
        return {
            **self.analysis_stats,
            "agent_status": await self.get_health_status(),
            "configuration": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "detect_contradictions": self.detect_contradictions,
                "analyze_causal_chains": self.analyze_causal_chains,
                "include_precedent_analysis": self.include_precedent_analysis
            }
        }

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process legal analysis task (implementation of abstract method)."""
        try:
            document_content = task_data.get('document_content', '')
            semantic_context = task_data.get('semantic_context', '')
            structural_context = task_data.get('structural_context', '')
            entities_context = task_data.get('entities_context', [])
            document_metadata = task_data.get('document_metadata', {})
            
            if not document_content:
                raise ValueError("No document content provided for legal analysis")
            
            result = await self.analyze_legal_document(
                document_content=document_content,
                semantic_context=semantic_context,
                structural_context=structural_context,
                entities_context=entities_context,
                document_metadata=document_metadata
            )
            
            return {
                "status": "success",
                "result": result.to_dict(),
                "metadata": {
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "analysis_depth": result.analysis_depth,
                    "issues_identified": len(result.irac_summary.get('issues', [])),
                    "contradictions_found": len(result.contradictions),
                    "causal_chains_found": len(result.causal_chains)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"processing_time": 0.0}
            }

    def apply_feedback_adjustments(self, feedback: List[Dict[str, Any]]):
        """Apply feedback adjustments to improve analysis quality."""
        adjustments = 0
        
        for fb in feedback:
            if fb.get("agent") != "legal_analysis":
                continue
                
            feedback_type = fb.get("type")
            
            if feedback_type == "incomplete_analysis":
                self.analysis_prompt_template = self.analysis_prompt_template.replace(
                    "Ensure thorough analysis", "Ensure *comprehensive and exhaustive* analysis"
                )
                self.logger.info("Feedback applied: Emphasizing comprehensive analysis")
                adjustments += 1
                
            elif feedback_type == "inaccurate_analysis":
                self.analysis_prompt_template = self.analysis_prompt_template.replace(
                    "Focus on legal accuracy", "Focus on *strict legal accuracy* with supporting evidence"
                )
                self.logger.info("Feedback applied: Emphasizing legal accuracy with evidence")
                adjustments += 1
                
            elif feedback_type == "missed_contradictions":
                self.detect_contradictions = True
                self.analysis_prompt_template = self.analysis_prompt_template.replace(
                    "Identify conflicting claims", "Identify *ALL* conflicting claims, statements, and evidence"
                )
                self.logger.info("Feedback applied: Enhanced contradiction detection")
                adjustments += 1
                
            elif feedback_type == "weak_causal_analysis":
                self.analyze_causal_chains = True
                self.analysis_prompt_template = self.analysis_prompt_template.replace(
                    "Map incident → violation → consequence", 
                    "Map *detailed* incident → violation → consequence sequences with legal basis"
                )
                self.logger.info("Feedback applied: Enhanced causal chain analysis")
                adjustments += 1
        
        if adjustments > 0:
            self.logger.info(f"Applied {adjustments} feedback adjustments to LegalAnalysisAgent")