"""
StructuralAnalysisAgent - IRAC component extraction and document structure analysis.

Extracts and analyzes the structural components of legal documents using the IRAC framework
(Issue, Rule, Application, Conclusion) with enhanced legal document structure recognition.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent
from ..utils.ontology import LegalEntityType
from ..core.llm_providers import LLMProviderManager
from ..core.model_switcher import ModelSwitcher, TaskComplexity


@dataclass
class StructuralAnalysisResult:
    """Results from structural analysis of legal document."""
    irac_components: Dict[str, Any]
    document_structure: Dict[str, Any]
    section_analysis: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    model_used: str
    structure_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "irac_components": self.irac_components,
            "document_structure": self.document_structure,
            "section_analysis": self.section_analysis,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "structure_type": self.structure_type,
            "analyzed_at": datetime.now().isoformat()
        }


class StructuralAnalysisAgent(BaseAgent):
    """
    Legal document structural analysis agent using IRAC framework.
    
    Features:
    - IRAC component extraction (Issue, Rule, Application, Conclusion)
    - Document structure recognition (headers, sections, paragraphs)
    - Legal document type classification
    - Section-by-section analysis with confidence scoring
    - Integration with legal ontology for enhanced accuracy
    """
    
    def __init__(self, llm_manager: LLMProviderManager, model_switcher: ModelSwitcher):
        super().__init__("StructuralAnalysisAgent")
        self.llm_manager = llm_manager
        self.model_switcher = model_switcher
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_confidence_threshold = 0.7
        self.max_sections_per_analysis = 20
        self.section_analysis_enabled = True
        self.detect_headers = True
        
        # IRAC structural analysis prompt
        self.structural_prompt_template = """Extract IRAC (Issue, Rule, Application, Conclusion) components and analyze document structure.

IRAC FRAMEWORK SCHEMA:
{irac_schema}

DOCUMENT STRUCTURE ANALYSIS:
Analyze the document for:
1. Legal Issues - Questions of law or fact requiring resolution
2. Legal Rules - Statutes, regulations, case law, precedents
3. Application - How rules apply to specific facts and circumstances
4. Conclusion - Legal outcomes, decisions, or recommendations

DOCUMENT TO ANALYZE:
{document_content}

ENTITY CONTEXT:
{entities_context}

SEMANTIC SUMMARY:
{semantic_summary}

ANALYSIS REQUIREMENTS:

1. IRAC COMPONENT EXTRACTION:
   Extract text passages that correspond to each IRAC component:
   - ISSUES: Legal questions, disputes, matters requiring determination
   - RULES: Applicable law, statutes, regulations, precedents, standards
   - APPLICATION: Analysis of how rules apply to facts, reasoning, arguments
   - CONCLUSION: Decisions, outcomes, recommendations, holdings

2. DOCUMENT STRUCTURE ANALYSIS:
   - Identify document type (brief, motion, opinion, contract, etc.)
   - Recognize structural elements (headers, sections, numbered paragraphs)
   - Classify document organization pattern
   - Assess completeness of IRAC framework

3. SECTION-BY-SECTION ANALYSIS:
   For each major section, identify:
   - Section purpose and function
   - IRAC component(s) contained
   - Key legal concepts
   - Relationship to other sections

Return analysis in structured JSON format:
{{
    "irac_components": {{
        "issues": [
            {{
                "text": "Exact text of the legal issue",
                "type": "constitutional|statutory|procedural|factual",
                "section": "Section where found",
                "confidence": 0.9
            }}
        ],
        "rules": [
            {{
                "text": "Statement of legal rule or authority",
                "source": "statute|case_law|regulation|precedent",
                "citation": "Legal citation if available",
                "section": "Section where found",
                "confidence": 0.85
            }}
        ],
        "application": [
            {{
                "text": "Application of rule to facts",
                "rule_reference": "Which rule is being applied",
                "facts_analyzed": "Specific facts being analyzed",
                "reasoning": "Legal reasoning employed",
                "section": "Section where found",
                "confidence": 0.8
            }}
        ],
        "conclusion": [
            {{
                "text": "Legal conclusion or outcome",
                "type": "holding|recommendation|decision|finding",
                "basis": "What the conclusion is based on",
                "section": "Section where found",
                "confidence": 0.9
            }}
        ]
    }},
    "document_structure": {{
        "document_type": "brief|motion|opinion|contract|statute|regulation",
        "organization_pattern": "chronological|topical|irac|other",
        "total_sections": 5,
        "has_headers": true,
        "numbered_paragraphs": true,
        "irac_completeness": "complete|partial|minimal",
        "structural_confidence": 0.85
    }},
    "section_analysis": [
        {{
            "section_number": 1,
            "header": "Section header or title",
            "purpose": "What this section accomplishes",
            "irac_components": ["issues", "rules"],
            "key_concepts": ["List of legal concepts"],
            "confidence": 0.8
        }}
    ],
    "overall_confidence": 0.82,
    "analysis_notes": "Additional observations about document structure"
}}

Ensure precise identification of IRAC components with confidence ≥{min_confidence}. Focus on clear structural boundaries and legal reasoning flow."""
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "avg_confidence": 0.0,
            "avg_irac_completeness": 0.0,
            "avg_sections_identified": 0.0,
            "processing_time_avg": 0.0,
            "document_types_seen": {}
        }

    async def analyze_document_structure(
        self,
        document_content: str,
        entities_context: List[Dict[str, Any]] = None,
        semantic_summary: str = "",
        document_metadata: Dict[str, Any] = None
    ) -> StructuralAnalysisResult:
        """
        Main structural analysis method.
        
        Args:
            document_content: Document text to analyze
            entities_context: Extracted entities for context
            semantic_summary: Semantic analysis results
            document_metadata: Document metadata
            
        Returns:
            StructuralAnalysisResult with IRAC components and structure analysis
        """
        start_time = datetime.now()
        
        try:
            # Assess analysis complexity and select model
            complexity = self._assess_structural_complexity(document_content)
            model_config = await self.model_switcher.get_optimal_model(complexity)
            
            self.logger.info(f"Starting structural analysis with {model_config['model']} for complexity {complexity}")
            
            # Prepare analysis context
            irac_schema = self._build_irac_schema()
            entities_json = json.dumps(entities_context[:10], indent=2) if entities_context else "None available"
            
            # Build analysis prompt
            prompt = self.structural_prompt_template.format(
                irac_schema=irac_schema,
                document_content=self._trim_content(document_content, 5000),
                entities_context=entities_json,
                semantic_summary=self._trim_content(semantic_summary, 1000),
                min_confidence=self.min_confidence_threshold
            )
            
            # Perform structural analysis
            response = await self.llm_manager.query(
                prompt=prompt,
                model=model_config['model'],
                provider=model_config['provider'],
                temperature=0.1,  # Low temperature for consistent structural identification
                max_tokens=4000
            )
            
            # Parse and validate analysis results
            analysis_data = self._parse_structural_response(response.content)
            
            # Calculate metrics
            confidence_score = analysis_data.get('overall_confidence', 0.0)
            structure_type = analysis_data.get('document_structure', {}).get('document_type', 'unknown')
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = StructuralAnalysisResult(
                irac_components=analysis_data.get('irac_components', {}),
                document_structure=analysis_data.get('document_structure', {}),
                section_analysis=analysis_data.get('section_analysis', []),
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_config['model'],
                structure_type=structure_type
            )
            
            # Update performance statistics
            self._update_analysis_stats(result)
            
            # Log analysis summary
            irac_components = result.irac_components
            self.logger.info(f"Structural analysis completed: "
                           f"{len(irac_components.get('issues', []))} issues, "
                           f"{len(irac_components.get('rules', []))} rules, "
                           f"{len(irac_components.get('application', []))} applications, "
                           f"{len(irac_components.get('conclusion', []))} conclusions, "
                           f"confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Structural analysis failed: {str(e)}")
            await self._record_error("structural_analysis_failed", {"error": str(e)})
            raise

    def _build_irac_schema(self) -> str:
        """Build IRAC schema with ontology guidance."""
        
        irac_components = {
            "LEGAL_ISSUE": "Legal questions or matters requiring determination",
            "RULE": "Legal principles, statutes, regulations, or precedents",
            "APPLICATION": "How legal rules apply to specific facts and circumstances", 
            "CONCLUSION": "Legal outcomes, decisions, holdings, or recommendations"
        }
        
        schema_lines = []
        for component, description in irac_components.items():
            try:
                # Try to get ontology hint if available
                entity_type = getattr(LegalEntityType, component, None)
                if entity_type:
                    hint = entity_type.prompt_hint
                else:
                    hint = description
                schema_lines.append(f"- {component}: {hint}")
            except AttributeError:
                schema_lines.append(f"- {component}: {description}")
        
        return '\n'.join(schema_lines)

    def _parse_structural_response(self, response_content: str) -> Dict[str, Any]:
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
                'irac_components': self._validate_irac_components(
                    analysis_data.get('irac_components', {})
                ),
                'document_structure': self._validate_document_structure(
                    analysis_data.get('document_structure', {})
                ),
                'section_analysis': self._validate_section_analysis(
                    analysis_data.get('section_analysis', [])
                ),
                'overall_confidence': float(analysis_data.get('overall_confidence', 0.0)),
                'analysis_notes': analysis_data.get('analysis_notes', '')
            }
            
            return validated_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse structural response: {str(e)}")
            
            # Return minimal structure on parsing failure
            return {
                'irac_components': {
                    'issues': [],
                    'rules': [],
                    'application': [],
                    'conclusion': []
                },
                'document_structure': {
                    'document_type': 'unknown',
                    'organization_pattern': 'unknown',
                    'irac_completeness': 'unknown'
                },
                'section_analysis': [],
                'overall_confidence': 0.0,
                'analysis_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_irac_components(self, irac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize IRAC component data."""
        
        components = ['issues', 'rules', 'application', 'conclusion']
        validated_irac = {}
        
        for component in components:
            component_list = irac_data.get(component, [])
            if not isinstance(component_list, list):
                component_list = []
            
            validated_items = []
            for item in component_list:
                if isinstance(item, dict) and 'text' in item:
                    validated_item = {
                        'text': item.get('text', ''),
                        'confidence': float(item.get('confidence', 0.0)),
                        'section': item.get('section', 'unknown')
                    }
                    
                    # Add component-specific fields
                    if component == 'issues':
                        validated_item['type'] = item.get('type', 'unknown')
                    elif component == 'rules':
                        validated_item['source'] = item.get('source', 'unknown')
                        validated_item['citation'] = item.get('citation', '')
                    elif component == 'application':
                        validated_item['rule_reference'] = item.get('rule_reference', '')
                        validated_item['facts_analyzed'] = item.get('facts_analyzed', '')
                        validated_item['reasoning'] = item.get('reasoning', '')
                    elif component == 'conclusion':
                        validated_item['type'] = item.get('type', 'unknown')
                        validated_item['basis'] = item.get('basis', '')
                    
                    # Only include items meeting confidence threshold
                    if validated_item['confidence'] >= self.min_confidence_threshold:
                        validated_items.append(validated_item)
            
            validated_irac[component] = validated_items
        
        return validated_irac

    def _validate_document_structure(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize document structure data."""
        
        return {
            'document_type': structure_data.get('document_type', 'unknown'),
            'organization_pattern': structure_data.get('organization_pattern', 'unknown'),
            'total_sections': int(structure_data.get('total_sections', 0)),
            'has_headers': bool(structure_data.get('has_headers', False)),
            'numbered_paragraphs': bool(structure_data.get('numbered_paragraphs', False)),
            'irac_completeness': structure_data.get('irac_completeness', 'unknown'),
            'structural_confidence': float(structure_data.get('structural_confidence', 0.0))
        }

    def _validate_section_analysis(self, section_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize section analysis data."""
        
        validated_sections = []
        
        for item in section_data:
            if isinstance(item, dict):
                section = {
                    'section_number': int(item.get('section_number', 0)),
                    'header': item.get('header', ''),
                    'purpose': item.get('purpose', ''),
                    'irac_components': item.get('irac_components', []),
                    'key_concepts': item.get('key_concepts', []),
                    'confidence': float(item.get('confidence', 0.0))
                }
                
                # Only include sections meeting confidence threshold
                if section['confidence'] >= self.min_confidence_threshold:
                    validated_sections.append(section)
        
        return validated_sections

    def _assess_structural_complexity(self, document_content: str) -> TaskComplexity:
        """Assess structural analysis complexity for model selection."""
        
        content_length = len(document_content)
        
        # Base complexity on length
        if content_length < 1500:
            complexity = TaskComplexity.SIMPLE
        elif content_length > 6000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        # Look for structural complexity indicators
        structure_indicators = [
            'section', 'subsection', 'paragraph', 'subparagraph',
            'whereas', 'therefore', 'furthermore', 'moreover',
            'issue', 'holding', 'reasoning', 'conclusion',
            'i.', 'ii.', 'iii.', 'a.', 'b.', 'c.',
            '1.', '2.', '3.', '(1)', '(2)', '(3)'
        ]
        
        content_lower = document_content.lower()
        indicator_count = sum(1 for indicator in structure_indicators if indicator in content_lower)
        
        # Upgrade complexity based on structural indicators
        if indicator_count >= 10:
            complexity = TaskComplexity.COMPLEX
        elif indicator_count >= 5 and complexity == TaskComplexity.SIMPLE:
            complexity = TaskComplexity.MODERATE
        
        return complexity

    def _trim_content(self, content: str, max_length: int) -> str:
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... [TRUNCATED]"

    def _update_analysis_stats(self, result: StructuralAnalysisResult):
        """Update performance statistics."""
        self.analysis_stats["total_analyses"] += 1
        
        # Update rolling averages
        total = self.analysis_stats["total_analyses"]
        
        # Confidence
        old_conf = self.analysis_stats["avg_confidence"]
        self.analysis_stats["avg_confidence"] = (
            old_conf * (total - 1) + result.confidence_score
        ) / total
        
        # IRAC completeness (based on component counts)
        irac = result.irac_components
        completeness_score = min(1.0, (
            len(irac.get('issues', [])) * 0.25 +
            len(irac.get('rules', [])) * 0.25 +
            len(irac.get('application', [])) * 0.25 +
            len(irac.get('conclusion', [])) * 0.25
        ))
        
        old_complete = self.analysis_stats["avg_irac_completeness"]
        self.analysis_stats["avg_irac_completeness"] = (
            old_complete * (total - 1) + completeness_score
        ) / total
        
        # Sections identified
        sections_count = len(result.section_analysis)
        old_sections = self.analysis_stats["avg_sections_identified"]
        self.analysis_stats["avg_sections_identified"] = (
            old_sections * (total - 1) + sections_count
        ) / total
        
        # Processing time
        old_time = self.analysis_stats["processing_time_avg"]
        self.analysis_stats["processing_time_avg"] = (
            old_time * (total - 1) + result.processing_time
        ) / total
        
        # Document types seen
        doc_type = result.structure_type
        if doc_type in self.analysis_stats["document_types_seen"]:
            self.analysis_stats["document_types_seen"][doc_type] += 1
        else:
            self.analysis_stats["document_types_seen"][doc_type] = 1

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get current structural analysis performance statistics."""
        return {
            **self.analysis_stats,
            "agent_status": await self.get_health_status(),
            "configuration": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "section_analysis_enabled": self.section_analysis_enabled,
                "detect_headers": self.detect_headers,
                "max_sections_per_analysis": self.max_sections_per_analysis
            }
        }

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process structural analysis task (implementation of abstract method)."""
        try:
            document_content = task_data.get('document_content', '')
            entities_context = task_data.get('entities_context', [])
            semantic_summary = task_data.get('semantic_summary', '')
            document_metadata = task_data.get('document_metadata', {})
            
            if not document_content:
                raise ValueError("No document content provided for structural analysis")
            
            result = await self.analyze_document_structure(
                document_content=document_content,
                entities_context=entities_context,
                semantic_summary=semantic_summary,
                document_metadata=document_metadata
            )
            
            # Calculate summary metrics
            irac_components = result.irac_components
            total_components = sum(len(components) for components in irac_components.values())
            
            return {
                "status": "success",
                "result": result.to_dict(),
                "metadata": {
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "structure_type": result.structure_type,
                    "total_irac_components": total_components,
                    "sections_analyzed": len(result.section_analysis),
                    "irac_breakdown": {
                        "issues": len(irac_components.get('issues', [])),
                        "rules": len(irac_components.get('rules', [])),
                        "applications": len(irac_components.get('application', [])),
                        "conclusions": len(irac_components.get('conclusion', []))
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"processing_time": 0.0}
            }

    def apply_feedback_adjustments(self, feedback: List[Dict[str, Any]]):
        """Apply feedback adjustments to improve structural analysis quality."""
        adjustments = 0
        
        for fb in feedback:
            if fb.get("agent") != "structural_analysis":
                continue
                
            feedback_type = fb.get("type")
            
            if feedback_type == "incorrect_irac":
                self.structural_prompt_template = self.structural_prompt_template.replace(
                    "Ensure precise identification", "Ensure *extremely precise* identification with careful verification"
                )
                self.logger.info("Feedback applied: Emphasizing IRAC precision")
                adjustments += 1
                
            elif feedback_type == "missing_irac_components":
                self.structural_prompt_template = self.structural_prompt_template.replace(
                    "Extract text passages", "Extract *ALL* text passages comprehensively"
                )
                self.logger.info("Feedback applied: Emphasizing comprehensive IRAC extraction")
                adjustments += 1
                
            elif feedback_type == "poor_section_analysis":
                self.section_analysis_enabled = True
                self.max_sections_per_analysis = min(30, self.max_sections_per_analysis + 5)
                self.logger.info(f"Feedback applied: Enhanced section analysis (max sections: {self.max_sections_per_analysis})")
                adjustments += 1
                
            elif feedback_type == "low_confidence_threshold":
                self.min_confidence_threshold = max(0.5, self.min_confidence_threshold - 0.1)
                self.logger.info(f"Feedback applied: Lowered confidence threshold to {self.min_confidence_threshold}")
                adjustments += 1
        
        if adjustments > 0:
            self.logger.info(f"Applied {adjustments} feedback adjustments to StructuralAnalysisAgent")