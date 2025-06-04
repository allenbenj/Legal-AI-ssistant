"""
TextCorrectionAgent - Legal document formatting and grammar correction.

Provides comprehensive text correction services for legal documents including
grammar correction, tone adjustment, role-based formatting, and legal writing
style enhancement with context-aware improvements.
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
class TextCorrectionResult:
    """Results from text correction analysis."""
    corrected_text: str
    corrections_made: List[Dict[str, Any]]
    formatting_improvements: List[Dict[str, Any]]
    style_adjustments: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrected_text": self.corrected_text,
            "corrections_made": self.corrections_made,
            "formatting_improvements": self.formatting_improvements,
            "style_adjustments": self.style_adjustments,
            "quality_metrics": self.quality_metrics,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "corrected_at": datetime.now().isoformat()
        }


class TextCorrectionAgent(BaseAgent):
    """
    Advanced text correction agent for legal documents.
    
    Features:
    - Grammar and spelling correction
    - Legal tone and formality adjustment
    - Role-based formatting (Judge, Attorney, Defendant, etc.)
    - Legal writing style enhancement
    - Citation format standardization
    - Context-aware corrections using entity information
    - Quality assessment and improvement suggestions
    """
    
    def __init__(self, llm_manager: LLMProviderManager, model_switcher: ModelSwitcher):
        super().__init__("TextCorrectionAgent")
        self.llm_manager = llm_manager
        self.model_switcher = model_switcher
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_confidence_threshold = 0.8
        self.preserve_legal_terminology = True
        self.enhance_formality = True
        self.standardize_citations = True
        self.role_based_formatting = True
        
        # Text correction prompt
        self.correction_prompt_template = """Correct and enhance the legal text for grammar, tone, formatting, and professional legal writing standards.

LEGAL ROLE SCHEMA:
{role_schema}

CORRECTION REQUIREMENTS:

1. GRAMMAR AND SPELLING:
   - Fix grammatical errors and typos
   - Correct punctuation and capitalization
   - Ensure proper sentence structure
   - Maintain legal terminology accuracy

2. TONE AND FORMALITY:
   - Enhance formal legal tone
   - Ensure appropriate professional language
   - Remove casual or colloquial expressions
   - Maintain respectful and objective tone

3. ROLE-BASED FORMATTING:
   - Apply appropriate formatting for legal roles
   - Use proper legal document structure
   - Ensure consistent voice and perspective
   - Apply role-specific language conventions

4. LEGAL WRITING STYLE:
   - Enhance clarity and precision
   - Improve sentence flow and readability
   - Ensure logical organization
   - Maintain legal argumentation structure

5. CITATION AND REFERENCE FORMATTING:
   - Standardize legal citation formats
   - Ensure proper case name formatting
   - Correct statutory and regulatory references
   - Verify citation completeness

ORIGINAL TEXT:
{raw_text}

KNOWN ENTITIES AND CONTEXT:
{entities_context}

DOCUMENT TYPE AND CONTEXT:
{document_context}

CORRECTION INSTRUCTIONS:
- Preserve all legal meanings and technical terms
- Maintain the original intent and arguments
- Enhance readability without changing substance
- Apply legal writing best practices
- Ensure professional presentation

Return corrections in structured JSON format:
{{
    "corrected_text": "The fully corrected and enhanced text",
    "corrections_made": [
        {{
            "type": "grammar|spelling|punctuation|syntax",
            "original": "Original problematic text",
            "corrected": "Corrected text",
            "explanation": "Why this correction was made",
            "confidence": 0.95
        }}
    ],
    "formatting_improvements": [
        {{
            "type": "structure|spacing|numbering|headers",
            "description": "Description of formatting improvement",
            "location": "Where the improvement was made",
            "justification": "Why this improvement enhances the document",
            "confidence": 0.9
        }}
    ],
    "style_adjustments": [
        {{
            "type": "tone|formality|clarity|flow",
            "original": "Original text segment",
            "improved": "Improved text segment",
            "explanation": "How this improves legal writing style",
            "confidence": 0.85
        }}
    ],
    "quality_metrics": {{
        "readability_improvement": "significant|moderate|minimal",
        "formality_level": "enhanced|maintained|unchanged",
        "grammar_errors_fixed": 5,
        "style_improvements": 3,
        "overall_quality_grade": "A|B|C|D|F"
    }},
    "overall_confidence": 0.88,
    "correction_notes": "Summary of corrections and recommendations"
}}

Ensure high-quality corrections with confidence ≥{min_confidence}. Focus on preserving legal accuracy while enhancing presentation."""
        
        # Performance tracking
        self.correction_stats = {
            "total_corrections": 0,
            "total_errors_fixed": 0,
            "avg_confidence": 0.0,
            "avg_improvements": 0.0,
            "processing_time_avg": 0.0,
            "correction_types": {
                "grammar": 0,
                "spelling": 0,
                "punctuation": 0,
                "formatting": 0,
                "style": 0
            }
        }

    async def correct_legal_text(
        self,
        raw_text: str,
        entities_context: List[Dict[str, Any]] = None,
        document_context: Dict[str, Any] = None
    ) -> TextCorrectionResult:
        """
        Main text correction method.
        
        Args:
            raw_text: Original text to correct
            entities_context: Extracted entities for context
            document_context: Document metadata and context
            
        Returns:
            TextCorrectionResult with corrected text and improvement details
        """
        start_time = datetime.now()
        
        try:
            # Assess correction complexity and select model
            complexity = self._assess_correction_complexity(raw_text)
            model_config = await self.model_switcher.get_optimal_model(complexity)
            
            self.logger.info(f"Starting text correction with {model_config['model']} for complexity {complexity}")
            
            # Prepare correction context
            role_schema = self._build_role_schema()
            entities_json = json.dumps(entities_context[:8], indent=2) if entities_context else "None available"
            doc_context = json.dumps(document_context, indent=2) if document_context else "General legal document"
            
            # Build correction prompt
            prompt = self.correction_prompt_template.format(
                role_schema=role_schema,
                raw_text=self._trim_content(raw_text, 4000),
                entities_context=entities_json,
                document_context=doc_context,
                min_confidence=self.min_confidence_threshold
            )
            
            # Perform text correction
            response = await self.llm_manager.query(
                prompt=prompt,
                model=model_config['model'],
                provider=model_config['provider'],
                temperature=0.2,  # Low temperature for consistent corrections
                max_tokens=4000
            )
            
            # Parse and validate correction results
            correction_data = self._parse_correction_response(response.content, raw_text)
            
            # Calculate metrics
            confidence_score = correction_data.get('overall_confidence', 0.0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TextCorrectionResult(
                corrected_text=correction_data.get('corrected_text', raw_text),
                corrections_made=correction_data.get('corrections_made', []),
                formatting_improvements=correction_data.get('formatting_improvements', []),
                style_adjustments=correction_data.get('style_adjustments', []),
                quality_metrics=correction_data.get('quality_metrics', {}),
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_config['model']
            )
            
            # Update performance statistics
            self._update_correction_stats(result)
            
            # Log correction summary
            total_corrections = len(result.corrections_made) + len(result.formatting_improvements) + len(result.style_adjustments)
            self.logger.info(f"Text correction completed: "
                           f"{total_corrections} total improvements, "
                           f"{len(result.corrections_made)} grammar/spelling fixes, "
                           f"confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text correction failed: {str(e)}")
            await self._record_error("text_correction_failed", {"error": str(e)})
            raise

    def _build_role_schema(self) -> str:
        """Build legal role schema for correction guidance."""
        
        legal_roles = ["JUDGE", "ATTORNEY", "DEFENDANT", "WITNESS", "PROSECUTOR", "DEFENSECOUNSEL"]
        
        role_descriptions = {
            "JUDGE": "Formal judicial language with authoritative tone and proper court etiquette",
            "ATTORNEY": "Professional legal advocacy with persuasive and analytical language",
            "DEFENDANT": "Respectful and formal responses with appropriate legal terminology",
            "WITNESS": "Clear and factual testimony with precise and honest language",
            "PROSECUTOR": "Formal prosecutorial language with evidence-based assertions",
            "DEFENSECOUNSEL": "Professional defense advocacy with client-protective language"
        }
        
        schema_lines = []
        for role in legal_roles:
            try:
                entity_type = getattr(LegalEntityType, role, None)
                if entity_type:
                    hint = entity_type.prompt_hint
                else:
                    hint = role_descriptions.get(role, f"Professional {role.lower()} language")
                schema_lines.append(f"- {role}: {hint}")
            except AttributeError:
                if role in role_descriptions:
                    schema_lines.append(f"- {role}: {role_descriptions[role]}")
        
        return '\n'.join(schema_lines)

    def _parse_correction_response(self, response_content: str, original_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured correction data."""
        
        try:
            # Handle JSON markdown blocks
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content:
                json_content = response_content.split('```')[1].split('```')[0]
            else:
                json_content = response_content
            
            correction_data = json.loads(json_content.strip())
            
            # Validate and normalize structure
            validated_data = {
                'corrected_text': correction_data.get('corrected_text', original_text),
                'corrections_made': self._validate_corrections(
                    correction_data.get('corrections_made', [])
                ),
                'formatting_improvements': self._validate_formatting_improvements(
                    correction_data.get('formatting_improvements', [])
                ),
                'style_adjustments': self._validate_style_adjustments(
                    correction_data.get('style_adjustments', [])
                ),
                'quality_metrics': self._validate_quality_metrics(
                    correction_data.get('quality_metrics', {})
                ),
                'overall_confidence': float(correction_data.get('overall_confidence', 0.0)),
                'correction_notes': correction_data.get('correction_notes', '')
            }
            
            return validated_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse correction response: {str(e)}")
            
            # Return minimal structure on parsing failure
            return {
                'corrected_text': original_text,
                'corrections_made': [],
                'formatting_improvements': [],
                'style_adjustments': [],
                'quality_metrics': {},
                'overall_confidence': 0.0,
                'correction_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_corrections(self, corrections_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize correction data."""
        validated_corrections = []
        
        for item in corrections_data:
            if not isinstance(item, dict):
                continue
                
            correction = {
                'type': item.get('type', 'unknown'),
                'original': item.get('original', ''),
                'corrected': item.get('corrected', ''),
                'explanation': item.get('explanation', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include corrections with meaningful content and confidence
            if (correction['original'] and 
                correction['corrected'] and 
                correction['confidence'] >= self.min_confidence_threshold):
                validated_corrections.append(correction)
        
        return validated_corrections

    def _validate_formatting_improvements(self, formatting_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize formatting improvement data."""
        validated_improvements = []
        
        for item in formatting_data:
            if not isinstance(item, dict):
                continue
                
            improvement = {
                'type': item.get('type', 'unknown'),
                'description': item.get('description', ''),
                'location': item.get('location', ''),
                'justification': item.get('justification', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include improvements with meaningful content and confidence
            if (improvement['description'] and 
                improvement['confidence'] >= self.min_confidence_threshold):
                validated_improvements.append(improvement)
        
        return validated_improvements

    def _validate_style_adjustments(self, style_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize style adjustment data."""
        validated_adjustments = []
        
        for item in style_data:
            if not isinstance(item, dict):
                continue
                
            adjustment = {
                'type': item.get('type', 'unknown'),
                'original': item.get('original', ''),
                'improved': item.get('improved', ''),
                'explanation': item.get('explanation', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include adjustments with meaningful content and confidence
            if (adjustment['original'] and 
                adjustment['improved'] and 
                adjustment['confidence'] >= self.min_confidence_threshold):
                validated_adjustments.append(adjustment)
        
        return validated_adjustments

    def _validate_quality_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize quality metrics data."""
        return {
            'readability_improvement': metrics_data.get('readability_improvement', 'unknown'),
            'formality_level': metrics_data.get('formality_level', 'unknown'),
            'grammar_errors_fixed': int(metrics_data.get('grammar_errors_fixed', 0)),
            'style_improvements': int(metrics_data.get('style_improvements', 0)),
            'overall_quality_grade': metrics_data.get('overall_quality_grade', 'unknown')
        }

    def _assess_correction_complexity(self, text: str) -> TaskComplexity:
        """Assess text correction complexity for model selection."""
        
        text_length = len(text)
        
        # Base complexity on length
        if text_length < 1000:
            complexity = TaskComplexity.SIMPLE
        elif text_length > 4000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        # Look for complexity indicators
        complexity_indicators = [
            'whereas', 'therefore', 'notwithstanding', 'heretofore',
            'aforementioned', 'pursuant', 'whereunder', 'hereby',
            'latin phrase', 'et al', 'inter alia', 'prima facie',
            'citation', 'statute', 'regulation', 'precedent'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in text_lower)
        
        # Check for potential errors (simple heuristics)
        error_indicators = [
            'teh ', 'hte ', 'adn ', 'recieve', 'seperate',
            'occured', 'judgement', 'loose' if 'lose' not in text_lower else '',
            '  ', '..', '??', '!!'  # Multiple spaces or punctuation
        ]
        
        error_count = sum(1 for error in error_indicators if error and error in text_lower)
        
        # Upgrade complexity based on indicators
        if indicator_count >= 5 or error_count >= 3:
            if complexity == TaskComplexity.SIMPLE:
                complexity = TaskComplexity.MODERATE
            elif complexity == TaskComplexity.MODERATE:
                complexity = TaskComplexity.COMPLEX
        
        return complexity

    def _trim_content(self, content: str, max_length: int) -> str:
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... [TRUNCATED]"

    def _update_correction_stats(self, result: TextCorrectionResult):
        """Update performance statistics."""
        self.correction_stats["total_corrections"] += 1
        
        # Count total errors fixed
        total_errors = len(result.corrections_made) + len(result.formatting_improvements) + len(result.style_adjustments)
        self.correction_stats["total_errors_fixed"] += total_errors
        
        # Update rolling averages
        total = self.correction_stats["total_corrections"]
        
        # Confidence
        old_conf = self.correction_stats["avg_confidence"]
        self.correction_stats["avg_confidence"] = (
            old_conf * (total - 1) + result.confidence_score
        ) / total
        
        # Improvements per correction
        old_improvements = self.correction_stats["avg_improvements"]
        self.correction_stats["avg_improvements"] = (
            old_improvements * (total - 1) + total_errors
        ) / total
        
        # Processing time
        old_time = self.correction_stats["processing_time_avg"]
        self.correction_stats["processing_time_avg"] = (
            old_time * (total - 1) + result.processing_time
        ) / total
        
        # Correction types
        for correction in result.corrections_made:
            correction_type = correction.get('type', 'unknown')
            if correction_type in self.correction_stats["correction_types"]:
                self.correction_stats["correction_types"][correction_type] += 1

    async def get_correction_statistics(self) -> Dict[str, Any]:
        """Get current text correction performance statistics."""
        return {
            **self.correction_stats,
            "agent_status": await self.get_health_status(),
            "configuration": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "preserve_legal_terminology": self.preserve_legal_terminology,
                "enhance_formality": self.enhance_formality,
                "standardize_citations": self.standardize_citations,
                "role_based_formatting": self.role_based_formatting
            }
        }

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process text correction task (implementation of abstract method)."""
        try:
            raw_text = task_data.get('raw_text', '')
            entities_context = task_data.get('entities_context', [])
            document_context = task_data.get('document_context', {})
            
            if not raw_text:
                raise ValueError("No text provided for correction")
            
            result = await self.correct_legal_text(
                raw_text=raw_text,
                entities_context=entities_context,
                document_context=document_context
            )
            
            # Calculate improvement metrics
            total_improvements = (len(result.corrections_made) + 
                                len(result.formatting_improvements) + 
                                len(result.style_adjustments))
            
            return {
                "status": "success",
                "result": result.to_dict(),
                "metadata": {
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "total_improvements": total_improvements,
                    "grammar_corrections": len(result.corrections_made),
                    "formatting_improvements": len(result.formatting_improvements),
                    "style_adjustments": len(result.style_adjustments),
                    "text_length_original": len(raw_text),
                    "text_length_corrected": len(result.corrected_text)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"processing_time": 0.0}
            }

    def apply_feedback_adjustments(self, feedback: List[Dict[str, Any]]):
        """Apply feedback adjustments to improve correction quality."""
        adjustments = 0
        
        for fb in feedback:
            if fb.get("agent") != "text_correction":
                continue
                
            feedback_type = fb.get("type")
            
            if feedback_type == "incorrect_tone":
                self.enhance_formality = True
                self.correction_prompt_template = self.correction_prompt_template.replace(
                    "Enhance formal legal tone", "Enhance *strict formal legal tone* with professional authority"
                )
                self.logger.info("Feedback applied: Enhanced formality requirements")
                adjustments += 1
                
            elif feedback_type == "formatting_errors":
                self.role_based_formatting = True
                self.correction_prompt_template = self.correction_prompt_template.replace(
                    "Apply appropriate formatting", "Apply *strict and precise* formatting standards"
                )
                self.logger.info("Feedback applied: Enhanced formatting strictness")
                adjustments += 1
                
            elif feedback_type == "preserved_errors":
                self.min_confidence_threshold = max(0.6, self.min_confidence_threshold - 0.1)
                self.correction_prompt_template = self.correction_prompt_template.replace(
                    "Fix grammatical errors", "Fix *ALL* grammatical errors and issues comprehensively"
                )
                self.logger.info("Feedback applied: More aggressive error correction")
                adjustments += 1
                
            elif feedback_type == "changed_meaning":
                self.preserve_legal_terminology = True
                self.correction_prompt_template = self.correction_prompt_template.replace(
                    "Preserve all legal meanings", "Preserve *exactly* all legal meanings and technical terms without any alteration"
                )
                self.logger.info("Feedback applied: Enhanced meaning preservation")
                adjustments += 1
        
        if adjustments > 0:
            self.logger.info(f"Applied {adjustments} feedback adjustments to TextCorrectionAgent")