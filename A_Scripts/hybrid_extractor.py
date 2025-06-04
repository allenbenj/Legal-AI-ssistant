"""Hybrid NER+LLM Legal Entity Extraction System with Confidence Calibration.

This module combines Named Entity Recognition (NER) with Large Language Model (LLM)
prompting for robust, context-aware legal entity extraction with confidence calibration,
validation, and cross-referencing capabilities.

Enhanced with confidence calibration system to resolve heterogeneous model outputs
and improve ensemble decision-making through normalized confidence scores.
"""

import asyncio
import spacy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict
import warnings
from pathlib import Path

# Try to import advanced NER models
try:
    import flair
    from flair.data import Sentence
    from flair.models import SequenceTagger
    FLAIR_AVAILABLE = True
except ImportError:
    FLAIR_AVAILABLE = False
    warnings.warn("Flair not available, using spaCy only")

try:
    # Blackstone is specialized for legal NLP
    import blackstone
    BLACKSTONE_AVAILABLE = True
except ImportError:
    BLACKSTONE_AVAILABLE = False
    warnings.warn("Blackstone not available, using standard models")

try:
    from ..utils.ontology import LegalEntityType, LegalRelationshipType
except ImportError:
    # Fallback for relative import issues
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.ontology import LegalEntityType, LegalRelationshipType
try:
    from ..core.confidence_calibration import (
        ConfidenceCalibrationManager, EntityPrediction, ValidationSample
    )
except ImportError:
    # Fallback for relative import issues
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.confidence_calibration import (
        ConfidenceCalibrationManager, EntityPrediction, ValidationSample
    )


@dataclass
class NERResult:
    """Result from NER extraction."""
    entity_text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    source_model: str  # 'spacy', 'flair', 'blackstone'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMExtractionResult:
    """Result from LLM-based extraction."""
    entity_text: str
    entity_type: str
    context: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from cross-validation between NER and LLM."""
    entity_text: str
    ner_type: Optional[str]
    llm_type: Optional[str]
    consensus_type: str
    confidence: float
    discrepancy: bool
    resolution_method: str  # 'ner_priority', 'llm_priority', 'manual_review'
    notes: str = ""


@dataclass
class HybridExtractionResult:
    """Complete result from hybrid extraction."""
    document_id: str
    ner_results: List[NERResult]
    llm_results: List[LLMExtractionResult]
    validated_entities: List[ValidationResult]
    targeted_extractions: Dict[str, List[Dict[str, Any]]]
    cross_references: Dict[str, List[str]]
    extraction_metadata: Dict[str, Any]
    processing_time: float


class HybridLegalExtractor:
    """
    Advanced hybrid extraction system combining NER and LLM capabilities.
    
    Features:
    - Multi-model NER (spaCy, Flair, Blackstone for legal text)
    - LLM-assisted contextual extraction with structured prompts
    - Targeted prompting for specific legal violations and patterns
    - Cross-validation between NER and LLM outputs
    - Neo4j cross-referencing and agent memory integration
    - Error handling and discrepancy resolution
    """
    
    def __init__(self, services, **config):
        self.services = services
        self.config = config
        self.logger = services.logger
        
        # Model configuration
        self.spacy_model_name = config.get('spacy_model', 'en_core_web_lg')
        self.use_blackstone = config.get('use_blackstone', BLACKSTONE_AVAILABLE)
        self.use_flair = config.get('use_flair', FLAIR_AVAILABLE)
        
        # Extraction configuration
        self.enable_ner = config.get('enable_ner', True)
        self.enable_llm_extraction = config.get('enable_llm_extraction', True)
        self.enable_targeted_prompting = config.get('enable_targeted_prompting', True)
        self.cross_validation_threshold = config.get('cross_validation_threshold', 0.7)
        
        # Models
        self.spacy_nlp = None
        self.blackstone_nlp = None
        self.flair_tagger = None
        
        # Confidence calibration system
        calibration_path = Path(config.get('calibration_storage_path', 'storage/calibration/'))
        self.calibration_manager = ConfidenceCalibrationManager(calibration_path)
        self.enable_confidence_calibration = config.get('enable_confidence_calibration', True)
        
        # Legal-specific patterns
        self.legal_patterns = self._setup_legal_patterns()
        self.targeted_prompts = self._setup_targeted_prompts()
        
        # Cross-referencing
        self.enable_cross_referencing = config.get('enable_cross_referencing', True)
        self.knowledge_graph_manager = None
        self.agent_memory = None
        
    async def initialize(self):
        """Initialize the hybrid extraction system."""
        await self._load_ner_models()
        await self._setup_cross_referencing()
        self.logger.info("Hybrid legal extractor initialized")
    
    async def _load_ner_models(self):
        """Load all available NER models."""
        try:
            # Load spaCy model
            if self.enable_ner:
                self.spacy_nlp = spacy.load(self.spacy_model_name)
                self.logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
                
                # Add custom legal entity patterns
                ruler = self.spacy_nlp.add_pipe("entity_ruler", before="ner")
                ruler.add_patterns(self.legal_patterns)
            
            # Load Blackstone (legal-specific)
            if self.use_blackstone and BLACKSTONE_AVAILABLE:
                try:
                    self.blackstone_nlp = spacy.load("en_blackstone_proto")
                    self.logger.info("Loaded Blackstone legal NLP model")
                except OSError:
                    self.logger.warning("Blackstone model not found, install with: pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz")
                    self.use_blackstone = False
            
            # Load Flair
            if self.use_flair and FLAIR_AVAILABLE:
                try:
                    self.flair_tagger = SequenceTagger.load('ner')
                    self.logger.info("Loaded Flair NER model")
                except Exception as e:
                    self.logger.warning(f"Failed to load Flair model: {e}")
                    self.use_flair = False
                    
        except Exception as e:
            self.logger.error(f"Failed to load NER models: {e}")
            raise
    
    def _setup_legal_patterns(self) -> List[Dict[str, Any]]:
        """Set up legal-specific entity patterns for spaCy."""
        patterns = [
            # Case citations
            {"label": "CASE_CITATION", "pattern": [
                {"TEXT": {"REGEX": r"\d+"}},
                {"TEXT": {"IN": ["F.", "F.2d", "F.3d", "F.Supp.", "U.S.", "S.Ct."]}},
                {"TEXT": {"REGEX": r"\d+"}}
            ]},
            
            # Statutes
            {"label": "STATUTE", "pattern": [
                {"TEXT": {"REGEX": r"\d+"}},
                {"TEXT": {"IN": ["U.S.C.", "USC", "CFR"]}},
                {"TEXT": {"REGEX": r"§?\s*\d+"}}
            ]},
            
            # Court names
            {"label": "COURT", "pattern": [
                {"TEXT": {"IN": ["United", "U.S.", "Supreme", "District", "Circuit", "Court"]}},
                {"TEXT": "Court", "OP": "?"}
            ]},
            
            # Legal violations
            {"label": "VIOLATION", "pattern": [
                {"LOWER": {"IN": ["brady", "misconduct", "violation", "tampering", "perjury"]}}
            ]},
            
            # Legal roles
            {"label": "LEGAL_ROLE", "pattern": [
                {"LOWER": {"IN": ["prosecutor", "defendant", "plaintiff", "judge", "attorney", "counsel"]}}
            ]},
            
            # Evidence types
            {"label": "EVIDENCE", "pattern": [
                {"LOWER": {"IN": ["exhibit", "evidence", "document", "testimony", "statement"]}},
                {"TEXT": {"REGEX": r"[A-Z]?\d*"}, "OP": "?"}
            ]}
        ]
        
        return patterns
    
    def _setup_targeted_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Set up targeted extraction prompts for specific legal issues."""
        return {
            "brady_violations": {
                "prompt": """
                Analyze this legal document for potential Brady violations. Brady violations occur when prosecutors fail to disclose exculpatory evidence to the defense.
                
                Look for:
                1. Evidence favorable to the defendant that was not disclosed
                2. Prosecutorial suppression of evidence
                3. Material evidence that could affect the outcome
                4. Late disclosure of evidence
                5. Failure to disclose impeachment evidence
                
                For each potential Brady violation found, provide:
                - Description of the violation
                - Evidence involved
                - Parties responsible
                - Impact on the case
                
                Document text: {text}
                
                Return results in JSON format:
                {{
                    "brady_violations": [
                        {{
                            "description": "...",
                            "evidence_type": "...",
                            "responsible_party": "...",
                            "materiality": "high|medium|low",
                            "text_reference": "...",
                            "confidence": 0.0-1.0
                        }}
                    ]
                }}
                """,
                "entity_types": ["PROSECUTOR", "EVIDENCE", "VIOLATION"],
                "confidence_threshold": 0.8
            },
            
            "prosecutorial_misconduct": {
                "prompt": """
                Identify instances of prosecutorial misconduct in this document. Look for:
                
                1. Withholding exculpatory evidence
                2. Presenting false or misleading evidence
                3. Improper arguments to the jury
                4. Vindictive prosecution
                5. Selective prosecution
                6. Coercing witnesses
                7. Making improper statements
                
                Document text: {text}
                
                Return JSON format:
                {{
                    "misconduct_instances": [
                        {{
                            "type": "...",
                            "description": "...",
                            "prosecutor": "...",
                            "severity": "high|medium|low",
                            "text_reference": "...",
                            "confidence": 0.0-1.0
                        }}
                    ]
                }}
                """,
                "entity_types": ["PROSECUTOR", "MISCONDUCT", "EVIDENCE"],
                "confidence_threshold": 0.75
            },
            
            "witness_tampering": {
                "prompt": """
                Examine this document for evidence of witness tampering or coercion:
                
                1. Threats or intimidation of witnesses
                2. Bribery or inducements
                3. Coaching witnesses to change testimony
                4. Suppression of witness statements
                5. Improper contact with witnesses
                
                Document text: {text}
                
                Return JSON format:
                {{
                    "tampering_instances": [
                        {{
                            "type": "...",
                            "witness_affected": "...",
                            "perpetrator": "...",
                            "method": "...",
                            "text_reference": "...",
                            "confidence": 0.0-1.0
                        }}
                    ]
                }}
                """,
                "entity_types": ["WITNESS", "PERSON", "VIOLATION"],
                "confidence_threshold": 0.8
            },
            
            "judge_shopping": {
                "prompt": """
                Look for evidence of judge shopping or forum shopping in this document:
                
                1. Strategic venue selection
                2. Multiple case filings in different jurisdictions
                3. Attempts to get specific judges
                4. Manipulation of case assignment
                5. Timing of filings to influence judge assignment
                
                Document text: {text}
                
                Return JSON format:
                {{
                    "judge_shopping_instances": [
                        {{
                            "description": "...",
                            "judges_involved": ["..."],
                            "courts_involved": ["..."],
                            "strategy_employed": "...",
                            "text_reference": "...",
                            "confidence": 0.0-1.0
                        }}
                    ]
                }}
                """,
                "entity_types": ["JUDGE", "COURT", "CASE"],
                "confidence_threshold": 0.7
            },
            
            "legal_statutes": {
                "prompt": """
                Extract all legal statutes, regulations, and legal references mentioned in this document:
                
                1. U.S. Code citations (e.g., 18 U.S.C. § 1001)
                2. Code of Federal Regulations (CFR) citations
                3. State statutes
                4. Constitutional amendments
                5. Rules of evidence or procedure
                6. Case law citations
                
                Document text: {text}
                
                Return JSON format:
                {{
                    "legal_references": [
                        {{
                            "citation": "...",
                            "type": "statute|regulation|case_law|constitutional",
                            "jurisdiction": "federal|state|local",
                            "description": "...",
                            "text_reference": "...",
                            "confidence": 0.0-1.0
                        }}
                    ]
                }}
                """,
                "entity_types": ["STATUTE", "CASE_CITATION", "COURT"],
                "confidence_threshold": 0.9
            }
        }
    
    async def _setup_cross_referencing(self):
        """Set up cross-referencing with knowledge graph and agent memory."""
        if self.enable_cross_referencing:
            try:
                # Get knowledge graph manager from services
                self.knowledge_graph_manager = self.services.get_service('knowledge_graph_manager')
                self.agent_memory = self.services.get_service('agent_memory')
                self.logger.info("Cross-referencing enabled")
            except Exception as e:
                self.logger.warning(f"Cross-referencing setup failed: {e}")
                self.enable_cross_referencing = False
    
    async def extract_hybrid(self, document_text: str, document_id: str, 
                           enable_targeted: bool = True) -> HybridExtractionResult:
        """Perform hybrid extraction with confidence calibration.
        
        Enhanced hybrid extraction combining NER and LLM approaches with
        confidence calibration for improved ensemble decision-making.
        
        Args:
            document_text: Text to analyze
            document_id: Document identifier
            enable_targeted: Whether to run targeted extractions
            
        Returns:
            HybridExtractionResult with calibrated confidence scores
        """
        start_time = datetime.now()
        
        try:
            # Phase 1: Multi-Model Entity Extraction
            self.logger.info("Starting multi-model entity extraction...")
            
            # Collect predictions from all models
            spacy_doc = None
            flair_sentence = None
            llm_response = ""
            blackstone_entities = []
            
            # spaCy extraction
            if self.spacy_nlp:
                spacy_doc = self.spacy_nlp(document_text)
                self.logger.debug(f"spaCy extracted {len(spacy_doc.ents)} entities")
            
            # Flair extraction
            if self.flair_tagger:
                flair_sentence = Sentence(document_text)
                self.flair_tagger.predict(flair_sentence)
                self.logger.debug(f"Flair extracted {len(flair_sentence.get_labels('ner'))} entities")
            
            # Blackstone extraction
            if self.blackstone_nlp:
                blackstone_doc = self.blackstone_nlp(document_text)
                blackstone_entities = [
                    {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.95  # Blackstone rules typically high confidence
                    }
                    for ent in blackstone_doc.ents
                ]
                self.logger.debug(f"Blackstone extracted {len(blackstone_entities)} entities")
            
            # LLM extraction
            if self.enable_llm_extraction:
                llm_response = await self._extract_with_llm_calibrated(document_text)
                self.logger.debug("LLM extraction completed")
            
            # Phase 2: Confidence Calibration and Ensemble Voting
            if self.enable_confidence_calibration:
                self.logger.info("Applying confidence calibration...")
                calibrated_entities = self.calibration_manager.process_predictions(
                    spacy_doc=spacy_doc,
                    flair_sentence=flair_sentence,
                    llm_response=llm_response,
                    source_text=document_text,
                    blackstone_entities=blackstone_entities
                )
                
                # Convert to legacy format for compatibility
                validated_entities = self._convert_calibrated_to_ner_results(calibrated_entities)
                
                self.logger.info(f"Calibrated ensemble produced {len(validated_entities)} entities")
            else:
                # Fallback to legacy cross-validation
                self.logger.info("Using legacy cross-validation...")
                ner_results = await self._extract_with_ner(document_text)
                llm_results = await self._extract_with_llm(document_text, ner_results)
                validated_entities = await self._cross_validate_results(ner_results, llm_results)
            
            # Phase 3: Targeted Extractions
            targeted_extractions = {}
            if enable_targeted and self.enable_targeted_prompting:
                self.logger.info("Running targeted extractions...")
                targeted_extractions = await self._run_targeted_extractions(document_text)
            
            # Phase 4: Cross-Referencing
            cross_references = {}
            if self.enable_cross_referencing:
                self.logger.info("Cross-referencing entities...")
                cross_references = await self._cross_reference_entities(validated_entities, document_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = HybridExtractionResult(
                document_id=document_id,
                ner_results=[] if self.enable_confidence_calibration else ner_results,
                llm_results=[] if self.enable_confidence_calibration else llm_results,
                validated_entities=validated_entities,
                targeted_extractions=targeted_extractions,
                cross_references=cross_references,
                extraction_metadata={
                    'models_used': self._get_models_used(),
                    'extraction_method': 'calibrated_hybrid' if self.enable_confidence_calibration else 'hybrid_ner_llm',
                    'confidence_calibration_enabled': self.enable_confidence_calibration,
                    'validated_entities': len(validated_entities),
                    'targeted_extractions_count': len(targeted_extractions)
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"Enhanced hybrid extraction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced hybrid extraction failed: {e}")
            raise
    
    async def _extract_with_ner(self, text: str) -> List[NERResult]:
        """Extract entities using NER models."""
        ner_results = []
        
        # spaCy extraction
        if self.spacy_nlp:
            spacy_results = await self._extract_with_spacy(text)
            ner_results.extend(spacy_results)
        
        # Blackstone extraction (legal-specific)
        if self.blackstone_nlp:
            blackstone_results = await self._extract_with_blackstone(text)
            ner_results.extend(blackstone_results)
        
        # Flair extraction
        if self.flair_tagger:
            flair_results = await self._extract_with_flair(text)
            ner_results.extend(flair_results)
        
        # Deduplicate and merge overlapping entities
        ner_results = self._deduplicate_ner_results(ner_results)
        
        return ner_results
    
    async def _extract_with_spacy(self, text: str) -> List[NERResult]:
        """Extract entities using spaCy."""
        results = []
        
        try:
            doc = self.spacy_nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our legal ontology
                mapped_type = self._map_spacy_label_to_legal(ent.label_)
                
                result = NERResult(
                    entity_text=ent.text,
                    entity_type=mapped_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy
                    source_model='spacy',
                    metadata={
                        'original_label': ent.label_,
                        'dependency': ent.root.dep_,
                        'pos': ent.root.pos_
                    }
                )
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"spaCy extraction failed: {e}")
        
        return results
    
    async def _extract_with_blackstone(self, text: str) -> List[NERResult]:
        """Extract entities using Blackstone legal model."""
        results = []
        
        try:
            doc = self.blackstone_nlp(text)
            
            for ent in doc.ents:
                # Blackstone has legal-specific labels
                mapped_type = self._map_blackstone_label_to_legal(ent.label_)
                
                result = NERResult(
                    entity_text=ent.text,
                    entity_type=mapped_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.85,  # Higher confidence for legal-specific model
                    source_model='blackstone',
                    metadata={
                        'original_label': ent.label_,
                        'legal_context': True
                    }
                )
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Blackstone extraction failed: {e}")
        
        return results
    
    async def _extract_with_flair(self, text: str) -> List[NERResult]:
        """Extract entities using Flair."""
        results = []
        
        try:
            sentence = Sentence(text)
            self.flair_tagger.predict(sentence)
            
            for entity in sentence.get_spans('ner'):
                mapped_type = self._map_flair_label_to_legal(entity.tag)
                
                result = NERResult(
                    entity_text=entity.text,
                    entity_type=mapped_type,
                    start_pos=entity.start_position,
                    end_pos=entity.end_position,
                    confidence=entity.score,
                    source_model='flair',
                    metadata={
                        'original_label': entity.tag
                    }
                )
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Flair extraction failed: {e}")
        
        return results
    
    def _map_spacy_label_to_legal(self, spacy_label: str) -> str:
        """Map spaCy entity labels to legal ontology."""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'PARTY',
            'GPE': 'JURISDICTION',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'MONEY': 'MONETARY_AMOUNT',
            'LAW': 'STATUTE',
            'CASE_CITATION': 'CASE_CITATION',
            'COURT': 'COURT',
            'VIOLATION': 'VIOLATION',
            'LEGAL_ROLE': 'LEGAL_ROLE',
            'EVIDENCE': 'EVIDENCEITEM'
        }
        
        return mapping.get(spacy_label, 'ENTITY')
    
    def _map_blackstone_label_to_legal(self, blackstone_label: str) -> str:
        """Map Blackstone entity labels to legal ontology."""
        mapping = {
            'JUDGE': 'JUDGE',
            'LAWYER': 'DEFENSECOUNSEL',
            'COURT': 'COURT',
            'CASENAME': 'CASE',
            'CITATION': 'CASE_CITATION',
            'LEGISLATION': 'STATUTE',
            'PROVISION': 'STATUTE',
            'INSTRUMENT': 'LEGALDOCUMENT'
        }
        
        return mapping.get(blackstone_label, 'ENTITY')
    
    def _map_flair_label_to_legal(self, flair_label: str) -> str:
        """Map Flair entity labels to legal ontology."""
        mapping = {
            'PER': 'PERSON',
            'ORG': 'PARTY',
            'LOC': 'JURISDICTION',
            'MISC': 'ENTITY'
        }
        
        return mapping.get(flair_label, 'ENTITY')
    
    def _deduplicate_ner_results(self, results: List[NERResult]) -> List[NERResult]:
        """Remove duplicate and overlapping NER results."""
        if not results:
            return results
        
        # Sort by start position
        results.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        deduplicated = []
        
        for result in results:
            # Check for overlap with existing results
            overlapping = False
            
            for existing in deduplicated:
                if (result.start_pos < existing.end_pos and 
                    result.end_pos > existing.start_pos):
                    # Overlap detected - keep the one with higher confidence
                    if result.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    overlapping = True
                    break
            
            if not overlapping:
                deduplicated.append(result)
        
        return deduplicated
    
    async def _extract_with_llm(self, text: str, ner_results: List[NERResult]) -> List[LLMExtractionResult]:
        """Extract entities using LLM with context awareness."""
        if not self.enable_llm_extraction or not self.services.llm_provider:
            return []
        
        # Create context from NER results
        ner_context = self._create_ner_context(ner_results)
        
        prompt = f"""
        Analyze this legal document and extract entities with contextual understanding.
        
        The following entities were already identified by NER:
        {ner_context}
        
        Please identify additional entities or provide better context for existing ones, focusing on:
        1. Legal relationships and connections
        2. Entities that NER might have missed
        3. Context and roles of identified entities
        4. Legal significance of entities
        
        Document text: {text[:3000]}...
        
        Return results in JSON format:
        {{
            "entities": [
                {{
                    "text": "entity text",
                    "type": "LEGAL_ENTITY_TYPE",
                    "context": "surrounding context",
                    "confidence": 0.0-1.0,
                    "reasoning": "why this is significant",
                    "legal_role": "role in legal context"
                }}
            ]
        }}
        """
        
        try:
            response = await self.services.llm_provider.generate_response(
                prompt=prompt,
                model_params={'temperature': 0.1, 'max_tokens': 2000}
            )
            
            return self._parse_llm_extraction_response(response)
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return []
    
    def _create_ner_context(self, ner_results: List[NERResult]) -> str:
        """Create context string from NER results."""
        if not ner_results:
            return "No entities identified by NER."
        
        context_lines = []
        for result in ner_results[:20]:  # Limit for prompt size
            context_lines.append(f"- {result.entity_text} ({result.entity_type}, confidence: {result.confidence:.2f})")
        
        return "\n".join(context_lines)
    
    def _parse_llm_extraction_response(self, response: str) -> List[LLMExtractionResult]:
        """Parse LLM extraction response."""
        results = []
        
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return results
            
            data = json.loads(json_match.group())
            entities = data.get('entities', [])
            
            for entity_data in entities:
                result = LLMExtractionResult(
                    entity_text=entity_data.get('text', ''),
                    entity_type=entity_data.get('type', 'ENTITY'),
                    context=entity_data.get('context', ''),
                    confidence=entity_data.get('confidence', 0.7),
                    reasoning=entity_data.get('reasoning', ''),
                    metadata={
                        'legal_role': entity_data.get('legal_role', ''),
                        'source': 'llm_extraction'
                    }
                )
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM extraction response: {e}")
        
        return results
    
    async def _cross_validate_results(self, ner_results: List[NERResult], 
                                    llm_results: List[LLMExtractionResult]) -> List[ValidationResult]:
        """Cross-validate NER and LLM results."""
        validated = []
        
        # Create lookup for efficient matching
        ner_lookup = {result.entity_text.lower(): result for result in ner_results}
        llm_lookup = {result.entity_text.lower(): result for result in llm_results}
        
        # Get all unique entity texts
        all_entity_texts = set(ner_lookup.keys()) | set(llm_lookup.keys())
        
        for entity_text in all_entity_texts:
            ner_result = ner_lookup.get(entity_text)
            llm_result = llm_lookup.get(entity_text)
            
            validation = self._validate_entity_pair(entity_text, ner_result, llm_result)
            validated.append(validation)
        
        return validated
    
    def _validate_entity_pair(self, entity_text: str, 
                            ner_result: Optional[NERResult],
                            llm_result: Optional[LLMExtractionResult]) -> ValidationResult:
        """Validate a pair of NER and LLM results for the same entity."""
        
        if ner_result and llm_result:
            # Both models identified this entity
            if ner_result.entity_type == llm_result.entity_type:
                # Agreement - high confidence
                return ValidationResult(
                    entity_text=entity_text,
                    ner_type=ner_result.entity_type,
                    llm_type=llm_result.entity_type,
                    consensus_type=ner_result.entity_type,
                    confidence=(ner_result.confidence + llm_result.confidence) / 2,
                    discrepancy=False,
                    resolution_method='consensus',
                    notes="NER and LLM agree on entity type"
                )
            else:
                # Disagreement - needs resolution
                if ner_result.confidence > llm_result.confidence:
                    chosen_type = ner_result.entity_type
                    method = 'ner_priority'
                else:
                    chosen_type = llm_result.entity_type
                    method = 'llm_priority'
                
                return ValidationResult(
                    entity_text=entity_text,
                    ner_type=ner_result.entity_type,
                    llm_type=llm_result.entity_type,
                    consensus_type=chosen_type,
                    confidence=max(ner_result.confidence, llm_result.confidence) * 0.8,  # Penalty for disagreement
                    discrepancy=True,
                    resolution_method=method,
                    notes=f"Type disagreement: NER={ner_result.entity_type}, LLM={llm_result.entity_type}"
                )
        
        elif ner_result:
            # Only NER identified this entity
            return ValidationResult(
                entity_text=entity_text,
                ner_type=ner_result.entity_type,
                llm_type=None,
                consensus_type=ner_result.entity_type,
                confidence=ner_result.confidence * 0.9,  # Slight penalty for single source
                discrepancy=False,
                resolution_method='ner_only',
                notes="Identified by NER only"
            )
        
        elif llm_result:
            # Only LLM identified this entity
            return ValidationResult(
                entity_text=entity_text,
                ner_type=None,
                llm_type=llm_result.entity_type,
                consensus_type=llm_result.entity_type,
                confidence=llm_result.confidence * 0.9,  # Slight penalty for single source
                discrepancy=False,
                resolution_method='llm_only',
                notes="Identified by LLM only"
            )
        
        else:
            # Should not happen
            return ValidationResult(
                entity_text=entity_text,
                ner_type=None,
                llm_type=None,
                consensus_type='UNKNOWN',
                confidence=0.0,
                discrepancy=True,
                resolution_method='error',
                notes="No extraction results found"
            )
    
    async def _run_targeted_extractions(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Run targeted extractions for specific legal patterns."""
        if not self.services.llm_provider:
            return {}
        
        targeted_results = {}
        
        for extraction_type, prompt_config in self.targeted_prompts.items():
            try:
                self.logger.debug(f"Running targeted extraction: {extraction_type}")
                
                # Format the prompt with the document text
                formatted_prompt = prompt_config['prompt'].format(text=text[:4000])  # Limit text size
                
                response = await self.services.llm_provider.generate_response(
                    prompt=formatted_prompt,
                    model_params={'temperature': 0.1, 'max_tokens': 1500}
                )
                
                # Parse response
                parsed_results = self._parse_targeted_response(response, extraction_type, prompt_config)
                
                if parsed_results:
                    targeted_results[extraction_type] = parsed_results
                    
            except Exception as e:
                self.logger.error(f"Targeted extraction failed for {extraction_type}: {e}")
        
        return targeted_results
    
    def _parse_targeted_response(self, response: str, extraction_type: str, 
                               prompt_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse targeted extraction response."""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            
            # Get the main results key based on extraction type
            key_mapping = {
                'brady_violations': 'brady_violations',
                'prosecutorial_misconduct': 'misconduct_instances',
                'witness_tampering': 'tampering_instances',
                'judge_shopping': 'judge_shopping_instances',
                'legal_statutes': 'legal_references'
            }
            
            results_key = key_mapping.get(extraction_type, extraction_type)
            results = data.get(results_key, [])
            
            # Filter by confidence threshold
            threshold = prompt_config.get('confidence_threshold', 0.7)
            filtered_results = [
                result for result in results 
                if result.get('confidence', 0) >= threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to parse targeted response for {extraction_type}: {e}")
            return []
    
    async def _cross_reference_entities(self, validated_entities: List[ValidationResult],
                                      document_id: str) -> Dict[str, List[str]]:
        """Cross-reference entities with knowledge graph and agent memory."""
        cross_references = {}
        
        if not self.enable_cross_referencing:
            return cross_references
        
        try:
            for entity in validated_entities:
                entity_refs = []
                
                # Check knowledge graph
                if self.knowledge_graph_manager:
                    graph_refs = await self._check_knowledge_graph(entity)
                    entity_refs.extend(graph_refs)
                
                # Check agent memory
                if self.agent_memory:
                    memory_refs = await self._check_agent_memory(entity)
                    entity_refs.extend(memory_refs)
                
                if entity_refs:
                    cross_references[entity.entity_text] = entity_refs
                    
        except Exception as e:
            self.logger.error(f"Cross-referencing failed: {e}")
        
        return cross_references
    
    async def _check_knowledge_graph(self, entity: ValidationResult) -> List[str]:
        """Check for entity references in knowledge graph."""
        # This would query the knowledge graph for similar entities
        # Implementation depends on the knowledge graph manager API
        return []
    
    async def _check_agent_memory(self, entity: ValidationResult) -> List[str]:
        """Check for entity references in agent memory."""
        # This would query agent memory for related entities
        # Implementation depends on the agent memory API
        return []
    
    def _get_models_used(self) -> List[str]:
        """Get list of models that were successfully loaded."""
        models = []
        
        if self.spacy_nlp:
            models.append('spacy')
        if self.blackstone_nlp:
            models.append('blackstone')
        if self.flair_tagger:
            models.append('flair')
        if self.enable_llm_extraction:
            models.append('llm')
        
        return models
    
    async def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction system statistics."""
        return {
            'models_available': {
                'spacy': self.spacy_nlp is not None,
                'blackstone': self.blackstone_nlp is not None,
                'flair': self.flair_tagger is not None,
                'llm': self.services.llm_provider is not None
            },
            'features_enabled': {
                'ner_extraction': self.enable_ner,
                'llm_extraction': self.enable_llm_extraction,
                'targeted_prompting': self.enable_targeted_prompting,
                'cross_referencing': self.enable_cross_referencing
            },
            'targeted_extractions_available': list(self.targeted_prompts.keys()),
            'legal_patterns_loaded': len(self.legal_patterns)
        }
    
    async def _extract_with_llm_calibrated(self, text: str) -> str:
        """Extract entities using LLM with structured prompt for calibration.
        
        Args:
            text: Text to analyze
            
        Returns:
            Raw LLM response containing entity information
        """
        if not self.services.llm_provider:
            return ""
        
        prompt = f"""
        Extract legal entities from the following text. Return a JSON array with objects containing:
        - text: the exact entity text
        - label: entity type (PERSON, ORGANIZATION, LOCATION, CASE, STATUTE, COURT, JUDGE, LAWYER, LEGAL_VIOLATION, etc.)
        - confidence: confidence score between 0.0 and 1.0
        
        Text: {text[:4000]}
        
        Response format: [{"text": "entity", "label": "TYPE", "confidence": 0.95}]
        """
        
        try:
            response = await self.services.llm_provider.generate_response(
                prompt=prompt,
                model_params={'temperature': 0.1, 'max_tokens': 2000}
            )
            return response
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return ""
    
    def _convert_calibrated_to_ner_results(self, calibrated_entities: List[EntityPrediction]) -> List[NERResult]:
        """Convert calibrated EntityPrediction objects to legacy NERResult format.
        
        Args:
            calibrated_entities: List of calibrated entity predictions
            
        Returns:
            List of NERResult objects for compatibility
        """
        ner_results = []
        
        for entity in calibrated_entities:
            ner_result = NERResult(
                entity_text=entity.text,
                entity_type=entity.label,
                start_pos=entity.start_pos,
                end_pos=entity.end_pos,
                confidence=entity.calibrated_confidence or entity.raw_confidence,
                source_model=f"calibrated_ensemble[{entity.model_source}]",
                metadata={
                    'raw_confidence': entity.raw_confidence,
                    'calibrated_confidence': entity.calibrated_confidence,
                    'original_model': entity.model_source,
                    'context': entity.context
                }
            )
            ner_results.append(ner_result)
        
        return ner_results
    
    async def train_confidence_calibration(self, validation_samples: List[ValidationSample]) -> None:
        """Train the confidence calibration system.
        
        Args:
            validation_samples: Validation data with ground truth annotations
        """
        if not self.enable_confidence_calibration:
            self.logger.info("Confidence calibration disabled, skipping training")
            return
        
        self.logger.info(f"Training confidence calibration with {len(validation_samples)} samples")
        self.calibration_manager.train_system(validation_samples)
        self.logger.info("Confidence calibration training completed")
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get confidence calibration metrics and status.
        
        Returns:
            Dictionary containing calibration system metrics
        """
        if not self.enable_confidence_calibration:
            return {'calibration_enabled': False}
        
        # Load calibrators to check their status
        self.calibration_manager.ensemble_voter.load_calibrators()
        
        calibrator_status = {}
        for model_name, calibrator in self.calibration_manager.ensemble_voter.calibrators.items():
            calibrator_status[model_name] = {
                'is_fitted': getattr(calibrator, 'is_fitted', False),
                'calibrator_type': type(calibrator).__name__
            }
        
        return {
            'calibration_enabled': True,
            'calibrator_status': calibrator_status,
            'model_weights': self.calibration_manager.ensemble_voter.model_weights,
            'legal_priority_labels': list(self.calibration_manager.ensemble_voter.legal_priority_labels),
            'storage_path': str(self.calibration_manager.storage_path)
        }

    async def close(self):
        """Close the hybrid extraction system."""
        # Clean up any resources if needed
        self.logger.info("Hybrid legal extractor closed")