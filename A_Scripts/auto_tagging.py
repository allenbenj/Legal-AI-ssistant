"""
Auto Tagging Agent - Learning-based Document Classification and Tagging

This agent automatically tags and classifies legal documents based on:
- Document type detection (motion, brief, order, statute, etc.)
- Subject matter classification (criminal, civil, constitutional, etc.)
- Legal domain tagging (family law, corporate law, immigration, etc.)
- Case status and procedural stage identification
- Legal entity extraction and tagging
- Importance and urgency scoring
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import re
from datetime import datetime
from collections import Counter
import hashlib

from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class AutoTaggingAgent(BaseAgent):
    """
    Intelligent auto-tagging agent that learns from patterns and user feedback
    """
    
    def __init__(self, services):
        super().__init__(
            agent_name="AutoTagging",
            agent_type="classification",
            services=services
        )
        
        # Initialize tagging frameworks
        self._init_tagging_frameworks()
        self._init_learning_system()
        
        logger.info("AutoTaggingAgent initialized")
    
    def _init_tagging_frameworks(self):
        """Initialize document classification and tagging frameworks"""
        
        # Document type patterns
        self.document_type_patterns = {
            'motion': [
                r'motion\s+to\s+\w+', r'plaintiff.*moves', r'defendant.*moves',
                r'motion\s+for\s+(summary\s+)?judgment', r'motion\s+to\s+dismiss',
                r'motion\s+to\s+suppress', r'motion\s+in\s+limine'
            ],
            'brief': [
                r'brief\s+in\s+(support|opposition)', r'memorandum\s+of\s+law',
                r'legal\s+brief', r'appellate\s+brief', r'trial\s+brief'
            ],
            'order': [
                r'court\s+orders?', r'it\s+is\s+(hereby\s+)?ordered',
                r'judgment\s+entered', r'ruling\s+on', r'decision\s+and\s+order'
            ],
            'complaint': [
                r'complaint\s+for', r'plaintiff\s+alleges', r'cause\s+of\s+action',
                r'wherefore.*judgment', r'jury\s+trial\s+demanded'
            ],
            'answer': [
                r'answer\s+to\s+complaint', r'defendant\s+admits',
                r'defendant\s+denies', r'affirmative\s+defenses'
            ],
            'discovery': [
                r'request\s+for\s+production', r'interrogatories',
                r'request\s+for\s+admission', r'deposition', r'subpoena'
            ],
            'contract': [
                r'this\s+agreement', r'parties\s+agree', r'terms\s+and\s+conditions',
                r'whereas.*now\s+therefore', r'consideration'
            ],
            'statute': [
                r'section\s+\d+', r'subsection\s+\([a-z]\)', r'pursuant\s+to.*u\.?s\.?c',
                r'code\s+section', r'statute\s+provides'
            ]
        }
        
        # Legal domain patterns
        self.legal_domain_patterns = {
            'criminal': [
                r'defendant\s+charged', r'criminal\s+case', r'prosecution',
                r'guilty\s+plea', r'sentencing', r'arraignment', r'indictment'
            ],
            'civil': [
                r'civil\s+action', r'damages', r'liability', r'breach\s+of\s+contract',
                r'negligence', r'tort\s+claim'
            ],
            'constitutional': [
                r'constitutional\s+right', r'amendment', r'due\s+process',
                r'equal\s+protection', r'first\s+amendment', r'fourth\s+amendment'
            ],
            'family': [
                r'divorce', r'custody', r'child\s+support', r'alimony',
                r'domestic\s+relations', r'family\s+court'
            ],
            'corporate': [
                r'corporation', r'shareholder', r'securities', r'merger',
                r'acquisition', r'corporate\s+governance'
            ],
            'immigration': [
                r'immigration', r'visa', r'deportation', r'asylum',
                r'citizenship', r'green\s+card'
            ],
            'employment': [
                r'employment', r'wrongful\s+termination', r'discrimination',
                r'harassment', r'wages', r'labor\s+law'
            ],
            'intellectual_property': [
                r'copyright', r'trademark', r'patent', r'trade\s+secret',
                r'intellectual\s+property', r'infringement'
            ],
            'real_estate': [
                r'real\s+estate', r'property', r'deed', r'mortgage',
                r'zoning', r'easement', r'title'
            ]
        }
        
        # Procedural stage patterns
        self.procedural_stage_patterns = {
            'pleading': [
                r'complaint', r'answer', r'counterclaim', r'cross[- ]claim'
            ],
            'discovery': [
                r'discovery', r'interrogatories', r'deposition', r'document\s+production'
            ],
            'pre_trial': [
                r'pre[- ]?trial', r'motion\s+for\s+summary', r'motion\s+in\s+limine'
            ],
            'trial': [
                r'trial', r'jury\s+selection', r'opening\s+statement', r'closing\s+argument'
            ],
            'post_trial': [
                r'post[- ]?trial', r'motion\s+for\s+new\s+trial', r'judgment'
            ],
            'appeal': [
                r'appeal', r'appellate', r'supreme\s+court', r'court\s+of\s+appeals'
            ]
        }
        
        # Importance indicators
        self.importance_indicators = {
            'high': [
                r'constitutional', r'supreme\s+court', r'class\s+action',
                r'injunctive\s+relief', r'emergency', r'urgent'
            ],
            'medium': [
                r'significant', r'important', r'material', r'substantial'
            ],
            'low': [
                r'minor', r'routine', r'administrative', r'procedural'
            ]
        }
        
        # Legal entity patterns
        self.entity_patterns = {
            'court': [
                r'supreme\s+court', r'court\s+of\s+appeals', r'district\s+court',
                r'circuit\s+court', r'trial\s+court', r'federal\s+court'
            ],
            'judge': [
                r'judge\s+\w+', r'justice\s+\w+', r'hon\.\s+\w+', r'chief\s+judge'
            ],
            'attorney': [
                r'attorney\s+for', r'counsel\s+for', r'esq\.', r'law\s+firm'
            ],
            'case': [
                r'\w+\s+v\.\s+\w+', r'case\s+no\.?\s*\d+', r'docket\s+no\.?\s*\d+'
            ]
        }
    
    def _init_learning_system(self):
        """Initialize the learning system for improving tagging accuracy"""
        
        # User feedback tracking
        self.feedback_history = {}
        self.tag_accuracy_scores = {}
        self.pattern_effectiveness = {}
        
        # Learning parameters
        self.learning_threshold = 0.1  # Minimum confidence change to update patterns
        self.feedback_weight = 0.3     # Weight of user feedback in learning
        
        logger.info("Learning system initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process document for automatic tagging and classification
        
        Args:
            input_data: Dictionary containing text and metadata for tagging
            
        Returns:
            AgentResult with tags, classifications, and confidence scores
        """
        try:
            logger.info(f"Processing auto-tagging for document: {input_data.get('doc_id', 'unknown')}")
            
            text = input_data.get('text', input_data.get('content', ''))
            doc_id = input_data.get('doc_id', input_data.get('id', 'unknown'))
            existing_tags = input_data.get('existing_tags', [])
            user_feedback = input_data.get('feedback', {})
            
            if not text:
                return AgentResult(
                    success=False,
                    data={},
                    metadata={'error': 'No text content provided'},
                    confidence=0.0,
                    processing_time=0.0
                )
            
            start_time = datetime.now()
            
            # Apply user feedback to improve tagging
            if user_feedback:
                await self._apply_user_feedback(doc_id, user_feedback)
            
            # Perform comprehensive tagging
            tagging_result = await self._perform_comprehensive_tagging(text, doc_id, existing_tags)
            
            # Learn from this tagging session
            await self._update_learning_model(doc_id, tagging_result, text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate overall confidence
            confidence = self._calculate_tagging_confidence(tagging_result)
            
            logger.info(f"Generated {len(tagging_result.get('all_tags', []))} tags with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                data=tagging_result,
                metadata={
                    'agent': self.agent_name,
                    'doc_id': doc_id,
                    'processing_time': processing_time,
                    'learning_applied': bool(user_feedback)
                },
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in auto-tagging: {e}")
            return AgentResult(
                success=False,
                data={},
                metadata={'error': str(e)},
                confidence=0.0,
                processing_time=0.0
            )
    
    async def _perform_comprehensive_tagging(self, text: str, doc_id: str, 
                                           existing_tags: List[str]) -> Dict[str, Any]:
        """Perform comprehensive document tagging and classification"""
        
        tagging_result = {
            'doc_id': doc_id,
            'document_type': await self._classify_document_type(text),
            'legal_domain': await self._classify_legal_domain(text),
            'procedural_stage': await self._identify_procedural_stage(text),
            'importance_level': await self._assess_importance(text),
            'legal_entities': await self._extract_legal_entities(text),
            'subject_tags': await self._generate_subject_tags(text),
            'confidence_scores': {},
            'all_tags': [],
            'suggested_tags': [],
            'existing_tags': existing_tags
        }
        
        # Combine all tags
        all_tags = []
        
        # Add classification tags
        if tagging_result['document_type']:
            all_tags.append(f"doc_type:{tagging_result['document_type']['type']}")
        
        if tagging_result['legal_domain']:
            all_tags.extend([f"domain:{domain}" for domain in tagging_result['legal_domain']])
        
        if tagging_result['procedural_stage']:
            all_tags.append(f"stage:{tagging_result['procedural_stage']['stage']}")
        
        if tagging_result['importance_level']:
            all_tags.append(f"importance:{tagging_result['importance_level']['level']}")
        
        # Add entity tags
        for entity_type, entities in tagging_result['legal_entities'].items():
            for entity in entities[:3]:  # Limit to top 3 per type
                all_tags.append(f"entity:{entity_type}:{entity['name']}")
        
        # Add subject tags
        all_tags.extend(tagging_result['subject_tags'])
        
        # Use LLM for additional tagging
        llm_tags = await self._llm_generate_tags(text, all_tags)
        all_tags.extend(llm_tags)
        
        # Remove duplicates and apply learning filters
        all_tags = list(set(all_tags))
        filtered_tags = await self._apply_learning_filters(all_tags, text)
        
        tagging_result['all_tags'] = filtered_tags
        tagging_result['suggested_tags'] = [tag for tag in filtered_tags if tag not in existing_tags]
        
        return tagging_result
    
    async def _classify_document_type(self, text: str) -> Optional[Dict[str, Any]]:
        """Classify the type of legal document"""
        
        type_scores = {}
        
        for doc_type, patterns in self.document_type_patterns.items():
            matches = self._find_pattern_matches(text, patterns)
            if matches:
                # Calculate score based on number and strength of matches
                score = len(matches) * 0.2 + sum(m['confidence'] for m in matches) / len(matches)
                type_scores[doc_type] = min(score, 1.0)
        
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return {
                'type': best_type[0],
                'confidence': best_type[1],
                'all_scores': type_scores
            }
        
        return None
    
    async def _classify_legal_domain(self, text: str) -> List[str]:
        """Classify the legal domain(s) of the document"""
        
        domain_scores = {}
        
        for domain, patterns in self.legal_domain_patterns.items():
            matches = self._find_pattern_matches(text, patterns)
            if matches:
                score = len(matches) * 0.1 + sum(m['confidence'] for m in matches) / len(matches)
                domain_scores[domain] = min(score, 1.0)
        
        # Return domains with score above threshold, sorted by score
        threshold = 0.3
        relevant_domains = [
            domain for domain, score in domain_scores.items() 
            if score >= threshold
        ]
        
        return sorted(relevant_domains, key=lambda d: domain_scores[d], reverse=True)
    
    async def _identify_procedural_stage(self, text: str) -> Optional[Dict[str, Any]]:
        """Identify the procedural stage of the case"""
        
        stage_scores = {}
        
        for stage, patterns in self.procedural_stage_patterns.items():
            matches = self._find_pattern_matches(text, patterns)
            if matches:
                score = len(matches) * 0.3 + sum(m['confidence'] for m in matches) / len(matches)
                stage_scores[stage] = min(score, 1.0)
        
        if stage_scores:
            best_stage = max(stage_scores.items(), key=lambda x: x[1])
            return {
                'stage': best_stage[0],
                'confidence': best_stage[1]
            }
        
        return None
    
    async def _assess_importance(self, text: str) -> Optional[Dict[str, Any]]:
        """Assess the importance level of the document"""
        
        importance_scores = {'high': 0, 'medium': 0, 'low': 0}
        
        for level, patterns in self.importance_indicators.items():
            matches = self._find_pattern_matches(text, patterns)
            if matches:
                score = len(matches) * 0.2 + sum(m['confidence'] for m in matches) / len(matches)
                importance_scores[level] = min(score, 1.0)
        
        # Determine overall importance
        if importance_scores['high'] > 0.5:
            level = 'high'
        elif importance_scores['medium'] > 0.3:
            level = 'medium'
        elif importance_scores['low'] > 0.2:
            level = 'low'
        else:
            level = 'medium'  # Default
        
        return {
            'level': level,
            'confidence': importance_scores[level],
            'all_scores': importance_scores
        }
    
    async def _extract_legal_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract legal entities from the document"""
        
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            entity_matches = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group().strip()
                    context = self._extract_context(text, match.start(), match.end())
                    
                    entity_matches.append({
                        'name': entity_name,
                        'context': context,
                        'confidence': 0.8,
                        'position': match.start()
                    })
            
            if entity_matches:
                # Remove duplicates and sort by confidence
                unique_entities = {}
                for entity in entity_matches:
                    name = entity['name'].lower()
                    if name not in unique_entities or entity['confidence'] > unique_entities[name]['confidence']:
                        unique_entities[name] = entity
                
                entities[entity_type] = sorted(unique_entities.values(), 
                                             key=lambda x: x['confidence'], reverse=True)
        
        return entities
    
    async def _generate_subject_tags(self, text: str) -> List[str]:
        """Generate subject matter tags from document content"""
        
        # Extract key legal terms and concepts
        legal_terms_patterns = [
            r'motion\s+to\s+\w+',
            r'\w+\s+violation',
            r'\w+\s+amendment',
            r'\w+\s+rights?',
            r'cause\s+of\s+action',
            r'affirmative\s+defense',
            r'statute\s+of\s+limitations'
        ]
        
        subject_tags = []
        
        for pattern in legal_terms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group().strip().lower().replace(' ', '_')
                if len(term) > 3 and term not in subject_tags:
                    subject_tags.append(f"subject:{term}")
        
        # Limit to most relevant tags
        return subject_tags[:10]
    
    async def _llm_generate_tags(self, text: str, existing_tags: List[str]) -> List[str]:
        """Use LLM to generate additional tags"""
        try:
            llm_manager = self.services.llm_manager
            
            existing_tags_str = ', '.join(existing_tags[:10])  # Limit context
            
            prompt = f"""
            Analyze this legal document and suggest additional relevant tags.
            
            Current tags: {existing_tags_str}
            
            Document text (first 1500 chars):
            {text[:1500]}
            
            Suggest 5-10 additional tags that would be useful for categorizing this document.
            Focus on:
            - Legal concepts not already covered
            - Specific legal procedures or doctrines
            - Parties or entities involved
            - Jurisdictional aspects
            
            Return only the tag names, one per line.
            """
            
            response = await llm_manager.complete(prompt, max_tokens=200)
            
            # Parse LLM response
            llm_tags = []
            for line in response.split('\n'):
                tag = line.strip()
                if tag and len(tag) > 2 and not tag.startswith('-'):
                    # Clean and format tag
                    clean_tag = re.sub(r'[^\w\s]', '', tag).lower().replace(' ', '_')
                    if clean_tag not in existing_tags:
                        llm_tags.append(f"llm:{clean_tag}")
            
            return llm_tags[:5]  # Limit to 5 LLM-generated tags
            
        except Exception as e:
            logger.warning(f"LLM tag generation failed: {e}")
            return []
    
    def _find_pattern_matches(self, text: str, patterns: List[str]) -> List[Dict[str, Any]]:
        """Find pattern matches in text"""
        matches = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,  # Base confidence
                    'pattern': pattern
                })
        
        return matches
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    async def _apply_learning_filters(self, tags: List[str], text: str) -> List[str]:
        """Apply learning-based filters to improve tag quality"""
        
        filtered_tags = []
        
        for tag in tags:
            # Get historical accuracy for this tag
            accuracy = self.tag_accuracy_scores.get(tag, 0.5)  # Default moderate accuracy
            
            # Apply learning threshold
            if accuracy >= 0.3:  # Keep tags with reasonable accuracy
                filtered_tags.append(tag)
        
        return filtered_tags
    
    async def _apply_user_feedback(self, doc_id: str, feedback: Dict[str, Any]):
        """Apply user feedback to improve future tagging"""
        
        # Store feedback
        feedback_key = f"{doc_id}_{datetime.now().isoformat()}"
        self.feedback_history[feedback_key] = feedback
        
        # Update tag accuracy scores
        correct_tags = feedback.get('correct_tags', [])
        incorrect_tags = feedback.get('incorrect_tags', [])
        
        for tag in correct_tags:
            current_score = self.tag_accuracy_scores.get(tag, 0.5)
            self.tag_accuracy_scores[tag] = min(1.0, current_score + self.feedback_weight)
        
        for tag in incorrect_tags:
            current_score = self.tag_accuracy_scores.get(tag, 0.5)
            self.tag_accuracy_scores[tag] = max(0.0, current_score - self.feedback_weight)
        
        logger.info(f"Applied user feedback for document {doc_id}")
    
    async def _update_learning_model(self, doc_id: str, tagging_result: Dict[str, Any], text: str):
        """Update the learning model based on tagging results"""
        
        # Create a signature for this document type/content
        content_signature = hashlib.md5(text[:500].encode()).hexdigest()
        
        # Store tagging patterns for learning
        patterns_used = {
            'document_type': tagging_result.get('document_type', {}).get('type'),
            'legal_domains': tagging_result.get('legal_domain', []),
            'procedural_stage': tagging_result.get('procedural_stage', {}).get('stage'),
            'tags_generated': len(tagging_result.get('all_tags', []))
        }
        
        # Update pattern effectiveness (simplified learning)
        for pattern_type, value in patterns_used.items():
            if value:
                key = f"{pattern_type}:{value}"
                current_effectiveness = self.pattern_effectiveness.get(key, 0.5)
                # Slight increase in effectiveness for patterns that are used
                self.pattern_effectiveness[key] = min(1.0, current_effectiveness + 0.05)
    
    def _calculate_tagging_confidence(self, tagging_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in the tagging results"""
        
        confidence_factors = []
        
        # Document type confidence
        doc_type = tagging_result.get('document_type')
        if doc_type:
            confidence_factors.append(doc_type.get('confidence', 0.5))
        
        # Legal domain confidence
        domains = tagging_result.get('legal_domain', [])
        if domains:
            confidence_factors.append(0.8)  # Base confidence for domain classification
        
        # Procedural stage confidence
        stage = tagging_result.get('procedural_stage')
        if stage:
            confidence_factors.append(stage.get('confidence', 0.5))
        
        # Importance level confidence
        importance = tagging_result.get('importance_level')
        if importance:
            confidence_factors.append(importance.get('confidence', 0.5))
        
        # Entity extraction confidence
        entities = tagging_result.get('legal_entities', {})
        if entities:
            avg_entity_confidence = 0.7  # Base confidence for entity extraction
            confidence_factors.append(avg_entity_confidence)
        
        # Tag count factor (more tags generally indicate better analysis)
        tag_count = len(tagging_result.get('all_tags', []))
        tag_confidence = min(0.9, 0.3 + (tag_count * 0.05))
        confidence_factors.append(tag_confidence)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        
        return {
            'feedback_sessions': len(self.feedback_history),
            'tracked_tags': len(self.tag_accuracy_scores),
            'pattern_effectiveness_entries': len(self.pattern_effectiveness),
            'average_tag_accuracy': sum(self.tag_accuracy_scores.values()) / len(self.tag_accuracy_scores) if self.tag_accuracy_scores else 0.0,
            'learning_parameters': {
                'learning_threshold': self.learning_threshold,
                'feedback_weight': self.feedback_weight
            }
        }