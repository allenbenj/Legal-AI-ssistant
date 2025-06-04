import logging
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime
import json
from uuid import uuid4

from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class NoteTakingAgent(BaseAgent):
    """
    Intelligent note-taking agent with legal context awareness
    """
    
    def __init__(self, services):
        super().__init__(
            agent_name="NoteTaking",
            agent_type="documentation",
            services=services
        )
        
        # Initialize note frameworks
        self._init_note_frameworks()
        
        logger.info("NoteTakingAgent initialized")
    
    def _init_note_frameworks(self):
        """Initialize note-taking frameworks and patterns"""
        
        # Note type patterns
        self.note_type_patterns = {
            'legal_issue': [
                r'issue\s+is\s+whether', r'question\s+presented', r'legal\s+issue',
                r'matter\s+in\s+dispute', r'key\s+question'
            ],
            'case_citation': [
                r'\w+\s+v\.\s+\w+', r'\d+\s+U\.S\.\s+\d+', r'\d+\s+F\.\d+d\s+\d+',
                r'see\s+also', r'citing', r'relying\s+on'
            ],
            'procedural_note': [
                r'motion\s+to', r'court\s+ordered', r'deadline', r'filing\s+date',
                r'hearing\s+scheduled', r'discovery\s+due'
            ],
            'evidence_note': [
                r'evidence\s+shows', r'exhibit', r'testimony', r'witness\s+stated',
                r'document\s+reveals', r'facts\s+demonstrate'
            ],
            'strategic_note': [
                r'strategy', r'approach', r'consider', r'recommend', r'suggest',
                r'next\s+steps', r'action\s+items'
            ],
            'research_note': [
                r'research\s+needed', r'look\s+into', r'investigate', r'verify',
                r'confirm', r'follow\s+up'
            ]
        }
        
        # Importance indicators for notes
        self.importance_indicators = {
            'critical': [
                r'constitutional', r'dispositive', r'outcome[- ]determinative',
                r'case[- ]dispositive', r'critical', r'essential'
            ],
            'high': [
                r'important', r'significant', r'material', r'substantial',
                r'key\s+issue', r'primary'
            ],
            'medium': [
                r'relevant', r'noteworthy', r'consideration', r'factor'
            ],
            'low': [
                r'minor', r'incidental', r'tangential', r'background'
            ]
        }
        
        # Legal citation patterns for note enhancement
        self.citation_patterns = [
            r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+.*?\d+',  # Case names with citations
            r'\d+\s+U\.S\.C\.\s+§\s*\d+',  # Federal statutes
            r'Fed\.\s*R\.\s*Civ\.\s*P\.\s*\d+',  # Federal Rules
            r'Fed\.\s*R\.\s*Evid\.\s*\d+'  # Evidence Rules
        ]
        
        # Note relationship types
        self.relationship_types = [
            'supports', 'contradicts', 'clarifies', 'extends', 'questions',
            'cites', 'distinguishes', 'follows_from', 'relates_to'
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process input for intelligent note-taking and management
        
        Args:
            input_data: Dictionary containing text, note_action, and context
            
        Returns:
            AgentResult with note suggestions and management operations
        """
        try:
            logger.info(f"Processing note-taking for document: {input_data.get('doc_id', 'unknown')}")
            
            text = input_data.get('text', input_data.get('content', ''))
            doc_id = input_data.get('doc_id', input_data.get('id', 'unknown'))
            note_action = input_data.get('action', 'suggest')  # suggest, create, update, link
            existing_notes = input_data.get('existing_notes', [])
            context = input_data.get('context', {})
            
            if not text and note_action == 'suggest':
                return AgentResult(
                    success=False,
                    data={},
                    metadata={'error': 'No text content provided for note suggestions'},
                    confidence=0.0,
                    processing_time=0.0
                )
            
            start_time = datetime.now()
            
            # Perform note-taking operations based on action
            result_data = await self._perform_note_operation(
                note_action, text, doc_id, existing_notes, context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence
            confidence = self._calculate_note_confidence(result_data, note_action)
            
            logger.info(f"Completed note operation '{note_action}' with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    'agent': self.agent_name,
                    'doc_id': doc_id,
                    'processing_time': processing_time,
                    'note_action': note_action
                },
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in note-taking: {e}")
            return AgentResult(
                success=False,
                data={},
                metadata={'error': str(e)},
                confidence=0.0,
                processing_time=0.0
            )
    
    async def _perform_note_operation(self, action: str, text: str, doc_id: str,
                                    existing_notes: List[Dict[str, Any]],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the requested note operation"""
        
        if action == 'suggest':
            return await self._suggest_notes(text, doc_id, context)
        elif action == 'create':
            return await self._create_note(text, doc_id, context)
        elif action == 'update':
            return await self._update_note(text, doc_id, existing_notes, context)
        elif action == 'link':
            return await self._link_notes(text, doc_id, existing_notes, context)
        elif action == 'consolidate':
            return await self._consolidate_notes(existing_notes, context)
        else:
            raise ValueError(f"Unknown note action: {action}")
    
    async def _suggest_notes(self, text: str, doc_id: str, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest contextually relevant notes based on document content"""
        
        note_suggestions = {
            'doc_id': doc_id,
            'suggested_notes': [],
            'note_opportunities': [],
            'citation_opportunities': [],
            'research_suggestions': [],
            'strategic_insights': []
        }
        
        # Analyze text for note opportunities
        note_opportunities = await self._identify_note_opportunities(text)
        note_suggestions['note_opportunities'] = note_opportunities
        
        # Generate specific note suggestions
        for opportunity in note_opportunities:
            note = await self._generate_note_suggestion(opportunity, text, doc_id)
            if note:
                note_suggestions['suggested_notes'].append(note)
        
        # Identify citation opportunities
        citations = self._extract_citation_opportunities(text)
        note_suggestions['citation_opportunities'] = citations
        
        # Generate research suggestions
        research_suggestions = await self._generate_research_suggestions(text, doc_id)
        note_suggestions['research_suggestions'] = research_suggestions
        
        # Generate strategic insights
        strategic_insights = await self._generate_strategic_insights(text, doc_id)
        note_suggestions['strategic_insights'] = strategic_insights
        
        # Use LLM for enhanced note suggestions
        llm_suggestions = await self._llm_note_suggestions(text, note_suggestions)
        if llm_suggestions:
            note_suggestions['suggested_notes'].extend(llm_suggestions)
        
        return note_suggestions
    
    async def _identify_note_opportunities(self, text: str) -> List[Dict[str, Any]]:
        """Identify opportunities for creating notes in the text"""
        
        opportunities = []
        
        for note_type, patterns in self.note_type_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    context_text = self._extract_context(text, match.start(), match.end())
                    importance = self._assess_note_importance(context_text)
                    
                    opportunity = {
                        'type': note_type,
                        'text_match': match.group(),
                        'context': context_text,
                        'position': match.start(),
                        'importance': importance,
                        'confidence': 0.8
                    }
                    opportunities.append(opportunity)
        
        # Sort by importance and remove duplicates
        opportunities = sorted(opportunities, key=lambda x: x['importance'], reverse=True)
        return opportunities[:10]  # Limit to top 10 opportunities
    
    async def _generate_note_suggestion(self, opportunity: Dict[str, Any], 
                                      text: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Generate a specific note suggestion based on an opportunity"""
        
        note_type = opportunity['type']
        context = opportunity['context']
        
        # Generate note content based on type
        if note_type == 'legal_issue':
            title = "Legal Issue Identified"
            content = f"Issue: {opportunity['text_match']}\n\nContext: {context}\n\nConsider: What law applies? What are the key facts? What arguments support each side?"
            
        elif note_type == 'case_citation':
            title = "Case Citation Found"
            content = f"Citation: {opportunity['text_match']}\n\nContext: {context}\n\nAction Items:\n- Verify citation accuracy\n- Review case for relevance\n- Check if case is still good law"
            
        elif note_type == 'procedural_note':
            title = "Procedural Matter"
            content = f"Procedure: {opportunity['text_match']}\n\nContext: {context}\n\nReminders:\n- Check applicable rules\n- Verify deadlines\n- Confirm proper procedure"
            
        elif note_type == 'evidence_note':
            title = "Evidence Note"
            content = f"Evidence: {opportunity['text_match']}\n\nContext: {context}\n\nConsiderations:\n- Admissibility\n- Authentication\n- Weight and relevance"
            
        elif note_type == 'strategic_note':
            title = "Strategic Consideration"
            content = f"Strategy: {opportunity['text_match']}\n\nContext: {context}\n\nNext Steps:\n- Evaluate options\n- Consider risks/benefits\n- Plan implementation"
            
        elif note_type == 'research_note':
            title = "Research Required"
            content = f"Research: {opportunity['text_match']}\n\nContext: {context}\n\nResearch Tasks:\n- Find relevant authority\n- Analyze precedents\n- Update research"
            
        else:
            return None
        
        return {
            'id': str(uuid4()),
            'title': title,
            'content': content,
            'type': note_type,
            'importance': opportunity['importance'],
            'doc_id': doc_id,
            'position': opportunity['position'],
            'created_at': datetime.now().isoformat(),
            'tags': [note_type, f"importance_{opportunity['importance']}"],
            'auto_generated': True
        }
    
    def _extract_citation_opportunities(self, text: str) -> List[Dict[str, Any]]:
        """Extract citation opportunities from text"""
        
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = {
                    'citation': match.group(),
                    'type': self._classify_citation_type(match.group()),
                    'context': self._extract_context(text, match.start(), match.end()),
                    'position': match.start(),
                    'verification_needed': True,
                    'research_suggestions': self._generate_citation_research_suggestions(match.group())
                }
                citations.append(citation)
        
        return citations
    
    def _classify_citation_type(self, citation: str) -> str:
        """Classify the type of citation"""
        if " v. " in citation:
            return "case"
        elif "U.S.C." in citation:
            return "statute"
        elif "Fed. R." in citation:
            return "rule"
        else:
            return "unknown"
    
    def _generate_citation_research_suggestions(self, citation: str) -> List[str]:
        """Generate research suggestions for a citation"""
        suggestions = [
            "Verify citation accuracy and format",
            "Check if case/statute is still good law",
            "Review for relevance to current matter"
        ]
        
        if " v. " in citation:
            suggestions.extend([
                "Check for subsequent history",
                "Review citing references",
                "Analyze holding and reasoning"
            ])
        elif "U.S.C." in citation:
            suggestions.extend([
                "Check for amendments",
                "Review implementing regulations",
                "Check judicial interpretations"
            ])
        
        return suggestions
    
    async def _generate_research_suggestions(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Generate research suggestions based on document content"""
        
        research_patterns = [
            r'need\s+to\s+research',
            r'unclear\s+whether',
            r'question\s+remains',
            r'further\s+investigation',
            r'verify\s+that',
            r'confirm\s+whether'
        ]
        
        suggestions = []
        
        for pattern in research_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._extract_context(text, match.start(), match.end())
                
                suggestion = {
                    'type': 'research_task',
                    'description': f"Research needed: {match.group()}",
                    'context': context,
                    'priority': 'medium',
                    'estimated_effort': 'moderate',
                    'suggested_resources': ['case law databases', 'statutes', 'secondary sources']
                }
                suggestions.append(suggestion)
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    async def _generate_strategic_insights(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Generate strategic insights and recommendations"""
        
        insights = []
        
        # Look for strategic indicators
        strategic_patterns = [
            (r'strong\s+argument', 'strength', 'This appears to be a strong argument - consider emphasizing'),
            (r'weak\s+position', 'weakness', 'Potential weakness identified - consider addressing or mitigating'),
            (r'precedent\s+supports', 'favorable_precedent', 'Favorable precedent - consider highlighting'),
            (r'distinguishable', 'distinction', 'Distinction from precedent - analyze carefully'),
            (r'constitutional\s+issue', 'constitutional', 'Constitutional issue - requires careful analysis')
        ]
        
        for pattern, insight_type, description in strategic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._extract_context(text, match.start(), match.end())
                
                insight = {
                    'type': insight_type,
                    'description': description,
                    'context': context,
                    'recommendation': self._generate_strategic_recommendation(insight_type),
                    'priority': 'high' if insight_type in ['constitutional', 'weakness'] else 'medium'
                }
                insights.append(insight)
        
        return insights[:5]  # Limit to 5 insights
    
    def _generate_strategic_recommendation(self, insight_type: str) -> str:
        """Generate strategic recommendations based on insight type"""
        
        recommendations = {
            'strength': 'Develop this argument further and place it prominently in your brief',
            'weakness': 'Address this weakness proactively or consider alternative approaches',
            'favorable_precedent': 'Emphasize this precedent and distinguish unfavorable cases',
            'distinction': 'Carefully analyze the distinguishing factors and their significance',
            'constitutional': 'Research constitutional precedents and consider level of scrutiny'
        }
        
        return recommendations.get(insight_type, 'Consider the strategic implications of this point')
    
    async def _create_note(self, text: str, doc_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new note with specified content"""
        
        note = {
            'id': str(uuid4()),
            'title': context.get('title', 'New Note'),
            'content': context.get('content', text[:500]),  # Use provided content or truncated text
            'type': context.get('type', 'general'),
            'doc_id': doc_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'tags': context.get('tags', []),
            'importance': context.get('importance', 'medium'),
            'auto_generated': False
        }
        
        # Store note in memory manager
        try:
            memory_manager = self.services.memory_manager
            await memory_manager.add_context_entry(
                session_id=doc_id,
                entry_type='note',
                content=json.dumps(note),
                importance_score=self._importance_to_score(note['importance'])
            )
        except Exception as e:
            logger.warning(f"Failed to store note in memory: {e}")
        
        return {
            'action': 'create',
            'note': note,
            'success': True
        }
    
    async def _update_note(self, text: str, doc_id: str, existing_notes: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing note"""
        
        note_id = context.get('note_id')
        if not note_id:
            raise ValueError("Note ID required for update operation")
        
        # Find the note to update
        note_to_update = None
        for note in existing_notes:
            if note.get('id') == note_id:
                note_to_update = note
                break
        
        if not note_to_update:
            raise ValueError(f"Note with ID {note_id} not found")
        
        # Update note fields
        updates = context.get('updates', {})
        for field, value in updates.items():
            if field in ['title', 'content', 'type', 'tags', 'importance']:
                note_to_update[field] = value
        
        note_to_update['updated_at'] = datetime.now().isoformat()
        
        return {
            'action': 'update',
            'note': note_to_update,
            'success': True
        }
    
    async def _link_notes(self, text: str, doc_id: str, existing_notes: List[Dict[str, Any]],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Link notes based on content relationships"""
        
        note_links = []
        
        # Find relationships between notes
        for i, note1 in enumerate(existing_notes):
            for j, note2 in enumerate(existing_notes[i+1:], i+1):
                relationship = self._analyze_note_relationship(note1, note2)
                if relationship:
                    note_links.append({
                        'from_note': note1['id'],
                        'to_note': note2['id'],
                        'relationship_type': relationship['type'],
                        'confidence': relationship['confidence'],
                        'description': relationship['description']
                    })
        
        return {
            'action': 'link',
            'links_created': len(note_links),
            'note_links': note_links,
            'success': True
        }
    
    async def _consolidate_notes(self, existing_notes: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate related notes into summaries"""
        
        # Group notes by topic/type
        note_groups = self._group_notes_by_topic(existing_notes)
        
        consolidated_notes = []
        
        for topic, notes in note_groups.items():
            if len(notes) > 1:  # Only consolidate if multiple notes
                consolidated = await self._create_consolidated_note(topic, notes)
                consolidated_notes.append(consolidated)
        
        return {
            'action': 'consolidate',
            'groups_found': len(note_groups),
            'consolidated_notes': consolidated_notes,
            'success': True
        }
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around a text position"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _assess_note_importance(self, context_text: str) -> str:
        """Assess the importance level of a potential note"""
        
        context_lower = context_text.lower()
        
        # Check for importance indicators
        for level, indicators in self.importance_indicators.items():
            for indicator in indicators:
                if re.search(indicator, context_lower):
                    return level
        
        return 'medium'  # Default importance
    
    def _importance_to_score(self, importance: str) -> float:
        """Convert importance level to numerical score"""
        importance_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        return importance_scores.get(importance, 0.6)
    
    def _analyze_note_relationship(self, note1: Dict[str, Any], 
                                 note2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze the relationship between two notes"""
        
        content1 = note1.get('content', '').lower()
        content2 = note2.get('content', '').lower()
        
        # Simple relationship detection based on content overlap
        common_words = set(content1.split()) & set(content2.split())
        overlap_ratio = len(common_words) / max(len(content1.split()), len(content2.split()), 1)
        
        if overlap_ratio > 0.3:
            return {
                'type': 'relates_to',
                'confidence': overlap_ratio,
                'description': f"Notes share {len(common_words)} common concepts"
            }
        
        # Check for citation relationships
        if any(word in content1 for word in ['citing', 'see also', 'compare']):
            if any(word in content2 for word in ['case', 'statute', 'rule']):
                return {
                    'type': 'cites',
                    'confidence': 0.8,
                    'description': "Note references legal authority mentioned in other note"
                }
        
        return None
    
    def _group_notes_by_topic(self, notes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group notes by topic based on content similarity"""
        
        groups = {}
        
        for note in notes:
            note_type = note.get('type', 'general')
            tags = note.get('tags', [])
            
            # Simple grouping by type and tags
            group_key = f"{note_type}_{':'.join(sorted(tags))}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(note)
        
        return groups
    
    async def _create_consolidated_note(self, topic: str, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a consolidated note from multiple related notes"""
        
        consolidated_content = f"Consolidated Notes: {topic}\n\n"
        
        for i, note in enumerate(notes, 1):
            consolidated_content += f"{i}. {note.get('title', 'Untitled')}\n"
            consolidated_content += f"   {note.get('content', '')[:200]}...\n\n"
        
        return {
            'id': str(uuid4()),
            'title': f"Consolidated: {topic}",
            'content': consolidated_content,
            'type': 'consolidated',
            'source_notes': [note['id'] for note in notes],
            'created_at': datetime.now().isoformat(),
            'tags': ['consolidated'] + list(set().union(*[note.get('tags', []) for note in notes])),
            'importance': 'high',
            'auto_generated': True
        }
    
    async def _llm_note_suggestions(self, text: str, 
                                  current_suggestions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to generate additional note suggestions"""
        try:
            llm_manager = self.services.llm_manager
            
            existing_count = len(current_suggestions.get('suggested_notes', []))
            
            prompt = f"""
            Analyze this legal document and suggest 3-5 additional notes that would be valuable for legal analysis.
            
            Document text (first 1500 chars):
            {text[:1500]}
            
            I already have {existing_count} note suggestions. Please suggest additional notes focusing on:
            - Key legal arguments or issues not already covered
            - Important procedural considerations
            - Strategic insights for case development
            - Research priorities
            
            For each note, provide:
            - A clear, specific title
            - Brief content explaining the note's importance
            - Suggested type (legal_issue, strategic, research, procedural, evidence)
            
            Format as a simple list.
            """
            
            response = await llm_manager.complete(prompt, max_tokens=500)
            
            # Parse LLM response into note suggestions
            return self._parse_llm_note_suggestions(response)
            
        except Exception as e:
            logger.warning(f"LLM note suggestions failed: {e}")
            return []
    
    def _parse_llm_note_suggestions(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured note suggestions"""
        
        suggestions = []
        
        # Simple parsing - in production would use structured output
        lines = response.split('\n')
        current_note = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for title patterns
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                if current_note:
                    suggestions.append(current_note)
                
                title = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                current_note = {
                    'id': str(uuid4()),
                    'title': title[:100],  # Limit title length
                    'content': f"LLM Suggestion: {title}",
                    'type': 'llm_suggestion',
                    'importance': 'medium',
                    'auto_generated': True,
                    'created_at': datetime.now().isoformat(),
                    'tags': ['llm_generated']
                }
            elif current_note and len(line) > 10:
                # Add to content
                current_note['content'] += f"\n{line}"
        
        # Add the last note
        if current_note:
            suggestions.append(current_note)
        
        return suggestions[:3]  # Limit to 3 LLM suggestions
    
    def _calculate_note_confidence(self, result_data: Dict[str, Any], action: str) -> float:
        """Calculate confidence in note-taking operations"""
        
        if action == 'suggest':
            # Base confidence on number and quality of suggestions
            suggestions = result_data.get('suggested_notes', [])
            opportunities = result_data.get('note_opportunities', [])
            
            if not suggestions and not opportunities:
                return 0.3
            
            # Higher confidence with more high-importance suggestions
            high_importance_count = sum(1 for s in suggestions if s.get('importance') in ['high', 'critical'])
            total_count = len(suggestions)
            
            if total_count == 0:
                return 0.5
            
            importance_ratio = high_importance_count / total_count
            return min(0.9, 0.5 + (importance_ratio * 0.3) + (total_count * 0.05))
            
        elif action in ['create', 'update']:
            return 0.9  # High confidence for explicit user actions
            
        elif action == 'link':
            links = result_data.get('note_links', [])
            if links:
                avg_confidence = sum(link.get('confidence', 0.5) for link in links) / len(links)
                return avg_confidence
            return 0.5
            
        elif action == 'consolidate':
            consolidated = result_data.get('consolidated_notes', [])
            return 0.8 if consolidated else 0.4
            
        else:
            return 0.5