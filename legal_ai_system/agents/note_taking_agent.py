# legal_ai_system/agents/note_taking/note_taking_agent.py
"""
NoteTakingAgent - Intelligent note-taking with legal context awareness.
"""

# import logging # Replaced by detailed_logging
from typing import Dict, List, Optional, Any
import asyncio
import re
from datetime import datetime, timezone  # Added timezone
import json
from uuid import uuid4
from dataclasses import dataclass, field, asdict  # Added

# Core imports from the new structure
from ..core.base_agent import BaseAgent
from ..core.llm_providers import LLMManager, LLMProviderError
from ..core.unified_exceptions import MemoryManagerError

from ..core.agent_unified_config import create_agent_memory_mixin

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()

# Assuming UnifiedMemoryManager is the service for memory
from ..core.unified_memory_manager import UnifiedMemoryManager

# Logger will be inherited from BaseAgent.

@dataclass
class Note: # Dataclass for a single note
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = "Untitled Note"
    content: str = ""
    note_type: str = "general" # e.g., legal_issue, case_citation, strategic
    importance: str = "medium" # low, medium, high, critical
    document_id: Optional[str] = None # Link to source document
    position_in_doc: Optional[int] = None # Character offset if applicable
    tags: List[str] = field(default_factory=list)
    related_note_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_auto_generated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NoteTakingOutput: # Dataclass for the output of this agent's operations
    action_performed: str # e.g., suggest, create, update, link
    success: bool = True
    notes: List[Note] = field(default_factory=list) # List of created/updated/suggested notes
    links_created_count: int = 0
    note_opportunities_found: int = 0 # For 'suggest' action
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # data['notes'] = [n.to_dict() for n in self.notes] # asdict handles this
        return data


class NoteTakingAgent(BaseAgent, MemoryMixin):
    """
    Intelligent note-taking agent with legal context awareness.
    """
    
    def __init__(self, service_container: Any, **config: Any): # Renamed services, added config
        super().__init__(service_container, name="NoteTakingAgent", agent_type="documentation")
        
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(
            f"NoteTakingAgent configured with model: {self.llm_config.get('llm_model', 'default')}"
        )
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.memory_manager: Optional[UnifiedMemoryManager] = self._get_service("unified_memory_manager")

        self._init_note_frameworks()
        
        # Configuration
        self.default_note_importance = config.get('default_note_importance', 'medium')
        self.max_llm_suggestions = int(config.get('max_llm_suggestions', 3))

        self.logger.info("NoteTakingAgent initialized.")
    
    def _init_note_frameworks(self):
        """Initialize note-taking frameworks and patterns."""
        # ... (pattern definitions remain largely the same)
        self.note_type_patterns = {
            'legal_issue': [r'issue\s+is\s+whether', r'question\s+presented'],
            'case_citation': [r'\w+\s+v\.\s+\w+', r'\d+\s+U\.S\.\s+\d+'],
            # ... more patterns
        }
        self.importance_indicators = {
            'critical': [r'constitutional', r'dispositive'], 'high': [r'important', r'significant'],
            # ... more indicators
        }
        self.citation_patterns_for_notes = [ # Renamed from self.citation_patterns
            r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+.*?\d+', r'\d+\s+U\.S\.C\.\s+§\s*\d+'
        ]
        self.note_relationship_types = [ # For linking notes
            'supports', 'contradicts', 'clarifies', 'extends', 'cites'
        ]
        self.logger.debug("Note-taking frameworks and patterns initialized.")

    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for intelligent note-taking and management.
        task_data: 'text_content', 'action' (suggest, create, etc.), 'existing_notes_data', 'context_info'
        metadata: 'document_id'
        """
        document_id = metadata.get('document_id', f"unknown_doc_{uuid4().hex[:8]}")
        text_content = task_data.get('text_content', task_data.get('text', task_data.get('content', ''))) # Flexible text input
        note_action = task_data.get('action', 'suggest').lower()
        context_info = task_data.get('context_info', {})  # For create/update actions

        self.logger.info(f"Processing note-taking action '{note_action}'.", 
                        parameters={'doc_id': document_id, 'text_len': len(text_content)})
        
        output = NoteTakingOutput(action_performed=note_action, success=False)

        if not text_content and note_action == 'suggest':
            self.logger.warning("No text content provided for note suggestions.", parameters={'doc_id': document_id})
            output.errors.append('No text content provided for note suggestions.')
            return output.to_dict()
            
        try:
            if note_action == 'suggest':
                suggest_output_data = await self._suggest_notes_async(text_content, document_id)
                output.notes = suggest_output_data.get('suggested_notes_obj', []) # Expect List[Note]
                output.note_opportunities_found = suggest_output_data.get('note_opportunities_count', 0)
                # Populate other fields of output from suggest_output_data if needed
                output.success = True
            elif note_action == 'create':
                created_note = await self._create_note_async(text_content, document_id, context_info) # Renamed
                if created_note:
                    output.notes = [created_note]
                    output.success = True
                else:
                    output.errors.append("Failed to create note.")
            # ... Implement other actions: update, link, consolidate
            else:
                self.logger.warning(f"Unknown note action requested: {note_action}")
                output.errors.append(f"Unknown note action: {note_action}")

            if output.success:
                self.logger.info(f"Note action '{note_action}' completed successfully.", 
                                parameters={'doc_id': document_id, 'notes_count': len(output.notes)})
            else:
                self.logger.warning(
                    f"Note action '{note_action}' failed or produced no results.",
                    parameters={'doc_id': document_id, 'errors': output.errors}
                )

        except Exception as e:
            self.logger.error(f"Error during note-taking action '{note_action}'.", 
                             parameters={'doc_id': document_id}, exception=e)
            output.errors.append(f"Note-taking failed: {str(e)}")
            # Consider raising AgentExecutionError(str(e), cause=e)
        
        return output.to_dict()

    async def _suggest_notes_async(self, text: str, doc_id: str) -> Dict[str, Any]:
        """Suggest contextually relevant notes based on document content (async wrapper)."""
        # Pattern-based suggestions are CPU-bound, LLM is IO-bound
        self.logger.debug("Generating note suggestions.", parameters={'doc_id': doc_id})
        
        # Run pattern matching in a background thread
        # asyncio.to_thread ensures the CPU-bound regex logic doesn't block the event loop
        pattern_opportunities = await asyncio.to_thread(
            self._identify_note_opportunities_sync, text
        )
        
        suggested_notes_list: List[Note] = []
        for opp_data in pattern_opportunities:
            note_obj = self._generate_note_from_opportunity_sync(opp_data, doc_id)
            if note_obj: suggested_notes_list.append(note_obj)
        
        # LLM for enhanced suggestions (async)
        if self.llm_manager and self.config.get('enable_llm_note_suggestions', True):
            try:
                llm_suggested_notes = await self._llm_generate_note_suggestions(text, suggested_notes_list, doc_id) # Renamed
                suggested_notes_list.extend(llm_suggested_notes)
            except Exception as llm_e:
                self.logger.error("LLM note suggestion generation failed.", parameters={'doc_id': doc_id}, exception=llm_e)

        # Further processing like citation opportunities, research suggestions, etc.
        # can be added here, similar to the original file's structure.
        # For brevity, focusing on the core suggestion flow.
        
        return {
            'suggested_notes_obj': suggested_notes_list, # List of Note objects
            'note_opportunities_count': len(pattern_opportunities)
            # ... other suggestion categories
        }

    def _identify_note_opportunities_sync(self, text: str) -> List[Dict[str, Any]]: # Renamed
        """Identify opportunities for creating notes (synchronous regex part)."""
        # ... (logic from original _identify_note_opportunities)
        opportunities = []
        for note_type, patterns in self.note_type_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                        context_text = self._extract_context_around_match(text, match.start(), match.end()) # Renamed
                        importance_level = self._assess_note_importance_level(context_text) # Renamed
                        opportunities.append({
                            'type': note_type, 'text_match': match.group(0), 'context': context_text,
                            'position': match.start(), 'importance': importance_level, 'confidence': 0.8 # Base confidence
                        })
                except re.error as re_err:
                    self.logger.warning("Regex error during note opportunity search.", parameters={'pattern': pattern, 'error': str(re_err)})
        
        # Sort by importance heuristic and limit
        opportunities.sort(key=lambda x: (x['importance'] != 'critical', x['importance'] != 'high', -x['confidence']))
        return opportunities[:15] # Limit opportunities

    def _generate_note_from_opportunity_sync(
        self, opportunity_data: Dict[str, Any], doc_id: str
    ) -> Optional[Note]:
        """Generate a Note object from an opportunity (synchronous)."""
        # ... (logic from original _generate_note_suggestion, but creates Note object)
        note_type = opportunity_data['type']
        title = f"Suggested Note: {note_type.replace('_', ' ').title()}"
        content_parts = [f"Potential {note_type.replace('_', ' ')} identified: '{opportunity_data['text_match']}'",
                         f"Context: {opportunity_data['context'][:200]}..."] # Limit context in note content
        
        # Add type-specific content prompts
        if note_type == 'legal_issue': content_parts.append("Consider: Applicable law? Key facts? Arguments?")
        elif note_type == 'case_citation': content_parts.append("Action: Verify citation, review relevance, check good law.")
        # ... more specific prompts

        return Note(
            title=title, content="\n".join(content_parts), note_type=note_type,
            importance=opportunity_data['importance'], document_id=doc_id,
            position_in_doc=opportunity_data['position'], is_auto_generated=True,
            tags=[note_type, f"importance:{opportunity_data['importance']}", "auto_suggested"]
        )

    async def _create_note_async(self, text_content_for_note: str, doc_id: str, context_info: Dict[str, Any]) -> Optional[Note]: # Renamed
        """Create a new note with specified content (async for memory manager call)."""
        # ... (logic from original _create_note, creates Note object, saves to UMM)
        note_obj = Note(
            title=context_info.get('title', f"Note for {doc_id}" if doc_id != "unknown_doc" else "New Note"),
            content=context_info.get('content', text_content_for_note[:1000]), # Limit content length
            note_type=context_info.get('type', 'general'),
            document_id=doc_id,
            importance=context_info.get('importance', self.default_note_importance),
            tags=context_info.get('tags', []),
            is_auto_generated=False # User-created or explicitly created by system
        )
        note_obj.updated_at = datetime.now(timezone.utc).isoformat() # Ensure updated_at is also set

        if self.memory_manager:
            try:
                # Adapt to UnifiedMemoryManager's expected storage method for notes
                # Example: UMM might have a generic store_object or specific store_note
                await self.memory_manager.store_agent_memory( # Using agent_memory as an example
                    session_id=doc_id, # Or a specific session_id if available
                    agent_name=self.name, # This agent's name
                    key=f"note_{note_obj.id}",
                    value=note_obj.to_dict(),
                    importance=self._importance_str_to_score(note_obj.importance)
                )
                self.logger.info("Note created and stored in memory.", parameters={'note_id': note_obj.id, 'doc_id': doc_id})
                return note_obj
            except MemoryManagerError as e:
                self.logger.error("Failed to store created note in UnifiedMemoryManager.", 
                                 parameters={'note_id': note_obj.id}, exception=e)
                return None # Indicate failure to store
            except Exception as e:  # Catch other unexpected errors
                self.logger.error(
                    "Unexpected error storing created note.",
                    parameters={'note_id': note_obj.id},
                    exception=e,
                )
                return None
        else:
            self.logger.warning("MemoryManager not available. Note created but not persisted.", parameters={'note_id': note_obj.id})
            return note_obj # Return note even if not persisted, caller can decide

    # ... Other methods like _update_note_async, _link_notes_async, _consolidate_notes_async
    # would follow a similar pattern: perform logic, create/update Note objects,
    # and interact with UnifiedMemoryManager.

    def _extract_context_around_match(self, text: str, start_idx: int, end_idx: int, window_size: int = 100) -> str: # Renamed
        """Extract context around a text match."""
        context_start = max(0, start_idx - window_size)
        context_end = min(len(text), end_idx + window_size)
        return text[context_start:context_end].strip().replace('\n', ' ') # Normalize context

    def _assess_note_importance_level(self, context_text: str) -> str: # Renamed
        """Assess the importance level of a potential note based on context keywords."""
        # ... (logic from original _assess_note_importance)
        context_lower = context_text.lower()
        for level, indicators in self.importance_indicators.items():
            if any(re.search(ind, context_lower) for ind in indicators):
                return level
        return self.default_note_importance

    def _importance_str_to_score(self, importance_str: str) -> float:
        """Convert importance string to a numerical score (0.0-1.0)."""
        mapping = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
        return mapping.get(importance_str.lower(), 0.5) # Default to 0.5

    async def _llm_generate_note_suggestions(self, text: str, 
                                           current_suggestions: List[Note], doc_id: str) -> List[Note]: # Renamed
        """Use LLM to generate additional Note objects."""
        # ... (logic from original _llm_note_suggestions, parse into Note objects)
        if not self.llm_manager: return []
        self.logger.debug("Using LLM for additional note suggestions.", parameters={'doc_id': doc_id})

        existing_titles = [note.title for note in current_suggestions[:5]] # Limit context
        prompt = f"""
        Analyze this legal document text excerpt and suggest up to {self.max_llm_suggestions} additional notes.
        Focus on key legal arguments, procedural points, strategic insights, or research needs NOT already covered by these titles: {', '.join(existing_titles) if existing_titles else "None"}.
        
        Document Text Excerpt (first 1500 chars):
        {text[:1500]}...
        
        For each suggested note, provide:
        - title: (string, concise and specific)
        - content: (string, brief explanation of the note's core idea or importance, max 150 chars)
        - type: (string, one of: legal_issue, strategic, research, procedural, evidence, general)
        - importance: (string, one of: low, medium, high, critical)
        
        Return a JSON array of these note objects. Example:
        [
          {{"title": "Potential Statute of Limitations Issue", "content": "Investigate if SOL has run for claim X.", "type": "research", "importance": "high"}}
        ]
        If no new valuable notes, return an empty array [].
        """
        try:
            llm_response_obj = await self.llm_manager.complete(prompt, model_params={'temperature': 0.6, 'max_tokens': 500 + (self.max_llm_suggestions * 100)})
            return self._parse_llm_note_suggestions_response(llm_response_obj.content, doc_id) # Renamed
        except LLMProviderError as e:
            self.logger.error("LLM API call for note suggestions failed.", parameters={'doc_id': doc_id}, exception=e)
            return []
        except Exception as e:
            self.logger.error("Unexpected error in LLM note suggestion generation.", parameters={'doc_id': doc_id}, exception=e)
            return []

    def _parse_llm_note_suggestions_response(self, response_content: str, doc_id: str) -> List[Note]: # Renamed
        """Parse LLM response into a list of Note objects."""
        # ... (logic from original _parse_llm_note_suggestions, create Note objects)
        llm_notes: List[Note] = []
        try:
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if not json_match:
                self.logger.warning("No JSON array found in LLM note suggestions response.", parameters={'doc_id': doc_id})
                return []
            
            notes_data_list = json.loads(json_match.group())

            for note_data in notes_data_list:
                if not isinstance(note_data, dict): continue
                if not all(k in note_data for k in ['title', 'content', 'type', 'importance']): continue # Ensure required fields

                note_obj = Note(
                    title=note_data['title'][:150], # Limit title length
                    content=note_data['content'][:500], # Limit content length
                    note_type=note_data['type'].lower(),
                    importance=note_data['importance'].lower(),
                    document_id=doc_id,
                    is_auto_generated=True,
                    tags=['llm_suggested', note_data['type'].lower()]
                )
                llm_notes.append(note_obj)
            
            return llm_notes[:self.max_llm_suggestions] # Adhere to max suggestions limit

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON from LLM note suggestions response.", parameters={'doc_id': doc_id}, exception=e)
        except Exception as e:
            self.logger.error("Unexpected error parsing LLM note suggestions.", parameters={'doc_id': doc_id}, exception=e)
        return []
