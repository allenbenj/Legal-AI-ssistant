"""
Reviewable Memory System for Legal AI.

This module implements a staging and review system for extracted legal information,
allowing human validation before permanent storage in agent memory.
"""

import asyncio
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid

from ..agents.ontology_extraction import ExtractedEntity, ExtractedRelationship, OntologyExtractionResult


class ReviewStatus(Enum):
    """Status of items in the review queue."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    AUTO_APPROVED = "auto_approved"


class ReviewPriority(Enum):
    """Priority levels for review items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReviewableItem:
    """An item awaiting review."""
    item_id: str
    item_type: str  # 'entity', 'relationship', 'finding'
    content: Dict[str, Any]
    confidence: float
    source_document: str
    extraction_context: str
    review_status: ReviewStatus
    review_priority: ReviewPriority
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewer_notes: str = ""
    original_content: Optional[Dict[str, Any]] = None  # For tracking modifications
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'item_type': self.item_type,
            'content': self.content,
            'confidence': self.confidence,
            'source_document': self.source_document,
            'extraction_context': self.extraction_context,
            'review_status': self.review_status.value,
            'review_priority': self.review_priority.value,
            'created_at': self.created_at.isoformat(),
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewer_notes': self.reviewer_notes,
            'original_content': self.original_content
        }


@dataclass
class ReviewDecision:
    """A review decision made by the user."""
    item_id: str
    decision: ReviewStatus
    modified_content: Optional[Dict[str, Any]] = None
    reviewer_notes: str = ""
    confidence_override: Optional[float] = None


@dataclass
class LegalFinding:
    """A significant legal finding that requires review."""
    finding_id: str
    finding_type: str  # 'violation', 'connection', 'inconsistency', 'pattern'
    description: str
    entities_involved: List[str]
    relationships_involved: List[str]
    evidence_sources: List[str]
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReviewableMemory:
    """
    Manages the review process for extracted legal information.
    
    Features:
    - Staging area for extracted entities and relationships
    - Configurable confidence thresholds for auto-approval
    - Priority assignment based on legal significance
    - Feedback loop to improve extraction confidence
    - Integration with agent_memory for permanent storage
    """
    
    def __init__(self, services, **config):
        self.services = services
        self.config = config
        self.logger = services.logger
        
        # Database setup
        self.db_path = config.get('review_db_path', 'data/review_memory.db')
        self.connection = None
        
        # Confidence thresholds (user-configurable)
        self.auto_approve_threshold = config.get('auto_approve_threshold', 0.9)
        self.review_threshold = config.get('review_threshold', 0.6)
        self.reject_threshold = config.get('reject_threshold', 0.4)
        
        # Review configuration
        self.enable_auto_approval = config.get('enable_auto_approval', True)
        self.max_auto_approvals_per_document = config.get('max_auto_approvals_per_document', 10)
        self.require_review_for_types = config.get('require_review_for_types', [
            'VIOLATION', 'SANCTION', 'CHARGED_WITH', 'FOUND_GUILTY_OF'
        ])
        
        # Feedback tracking
        self.feedback_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the reviewable memory system."""
        await self._setup_database()
        self.logger.info("Reviewable memory system initialized")
    
    async def _setup_database(self):
        """Set up SQLite database for review queue."""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Review items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_items (
                item_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_document TEXT NOT NULL,
                extraction_context TEXT,
                review_status TEXT NOT NULL,
                review_priority TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewer_notes TEXT,
                original_content TEXT
            )
        ''')
        
        # Legal findings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS legal_findings (
                finding_id TEXT PRIMARY KEY,
                finding_type TEXT NOT NULL,
                description TEXT NOT NULL,
                entities_involved TEXT NOT NULL,
                relationships_involved TEXT NOT NULL,
                evidence_sources TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                created_at TEXT NOT NULL,
                review_status TEXT NOT NULL
            )
        ''')
        
        # Feedback history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_history (
                feedback_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                original_confidence REAL,
                review_decision TEXT NOT NULL,
                confidence_adjustment REAL,
                feedback_notes TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        self.connection.commit()
    
    async def process_extraction_result(self, result: OntologyExtractionResult, 
                                      document_path: str) -> Dict[str, Any]:
        """
        Process extraction results and add items to review queue.
        
        Args:
            result: Extraction results from OntologyExtractionAgent
            document_path: Source document path
            
        Returns:
            Dictionary with review statistics
        """
        auto_approved = 0
        queued_for_review = 0
        rejected = 0
        findings_detected = 0
        
        # Process entities
        for entity in result.entities:
            review_item = await self._create_entity_review_item(entity, document_path, result)
            
            if await self._should_auto_approve(review_item):
                await self._auto_approve_item(review_item)
                auto_approved += 1
            elif await self._should_reject(review_item):
                await self._auto_reject_item(review_item)
                rejected += 1
            else:
                await self._queue_for_review(review_item)
                queued_for_review += 1
        
        # Process relationships
        for relationship in result.relationships:
            review_item = await self._create_relationship_review_item(relationship, document_path, result)
            
            if await self._should_auto_approve(review_item):
                await self._auto_approve_item(review_item)
                auto_approved += 1
            elif await self._should_reject(review_item):
                await self._auto_reject_item(review_item)
                rejected += 1
            else:
                await self._queue_for_review(review_item)
                queued_for_review += 1
        
        # Detect legal findings
        findings = await self._detect_legal_findings(result, document_path)
        for finding in findings:
            await self._queue_finding_for_review(finding)
            findings_detected += 1
        
        return {
            'auto_approved': auto_approved,
            'queued_for_review': queued_for_review,
            'rejected': rejected,
            'findings_detected': findings_detected,
            'total_processed': len(result.entities) + len(result.relationships)
        }
    
    async def _create_entity_review_item(self, entity: ExtractedEntity, 
                                       document_path: str, result: OntologyExtractionResult) -> ReviewableItem:
        """Create a review item for an extracted entity."""
        item_id = str(uuid.uuid4())
        
        # Determine priority based on entity type and confidence
        priority = await self._calculate_priority(entity.entity_type, entity.confidence, entity.source_text)
        
        return ReviewableItem(
            item_id=item_id,
            item_type='entity',
            content={
                'entity_type': entity.entity_type,
                'entity_id': entity.entity_id,
                'attributes': entity.attributes,
                'source_text': entity.source_text,
                'span': entity.span
            },
            confidence=entity.confidence,
            source_document=document_path,
            extraction_context=result.extraction_metadata.get('extraction_method', ''),
            review_status=ReviewStatus.PENDING,
            review_priority=priority,
            created_at=datetime.now()
        )
    
    async def _create_relationship_review_item(self, relationship: ExtractedRelationship,
                                             document_path: str, result: OntologyExtractionResult) -> ReviewableItem:
        """Create a review item for an extracted relationship."""
        item_id = str(uuid.uuid4())
        
        # Higher priority for legal action relationships
        priority = await self._calculate_priority(relationship.relationship_type, relationship.confidence, relationship.source_text)
        
        return ReviewableItem(
            item_id=item_id,
            item_type='relationship',
            content={
                'relationship_type': relationship.relationship_type,
                'source_entity': relationship.source_entity,
                'target_entity': relationship.target_entity,
                'properties': relationship.properties,
                'source_text': relationship.source_text,
                'span': relationship.span
            },
            confidence=relationship.confidence,
            source_document=document_path,
            extraction_context=result.extraction_metadata.get('extraction_method', ''),
            review_status=ReviewStatus.PENDING,
            review_priority=priority,
            created_at=datetime.now()
        )
    
    async def _calculate_priority(self, item_type: str, confidence: float, source_text: str) -> ReviewPriority:
        """Calculate review priority based on legal significance."""
        # Critical priority for legal violations and serious matters
        critical_keywords = ['violation', 'misconduct', 'guilty', 'conviction', 'sanction', 'penalty']
        high_keywords = ['charged', 'accused', 'indicted', 'evidence', 'witness', 'testimony']
        
        source_lower = source_text.lower()
        
        if any(keyword in source_lower for keyword in critical_keywords):
            return ReviewPriority.CRITICAL
        elif any(keyword in source_lower for keyword in high_keywords):
            return ReviewPriority.HIGH
        elif confidence < 0.7:
            return ReviewPriority.HIGH  # Low confidence needs review
        elif item_type in self.require_review_for_types:
            return ReviewPriority.HIGH
        else:
            return ReviewPriority.MEDIUM if confidence < 0.8 else ReviewPriority.LOW
    
    async def _should_auto_approve(self, item: ReviewableItem) -> bool:
        """Determine if item should be auto-approved."""
        if not self.enable_auto_approval:
            return False
        
        # Never auto-approve critical items
        if item.review_priority == ReviewPriority.CRITICAL:
            return False
        
        # Never auto-approve items requiring review
        if item.content.get('entity_type') in self.require_review_for_types:
            return False
        if item.content.get('relationship_type') in self.require_review_for_types:
            return False
        
        # Auto-approve high confidence items
        return item.confidence >= self.auto_approve_threshold
    
    async def _should_reject(self, item: ReviewableItem) -> bool:
        """Determine if item should be auto-rejected."""
        return item.confidence < self.reject_threshold
    
    async def _auto_approve_item(self, item: ReviewableItem):
        """Auto-approve an item and send to permanent storage."""
        item.review_status = ReviewStatus.AUTO_APPROVED
        item.reviewed_at = datetime.now()
        
        await self._store_review_item(item)
        await self._send_to_agent_memory(item)
        
        self.logger.debug(f"Auto-approved {item.item_type}: {item.content.get('source_text', 'N/A')}")
    
    async def _auto_reject_item(self, item: ReviewableItem):
        """Auto-reject a low-confidence item."""
        item.review_status = ReviewStatus.REJECTED
        item.reviewed_at = datetime.now()
        item.reviewer_notes = "Auto-rejected due to low confidence"
        
        await self._store_review_item(item)
        
        self.logger.debug(f"Auto-rejected {item.item_type}: {item.content.get('source_text', 'N/A')}")
    
    async def _queue_for_review(self, item: ReviewableItem):
        """Queue item for human review."""
        await self._store_review_item(item)
        
        self.logger.info(f"Queued for review ({item.review_priority.value}): {item.item_type}")
    
    async def _store_review_item(self, item: ReviewableItem):
        """Store review item in database."""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO review_items 
            (item_id, item_type, content, confidence, source_document, extraction_context,
             review_status, review_priority, created_at, reviewed_at, reviewer_notes, original_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.item_id,
            item.item_type,
            json.dumps(item.content),
            item.confidence,
            item.source_document,
            item.extraction_context,
            item.review_status.value,
            item.review_priority.value,
            item.created_at.isoformat(),
            item.reviewed_at.isoformat() if item.reviewed_at else None,
            item.reviewer_notes,
            json.dumps(item.original_content) if item.original_content else None
        ))
        self.connection.commit()
    
    async def _detect_legal_findings(self, result: OntologyExtractionResult, 
                                   document_path: str) -> List[LegalFinding]:
        """Detect significant legal findings that require special attention."""
        findings = []
        
        # Look for violation patterns
        violation_entities = [e for e in result.entities if 'violation' in e.source_text.lower()]
        misconduct_entities = [e for e in result.entities if 'misconduct' in e.source_text.lower()]
        
        if violation_entities or misconduct_entities:
            finding = LegalFinding(
                finding_id=str(uuid.uuid4()),
                finding_type='violation',
                description=f"Potential legal violation detected in {document_path}",
                entities_involved=[e.entity_id for e in violation_entities + misconduct_entities],
                relationships_involved=[],
                evidence_sources=[document_path],
                confidence=max([e.confidence for e in violation_entities + misconduct_entities]),
                severity='high'
            )
            findings.append(finding)
        
        # Look for contradiction patterns
        contradiction_rels = [r for r in result.relationships if r.relationship_type == 'CONTRADICTS']
        if contradiction_rels:
            finding = LegalFinding(
                finding_id=str(uuid.uuid4()),
                finding_type='inconsistency',
                description=f"Contradictory statements detected in {document_path}",
                entities_involved=[],
                relationships_involved=[r.relationship_type for r in contradiction_rels],
                evidence_sources=[document_path],
                confidence=max([r.confidence for r in contradiction_rels]),
                severity='medium'
            )
            findings.append(finding)
        
        return findings
    
    async def _queue_finding_for_review(self, finding: LegalFinding):
        """Queue a legal finding for review."""
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO legal_findings 
            (finding_id, finding_type, description, entities_involved, relationships_involved,
             evidence_sources, confidence, severity, created_at, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            finding.finding_id,
            finding.finding_type,
            finding.description,
            json.dumps(finding.entities_involved),
            json.dumps(finding.relationships_involved),
            json.dumps(finding.evidence_sources),
            finding.confidence,
            finding.severity,
            datetime.now().isoformat(),
            ReviewStatus.PENDING.value
        ))
        self.connection.commit()
    
    async def get_pending_reviews(self, priority: Optional[ReviewPriority] = None) -> List[ReviewableItem]:
        """Get items pending review, optionally filtered by priority."""
        cursor = self.connection.cursor()
        
        if priority:
            cursor.execute('''
                SELECT * FROM review_items 
                WHERE review_status = ? AND review_priority = ?
                ORDER BY created_at DESC
            ''', (ReviewStatus.PENDING.value, priority.value))
        else:
            cursor.execute('''
                SELECT * FROM review_items 
                WHERE review_status = ?
                ORDER BY 
                    CASE review_priority 
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2  
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END,
                    created_at DESC
            ''', (ReviewStatus.PENDING.value,))
        
        rows = cursor.fetchall()
        items = []
        
        for row in rows:
            item = ReviewableItem(
                item_id=row[0],
                item_type=row[1],
                content=json.loads(row[2]),
                confidence=row[3],
                source_document=row[4],
                extraction_context=row[5],
                review_status=ReviewStatus(row[6]),
                review_priority=ReviewPriority(row[7]),
                created_at=datetime.fromisoformat(row[8]),
                reviewed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                reviewer_notes=row[10] or "",
                original_content=json.loads(row[11]) if row[11] else None
            )
            items.append(item)
        
        return items
    
    async def submit_review_decision(self, decision: ReviewDecision) -> bool:
        """Submit a review decision and update the item."""
        try:
            # Get the item being reviewed
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM review_items WHERE item_id = ?', (decision.item_id,))
            row = cursor.fetchone()
            
            if not row:
                self.logger.error(f"Review item {decision.item_id} not found")
                return False
            
            # Create reviewable item from database row
            item = ReviewableItem(
                item_id=row[0],
                item_type=row[1],
                content=json.loads(row[2]),
                confidence=row[3],
                source_document=row[4],
                extraction_context=row[5],
                review_status=ReviewStatus(row[6]),
                review_priority=ReviewPriority(row[7]),
                created_at=datetime.fromisoformat(row[8]),
                reviewed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                reviewer_notes=row[10] or "",
                original_content=json.loads(row[11]) if row[11] else None
            )
            
            # Store original content if modifying
            if decision.decision == ReviewStatus.MODIFIED and not item.original_content:
                item.original_content = item.content.copy()
            
            # Update item based on decision
            item.review_status = decision.decision
            item.reviewed_at = datetime.now()
            item.reviewer_notes = decision.reviewer_notes
            
            if decision.modified_content:
                item.content = decision.modified_content
            
            if decision.confidence_override:
                item.confidence = decision.confidence_override
            
            # Store updated item
            await self._store_review_item(item)
            
            # Record feedback for learning
            await self._record_feedback(item, decision)
            
            # If approved or modified, send to agent memory
            if decision.decision in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]:
                await self._send_to_agent_memory(item)
            
            self.logger.info(f"Review decision processed: {decision.decision.value} for {item.item_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing review decision: {e}")
            return False
    
    async def _record_feedback(self, item: ReviewableItem, decision: ReviewDecision):
        """Record feedback for improving the system."""
        feedback_id = str(uuid.uuid4())
        
        # Calculate confidence adjustment
        original_confidence = item.confidence
        new_confidence = decision.confidence_override or item.confidence
        confidence_adjustment = new_confidence - original_confidence
        
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO feedback_history 
            (feedback_id, item_id, original_confidence, review_decision, 
             confidence_adjustment, feedback_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            item.item_id,
            original_confidence,
            decision.decision.value,
            confidence_adjustment,
            decision.reviewer_notes,
            datetime.now().isoformat()
        ))
        self.connection.commit()
        
        # Store in memory for immediate learning
        self.feedback_history.append({
            'item_type': item.item_type,
            'original_confidence': original_confidence,
            'review_decision': decision.decision.value,
            'confidence_adjustment': confidence_adjustment,
            'context': item.extraction_context
        })
    
    async def _send_to_agent_memory(self, item: ReviewableItem):
        """Send approved item to permanent agent memory storage."""
        try:
            # This would integrate with the existing agent_memory system
            # For now, just log the action
            self.logger.info(f"Sending {item.item_type} to agent memory: {item.content.get('source_text', 'N/A')}")
            
            # TODO: Implement actual agent_memory integration
            # agent_memory = self.services.get_service('agent_memory')
            # await agent_memory.store_reviewed_item(item)
            
        except Exception as e:
            self.logger.error(f"Failed to send item to agent memory: {e}")
    
    async def get_review_stats(self) -> Dict[str, Any]:
        """Get review queue statistics."""
        cursor = self.connection.cursor()
        
        # Count by status
        cursor.execute('''
            SELECT review_status, COUNT(*) FROM review_items GROUP BY review_status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Count by priority
        cursor.execute('''
            SELECT review_priority, COUNT(*) FROM review_items 
            WHERE review_status = 'pending' GROUP BY review_priority
        ''')
        priority_counts = dict(cursor.fetchall())
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM review_items 
            WHERE created_at > datetime('now', '-24 hours')
        ''')
        recent_items = cursor.fetchone()[0]
        
        return {
            'status_counts': status_counts,
            'priority_counts': priority_counts,
            'recent_items_24h': recent_items,
            'pending_reviews': status_counts.get('pending', 0),
            'auto_approval_enabled': self.enable_auto_approval,
            'thresholds': {
                'auto_approve': self.auto_approve_threshold,
                'review': self.review_threshold,
                'reject': self.reject_threshold
            }
        }
    
    async def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update confidence thresholds based on user preferences."""
        if 'auto_approve_threshold' in new_thresholds:
            self.auto_approve_threshold = new_thresholds['auto_approve_threshold']
        if 'review_threshold' in new_thresholds:
            self.review_threshold = new_thresholds['review_threshold']
        if 'reject_threshold' in new_thresholds:
            self.reject_threshold = new_thresholds['reject_threshold']
        
        self.logger.info(f"Updated confidence thresholds: {new_thresholds}")
    
    async def close(self):
        """Close database connections."""
        if self.connection:
            self.connection.close()
        self.logger.info("Reviewable memory system closed")