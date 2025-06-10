# legal_ai_system/memory/cccccccccccccccccccccccc
"""
Reviewable Memory System for Legal AI.

Implements a staging and review system for extracted legal information,
allowing human validation before permanent storage.
"""

import asyncio
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import uuid
from pathlib import Path
import threading

# Use detailed_logging
from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from ..core.unified_exceptions import MemoryManagerError
from typing import TYPE_CHECKING, Any

# Import types from agents only for type checking to avoid heavy dependencies at
# runtime during tests.
if TYPE_CHECKING:  # pragma: no cover - hints only
    from ..agents.ontology_extraction_agent import (
        ExtractedEntity,
        ExtractedRelationship,
        OntologyExtractionOutput,
    )
else:  # pragma: no cover - simplified fallbacks
    ExtractedEntity = Any  # type: ignore
    ExtractedRelationship = Any  # type: ignore
    OntologyExtractionOutput = Any  # type: ignore

# Initialize logger for this module
review_mem_logger = get_detailed_logger("ReviewableMemory", LogCategory.DATABASE)


class ReviewStatus(Enum):
    PENDING = "pending"; AUTO_APPROVED = "auto_approved"; APPROVED = "approved"
    REJECTED = "rejected"; MODIFIED = "modified"
    
class ReviewPriority(Enum):
    LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"

@dataclass
class ReviewableItem:
    item_type: str  # 'entity', 'relationship', 'finding'
    content: Dict[str, Any]  # Original extracted content
    confidence: float
    source_document_id: str  # Changed from source_document (path) to an ID
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extraction_context: Optional[Dict[str, Any]] = field(default_factory=dict) # Changed to dict
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_priority: ReviewPriority = ReviewPriority.MEDIUM
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None # Added
    reviewer_notes: str = ""
    original_content_on_modify: Optional[Dict[str, Any]] = None # Renamed, stores original if modified
    
    def to_dict(self) -> Dict[str, Any]: # For serialization
        data = asdict(self)
        data['review_status'] = self.review_status.value
        data['review_priority'] = self.review_priority.value
        data['created_at'] = self.created_at.isoformat()
        data['reviewed_at'] = self.reviewed_at.isoformat() if self.reviewed_at else None
        return data

@dataclass
class ReviewDecision:
    item_id: str
    decision: ReviewStatus  # The new status after review
    reviewer_id: str  # ID of the user/agent making the review
    modified_content: Optional[Dict[str, Any]] = None  # If decision is MODIFIED
    reviewer_notes: str = ""
    confidence_override: Optional[float] = None

@dataclass
class LegalFindingItem: # Renamed from LegalFinding to avoid confusion with a potential domain object
    document_id: str  # Added
    finding_type: str  # 'violation', 'connection', 'inconsistency', 'pattern'
    description: str
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical' - consider Enum
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entities_involved_ids: List[str] = field(default_factory=list)  # Changed to IDs
    relationships_involved_ids: List[str] = field(default_factory=list)  # Changed to IDs
    evidence_source_refs: List[str] = field(default_factory=list)  # Changed, e.g. doc_id + snippet_ref
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_status: ReviewStatus = ReviewStatus.PENDING # Findings also go through review
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['review_status'] = self.review_status.value
        return data

class ReviewableMemory:
    """Manages the review process for extracted legal information."""
    
    @detailed_log_function(LogCategory.DATABASE)
    def __init__(self, 
                 db_path_str: str = "./storage/databases/review_memory.db", # Renamed param
                 unified_memory_manager: Optional[Any] = None, # For storing approved items
                 service_config: Optional[Dict[str, Any]] = None): # For config injection
        review_mem_logger.info("Initializing ReviewableMemory.")
        self.config = service_config or {}
        
        self.db_path = Path(db_path_str)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: Optional[sqlite3.Connection] = None # Initialized in async init
        self._lock = threading.RLock() # For DB operations
        self._initialized = False
        
        self.unified_memory_manager = unified_memory_manager # To store approved items

        # Confidence thresholds
        self.auto_approve_threshold: float = self.config.get('auto_approve_threshold', 0.9)
        self.review_threshold: float = self.config.get('review_threshold', 0.6) # Items below this might need review
        self.reject_threshold: float = self.config.get('reject_threshold', 0.4) # Items below this might be auto-rejected
        
        self.enable_auto_approval: bool = self.config.get('enable_auto_approval', True)
        # max_auto_approvals_per_document is complex to implement without more context on processing flow
        
        # Types that always require review, regardless of confidence
        self.require_review_for_types: Set[str] = set(self.config.get('require_review_for_types', [
            'VIOLATION', 'SANCTION', 'CHARGED_WITH', 'FOUND_GUILTY_OF', # Example types
            'Brady Violation', 'Prosecutorial Misconduct' # From hybrid_extractor targeted prompts
        ]))
        
        self.feedback_history_cache: List[Dict[str, Any]] = [] # In-memory cache for recent feedback
        review_mem_logger.info("ReviewableMemory initialized.", parameters=self.get_config_summary())

    def get_config_summary(self) -> Dict[str, Any]:
        return {
            'db_path': str(self.db_path),
            'auto_approve_thresh': self.auto_approve_threshold,
            'review_thresh': self.review_threshold,
            'reject_thresh': self.reject_threshold,
            'auto_approval_on': self.enable_auto_approval,
            'types_always_review': list(self.require_review_for_types)
        }

    @detailed_log_function(LogCategory.DATABASE)
    async def initialize(self): # Made async
        """Initialize the reviewable memory system and database."""
        if self._initialized:
            review_mem_logger.warning("ReviewableMemory already initialized.")
            return self
        review_mem_logger.info("Starting ReviewableMemory initialization.")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._setup_database_sync)
            self._initialized = True
            review_mem_logger.info("ReviewableMemory initialized successfully.")
        except Exception as e:
            review_mem_logger.critical("ReviewableMemory initialization failed.", exception=e)
            self._initialized = False
            raise MemoryManagerError("Failed to initialize ReviewableMemory database.", cause=e)
        return self

    def _get_db_connection(self) -> sqlite3.Connection:
        """Gets a new SQLite connection for the current thread."""
        try:
            # check_same_thread=False is needed if connections are passed across threads,
            # but typically each thread (from executor) should get its own connection.
            # For simplicity in a threaded executor model, new connection per call is safer.
            return sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        except sqlite3.Error as e:
            review_mem_logger.error("Failed to connect to ReviewableMemory SQLite database.", exception=e)
            raise MemoryManagerError("ReviewableMemory database connection failed.", cause=e)

    def _setup_database_sync(self): # Renamed
        """Set up SQLite database for review queue (synchronous part)."""
        review_mem_logger.debug("Initializing ReviewableMemory database schema.")
        try:
            with self._get_db_connection() as conn:
                conn.executescript('''
                    CREATE TABLE IF NOT EXISTS review_items (
                        item_id TEXT PRIMARY KEY, item_type TEXT NOT NULL, content TEXT NOT NULL,
                        confidence REAL NOT NULL, source_document_id TEXT NOT NULL,
                        extraction_context TEXT, review_status TEXT NOT NULL,
                        review_priority TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                        reviewed_at TIMESTAMPTZ, reviewer_id TEXT, reviewer_notes TEXT,
                        original_content_on_modify TEXT -- Renamed
                    );
                    CREATE INDEX IF NOT EXISTS idx_rev_item_status_priority ON review_items(review_status, review_priority, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_rev_item_doc_id ON review_items(source_document_id);

                    CREATE TABLE IF NOT EXISTS legal_findings_review ( -- Renamed table
                        finding_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, finding_type TEXT NOT NULL,
                        description TEXT NOT NULL, entities_involved_ids TEXT, relationships_involved_ids TEXT,
                        evidence_source_refs TEXT, confidence REAL NOT NULL, severity TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL, review_status TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_legal_finding_doc_status ON legal_findings_review(document_id, review_status);

                    CREATE TABLE IF NOT EXISTS review_feedback_history ( -- Renamed table
                        feedback_id TEXT PRIMARY KEY, item_id TEXT NOT NULL, item_type_reviewed TEXT NOT NULL,
                        original_confidence REAL, review_decision TEXT NOT NULL,
                        confidence_adjustment REAL, feedback_notes TEXT, reviewer_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_feedback_item ON review_feedback_history(item_id);
                ''')
                conn.commit()
            review_mem_logger.info("ReviewableMemory database schema initialized/verified.")
        except sqlite3.Error as e:
            review_mem_logger.error("SQLite error during ReviewableMemory schema setup.", exception=e)
            raise MemoryManagerError("ReviewableMemory DB schema setup failed.", cause=e)

    @detailed_log_function(LogCategory.WORKFLOW)
    async def process_extraction_result(self,
                                      extraction: OntologyExtractionOutput,  # Renamed from result
                                      document_id: str, # Changed from document_path
                                      extraction_source_info: Optional[Dict[str,Any]] = None) -> Dict[str, int]: # Renamed param & type
        """Process extraction results and add items to review queue."""
        if not self._initialized: await self.initialize()
        review_mem_logger.info("Processing extraction result for review.", parameters={'doc_id': document_id, 'num_entities': len(extraction.entities)})
        
        stats = {'auto_approved': 0, 'queued_for_review': 0, 'auto_rejected': 0, 'findings_added': 0}
        extraction_context_dict = extraction.extraction_metadata or {}
        if extraction_source_info: extraction_context_dict.update(extraction_source_info)

        # Process entities
        for entity_obj in extraction.entities: # Renamed var
            review_item = await self._create_review_item_from_entity(entity_obj, document_id, extraction_context_dict)
            await self._handle_review_item_decision(review_item, stats)
        
        # Process relationships
        for rel_obj in extraction.relationships: # Renamed var
            review_item = await self._create_review_item_from_relationship(rel_obj, document_id, extraction_context_dict)
            await self._handle_review_item_decision(review_item, stats)
        
        # Detect and queue legal findings
        findings = await self._detect_and_create_findings(extraction, document_id)
        for finding_item in findings: # Renamed var
            await self._queue_finding_for_review_async(finding_item) # Renamed
            stats['findings_added'] += 1
        
        review_mem_logger.info("Extraction result processing complete.", parameters={'doc_id': document_id, 'stats': stats})
        return stats

    async def _handle_review_item_decision(self, item: ReviewableItem, stats: Dict[str, int]):
        """Applies auto-approve/reject logic or queues item."""
        if await self._should_auto_approve(item):
            await self._auto_approve_item_async(item) # Renamed
            stats['auto_approved'] += 1
        elif await self._should_auto_reject(item):
            await self._auto_reject_item_async(item) # Renamed
            stats['auto_rejected'] += 1
        else:
            await self._queue_item_for_review_async(item) # Renamed
            stats['queued_for_review'] += 1
            
    async def _create_review_item_from_entity(self, entity: ExtractedEntity, 
                                            document_id: str, extraction_context: Dict[str,Any]) -> ReviewableItem:
        priority = await self._calculate_item_priority(
            entity.entity_type, entity.confidence, entity.source_text_snippet
        )
        return ReviewableItem(
            item_type='entity',
            content=entity.to_dict(), # Store full entity data
            confidence=entity.confidence,
            source_document_id=document_id,
            extraction_context=extraction_context,
            review_priority=priority
        )

    async def _create_review_item_from_relationship(self, rel: ExtractedRelationship,
                                                  document_id: str, extraction_context: Dict[str,Any]) -> ReviewableItem:
        priority = await self._calculate_item_priority(
            rel.relationship_type, rel.confidence, rel.evidence_text_snippet
        )
        return ReviewableItem(
            item_type='relationship',
            content=rel.to_dict(), # Store full relationship data
            confidence=rel.confidence,
            source_document_id=document_id,
            extraction_context=extraction_context,
            review_priority=priority
        )

    async def _calculate_item_priority(self, item_type_str: str, confidence: float, text_context: str) -> ReviewPriority: # Renamed param
        """Calculate review priority based on item type, confidence, and context keywords."""
        # This logic can be significantly enhanced
        text_context_lower = text_context.lower()
        if any(kw in text_context_lower for kw in ['violation', 'misconduct', 'fraud', 'brady']):
            return ReviewPriority.CRITICAL
        if item_type_str.upper() in self.require_review_for_types:
            return ReviewPriority.HIGH
        if confidence < self.review_threshold: # If below general review threshold but not reject threshold
            return ReviewPriority.HIGH 
        if confidence < (self.review_threshold + self.auto_approve_threshold) / 2: # Mid-range
            return ReviewPriority.MEDIUM
        return ReviewPriority.LOW

    async def _should_auto_approve(self, item: ReviewableItem) -> bool:
        if not self.enable_auto_approval: return False
        if item.review_priority == ReviewPriority.CRITICAL: return False
        
        item_main_type = item.content.get('entity_type', item.content.get('relationship_type', item.content.get('finding_type', '')))
        if item_main_type.upper() in self.require_review_for_types: return False
        
        return item.confidence >= self.auto_approve_threshold

    async def _should_auto_reject(self, item: ReviewableItem) -> bool:
        # Avoid rejecting critical/high priority items automatically even if confidence is low
        if item.review_priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]:
            return False
        return item.confidence < self.reject_threshold

    async def _auto_approve_item_async(self, item: ReviewableItem): # Renamed
        item.review_status = ReviewStatus.AUTO_APPROVED
        item.reviewed_at = datetime.now(timezone.utc)
        await self._store_review_item_async(item) # Renamed
        if self.unified_memory_manager: # Check if UMM is available
            await self._send_to_unified_memory(item) # Renamed
        review_mem_logger.debug(f"Auto-approved item.", parameters={'item_id': item.item_id, 'type': item.item_type})

    async def _auto_reject_item_async(self, item: ReviewableItem): # Renamed
        item.review_status = ReviewStatus.REJECTED
        item.reviewed_at = datetime.now(timezone.utc)
        item.reviewer_notes = "Auto-rejected due to low confidence and low priority."
        await self._store_review_item_async(item)
        review_mem_logger.debug(f"Auto-rejected item.", parameters={'item_id': item.item_id, 'type': item.item_type})

    async def _queue_item_for_review_async(self, item: ReviewableItem): # Renamed
        item.review_status = ReviewStatus.PENDING # Ensure it's pending
        await self._store_review_item_async(item)
        review_mem_logger.info(f"Item queued for review.", parameters={'item_id': item.item_id, 'type': item.item_type, 'priority': item.review_priority.value})

    async def _store_review_item_async(self, item: ReviewableItem): # Renamed
        """Store review item in database asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_review_item_sync, item)

    def _store_review_item_sync(self, item: ReviewableItem): # New sync helper
        with self._lock, self._get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO review_items 
                (item_id, item_type, content, confidence, source_document_id, extraction_context,
                 review_status, review_priority, created_at, reviewed_at, reviewer_id, reviewer_notes, original_content_on_modify)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.item_id, item.item_type, json.dumps(item.content, default=str), item.confidence,
                item.source_document_id, json.dumps(item.extraction_context, default=str),
                item.review_status.value, item.review_priority.value,
                item.created_at.isoformat(), item.reviewed_at.isoformat() if item.reviewed_at else None,
                item.reviewer_id, item.reviewer_notes,
                json.dumps(item.original_content_on_modify, default=str) if item.original_content_on_modify else None
            ))
            conn.commit()

    async def _detect_and_create_findings(self, extraction: OntologyExtractionOutput, document_id: str) -> List[LegalFindingItem]:
        """Detect significant legal findings that require special attention."""
        findings_list: List[LegalFindingItem] = []
        # Example: Detect potential Brady violations if specific entities/relations appear
        brady_keywords = {"brady", "exculpatory", "failure to disclose", "withheld evidence"}
        text_for_finding_scan = " ".join(
            [e.source_text_snippet for e in extraction.entities] +
            [r.evidence_text_snippet for r in extraction.relationships]
        ).lower()

        if any(keyword in text_for_finding_scan for keyword in brady_keywords):
            critical_entities = [e.entity_id for e in extraction.entities if e.confidence > 0.7 and "prosecutor" in e.entity_type.lower()]
            
            finding = LegalFindingItem(
                document_id=document_id,
                finding_type='PotentialBradyViolation',
                description=f"Potential Brady material detected in document {document_id} related to entities: {critical_entities}",
                entities_involved_ids=critical_entities,
                confidence=0.85, # High confidence this *needs review*
                severity='CRITICAL'
            )
            findings_list.append(finding)
            review_mem_logger.warning("Potential critical finding (Brady) detected.", parameters={'doc_id': document_id})
        return findings_list

    async def _queue_finding_for_review_async(self, finding: LegalFindingItem): # Renamed
        """Queue a legal finding for review asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._queue_finding_for_review_sync, finding)

    def _queue_finding_for_review_sync(self, finding: LegalFindingItem): # New sync helper
        with self._lock, self._get_db_connection() as conn:
            conn.execute('''
                INSERT INTO legal_findings_review 
                (finding_id, document_id, finding_type, description, entities_involved_ids, 
                 relationships_involved_ids, evidence_source_refs, confidence, severity, created_at, review_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                finding.finding_id, finding.document_id, finding.finding_type, finding.description,
                json.dumps(finding.entities_involved_ids), json.dumps(finding.relationships_involved_ids),
                json.dumps(finding.evidence_source_refs), finding.confidence, finding.severity,
                finding.created_at.isoformat(), finding.review_status.value
            ))
            conn.commit()
        review_mem_logger.info("Legal finding queued for review.", parameters={'finding_id': finding.finding_id, 'type': finding.finding_type})

    @detailed_log_function(LogCategory.DATABASE)
    async def get_pending_reviews_async(self, # Renamed
                                      priority: Optional[ReviewPriority] = None, 
                                      limit: int = 50) -> List[ReviewableItem]:
        """Get items pending review, optionally filtered by priority, async."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_pending_reviews_sync, priority, limit)

    def _get_pending_reviews_sync(self, priority: Optional[ReviewPriority], limit: int) -> List[ReviewableItem]: # New sync helper
        with self._lock, self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM review_items WHERE review_status = ? "
            params: List[Any] = [ReviewStatus.PENDING.value]

            if priority:
                query += "AND review_priority = ? "
                params.append(priority.value)
            
            query += """
                ORDER BY 
                    CASE review_priority 
                        WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 
                    END, created_at ASC LIMIT ? 
            """ # Order by asc created_at to review oldest first within priority
            params.append(limit)
            
            cursor = conn.execute(query, tuple(params))
            items = [self._row_to_reviewable_item(row) for row in cursor.fetchall()]
            review_mem_logger.debug(f"Fetched {len(items)} pending review items.")
            return items

    def _row_to_reviewable_item(self, row: sqlite3.Row) -> ReviewableItem:
        """Converts a database row to a ReviewableItem object."""
        return ReviewableItem(
            item_id=row['item_id'], item_type=row['item_type'],
            content=json.loads(row['content']), confidence=row['confidence'],
            source_document_id=row['source_document_id'],
            extraction_context=json.loads(row['extraction_context']) if row['extraction_context'] else {},
            review_status=ReviewStatus(row['review_status']),
            review_priority=ReviewPriority(row['review_priority']),
            created_at=datetime.fromisoformat(row['created_at'].replace("Z", "+00:00")),
            reviewed_at=datetime.fromisoformat(row['reviewed_at'].replace("Z", "+00:00")) if row['reviewed_at'] else None,
            reviewer_id=row['reviewer_id'],
            reviewer_notes=row['reviewer_notes'] or "",
            original_content_on_modify=json.loads(row['original_content_on_modify']) if row['original_content_on_modify'] else None
        )

    @detailed_log_function(LogCategory.DATABASE)
    async def submit_review_decision_async(self, decision: ReviewDecision) -> bool: # Renamed
        """Submit a review decision and update the item, async."""
        if not self._initialized: await self.initialize()
        review_mem_logger.info("Submitting review decision.", parameters=decision.__dict__)
        loop = asyncio.get_event_loop()
        try:
            success = await loop.run_in_executor(None, self._submit_review_decision_sync, decision)
            if success and self.unified_memory_manager and decision.decision in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]:
                # Fetch the updated item to send to UMM
                item_to_store = await loop.run_in_executor(None, self._get_review_item_sync, decision.item_id)
                if item_to_store:
                    await self._send_to_unified_memory(item_to_store)
            return success
        except Exception as e:
            review_mem_logger.error("Error processing review decision.", parameters={'item_id': decision.item_id}, exception=e)
            return False

    def _get_review_item_sync(self, item_id: str) -> Optional[ReviewableItem]: # New sync helper
        with self._lock, self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM review_items WHERE item_id = ?", (item_id,)).fetchone()
            return self._row_to_reviewable_item(row) if row else None

    def _submit_review_decision_sync(self, decision: ReviewDecision) -> bool: # New sync helper
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, confidence FROM review_items WHERE item_id = ?", (decision.item_id,))
            row = cursor.fetchone()
            if not row:
                review_mem_logger.error(f"Review item not found for decision.", parameters={'item_id': decision.item_id})
                return False

            original_content_str, original_confidence = row
            original_content = json.loads(original_content_str)
            
            new_content = decision.modified_content or original_content
            new_confidence = decision.confidence_override if decision.confidence_override is not None else original_confidence
            original_content_for_log = original_content if decision.decision == ReviewStatus.MODIFIED else None

            cursor.execute("""
                UPDATE review_items SET review_status = ?, reviewed_at = ?, reviewer_id = ?, 
                                       reviewer_notes = ?, content = ?, confidence = ?,
                                       original_content_on_modify = ?
                WHERE item_id = ?
            """, (
                decision.decision.value, datetime.now(timezone.utc).isoformat(), decision.reviewer_id,
                decision.reviewer_notes, json.dumps(new_content, default=str), new_confidence,
                json.dumps(original_content_for_log, default=str) if original_content_for_log else None,
                decision.item_id
            ))
            conn.commit()
            self._record_feedback_sync(decision.item_id, ReviewableItem( # Dummy item for original confidence
                item_id=decision.item_id, item_type="unknown", content={}, confidence=original_confidence,
                source_document_id="unknown" # Not strictly needed for feedback record context here
            ), decision)
            return True

    def _record_feedback_sync(self, item_id: str, original_item_data_for_confidence: ReviewableItem, decision: ReviewDecision): # Renamed and clarified param
        """Record feedback for improving the system (synchronous part)."""
        feedback_id = str(uuid.uuid4())
        confidence_adjustment = (decision.confidence_override or original_item_data_for_confidence.confidence) - original_item_data_for_confidence.confidence
        
        with self._lock, self._get_db_connection() as conn:
            conn.execute('''
                INSERT INTO review_feedback_history 
                (feedback_id, item_id, item_type_reviewed, original_confidence, review_decision, 
                 confidence_adjustment, feedback_notes, reviewer_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id, item_id, original_item_data_for_confidence.item_type, # Assuming item_type is on original
                original_item_data_for_confidence.confidence, decision.decision.value,
                confidence_adjustment, decision.reviewer_notes, decision.reviewer_id,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
        review_mem_logger.debug("Feedback recorded for item.", parameters={'item_id': item_id, 'decision': decision.decision.value})
    
    async def _send_to_unified_memory(self, item: ReviewableItem): # Renamed
        """Send approved/modified item to UnifiedMemoryManager."""
        if not self.unified_memory_manager:
            review_mem_logger.warning("UnifiedMemoryManager not configured. Cannot send item to permanent memory.", parameters={'item_id': item.item_id})
            return

        review_mem_logger.info(f"Sending item to UnifiedMemoryManager.", parameters={'item_id': item.item_id, 'item_type': item.item_type})
        try:
            # Adapt ReviewableItem to a format UMM expects
            # This is a placeholder for actual integration logic.
            # Example: if item.item_type == 'entity':
            #    await self.unified_memory_manager.store_session_entity(
            #        session_id=item.source_document_id,  # Or a relevant session_id
            #        name=item.content.get('entity_id', item.content.get('source_text_snippet')),
            #        entity_type=item.content.get('entity_type'),
            #        attributes=item.content.get('attributes'),
            #        confidence=item.confidence,
            #        source="human_review"
            #    )
            pass # Placeholder for UMM integration
            review_mem_logger.debug("Item sent to UnifiedMemoryManager.", parameters={'item_id': item.item_id})
        except Exception as e:
            review_mem_logger.error(f"Failed to send item to UnifiedMemoryManager.", parameters={'item_id': item.item_id}, exception=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_review_stats_async(self) -> Dict[str, Any]: # Renamed
        """Get review queue statistics asynchronously."""
        if not self._initialized: await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_review_stats_sync)

    def _get_review_stats_sync(self) -> Dict[str, Any]: # New sync helper
        with self._lock, self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row # Ensure dict-like rows
            stats: Dict[str, Any] = {'status_counts': {}, 'priority_counts_pending': {}} # type hint

            cursor = conn.execute("SELECT review_status, COUNT(*) FROM review_items GROUP BY review_status")
            stats['status_counts'] = {row['review_status']: row['COUNT(*)'] for row in cursor.fetchall()}
            
            cursor = conn.execute("SELECT review_priority, COUNT(*) FROM review_items WHERE review_status = 'pending' GROUP BY review_priority")
            stats['priority_counts_pending'] = {row['review_priority']: row['COUNT(*)'] for row in cursor.fetchall()}
            
            stats['pending_reviews_total'] = stats['status_counts'].get(ReviewStatus.PENDING.value, 0)
            
            # Recent items (e.g., last 24 hours)
            twenty_four_hours_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            cursor = conn.execute("SELECT COUNT(*) FROM review_items WHERE created_at > ?", (twenty_four_hours_ago,))
            stats['new_items_last_24h'] = cursor.fetchone()[0]
            
            stats.update(self.get_config_summary()) # Add config to stats
            review_mem_logger.debug("Review stats retrieved.")
            return stats

    @detailed_log_function(LogCategory.SYSTEM)
    async def update_thresholds_async(self, new_thresholds: Dict[str, float]): # Renamed
        """Update confidence thresholds (can be done at runtime)."""
        review_mem_logger.info("Updating confidence thresholds.", parameters=new_thresholds)
        if 'auto_approve_threshold' in new_thresholds: self.auto_approve_threshold = new_thresholds['auto_approve_threshold']
        if 'review_threshold' in new_thresholds: self.review_threshold = new_thresholds['review_threshold']
        if 'reject_threshold' in new_thresholds: self.reject_threshold = new_thresholds['reject_threshold']
        review_mem_logger.info("Confidence thresholds updated.", parameters=self.get_config_summary())
    
    async def get_service_status(self) -> Dict[str, Any]: # For service container
        review_mem_logger.debug("Performing ReviewableMemory health check for service status.")
        db_ok = False
        try:
            if not self._initialized: await self.initialize() # Ensure init
            conn = self._get_db_connection()
            conn.execute("SELECT 1;")
            conn.close()
            db_ok = True
        except Exception: pass

        stats = await self.get_review_stats_async()
        status_report = {
            "status": "healthy" if self._initialized and db_ok else "degraded",
            "initialized": self._initialized,
            "database_connection": "ok" if db_ok else "error",
            **stats
        }
        review_mem_logger.info("ReviewableMemory health check complete.", parameters=status_report)
        return status_report

    async def close(self): # For service container
        review_mem_logger.info("Closing ReviewableMemory.")
        # SQLite connections are opened/closed per operation. No explicit pool to close.
        self._initialized = False
        review_mem_logger.info("ReviewableMemory closed.")

# Factory for service container
def create_reviewable_memory(service_config: Optional[Dict[str, Any]] = None, 
                             unified_memory_manager: Optional[Any] = None) -> ReviewableMemory:
    cfg = service_config.get("reviewable_memory_config", {}) if service_config else {}
    # db_path = cfg.get("DB_PATH", global_settings.data_dir / "databases" / "review_memory.db")
    db_path = cfg.get("DB_PATH", "./storage/databases/review_memory.db") # Simpler default for now

    return ReviewableMemory(
        db_path_str=str(db_path),
        unified_memory_manager=unified_memory_manager,
        service_config=cfg
    )
