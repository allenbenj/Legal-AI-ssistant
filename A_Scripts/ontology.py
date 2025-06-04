"""
Legal ontology definitions with comprehensive prompt hints for LLM-assisted extraction.

This module provides the legal domain ontology with enhanced metadata to guide
AI systems in extracting structured information from legal documents.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Iterable
from collections import namedtuple

EntityMeta = namedtuple('EntityMeta', ['label', 'attributes', 'prompt_hint'])
RelMeta = namedtuple('RelMeta', ['label', 'properties', 'prompt_hint'])

def _EntityMeta(data):
    """Helper function to handle the enum definition from source ontology."""
    return EntityMeta(data[0], data[1], data[2])

def _RelMeta(data):
    """Helper function to handle relationship enum definition."""
    return RelMeta(data[0], data[1], data[2])


class LegalEntityType(Enum):
    """
    Comprehensive legal entity types with AI-friendly prompt hints.
    Each entity includes attributes to extract and context hints for LLM guidance.
    """
    
    # Core People and Parties
    PERSON = _EntityMeta(("Person", ["name", "role"], 
                         "Any individual involved in the case - extract full names and their role/title."))
    WITNESS = _EntityMeta(("Witness", ["name", "contact_information"], 
                          "Individual who provides testimony - look for phrases like 'testified', 'stated', 'declared'."))
    PARTY = _EntityMeta(("Party", ["name", "role"], 
                        "Collective entity like company, agency, or organization involved in the case."))
    JUDGE = _EntityMeta(("Judge", ["name", "court", "jurisdiction"], 
                        "Presiding judge - look for 'Judge', 'Justice', 'Hon.', 'Honorable' titles."))
    PROSECUTOR = _EntityMeta(("Prosecutor", ["name", "office", "jurisdiction"], 
                             "Prosecuting attorney - look for 'DA', 'District Attorney', 'State Attorney'."))
    DEFENSECOUNSEL = _EntityMeta(("DefenseCounsel", ["name", "firm"], 
                                 "Defense attorney - look for 'Attorney for', 'Counsel', 'Esq.', law firm names."))
    EXPERTWITNESS = _EntityMeta(("ExpertWitness", ["name", "field"], 
                                "Specialist witness - look for professional titles, PhD, MD, certifications."))
    VICTIM = _EntityMeta(("Victim", ["name", "case_id"], 
                         "Victim in the case - often referenced as 'victim', 'complainant', 'injured party'."))
    
    # Legal Documents and Filings
    LEGALDOCUMENT = _EntityMeta(("LegalDocument", ["title", "filed_date"], 
                                "Formal pleadings, orders, briefs - look for document types like Motion, Order, Brief."))
    MOTION = _EntityMeta(("Motion", ["filed_on", "status", "result_summary"], 
                         "Filed motion - look for 'Motion to', 'Motion for', filing dates and outcomes."))
    ORDER = _EntityMeta(("Order", ["ruled_on", "status", "result_summary"], 
                        "Court order - look for 'Orders', 'Decrees', judicial rulings and their dates."))
    STATEMENT = _EntityMeta(("Statement", ["speaker", "timestamp", "medium", "verbatim"], 
                            "Discrete testimony - extract who said what, when, and in what context."))
    
    # Case and Procedural Elements
    CASE = _EntityMeta(("Case", ["title", "status", "jurisdiction"], 
                       "Legal case container - look for case numbers, 'v.' or 'vs.', docket numbers."))
    HEARING = _EntityMeta(("Hearing", ["date", "location", "jurisdiction"], 
                          "Court session - look for hearing dates, courtroom numbers, session types."))
    LEGALISSUE = _EntityMeta(("LegalIssue", ["issue", "status"], 
                             "Specific legal issue - constitutional questions, statutory interpretations, disputes."))
    EVENT = _EntityMeta(("Event", ["name", "date"], 
                        "Generic legal event - incidents, meetings, deadlines, significant occurrences."))
    CASEEVENT = _EntityMeta(("CaseEvent", ["name", "date", "event_type"], 
                            "Timeline event - arraignments, depositions, settlements, key case milestones."))
    
    # Evidence and Investigation
    EVIDENCEITEM = _EntityMeta(("EvidenceItem", ["description", "subtype", "collected_date", "source", "hash", "location_found", "integrity_score"], 
                               "Physical or digital evidence - documents, photos, recordings, physical objects with chain of custody."))
    
    # Charges and Legal Violations
    INDICTMENTCOUNT = _EntityMeta(("IndictmentCount", ["count_id", "description", "statute"], 
                                  "Specific charge - look for 'Count I', 'Count 1', numbered charges with statutory citations."))
    OFFENSE = _EntityMeta(("Offense", ["description", "statute"], 
                          "Criminal offense - crimes, violations, infractions with legal code references."))
    
    # Institutional Entities
    COURT = _EntityMeta(("Court", ["name", "level", "jurisdiction"], 
                        "Court entity - District, Superior, Circuit, Federal courts with jurisdictional info."))
    LAWENFORCEMENTAGENCY = _EntityMeta(("LawEnforcementAgency", ["name", "jurisdiction"], 
                                       "Police or similar agency - FBI, local police, sheriff departments, regulatory agencies."))
    
    # Agreements and Resolutions
    PLEADEAL = _EntityMeta(("PleaDeal", ["agreement_date", "terms"], 
                           "Plea agreement - look for 'plea bargain', 'plea agreement', negotiated settlements."))
    SANCTION = _EntityMeta(("Sanction", ["imposed_on", "reason", "severity"], 
                           "Penalty imposed - fines, suspensions, disciplinary actions with reasoning."))
    
    # Task Management
    TASK = _EntityMeta(("Task", ["description", "due_date", "assigned_to", "status"], 
                       "Action item or deadline - things to be done, filing deadlines, court-ordered actions."))

    def __str__(self): 
        return self.value.label
    
    @property
    def attributes(self): 
        return self.value.attributes
    
    @property
    def prompt_hint(self): 
        return self.value.prompt_hint

    @classmethod
    def validate_attrs(cls, ent: 'LegalEntityType', attrs: Dict[str, Any]):
        missing = [a for a in ent.attributes if a not in attrs]
        if missing: 
            raise ValueError(f"{ent} missing {missing}")


class LegalRelationshipType(Enum):
    """
    Legal relationship types with AI-friendly prompt hints for extraction guidance.
    Each relationship connects entities and includes properties to extract.
    """
    
    # Document and Filing Relationships
    FILED_BY = _RelMeta(("Filed_By", ["filed_date"], 
                        "Document filed by entity - look for 'filed by', 'submitted by', filing timestamps."))
    RULED_BY = _RelMeta(("Ruled_By", ["ruled_date"], 
                        "Order ruled by judge/court - look for 'ruled', 'decided', 'ordered by' with dates."))
    PRESIDED_BY = _RelMeta(("Presided_By", ["session_date"], 
                           "Judge presiding over hearing/case - 'presided over', 'heard before Judge'."))
    ADDRESSES = _RelMeta(("Addresses", ["relevance"], 
                         "Motion/document addresses issue - what legal issues are being tackled."))
    
    # Evidence and Argumentation
    SUPPORTS = _RelMeta(("Supports", ["confidence", "analysis_method", "notes"], 
                        "Evidence supports claim - look for 'supports', 'corroborates', 'proves', 'demonstrates'."))
    REFUTES = _RelMeta(("Refutes", ["confidence", "analysis_method", "notes"], 
                       "Evidence refutes claim - look for 'refutes', 'disproves', 'contradicts', 'undermines'."))
    CHALLENGES = _RelMeta(("Challenges", ["argument_summary"], 
                          "Challenges evidence/claim - 'challenges', 'disputes', 'questions the validity of'."))
    CONTRADICTS = _RelMeta(("Contradicts", ["confidence", "notes"], 
                           "Evidence contradicts other evidence - conflicting statements or facts."))
    
    # Citations and References
    CITES = _RelMeta(("Cites", ["citation_date"], 
                     "Document cites legal precedent - 'cites', 'references case', 'pursuant to'."))
    REFERENCES = _RelMeta(("References", ["reference_date"], 
                          "References another document - 'see attached', 'as referenced in', cross-references."))
    
    # Procedural Relationships
    CHAIN_OF_CUSTODY = _RelMeta(("Chain_Of_Custody", ["from_role", "to_role", "timestamp", "method"], 
                                "Evidence custody transfer - tracking who handled evidence when and how."))
    PARTICIPATED_IN = _RelMeta(("Participated_In", ["role"], 
                               "Entity participated in event - 'attended', 'participated in', 'was present at'."))
    OCCURRED_AT = _RelMeta(("Occurred_At", ["location", "date"], 
                           "Event occurred at location - 'took place at', 'occurred at', 'happened in'."))
    OCCURRED_ON = _RelMeta(("Occurred_On", ["date"], 
                           "Event occurred on date - temporal relationships, 'on the date of', 'occurred on'."))
    
    # Legal Actions and Proceedings
    CHARGED_WITH = _RelMeta(("Charged_With", ["charge_date"], 
                            "Person charged with offense - 'charged with', 'accused of', 'indicted for'."))
    DISMISSED_BY = _RelMeta(("Dismissed_By", ["dismissal_date"], 
                            "Charge dismissed by authority - 'dismissed', 'dropped charges', 'case closed'."))
    PLEADS_TO = _RelMeta(("Pleads_To", ["plea_date"], 
                         "Person pleads to charge - 'pleads guilty', 'pleads not guilty', 'enters plea'."))
    SANCTIONED_BY = _RelMeta(("Sanctioned_By", ["sanction_date", "reason"], 
                             "Person sanctioned by authority - disciplinary actions, penalties, punishments."))
    
    # Testimony and Statements
    GAVE_STATEMENT = _RelMeta(("Gave_Statement", ["under_oath", "location"], 
                              "Witness gave statement - 'testified', 'stated under oath', 'deposed'."))
    STATEMENT_IN = _RelMeta(("Statement_In", [], 
                            "Links statement to case/hearing - contextual relationship of testimony."))
    WITNESS_IN = _RelMeta(("Witness_In", ["statement_date", "relevance"], 
                          "Person witnessed event/case - 'witnessed', 'observed', 'was present during'."))
    
    # Verdict and Resolution
    FOUND_GUILTY_OF = _RelMeta(("Found_Guilty_Of", ["verdict_date", "severity"], 
                               "Person found guilty - 'found guilty', 'convicted of', 'verdict of guilty'."))
    FOUND_NOT_GUILTY_OF = _RelMeta(("Found_Not_Guilty_Of", ["verdict_date"], 
                                   "Person found not guilty - 'acquitted', 'found not guilty', 'verdict of not guilty'."))
    APPEALED_TO = _RelMeta(("Appealed_To", ["appeal_date"], 
                           "Case appealed to higher court - 'appealed to', 'petition for review', 'appellate court'."))
    
    # Task and Assignment
    HAS_TASK = _RelMeta(("Has_Task", ["assignment_date", "status"], 
                        "Case has associated task - court orders, deadlines, required actions."))
    ASSIGNED_TO = _RelMeta(("Assigned_To", [], 
                           "Task assigned to person/entity - 'assigned to', 'responsibility of', 'delegated to'."))
    
    # General Relationships
    RELATED_TO = _RelMeta(("Related_To", ["relationship_type", "description"], 
                          "Generic relationship - any connection not covered by specific relationship types."))

    def __str__(self): 
        return self.value.label
    
    @property
    def properties(self): 
        return self.value.properties
    
    @property
    def prompt_hint(self): 
        return self.value.prompt_hint


def get_entity_types_for_prompt() -> str:
    """Generate prompt-friendly list of entity types with extraction guidance."""
    lines = []
    for entity_type in LegalEntityType:
        attrs = ', '.join(entity_type.attributes)
        lines.append(f"- {entity_type.value.label}: {{{attrs}}}  # {entity_type.prompt_hint}")
    return '\n'.join(lines)


def get_relationship_types_for_prompt() -> str:
    """Generate prompt-friendly list of relationship types with extraction guidance."""
    lines = []
    for rel_type in LegalRelationshipType:
        props = ', '.join(rel_type.properties) if rel_type.properties else 'none'
        lines.append(f"- {rel_type.value.label}: {{{props}}}  # {rel_type.prompt_hint}")
    return '\n'.join(lines)


def get_extraction_prompt() -> str:
    """Generate comprehensive extraction prompt for LLM."""
    return f"""
LEGAL ENTITY EXTRACTION GUIDELINES:

ENTITY TYPES:
{get_entity_types_for_prompt()}

RELATIONSHIP TYPES:
{get_relationship_types_for_prompt()}

EXTRACTION INSTRUCTIONS:
1. Look for exact phrases and context clues mentioned in prompt hints
2. Extract all required attributes for each entity type
3. Identify relationships using the connecting phrases specified
4. Maintain high confidence (>0.7) for legal accuracy
5. Preserve exact spellings of names, dates, and legal citations
6. Note jurisdictional information when available
7. Track temporal sequences for proper case timeline construction
"""


def prompt_lines(types: Iterable[Enum]) -> str:
    """Generate prompt lines for a list of entity or relationship types."""
    lines = []
    for t in types:
        if hasattr(t, 'attributes'):  # Entity type
            elems = t.attributes
        else:  # Relationship type
            elems = t.properties
        lines.append(f"- {t.value.label}: {{{', '.join(elems)}}}  # {t.prompt_hint}")
    return '\n'.join(lines)


def validate_entity_attributes(entity_type: LegalEntityType, attributes: Dict[str, Any]) -> bool:
    """Validate that an entity has all required attributes."""
    required_attrs = entity_type.attributes
    missing = [attr for attr in required_attrs if attr not in attributes]
    return len(missing) == 0


def validate_relationship_properties(rel_type: LegalRelationshipType, properties: Dict[str, Any]) -> bool:
    """Validate that a relationship has all required properties."""
    required_props = rel_type.properties
    missing = [prop for prop in required_props if prop not in properties]
    return len(missing) == 0


# Convenience mappings for quick lookup
ENTITY_TYPE_MAPPING = {et.value.label: et for et in LegalEntityType}
RELATIONSHIP_TYPE_MAPPING = {rt.value.label: rt for rt in LegalRelationshipType}


def get_entity_type_by_label(label: str) -> LegalEntityType:
    """Get entity type by its label string."""
    return ENTITY_TYPE_MAPPING.get(label)


def get_relationship_type_by_label(label: str) -> LegalRelationshipType:
    """Get relationship type by its label string."""
    return RELATIONSHIP_TYPE_MAPPING.get(label)