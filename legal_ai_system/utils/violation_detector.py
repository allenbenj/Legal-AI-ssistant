"""
Violation Detector Agent - Specialized Legal Violation Detection

This agent identifies and analyzes various types of legal violations including:
- Constitutional violations (Brady, 4th Amendment, 5th Amendment, etc.)
- Prosecutorial misconduct
- Law enforcement conduct violations
- Evidence tampering and chain of custody issues
- Witness intimidation and tampering
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime

from .base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class ViolationDetectorAgent(BaseAgent):
    """
    Specialized agent for detecting legal violations in documents and case materials
    """
    
    def __init__(self, services):
        super().__init__(
            agent_name="ViolationDetector",
            agent_type="legal_analysis",
            services=services
        )
        
        # Initialize violation patterns and keywords
        self._init_violation_patterns()
        
        logger.info("ViolationDetectorAgent initialized")
    
    def _init_violation_patterns(self):
        """Initialize violation detection patterns and keywords"""
        
        # Brady violation keywords and patterns
        self.brady_patterns = [
            r"brady\s+violation",
            r"exculpatory\s+evidence",
            r"material\s+evidence\s+withheld",
            r"prosecut\w+\s+misconduct",
            r"evidence\s+suppression",
            r"failure\s+to\s+disclose",
            r"impeachment\s+evidence"
        ]
        
        # 4th Amendment violation patterns
        self.fourth_amendment_patterns = [
            r"unreasonable\s+search",
            r"warrantless\s+search",
            r"illegal\s+search",
            r"fourth\s+amendment",
            r"search\s+and\s+seizure",
            r"probable\s+cause\s+lacking",
            r"unlawful\s+detention"
        ]
        
        # 5th Amendment (Miranda) patterns
        self.fifth_amendment_patterns = [
            r"miranda\s+rights?",
            r"right\s+to\s+remain\s+silent",
            r"self[- ]incrimination",
            r"coerced\s+confession",
            r"involuntary\s+statement",
            r"miranda\s+violation"
        ]
        
        # Law enforcement misconduct patterns
        self.leo_misconduct_patterns = [
            r"police\s+brutality",
            r"excessive\s+force",
            r"false\s+arrest",
            r"malicious\s+prosecution",
            r"perjury\s+by\s+officer",
            r"planted\s+evidence",
            r"falsified\s+report"
        ]
        
        # Evidence tampering patterns
        self.evidence_tampering_patterns = [
            r"evidence\s+tampering",
            r"chain\s+of\s+custody",
            r"contaminated\s+evidence",
            r"altered\s+evidence",
            r"destroyed\s+evidence",
            r"missing\s+evidence"
        ]
        
        # Witness tampering patterns
        self.witness_tampering_patterns = [
            r"witness\s+intimidation",
            r"witness\s+tampering",
            r"coaching\s+witness",
            r"threatened\s+witness",
            r"influenced\s+testimony"
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process input to detect legal violations
        
        Args:
            input_data: Dictionary containing text and metadata to analyze
            
        Returns:
            AgentResult with detected violations and analysis
        """
        try:
            logger.info(f"Processing violation detection for document: {input_data.get('doc_id', 'unknown')}")
            
            text = input_data.get('text', input_data.get('content', ''))
            doc_id = input_data.get('doc_id', input_data.get('id', 'unknown'))
            
            if not text:
                return AgentResult(
                    success=False,
                    data={},
                    metadata={'error': 'No text content provided'},
                    confidence=0.0,
                    processing_time=0.0
                )
            
            start_time = datetime.now()
            
            # Detect all violation types
            violations = await self._detect_all_violations(text, doc_id)
            
            # Analyze severity and impact
            analysis = await self._analyze_violations(violations, text)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(violations, analysis)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result_data = {
                'doc_id': doc_id,
                'violations': violations,
                'analysis': analysis,
                'recommendations': recommendations,
                'total_violations': len(violations),
                'severity_distribution': self._get_severity_distribution(violations)
            }
            
            # Calculate overall confidence based on pattern matches and LLM validation
            confidence = self._calculate_confidence(violations)
            
            logger.info(f"Detected {len(violations)} violations with confidence {confidence:.2f}")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    'agent': self.agent_name,
                    'doc_id': doc_id,
                    'processing_time': processing_time,
                    'violation_types': list(set(v['type'] for v in violations))
                },
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in violation detection: {e}")
            return AgentResult(
                success=False,
                data={},
                metadata={'error': str(e)},
                confidence=0.0,
                processing_time=0.0
            )
    
    async def _detect_all_violations(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Detect all types of violations in the text"""
        violations = []
        
        # Pattern-based detection for each violation type
        violation_types = [
            ('Brady Violation', self.brady_patterns),
            ('4th Amendment Violation', self.fourth_amendment_patterns),
            ('5th Amendment Violation', self.fifth_amendment_patterns),
            ('Law Enforcement Misconduct', self.leo_misconduct_patterns),
            ('Evidence Tampering', self.evidence_tampering_patterns),
            ('Witness Tampering', self.witness_tampering_patterns)
        ]
        
        for violation_type, patterns in violation_types:
            matches = self._find_pattern_matches(text, patterns)
            
            for match in matches:
                violation = {
                    'type': violation_type,
                    'description': match['matched_text'],
                    'context': match['context'],
                    'confidence': match['confidence'],
                    'start_pos': match['start'],
                    'end_pos': match['end'],
                    'severity': self._assess_severity(violation_type, match['matched_text']),
                    'detected_by': 'pattern_matching'
                }
                violations.append(violation)
        
        # LLM-based validation and additional detection
        if violations:
            llm_violations = await self._llm_validate_violations(text, violations)
            violations.extend(llm_violations)
        
        return violations
    
    def _find_pattern_matches(self, text: str, patterns: List[str]) -> List[Dict[str, Any]]:
        """Find pattern matches in text with context"""
        matches = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # Extract context around the match
                context_start = max(0, start - 100)
                context_end = min(len(text), end + 100)
                context = text[context_start:context_end].strip()
                
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_pattern_confidence(pattern, match.group())
                
                matches.append({
                    'matched_text': match.group(),
                    'context': context,
                    'start': start,
                    'end': end,
                    'confidence': confidence,
                    'pattern': pattern
                })
        
        return matches
    
    def _calculate_pattern_confidence(self, pattern: str, matched_text: str) -> float:
        """Calculate confidence score for pattern matches"""
        base_confidence = 0.7
        
        # Boost confidence for more specific patterns
        if len(pattern) > 20:
            base_confidence += 0.1
        
        # Boost confidence for exact legal terms
        legal_terms = ['brady', 'miranda', 'amendment', 'constitutional', 'violation']
        if any(term in matched_text.lower() for term in legal_terms):
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _assess_severity(self, violation_type: str, description: str) -> str:
        """Assess the severity of a violation"""
        high_severity_keywords = [
            'constitutional', 'brady', 'perjury', 'fabricated', 'planted',
            'destroyed', 'suppressed', 'withheld', 'malicious'
        ]
        
        medium_severity_keywords = [
            'misconduct', 'violation', 'unlawful', 'improper', 'irregular'
        ]
        
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in high_severity_keywords):
            return 'high'
        elif any(keyword in description_lower for keyword in medium_severity_keywords):
            return 'medium'
        else:
            return 'low'
    
    async def _llm_validate_violations(self, text: str, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to validate and find additional violations"""
        try:
            llm_manager = self.services.llm_manager
            
            # Prepare validation prompt
            violations_summary = "\n".join([
                f"- {v['type']}: {v['description'][:100]}..." 
                for v in violations[:5]  # Limit to first 5 for context
            ])
            
            prompt = f"""
            Analyze the following legal document text for violations. I've detected these potential violations:
            
            {violations_summary}
            
            Document text (first 2000 chars):
            {text[:2000]}
            
            Please:
            1. Validate the detected violations (confirm or dispute each)
            2. Identify any additional legal violations not caught by pattern matching
            3. Provide confidence scores (0.0-1.0) for each violation
            4. Focus on: Brady violations, Constitutional violations, prosecutorial misconduct, evidence tampering
            
            Return a JSON list of violations with fields: type, description, confidence, severity, reasoning
            """
            
            response = await llm_manager.complete(prompt, max_tokens=1000)
            
            # Parse LLM response (simplified - in production would use structured output)
            additional_violations = self._parse_llm_violations(response)
            
            return additional_violations
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return []
    
    def _parse_llm_violations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for additional violations"""
        violations = []
        
        # Simplified parsing - in production would use structured output
        try:
            import json
            
            # Try to extract JSON from response
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed_violations = json.loads(json_str)
                
                for v in parsed_violations:
                    if isinstance(v, dict) and 'type' in v:
                        violation = {
                            'type': v.get('type', 'Unknown'),
                            'description': v.get('description', ''),
                            'confidence': float(v.get('confidence', 0.5)),
                            'severity': v.get('severity', 'medium'),
                            'detected_by': 'llm_analysis',
                            'reasoning': v.get('reasoning', ''),
                            'start_pos': -1,
                            'end_pos': -1,
                            'context': v.get('description', '')[:200]
                        }
                        violations.append(violation)
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM violations: {e}")
        
        return violations
    
    async def _analyze_violations(self, violations: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Analyze the detected violations for patterns and impact"""
        
        analysis = {
            'total_violations': len(violations),
            'violation_types': {},
            'severity_analysis': {'high': 0, 'medium': 0, 'low': 0},
            'potential_impact': [],
            'legal_precedents': [],
            'case_strength_impact': 'unknown'
        }
        
        # Count violations by type
        for violation in violations:
            v_type = violation['type']
            analysis['violation_types'][v_type] = analysis['violation_types'].get(v_type, 0) + 1
            
            severity = violation.get('severity', 'medium')
            analysis['severity_analysis'][severity] += 1
        
        # Assess potential impact
        if analysis['severity_analysis']['high'] > 0:
            analysis['potential_impact'].append('Case dismissal possible')
            analysis['potential_impact'].append('Serious constitutional issues')
            analysis['case_strength_impact'] = 'severe_negative'
        elif analysis['severity_analysis']['medium'] > 2:
            analysis['potential_impact'].append('Significant case weakness')
            analysis['case_strength_impact'] = 'moderate_negative'
        
        # Add specific legal precedents for detected violations
        for v_type in analysis['violation_types'].keys():
            precedents = self._get_legal_precedents(v_type)
            analysis['legal_precedents'].extend(precedents)
        
        return analysis
    
    def _get_legal_precedents(self, violation_type: str) -> List[str]:
        """Get relevant legal precedents for violation types"""
        precedents_map = {
            'Brady Violation': [
                'Brady v. Maryland (1963)',
                'Giglio v. United States (1972)',
                'United States v. Bagley (1985)'
            ],
            '4th Amendment Violation': [
                'Mapp v. Ohio (1961)',
                'Terry v. Ohio (1968)',
                'Katz v. United States (1967)'
            ],
            '5th Amendment Violation': [
                'Miranda v. Arizona (1966)',
                'Dickerson v. United States (2000)',
                'Berghuis v. Thompkins (2010)'
            ],
            'Law Enforcement Misconduct': [
                'Monroe v. Pape (1961)',
                'Tennessee v. Garner (1985)',
                'Graham v. Connor (1989)'
            ]
        }
        
        return precedents_map.get(violation_type, [])
    
    async def _generate_recommendations(self, violations: List[Dict[str, Any]], 
                                      analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate legal recommendations based on detected violations"""
        recommendations = []
        
        if analysis['severity_analysis']['high'] > 0:
            recommendations.append({
                'priority': 'urgent',
                'action': 'File motion to dismiss',
                'reasoning': 'High-severity constitutional violations detected',
                'timeline': 'immediate'
            })
        
        if 'Brady Violation' in analysis['violation_types']:
            recommendations.append({
                'priority': 'high',
                'action': 'File Brady motion for sanctions',
                'reasoning': 'Prosecutorial misconduct in evidence disclosure',
                'timeline': 'within 30 days'
            })
        
        if '4th Amendment Violation' in analysis['violation_types']:
            recommendations.append({
                'priority': 'high',
                'action': 'File motion to suppress evidence',
                'reasoning': 'Evidence obtained through unlawful search',
                'timeline': 'pre-trial'
            })
        
        if '5th Amendment Violation' in analysis['violation_types']:
            recommendations.append({
                'priority': 'high',
                'action': 'File motion to suppress statements',
                'reasoning': 'Statements obtained in violation of Miranda rights',
                'timeline': 'pre-trial'
            })
        
        # Add general recommendations
        if violations:
            recommendations.append({
                'priority': 'medium',
                'action': 'Comprehensive case review',
                'reasoning': 'Multiple violations detected requiring thorough analysis',
                'timeline': 'ongoing'
            })
        
        return recommendations
    
    def _get_severity_distribution(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of violation severities"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for violation in violations:
            severity = violation.get('severity', 'medium')
            distribution[severity] += 1
        
        return distribution
    
    def _calculate_confidence(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in violation detection"""
        if not violations:
            return 0.0
        
        # Average confidence of all violations, weighted by severity
        total_weight = 0
        weighted_confidence = 0
        
        severity_weights = {'high': 1.0, 'medium': 0.8, 'low': 0.6}
        
        for violation in violations:
            confidence = violation.get('confidence', 0.5)
            severity = violation.get('severity', 'medium')
            weight = severity_weights[severity]
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0