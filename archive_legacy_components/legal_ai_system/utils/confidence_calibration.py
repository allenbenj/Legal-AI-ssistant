"""Confidence Calibration System for Hybrid NER Models.

Implements confidence calibration to resolve issues with heterogeneous model outputs
and uncalibrated confidence scores in the hybrid legal extraction system.

This module provides temperature scaling for LLMs and Platt scaling for traditional
NER models to create uniform confidence scales for improved ensemble decisions.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .constants import Constants
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Initialize loggers
calibration_logger = get_detailed_logger("Confidence_Calibration", LogCategory.SYSTEM)
model_logger = get_detailed_logger("Model_Calibration", LogCategory.SYSTEM)
validation_logger = get_detailed_logger("Calibration_Validation", LogCategory.SYSTEM)


@dataclass
class EntityPrediction:
    """Standardized entity prediction with calibrated confidence.
    
    Attributes:
        text: The extracted entity text.
        label: The predicted entity label/type.
        start_pos: Starting character position in source text.
        end_pos: Ending character position in source text.
        raw_confidence: Original model confidence score.
        calibrated_confidence: Calibrated confidence score.
        model_source: Name of the model that made this prediction.
        context: Surrounding text context for disambiguation.
    """
    text: str
    label: str
    start_pos: int
    end_pos: int
    raw_confidence: float
    calibrated_confidence: Optional[float] = None
    model_source: str = "unknown"
    context: str = ""


@dataclass
class ValidationSample:
    """Sample for confidence calibration validation.
    
    Attributes:
        text: Input text.
        true_entities: Ground truth entities.
        predictions: Model predictions with confidence scores.
    """
    text: str
    true_entities: List[EntityPrediction]
    predictions: List[EntityPrediction]


class ConfidenceCalibrator(ABC):
    """Abstract base class for confidence calibration methods."""
    
    @abstractmethod
    def fit(self, predictions: List[float], true_labels: List[bool]) -> None:
        """Train the calibration model.
        
        Args:
            predictions: Raw confidence scores from the model.
            true_labels: Binary labels indicating if predictions were correct.
        """
        pass
    
    @abstractmethod
    def calibrate(self, confidence: float) -> float:
        """Apply calibration to a confidence score.
        
        Args:
            confidence: Raw confidence score.
            
        Returns:
            Calibrated confidence score.
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save calibration model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load calibration model from disk."""
        pass


class TemperatureScaling(ConfidenceCalibrator):
    """Temperature scaling calibration for neural network outputs.
    
    Applies a learned temperature parameter to logits before softmax
    to calibrate confidence scores, particularly effective for LLMs.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self):
        """Initialize temperature scaling calibrator."""
        self.temperature = 1.0
        self.is_fitted = False
        calibration_logger.info("Initialized TemperatureScaling calibrator")
    
    def fit(self, predictions: List[float], true_labels: List[bool]) -> None:
        """Fit temperature parameter using validation data.
        
        Args:
            predictions: Raw confidence scores (post-softmax probabilities).
            true_labels: Binary labels indicating correctness.
        """
        if not SKLEARN_AVAILABLE:
            calibration_logger.warning("scikit-learn not available, using default temperature=1.0")
            return
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels, dtype=int)
        
        # Convert probabilities back to logits (approximate)
        epsilon = 1e-15
        logits = np.log(np.clip(predictions, epsilon, 1 - epsilon) / 
                       np.clip(1 - predictions, epsilon, 1 - epsilon))
        
        # Find optimal temperature using cross-entropy loss
        from scipy.optimize import minimize_scalar
        
        def temperature_loss(temp):
            calibrated_probs = 1 / (1 + np.exp(-logits / temp))
            return log_loss(true_labels, calibrated_probs)
        
        result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        calibration_logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}")
    
    def calibrate(self, confidence: float) -> float:
        """Apply temperature scaling to confidence score.
        
        Args:
            confidence: Raw confidence score.
            
        Returns:
            Temperature-scaled confidence score.
        """
        if not self.is_fitted:
            return confidence
        
        # Convert to logit, apply temperature, convert back
        epsilon = 1e-15
        logit = np.log(np.clip(confidence, epsilon, 1 - epsilon) / 
                      np.clip(1 - confidence, epsilon, 1 - epsilon))
        
        calibrated_logit = logit / self.temperature
        calibrated_confidence = 1 / (1 + np.exp(-calibrated_logit))
        
        return float(calibrated_confidence)
    
    def save(self, path: Path) -> None:
        """Save temperature parameter to disk."""
        calibration_data = {
            'temperature': self.temperature,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(calibration_data, f)
        calibration_logger.info(f"Temperature scaling saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load temperature parameter from disk."""
        with open(path, 'rb') as f:
            calibration_data = pickle.load(f)
        self.temperature = calibration_data['temperature']
        self.is_fitted = calibration_data['is_fitted']
        calibration_logger.info(f"Temperature scaling loaded from {path}")


class PlattScaling(ConfidenceCalibrator):
    """Platt scaling calibration using sigmoid function.
    
    Fits a sigmoid function to map raw confidence scores to calibrated
    probabilities, effective for traditional ML models like Flair.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self):
        """Initialize Platt scaling calibrator."""
        self.calibrator = None
        self.is_fitted = False
        calibration_logger.info("Initialized PlattScaling calibrator")
    
    def fit(self, predictions: List[float], true_labels: List[bool]) -> None:
        """Fit Platt scaling using logistic regression.
        
        Args:
            predictions: Raw confidence scores from the model.
            true_labels: Binary labels indicating correctness.
        """
        if not SKLEARN_AVAILABLE:
            calibration_logger.warning("scikit-learn not available, skipping Platt scaling")
            return
        
        predictions = np.array(predictions).reshape(-1, 1)
        true_labels = np.array(true_labels, dtype=int)
        
        # Use CalibratedClassifierCV with sigmoid method (Platt scaling)
        base_classifier = LogisticRegression()
        self.calibrator = CalibratedClassifierCV(
            base_classifier, 
            method="sigmoid",
            cv=3  # 3-fold cross-validation
        )
        
        try:
            self.calibrator.fit(predictions, true_labels)
            self.is_fitted = True
            calibration_logger.info("Platt scaling fitted successfully")
        except Exception as e:
            calibration_logger.error(f"Platt scaling fit failed: {e}")
            self.is_fitted = False
    
    def calibrate(self, confidence: float) -> float:
        """Apply Platt scaling to confidence score.
        
        Args:
            confidence: Raw confidence score.
            
        Returns:
            Platt-scaled confidence score.
        """
        if not self.is_fitted or self.calibrator is None:
            return confidence
        
        try:
            calibrated_prob = self.calibrator.predict_proba([[confidence]])[0][1]
            return float(calibrated_prob)
        except Exception as e:
            calibration_logger.warning(f"Platt calibration failed: {e}")
            return confidence
    
    def save(self, path: Path) -> None:
        """Save Platt scaling model to disk."""
        calibration_data = {
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(calibration_data, f)
        calibration_logger.info(f"Platt scaling saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load Platt scaling model from disk."""
        with open(path, 'rb') as f:
            calibration_data = pickle.load(f)
        self.calibrator = calibration_data['calibrator']
        self.is_fitted = calibration_data['is_fitted']
        calibration_logger.info(f"Platt scaling loaded from {path}")


class ModelOutputNormalizer:
    """Normalizes heterogeneous model outputs to standardized format.
    
    Handles different output formats from spaCy (IOB), Flair (spans),
    Blackstone (rules), and LLMs (unstructured) to create uniform
    EntityPrediction objects.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self):
        """Initialize model output normalizer."""
        self.label_mappings = {
            # Standard to legal entity mapping
            'PERSON': 'PERSON',
            'PER': 'PERSON', 
            'ORGANIZATION': 'ORGANIZATION',
            'ORG': 'ORGANIZATION',
            'LOCATION': 'LOCATION',
            'LOC': 'LOCATION',
            'MISC': 'CONCEPT',
            
            # Legal-specific mappings
            'OFFENSE': 'LEGAL_VIOLATION',
            'STATUTE': 'STATUTE',
            'REGULATION': 'REGULATION',
            'CASE': 'CASE',
            'COURT': 'COURT',
            'JUDGE': 'JUDGE',
            'LAWYER': 'LAWYER'
        }
        
        model_logger.info("Initialized ModelOutputNormalizer with legal entity mappings")
    
    def normalize_spacy_output(self, doc, model_name: str = "spacy") -> List[EntityPrediction]:
        """Normalize spaCy IOB tagged output.
        
        Args:
            doc: spaCy Doc object with NER annotations.
            model_name: Name of the spaCy model used.
            
        Returns:
            List of normalized EntityPrediction objects.
        """
        predictions = []
        
        for ent in doc.ents:
            normalized_label = self.label_mappings.get(ent.label_, ent.label_)
            
            prediction = EntityPrediction(
                text=ent.text,
                label=normalized_label,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                raw_confidence=getattr(ent, 'confidence', 0.8),  # spaCy doesn't provide confidence
                model_source=model_name,
                context=doc.text[max(0, ent.start_char-50):ent.end_char+50]
            )
            predictions.append(prediction)
        
        model_logger.debug(f"Normalized {len(predictions)} spaCy entities")
        return predictions
    
    def normalize_flair_output(self, sentence, model_name: str = "flair") -> List[EntityPrediction]:
        """Normalize Flair multi-label output.
        
        Args:
            sentence: Flair Sentence object with NER labels.
            model_name: Name of the Flair model used.
            
        Returns:
            List of normalized EntityPrediction objects.
        """
        predictions = []
        
        for entity in sentence.get_labels("ner"):
            normalized_label = self.label_mappings.get(entity.tag, entity.tag)
            
            prediction = EntityPrediction(
                text=entity.data_point.text,
                label=normalized_label,
                start_pos=entity.data_point.start_position,
                end_pos=entity.data_point.end_position,
                raw_confidence=entity.score,
                model_source=model_name,
                context=sentence.to_original_text()
            )
            predictions.append(prediction)
        
        model_logger.debug(f"Normalized {len(predictions)} Flair entities")
        return predictions
    
    def normalize_llm_output(self, llm_response: str, source_text: str, 
                           model_name: str = "llm") -> List[EntityPrediction]:
        """Normalize unstructured LLM output.
        
        Args:
            llm_response: Raw LLM response containing entity information.
            source_text: Original text that was analyzed.
            model_name: Name of the LLM model used.
            
        Returns:
            List of normalized EntityPrediction objects.
        """
        predictions = []
        
        try:
            # Parse JSON from LLM response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', llm_response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*?\}', llm_response, re.DOTALL)
            
            if json_match:
                entities_data = json.loads(json_match.group())
                
                # Handle both list and single object formats
                if isinstance(entities_data, dict):
                    entities_data = [entities_data]
                
                for entity_data in entities_data:
                    entity_text = entity_data.get('text', '')
                    entity_label = entity_data.get('label', 'UNKNOWN')
                    confidence = entity_data.get('confidence', 0.9)
                    
                    # Find entity position in source text
                    start_pos = source_text.find(entity_text)
                    if start_pos != -1:
                        end_pos = start_pos + len(entity_text)
                        
                        normalized_label = self.label_mappings.get(entity_label, entity_label)
                        
                        prediction = EntityPrediction(
                            text=entity_text,
                            label=normalized_label,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            raw_confidence=confidence,
                            model_source=model_name,
                            context=source_text[max(0, start_pos-50):end_pos+50]
                        )
                        predictions.append(prediction)
            
        except Exception as e:
            model_logger.warning(f"LLM output normalization failed: {e}")
        
        model_logger.debug(f"Normalized {len(predictions)} LLM entities")
        return predictions


class CalibratedEnsembleVoter:
    """Ensemble voting system using calibrated confidence scores.
    
    Resolves conflicts between different models using calibrated confidence
    scores, with domain-specific rules for legal entity prioritization.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self, calibration_storage_path: Optional[Path] = None):
        """Initialize calibrated ensemble voter.
        
        Args:
            calibration_storage_path: Path to store/load calibration models.
        """
        self.calibrators: Dict[str, ConfidenceCalibrator] = {}
        self.model_weights = {
            'blackstone': 1.5,  # Higher weight for legal-specific model
            'flair': 1.2,       # Good general NER performance
            'spacy': 1.0,       # Baseline weight
            'llm': 1.3          # Strong contextual understanding
        }
        
        self.storage_path = calibration_storage_path or Path("storage/calibration/")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Legal entity prioritization rules
        self.legal_priority_labels = {
            'LEGAL_VIOLATION', 'STATUTE', 'REGULATION', 'CASE', 
            'COURT', 'JUDGE', 'LAWYER', 'EVIDENCE'
        }
        
        validation_logger.info("Initialized CalibratedEnsembleVoter")
    
    def add_calibrator(self, model_name: str, calibrator_type: str = "platt") -> None:
        """Add a calibrator for a specific model.
        
        Args:
            model_name: Name of the model to calibrate.
            calibrator_type: Type of calibration ('platt' or 'temperature').
        """
        if calibrator_type == "temperature":
            calibrator = TemperatureScaling()
        else:
            calibrator = PlattScaling()
        
        self.calibrators[model_name] = calibrator
        calibration_logger.info(f"Added {calibrator_type} calibrator for {model_name}")
    
    def train_calibrators(self, validation_samples: List[ValidationSample]) -> None:
        """Train all calibrators using validation data.
        
        Args:
            validation_samples: Validation samples with ground truth.
        """
        model_data = {}
        
        # Collect predictions by model
        for sample in validation_samples:
            for pred in sample.predictions:
                model_name = pred.model_source
                if model_name not in model_data:
                    model_data[model_name] = {'predictions': [], 'labels': []}
                
                # Check if prediction matches any ground truth entity
                is_correct = any(
                    pred.text.lower() == true_ent.text.lower() and 
                    pred.label == true_ent.label
                    for true_ent in sample.true_entities
                )
                
                model_data[model_name]['predictions'].append(pred.raw_confidence)
                model_data[model_name]['labels'].append(is_correct)
        
        # Train calibrators for each model
        for model_name, data in model_data.items():
            if model_name in self.calibrators and len(data['predictions']) > 10:
                self.calibrators[model_name].fit(
                    data['predictions'], 
                    data['labels']
                )
                
                # Save calibrator
                save_path = self.storage_path / f"{model_name}_calibrator.pkl"
                self.calibrators[model_name].save(save_path)
                
                calibration_logger.info(
                    f"Trained and saved calibrator for {model_name} "
                    f"with {len(data['predictions'])} samples"
                )
    
    def load_calibrators(self) -> None:
        """Load all saved calibrators from disk."""
        for model_name in self.calibrators:
            save_path = self.storage_path / f"{model_name}_calibrator.pkl"
            if save_path.exists():
                try:
                    self.calibrators[model_name].load(save_path)
                    calibration_logger.info(f"Loaded calibrator for {model_name}")
                except Exception as e:
                    calibration_logger.warning(f"Failed to load calibrator for {model_name}: {e}")
    
    def vote_on_entities(self, model_predictions: Dict[str, List[EntityPrediction]]) -> List[EntityPrediction]:
        """Perform ensemble voting using calibrated confidence scores.
        
        Args:
            model_predictions: Dictionary mapping model names to their predictions.
            
        Returns:
            List of final ensemble predictions.
        """
        # Apply calibration to all predictions
        calibrated_predictions = {}
        for model_name, predictions in model_predictions.items():
            calibrated_preds = []
            
            for pred in predictions:
                calibrated_pred = EntityPrediction(
                    text=pred.text,
                    label=pred.label,
                    start_pos=pred.start_pos,
                    end_pos=pred.end_pos,
                    raw_confidence=pred.raw_confidence,
                    calibrated_confidence=self._calibrate_prediction(model_name, pred),
                    model_source=pred.model_source,
                    context=pred.context
                )
                calibrated_preds.append(calibrated_pred)
            
            calibrated_predictions[model_name] = calibrated_preds
        
        # Group overlapping entities
        entity_groups = self._group_overlapping_entities(calibrated_predictions)
        
        # Vote on each group
        final_predictions = []
        for group in entity_groups:
            winner = self._resolve_entity_group(group)
            if winner:
                final_predictions.append(winner)
        
        validation_logger.info(f"Ensemble voting produced {len(final_predictions)} final entities")
        return final_predictions
    
    def _calibrate_prediction(self, model_name: str, prediction: EntityPrediction) -> float:
        """Apply calibration to a single prediction.
        
        Args:
            model_name: Name of the model that made the prediction.
            prediction: The prediction to calibrate.
            
        Returns:
            Calibrated confidence score.
        """
        if model_name in self.calibrators:
            return self.calibrators[model_name].calibrate(prediction.raw_confidence)
        return prediction.raw_confidence
    
    def _group_overlapping_entities(self, 
                                  model_predictions: Dict[str, List[EntityPrediction]]) -> List[List[EntityPrediction]]:
        """Group overlapping entity predictions from different models.
        
        Args:
            model_predictions: Calibrated predictions from all models.
            
        Returns:
            List of entity groups (overlapping predictions).
        """
        all_predictions = []
        for predictions in model_predictions.values():
            all_predictions.extend(predictions)
        
        # Sort by start position
        all_predictions.sort(key=lambda x: x.start_pos)
        
        groups = []
        current_group = []
        
        for pred in all_predictions:
            if not current_group:
                current_group.append(pred)
            else:
                # Check if this prediction overlaps with any in current group
                overlaps = any(
                    self._entities_overlap(pred, existing)
                    for existing in current_group
                )
                
                if overlaps:
                    current_group.append(pred)
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [pred]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _entities_overlap(self, ent1: EntityPrediction, ent2: EntityPrediction) -> bool:
        """Check if two entities overlap in text position.
        
        Args:
            ent1: First entity prediction.
            ent2: Second entity prediction.
            
        Returns:
            True if entities overlap, False otherwise.
        """
        return not (ent1.end_pos <= ent2.start_pos or ent2.end_pos <= ent1.start_pos)
    
    def _resolve_entity_group(self, group: List[EntityPrediction]) -> Optional[EntityPrediction]:
        """Resolve conflicts within an entity group using voting.
        
        Args:
            group: List of overlapping entity predictions.
            
        Returns:
            Winning entity prediction or None.
        """
        if not group:
            return None
        
        if len(group) == 1:
            return group[0]
        
        # Apply legal entity prioritization
        legal_entities = [ent for ent in group if ent.label in self.legal_priority_labels]
        if legal_entities:
            # Among legal entities, choose highest calibrated confidence
            return max(legal_entities, key=lambda x: (x.calibrated_confidence or 0) * 
                      self.model_weights.get(x.model_source, 1.0))
        
        # For non-legal entities, use weighted confidence voting
        scores = {}
        for pred in group:
            key = (pred.text.lower(), pred.label)
            weight = self.model_weights.get(pred.model_source, 1.0)
            confidence = pred.calibrated_confidence or pred.raw_confidence
            score = confidence * weight
            
            if key not in scores:
                scores[key] = {'score': 0, 'prediction': pred}
            
            if score > scores[key]['score']:
                scores[key] = {'score': score, 'prediction': pred}
        
        if scores:
            winner = max(scores.values(), key=lambda x: x['score'])
            return winner['prediction']
        
        return None


class ConfidenceCalibrationManager:
    """Main manager for confidence calibration system.
    
    Coordinates calibration training, model output normalization,
    and ensemble voting for the hybrid legal extraction system.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize confidence calibration manager.
        
        Args:
            storage_path: Path for storing calibration models and data.
        """
        self.storage_path = storage_path or Path("storage/calibration/")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.normalizer = ModelOutputNormalizer()
        self.ensemble_voter = CalibratedEnsembleVoter(self.storage_path)
        
        # Add calibrators for supported models
        self.ensemble_voter.add_calibrator("spacy", "platt")
        self.ensemble_voter.add_calibrator("flair", "platt")
        self.ensemble_voter.add_calibrator("blackstone", "platt")
        self.ensemble_voter.add_calibrator("llm", "temperature")
        
        calibration_logger.info("ConfidenceCalibrationManager initialized")
    
    def train_system(self, validation_samples: List[ValidationSample]) -> None:
        """Train the entire calibration system.
        
        Args:
            validation_samples: Validation data with ground truth annotations.
        """
        calibration_logger.info(f"Training calibration system with {len(validation_samples)} samples")
        
        # Train ensemble voter calibrators
        self.ensemble_voter.train_calibrators(validation_samples)
        
        # Evaluate calibration quality
        self._evaluate_calibration_quality(validation_samples)
        
        calibration_logger.info("Calibration training completed")
    
    def process_predictions(self, 
                          spacy_doc=None,
                          flair_sentence=None, 
                          llm_response: str = "",
                          source_text: str = "",
                          blackstone_entities: List = None) -> List[EntityPrediction]:
        """Process predictions from all models and return ensemble results.
        
        Args:
            spacy_doc: spaCy Doc object with NER annotations.
            flair_sentence: Flair Sentence object with NER labels.
            llm_response: Raw LLM response containing entity information.
            source_text: Original text that was analyzed.
            blackstone_entities: Blackstone rule-based entity extractions.
            
        Returns:
            List of final ensemble predictions with calibrated confidence.
        """
        model_predictions = {}
        
        # Normalize outputs from each model
        if spacy_doc:
            model_predictions["spacy"] = self.normalizer.normalize_spacy_output(spacy_doc)
        
        if flair_sentence:
            model_predictions["flair"] = self.normalizer.normalize_flair_output(flair_sentence)
        
        if llm_response and source_text:
            model_predictions["llm"] = self.normalizer.normalize_llm_output(
                llm_response, source_text
            )
        
        if blackstone_entities:
            # Normalize Blackstone entities (assuming they're in a specific format)
            normalized_blackstone = []
            for ent in blackstone_entities:
                prediction = EntityPrediction(
                    text=ent.get('text', ''),
                    label=ent.get('label', ''),
                    start_pos=ent.get('start', 0),
                    end_pos=ent.get('end', 0),
                    raw_confidence=ent.get('confidence', 0.95),  # Blackstone rules are typically high confidence
                    model_source="blackstone"
                )
                normalized_blackstone.append(prediction)
            model_predictions["blackstone"] = normalized_blackstone
        
        # Perform ensemble voting
        if model_predictions:
            return self.ensemble_voter.vote_on_entities(model_predictions)
        
        return []
    
    def _evaluate_calibration_quality(self, validation_samples: List[ValidationSample]) -> None:
        """Evaluate the quality of confidence calibration.
        
        Args:
            validation_samples: Validation samples for evaluation.
        """
        if not SKLEARN_AVAILABLE:
            validation_logger.warning("scikit-learn not available, skipping calibration evaluation")
            return
        
        total_predictions = 0
        correct_predictions = 0
        calibration_errors = []
        
        for sample in validation_samples:
            # Get ensemble predictions
            model_preds = {}
            for pred in sample.predictions:
                model_name = pred.model_source
                if model_name not in model_preds:
                    model_preds[model_name] = []
                model_preds[model_name].append(pred)
            
            ensemble_preds = self.ensemble_voter.vote_on_entities(model_preds)
            
            for pred in ensemble_preds:
                total_predictions += 1
                
                # Check if prediction is correct
                is_correct = any(
                    pred.text.lower() == true_ent.text.lower() and 
                    pred.label == true_ent.label
                    for true_ent in sample.true_entities
                )
                
                if is_correct:
                    correct_predictions += 1
                
                # Calculate calibration error
                confidence = pred.calibrated_confidence or pred.raw_confidence
                calibration_error = abs(confidence - (1.0 if is_correct else 0.0))
                calibration_errors.append(calibration_error)
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            mean_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
            
            validation_logger.info(
                f"Calibration evaluation: Accuracy={accuracy:.3f}, "
                f"Mean Calibration Error={mean_calibration_error:.3f}"
            )
        else:
            validation_logger.warning("No predictions found during calibration evaluation")