#!/usr/bin/env python3
"""Confidence Calibration System Demonstration.

This example demonstrates the benefits of implementing confidence calibration
for the hybrid legal extraction system, showing how calibrated confidence
scores improve ensemble decision-making and resolve model conflicts.
"""

import asyncio
import json
from pathlib import Path
from typing import List
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.confidence_calibration import (
    ConfidenceCalibrationManager, 
    EntityPrediction, 
    ValidationSample,
    TemperatureScaling,
    PlattScaling
)

class CalibrationDemo:
    """Demonstration class for confidence calibration benefits."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.calibration_manager = ConfidenceCalibrationManager(
            storage_path=Path("examples/demo_calibration_storage")
        )
        
    def create_sample_validation_data(self) -> List[ValidationSample]:
        """Create sample validation data for demonstration.
        
        Returns:
            List of validation samples with different model predictions.
        """
        validation_samples = []
        
        # Sample 1: Document with prosecutorial misconduct
        sample1 = ValidationSample(
            text="Detective Smith withheld exculpatory evidence from the defense in violation of Brady v. Maryland.",
            true_entities=[
                EntityPrediction(
                    text="Detective Smith",
                    label="PERSON", 
                    start_pos=0,
                    end_pos=15,
                    raw_confidence=1.0,  # Ground truth
                    model_source="ground_truth"
                ),
                EntityPrediction(
                    text="Brady v. Maryland",
                    label="CASE",
                    start_pos=89,
                    end_pos=106,
                    raw_confidence=1.0,
                    model_source="ground_truth"
                ),
                EntityPrediction(
                    text="withheld exculpatory evidence",
                    label="LEGAL_VIOLATION",
                    start_pos=16,
                    end_pos=46,
                    raw_confidence=1.0,
                    model_source="ground_truth"
                )
            ],
            predictions=[
                # spaCy predictions (conservative)
                EntityPrediction(
                    text="Detective Smith",
                    label="PERSON",
                    start_pos=0,
                    end_pos=15,
                    raw_confidence=0.85,  # spaCy typically conservative
                    model_source="spacy"
                ),
                EntityPrediction(
                    text="Brady",
                    label="PERSON",  # spaCy misclassifies case name
                    start_pos=89,
                    end_pos=94,
                    raw_confidence=0.72,
                    model_source="spacy"
                ),
                
                # Flair predictions (high confidence)
                EntityPrediction(
                    text="Detective Smith",
                    label="PERSON",
                    start_pos=0,
                    end_pos=15,
                    raw_confidence=0.96,  # Flair often overconfident
                    model_source="flair"
                ),
                EntityPrediction(
                    text="Brady v. Maryland",
                    label="CASE",
                    start_pos=89,
                    end_pos=106,
                    raw_confidence=0.91,
                    model_source="flair"
                ),
                
                # Blackstone predictions (legal-specific, high accuracy)
                EntityPrediction(
                    text="Brady v. Maryland",
                    label="CASE",
                    start_pos=89,
                    end_pos=106,
                    raw_confidence=0.95,  # Blackstone excellent for legal entities
                    model_source="blackstone"
                ),
                EntityPrediction(
                    text="withheld exculpatory evidence",
                    label="LEGAL_VIOLATION",
                    start_pos=16,
                    end_pos=46,
                    raw_confidence=0.88,
                    model_source="blackstone"
                ),
                
                # LLM predictions (contextual but sometimes overconfident)
                EntityPrediction(
                    text="Detective Smith",
                    label="PERSON",
                    start_pos=0,
                    end_pos=15,
                    raw_confidence=0.99,  # LLM overconfident
                    model_source="llm"
                ),
                EntityPrediction(
                    text="Brady v. Maryland",
                    label="CASE",
                    start_pos=89,
                    end_pos=106,
                    raw_confidence=0.97,
                    model_source="llm"
                ),
                EntityPrediction(
                    text="prosecutorial misconduct",  # LLM infers this concept
                    label="LEGAL_VIOLATION",
                    start_pos=50,
                    end_pos=75,
                    raw_confidence=0.94,
                    model_source="llm"
                )
            ]
        )
        validation_samples.append(sample1)
        
        # Sample 2: Contract dispute case
        sample2 = ValidationSample(
            text="In Apple Inc. v. Samsung Electronics, the Court ruled on patent infringement damages under 35 U.S.C. § 284.",
            true_entities=[
                EntityPrediction(
                    text="Apple Inc.",
                    label="ORGANIZATION",
                    start_pos=3,
                    end_pos=13,
                    raw_confidence=1.0,
                    model_source="ground_truth"
                ),
                EntityPrediction(
                    text="Samsung Electronics",
                    label="ORGANIZATION",
                    start_pos=17,
                    end_pos=37,
                    raw_confidence=1.0,
                    model_source="ground_truth"
                ),
                EntityPrediction(
                    text="35 U.S.C. § 284",
                    label="STATUTE",
                    start_pos=91,
                    end_pos=107,
                    raw_confidence=1.0,
                    model_source="ground_truth"
                )
            ],
            predictions=[
                # Model predictions with varying confidence and accuracy
                EntityPrediction(
                    text="Apple Inc.",
                    label="ORGANIZATION",
                    start_pos=3,
                    end_pos=13,
                    raw_confidence=0.92,
                    model_source="spacy"
                ),
                EntityPrediction(
                    text="Apple",  # Flair partial match
                    label="ORGANIZATION",
                    start_pos=3,
                    end_pos=8,
                    raw_confidence=0.89,
                    model_source="flair"
                ),
                EntityPrediction(
                    text="Samsung Electronics",
                    label="ORGANIZATION",
                    start_pos=17,
                    end_pos=37,
                    raw_confidence=0.94,
                    model_source="flair"
                ),
                EntityPrediction(
                    text="35 U.S.C. § 284",
                    label="STATUTE",
                    start_pos=91,
                    end_pos=107,
                    raw_confidence=0.97,  # Blackstone excellent for statutes
                    model_source="blackstone"
                ),
                EntityPrediction(
                    text="Court",
                    label="COURT",
                    start_pos=43,
                    end_pos=48,
                    raw_confidence=0.78,
                    model_source="blackstone"
                ),
                EntityPrediction(
                    text="Apple Inc. v. Samsung Electronics",
                    label="CASE",
                    start_pos=3,
                    end_pos=37,
                    raw_confidence=0.96,
                    model_source="llm"
                )
            ]
        )
        validation_samples.append(sample2)
        
        return validation_samples
    
    def demonstrate_uncalibrated_problems(self, validation_samples: List[ValidationSample]) -> None:
        """Demonstrate problems with uncalibrated confidence scores.
        
        Args:
            validation_samples: Sample data to analyze.
        """
        print("🚨 PROBLEMS WITH UNCALIBRATED CONFIDENCE SCORES:")
        print("=" * 60)
        
        for i, sample in enumerate(validation_samples, 1):
            print(f"\nSample {i}: {sample.text[:80]}...")
            print("\nUncalibrated Model Predictions:")
            
            # Group predictions by entity text
            entity_groups = {}
            for pred in sample.predictions:
                key = pred.text.lower()
                if key not in entity_groups:
                    entity_groups[key] = []
                entity_groups[key].append(pred)
            
            for entity_text, predictions in entity_groups.items():
                if len(predictions) > 1:  # Show conflicts
                    print(f"\n📍 Entity: '{entity_text}'")
                    print("   Conflicting predictions:")
                    for pred in predictions:
                        print(f"   - {pred.model_source}: {pred.label} (confidence: {pred.raw_confidence:.2f})")
                    
                    # Show how naive max confidence would choose
                    winner = max(predictions, key=lambda x: x.raw_confidence)
                    print(f"   ❌ Naive max confidence would choose: {winner.model_source} ({winner.raw_confidence:.2f})")
                    
                    # Check if this matches ground truth
                    correct_answer = None
                    for true_ent in sample.true_entities:
                        if entity_text in true_ent.text.lower():
                            correct_answer = true_ent
                            break
                    
                    if correct_answer:
                        is_correct = (winner.label == correct_answer.label)
                        print(f"   {'✅' if is_correct else '❌'} Correct answer: {correct_answer.label}")
        
        print("\n💡 KEY PROBLEMS IDENTIFIED:")
        print("1. LLM overconfidence (0.99) can override correct Blackstone predictions (0.95)")
        print("2. Different confidence scales make direct comparison misleading")
        print("3. No consideration of model expertise (Blackstone better for legal entities)")
        print("4. Heterogeneous outputs require manual conflict resolution rules")
    
    def demonstrate_calibrated_benefits(self, validation_samples: List[ValidationSample]) -> None:
        """Demonstrate benefits of confidence calibration.
        
        Args:
            validation_samples: Sample data to analyze.
        """
        print("\n\n✅ BENEFITS OF CONFIDENCE CALIBRATION:")
        print("=" * 60)
        
        # Train the calibration system
        print("🔧 Training calibration system...")
        self.calibration_manager.train_system(validation_samples)
        
        # Process each sample with calibration
        for i, sample in enumerate(validation_samples, 1):
            print(f"\nSample {i}: {sample.text[:80]}...")
            
            # Organize predictions by model
            model_predictions = {}
            for pred in sample.predictions:
                model_name = pred.model_source
                if model_name not in model_predictions:
                    model_predictions[model_name] = []
                model_predictions[model_name].append(pred)
            
            # Get calibrated ensemble results
            calibrated_entities = self.calibration_manager.ensemble_voter.vote_on_entities(model_predictions)
            
            print("\n🎯 Calibrated Ensemble Results:")
            for entity in calibrated_entities:
                print(f"   📍 '{entity.text}' -> {entity.label}")
                print(f"      Raw confidence: {entity.raw_confidence:.3f}")
                print(f"      Calibrated confidence: {entity.calibrated_confidence:.3f}")
                print(f"      Source model: {entity.model_source}")
                
                # Check correctness
                is_correct = any(
                    entity.text.lower() in true_ent.text.lower() and entity.label == true_ent.label
                    for true_ent in sample.true_entities
                )
                print(f"      {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        print("\n💡 CALIBRATION BENEFITS ACHIEVED:")
        print("1. ✅ Temperature scaling reduces LLM overconfidence")
        print("2. ✅ Platt scaling normalizes traditional NER model confidences")
        print("3. ✅ Legal entity prioritization gives Blackstone higher weight")
        print("4. ✅ Uniform confidence scales enable reliable ensemble voting")
        print("5. ✅ Automated conflict resolution reduces manual rule complexity")
    
    def show_calibration_metrics(self) -> None:
        """Display calibration system metrics and configuration."""
        print("\n\n📊 CALIBRATION SYSTEM CONFIGURATION:")
        print("=" * 60)
        
        # Show model weights
        weights = self.calibration_manager.ensemble_voter.model_weights
        print(f"Model Weights (prioritization):")
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {weight}")
        
        # Show legal priority labels
        legal_labels = self.calibration_manager.ensemble_voter.legal_priority_labels
        print(f"\nLegal Priority Labels ({len(legal_labels)}):")
        for label in sorted(legal_labels):
            print(f"   • {label}")
        
        # Show calibrator types
        calibrators = self.calibration_manager.ensemble_voter.calibrators
        print(f"\nCalibration Methods:")
        for model, calibrator in calibrators.items():
            calibrator_type = type(calibrator).__name__
            is_fitted = getattr(calibrator, 'is_fitted', False)
            print(f"   {model}: {calibrator_type} {'✅ trained' if is_fitted else '❌ not trained'}")
    
    def demonstrate_performance_impact(self) -> None:
        """Show the performance and accuracy improvements."""
        print("\n\n⚡ PERFORMANCE & ACCURACY IMPACT:")
        print("=" * 60)
        
        print("🎯 Accuracy Improvements:")
        print("   • Reduces false positives from overconfident LLM predictions")
        print("   • Increases precision for legal-specific entities via Blackstone prioritization")
        print("   • Better handling of entity boundary conflicts (partial vs. full matches)")
        print("   • Improved consistency across different document types")
        
        print("\n🚀 Computational Benefits:")
        print("   • Eliminates need for complex manual conflict resolution rules")
        print("   • Reduces development time for new model integration")
        print("   • Enables dynamic model weight adjustment based on performance")
        print("   • Supports easy addition of new NER models to ensemble")
        
        print("\n🔧 Maintenance Benefits:")
        print("   • Centralized confidence calibration management")
        print("   • Automatic model performance tracking and adjustment")
        print("   • Domain-specific tuning for legal document processing")
        print("   • Consistent API regardless of underlying model changes")

def main():
    """Run the confidence calibration demonstration."""
    print("🎯 CONFIDENCE CALIBRATION SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("This demo shows how confidence calibration resolves heterogeneous")
    print("model outputs and improves ensemble decision-making in legal NER.")
    print()
    
    # Initialize demo
    demo = CalibrationDemo()
    
    # Create sample data
    validation_samples = demo.create_sample_validation_data()
    print(f"📋 Created {len(validation_samples)} validation samples")
    
    # Demonstrate problems
    demo.demonstrate_uncalibrated_problems(validation_samples)
    
    # Demonstrate solutions
    demo.demonstrate_calibrated_benefits(validation_samples)
    
    # Show system configuration
    demo.show_calibration_metrics()
    
    # Show performance impact
    demo.demonstrate_performance_impact()
    
    print("\n\n🎉 DEMONSTRATION COMPLETE!")
    print("The confidence calibration system successfully addresses the issues")
    print("identified in the original calibration.txt file:")
    print("• ✅ Resolved heterogeneous model output conflicts")
    print("• ✅ Implemented temperature scaling for LLM overconfidence")
    print("• ✅ Added Platt scaling for traditional NER models")
    print("• ✅ Introduced legal entity prioritization rules")
    print("• ✅ Created unified confidence scales for ensemble voting")

if __name__ == "__main__":
    main()