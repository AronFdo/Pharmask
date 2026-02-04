"""Evaluation runner to orchestrate tests across all systems."""

import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from app.evaluation.schemas import (
    EvalQuestion,
    EvalResult,
    SystemEvaluation,
    EvaluationReport,
)
from app.evaluation.baselines.base import BaseSystem
from app.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

# Default output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"


class EvaluationRunner:
    """
    Orchestrates evaluation across multiple systems.
    
    Runs each system against the same dataset, collects results,
    and computes comparative metrics.
    """
    
    def __init__(
        self,
        systems: dict[str, BaseSystem],
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            systems: Dict mapping system name to BaseSystem instance
            output_dir: Directory to save results
        """
        self.systems = systems
        self.output_dir = Path(output_dir)
        self.metrics_calculator = MetricsCalculator()
    
    async def run_evaluation(
        self,
        questions: list[EvalQuestion],
        save_results: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> EvaluationReport:
        """
        Run evaluation on all systems.
        
        Args:
            questions: List of questions to evaluate
            save_results: Whether to save results to files
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationReport with all results
        """
        logger.info(f"Starting evaluation with {len(questions)} questions across {len(self.systems)} systems")
        
        evaluations = {}
        
        for system_name, system in self.systems.items():
            logger.info(f"Evaluating system: {system_name}")
            
            # Initialize the system
            await system.initialize()
            
            # Run all questions
            results = await self._evaluate_system(
                system=system,
                questions=questions,
                progress_callback=progress_callback,
            )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate(results)
            confusion_matrix = self.metrics_calculator.build_confusion_matrix(results)
            
            # Create system evaluation
            evaluations[system_name] = SystemEvaluation(
                system_name=system_name,
                metrics=metrics,
                results=results,
                confusion_matrix=confusion_matrix,
            )
            
            # Cleanup
            await system.cleanup()
            
            logger.info(
                f"  {system_name}: Accuracy={metrics.accuracy:.2%}, "
                f"F1={metrics.macro_f1:.3f}, Cost=${metrics.total_cost_usd:.4f}"
            )
        
        # Create report
        report = EvaluationReport(
            evaluations=evaluations,
            dataset_name="PubMedQA",
            dataset_size=len(questions),
        )
        
        # Save results
        if save_results:
            self._save_results(report)
        
        return report
    
    async def _evaluate_system(
        self,
        system: BaseSystem,
        questions: list[EvalQuestion],
        progress_callback: Optional[callable] = None,
    ) -> list[EvalResult]:
        """
        Evaluate a single system on all questions.
        
        Args:
            system: The system to evaluate
            questions: List of questions
            progress_callback: Optional progress callback
            
        Returns:
            List of EvalResult objects
        """
        results = []
        total = len(questions)
        
        for idx, question in enumerate(questions):
            try:
                # Get answer from system
                response = await system.answer(question)
                
                # Check if correct
                is_correct = response.predicted_label == question.ground_truth
                
                # Create result
                result = EvalResult(
                    question_id=question.id,
                    question=question.question,
                    ground_truth=question.ground_truth,
                    predicted_label=response.predicted_label,
                    is_correct=is_correct,
                    answer=response.answer,
                    latency_ms=response.latency_ms,
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    sources=response.sources,
                )
                
            except Exception as e:
                logger.error(f"Error evaluating question {question.id}: {e}")
                
                # Create error result
                result = EvalResult(
                    question_id=question.id,
                    question=question.question,
                    ground_truth=question.ground_truth,
                    predicted_label="error",
                    is_correct=False,
                    answer=f"Error: {str(e)}",
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    sources=[],
                )
            
            results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(system.name, idx + 1, total)
            
            # Log progress every 10 questions
            if (idx + 1) % 10 == 0:
                logger.info(f"  Progress: {idx + 1}/{total} questions")
        
        return results
    
    def _save_results(self, report: EvaluationReport) -> None:
        """Save evaluation results to files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full JSON report
        json_path = self.output_dir / f"eval_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._report_to_dict(report), f, indent=2, default=str)
        
        logger.info(f"Saved JSON results to {json_path}")
    
    def _report_to_dict(self, report: EvaluationReport) -> dict:
        """Convert report to serializable dict."""
        return {
            "dataset_name": report.dataset_name,
            "dataset_size": report.dataset_size,
            "timestamp": report.timestamp,
            "evaluations": {
                name: self._system_eval_to_dict(eval)
                for name, eval in report.evaluations.items()
            }
        }
    
    def _system_eval_to_dict(self, eval: SystemEvaluation) -> dict:
        """Convert system evaluation to dict."""
        return {
            "system_name": eval.system_name,
            "timestamp": eval.timestamp,
            "metrics": {
                "accuracy": eval.metrics.accuracy,
                "macro_f1": eval.metrics.macro_f1,
                "macro_precision": eval.metrics.macro_precision,
                "macro_recall": eval.metrics.macro_recall,
                "total_cost_usd": eval.metrics.total_cost_usd,
                "avg_cost_usd": eval.metrics.avg_cost_usd,
                "avg_latency_ms": eval.metrics.avg_latency_ms,
                "latency_p50_ms": eval.metrics.latency_p50_ms,
                "latency_p95_ms": eval.metrics.latency_p95_ms,
                "latency_p99_ms": eval.metrics.latency_p99_ms,
                "total_tokens": eval.metrics.total_tokens,
                "avg_tokens": eval.metrics.avg_tokens,
                "total_samples": eval.metrics.total_samples,
                "correct_samples": eval.metrics.correct_samples,
                "class_metrics": {
                    label: {
                        "precision": cm.precision,
                        "recall": cm.recall,
                        "f1": cm.f1,
                        "support": cm.support,
                    }
                    for label, cm in eval.metrics.class_metrics.items()
                }
            },
            "confusion_matrix": eval.confusion_matrix,
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                    "ground_truth": r.ground_truth,
                    "predicted_label": r.predicted_label,
                    "is_correct": r.is_correct,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used,
                    "cost_usd": r.cost_usd,
                }
                for r in eval.results
            ]
        }


async def run_quick_evaluation(
    systems: dict[str, BaseSystem],
    questions: list[EvalQuestion],
    max_questions: int = 10,
) -> EvaluationReport:
    """
    Run a quick evaluation for testing purposes.
    
    Args:
        systems: Systems to evaluate
        questions: Full question list
        max_questions: Maximum questions to use
        
    Returns:
        EvaluationReport
    """
    # Limit questions
    if len(questions) > max_questions:
        questions = questions[:max_questions]
    
    runner = EvaluationRunner(systems)
    return await runner.run_evaluation(questions, save_results=False)
