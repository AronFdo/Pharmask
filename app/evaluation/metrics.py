"""Metrics calculator for evaluation framework."""

import logging
from collections import defaultdict
from typing import Optional
import statistics

from app.evaluation.schemas import (
    EvalResult,
    EvalMetrics,
    ClassMetrics,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate evaluation metrics from prediction results.
    
    Computes:
    - Accuracy (exact match)
    - Precision, Recall, F1 per class
    - Macro-averaged F1
    - Cost statistics
    - Latency statistics (P50, P95, P99)
    - Token statistics
    """
    
    LABELS = ["yes", "no", "maybe"]
    
    def calculate(self, results: list[EvalResult]) -> EvalMetrics:
        """
        Calculate all metrics from evaluation results.
        
        Args:
            results: List of EvalResult objects
            
        Returns:
            EvalMetrics with all computed statistics
        """
        if not results:
            raise ValueError("Cannot calculate metrics on empty results")
        
        # Extract predictions and ground truth
        predictions = [r.predicted_label for r in results]
        ground_truth = [r.ground_truth for r in results]
        
        # Calculate accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results)
        
        # Calculate per-class metrics
        class_metrics = self._calculate_class_metrics(predictions, ground_truth)
        
        # Calculate macro-averaged metrics
        precisions = [cm.precision for cm in class_metrics.values() if cm.support > 0]
        recalls = [cm.recall for cm in class_metrics.values() if cm.support > 0]
        f1s = [cm.f1 for cm in class_metrics.values() if cm.support > 0]
        
        macro_precision = statistics.mean(precisions) if precisions else 0.0
        macro_recall = statistics.mean(recalls) if recalls else 0.0
        macro_f1 = statistics.mean(f1s) if f1s else 0.0
        
        # Calculate cost metrics
        costs = [r.cost_usd for r in results]
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs) if costs else 0.0
        
        # Calculate latency metrics
        latencies = [r.latency_ms for r in results]
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        latency_p50 = self._percentile(latencies, 50)
        latency_p95 = self._percentile(latencies, 95)
        latency_p99 = self._percentile(latencies, 99)
        
        # Calculate token metrics
        tokens = [r.tokens_used for r in results]
        total_tokens = sum(tokens)
        avg_tokens = statistics.mean(tokens) if tokens else 0.0
        
        return EvalMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            class_metrics=class_metrics,
            total_cost_usd=total_cost,
            avg_cost_usd=avg_cost,
            avg_latency_ms=avg_latency,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            total_tokens=total_tokens,
            avg_tokens=avg_tokens,
            total_samples=len(results),
            correct_samples=correct,
        )
    
    def _calculate_class_metrics(
        self,
        predictions: list[str],
        ground_truth: list[str],
    ) -> dict[str, ClassMetrics]:
        """Calculate precision, recall, F1 for each class."""
        
        # Count true positives, false positives, false negatives per class
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        support = defaultdict(int)
        
        for pred, gt in zip(predictions, ground_truth):
            support[gt] += 1
            
            if pred == gt:
                tp[gt] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1
        
        # Calculate metrics per class
        class_metrics = {}
        for label in self.LABELS:
            precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[label] = ClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=support[label],
            )
        
        return class_metrics
    
    def _percentile(self, data: list[float], p: int) -> float:
        """Calculate percentile of a list."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n == 1:
            return sorted_data[0]
        
        # Calculate the percentile index
        k = (n - 1) * p / 100
        f = int(k)
        c = k - f
        
        if f + 1 < n:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        else:
            return sorted_data[f]
    
    def build_confusion_matrix(
        self,
        results: list[EvalResult],
    ) -> dict[str, dict[str, int]]:
        """
        Build confusion matrix from results.
        
        Returns:
            Dict mapping ground_truth -> predicted -> count
        """
        matrix = {gt: {pred: 0 for pred in self.LABELS} for gt in self.LABELS}
        
        for r in results:
            gt = r.ground_truth
            pred = r.predicted_label
            
            if gt in matrix and pred in matrix[gt]:
                matrix[gt][pred] += 1
        
        return matrix
    
    def calculate_cer(
        self,
        results: list[EvalResult],
        accuracy: Optional[float] = None,
    ) -> float:
        """
        Calculate Cost-Efficiency Ratio (CER).
        
        CER = Total Cost / Accuracy Score
        Lower is better (more cost-efficient).
        
        Args:
            results: Evaluation results
            accuracy: Pre-computed accuracy (optional)
            
        Returns:
            CER value
        """
        if accuracy is None:
            correct = sum(1 for r in results if r.is_correct)
            accuracy = correct / len(results) if results else 0.0
        
        total_cost = sum(r.cost_usd for r in results)
        
        # Avoid division by zero
        if accuracy == 0:
            return float('inf')
        
        return total_cost / accuracy
