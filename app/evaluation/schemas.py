"""Schemas for the evaluation framework."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class EvalQuestion:
    """A single evaluation question from PubMedQA."""
    id: str
    question: str
    context: str  # Abstract without conclusion
    ground_truth: str  # yes/no/maybe
    long_answer: str  # Full conclusion text
    meshes: list[str] = field(default_factory=list)


@dataclass
class BaselineResponse:
    """Response from any baseline system."""
    answer: str
    predicted_label: str  # yes/no/maybe
    latency_ms: float
    tokens_used: int
    cost_usd: float
    sources: list[str] = field(default_factory=list)
    raw_response: Optional[dict] = None


@dataclass
class ClassMetrics:
    """Metrics for a single class (yes/no/maybe)."""
    precision: float
    recall: float
    f1: float
    support: int  # Number of samples


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for a system."""
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    
    # Per-class metrics
    class_metrics: dict[str, ClassMetrics]
    
    # Cost metrics
    total_cost_usd: float
    avg_cost_usd: float
    
    # Latency metrics
    avg_latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Token metrics
    total_tokens: int
    avg_tokens: float
    
    # Sample counts
    total_samples: int
    correct_samples: int


@dataclass
class EvalResult:
    """Complete evaluation result for a single question."""
    question_id: str
    question: str
    ground_truth: str
    predicted_label: str
    is_correct: bool
    answer: str
    latency_ms: float
    tokens_used: int
    cost_usd: float
    sources: list[str] = field(default_factory=list)


@dataclass
class SystemEvaluation:
    """Complete evaluation results for a system."""
    system_name: str
    metrics: EvalMetrics
    results: list[EvalResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Confusion matrix data
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report comparing multiple systems."""
    evaluations: dict[str, SystemEvaluation]
    dataset_name: str
    dataset_size: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
