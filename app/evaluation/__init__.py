"""Evaluation framework for the Pharmaceutical RAG Agent."""

from .dataset import PubMedQADataset
from .schemas import (
    EvalQuestion,
    BaselineResponse,
    EvalMetrics,
    EvalResult,
    ClassMetrics,
    SystemEvaluation,
    EvaluationReport,
)
from .metrics import MetricsCalculator
from .runner import EvaluationRunner
from .report import ReportGenerator
from .baselines import (
    BaseSystem,
    BM25Baseline,
    DirectLLMBaseline,
    HybridRAGAdapter,
)

__all__ = [
    # Dataset
    "PubMedQADataset",
    # Schemas
    "EvalQuestion",
    "BaselineResponse",
    "EvalMetrics",
    "EvalResult",
    "ClassMetrics",
    "SystemEvaluation",
    "EvaluationReport",
    # Core
    "MetricsCalculator",
    "EvaluationRunner",
    "ReportGenerator",
    # Baselines
    "BaseSystem",
    "BM25Baseline",
    "DirectLLMBaseline",
    "HybridRAGAdapter",
]
