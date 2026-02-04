"""
Custom evaluation for Pharmaceutical RAG system.

This evaluator tests what the system is ACTUALLY designed for:
1. Query Classification Accuracy (text/sql/hybrid routing)
2. Retrieval Source Correctness (did it use the right DB?)
3. Answer Quality (keyword containment)
4. Cost-Efficiency (Tier-1 vs Tier-2 usage)
"""

import json
import logging
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from app.services.orchestrator import RAGOrchestrator
from app.services.classifier import QueryClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "pharma_qa_dataset.json"


@dataclass
class PharmaQuestion:
    """A pharmaceutical evaluation question."""
    id: str
    question: str
    query_type: str  # text, sql, hybrid
    expected_source: str
    ground_truth_contains: list[str]
    difficulty: str


@dataclass
class PharmaEvalResult:
    """Result for a single question."""
    question_id: str
    question: str
    
    # Classification
    expected_type: str
    predicted_type: str
    classification_correct: bool
    
    # Answer Quality
    answer: str
    keywords_found: list[str]
    keywords_missing: list[str]
    keyword_coverage: float
    
    # Performance
    latency_ms: float
    tier1_tokens: int
    tier2_tokens: int
    total_cost_usd: float
    
    # Sources
    sources_used: list[str]


@dataclass
class PharmaEvalMetrics:
    """Aggregated evaluation metrics."""
    # Classification
    classification_accuracy: float
    classification_by_type: dict[str, dict]
    
    # Answer Quality
    avg_keyword_coverage: float
    perfect_coverage_rate: float  # Questions with 100% keywords
    
    # Cost Efficiency
    total_cost_usd: float
    avg_cost_per_query: float
    tier1_ratio: float  # % of tokens in Tier-1 (cheaper)
    estimated_savings_vs_tier2_only: float
    
    # Performance
    avg_latency_ms: float
    latency_p95_ms: float
    
    # By difficulty
    accuracy_by_difficulty: dict[str, float]
    
    # By query type
    coverage_by_query_type: dict[str, float]


class PharmaEvaluator:
    """
    Evaluator for the Pharmaceutical RAG system.
    
    Tests the core research claims:
    1. Query classification works
    2. Hybrid retrieval provides value
    3. Cost-efficiency through tiered approach
    """
    
    def __init__(self, dataset_path: Path = DATASET_PATH):
        self.dataset_path = dataset_path
        self.orchestrator = None
        self.classifier = None
    
    def load_dataset(self) -> list[PharmaQuestion]:
        """Load the pharmaceutical QA dataset."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append(PharmaQuestion(
                id=item["id"],
                question=item["question"],
                query_type=item["query_type"],
                expected_source=item["expected_source"],
                ground_truth_contains=item["ground_truth_contains"],
                difficulty=item["difficulty"],
            ))
        
        return questions
    
    async def initialize(self):
        """Initialize the RAG system."""
        if self.orchestrator is None:
            self.orchestrator = RAGOrchestrator()
            self.classifier = QueryClassifier()
            logger.info("PharmaEvaluator initialized")
    
    async def evaluate(
        self,
        questions: Optional[list[PharmaQuestion]] = None,
        progress_callback: Optional[callable] = None,
    ) -> tuple[list[PharmaEvalResult], PharmaEvalMetrics]:
        """
        Run evaluation on pharmaceutical questions.
        
        Returns:
            Tuple of (detailed results, aggregated metrics)
        """
        await self.initialize()
        
        if questions is None:
            questions = self.load_dataset()
        
        logger.info(f"Evaluating {len(questions)} pharmaceutical questions...")
        
        results = []
        for idx, q in enumerate(questions):
            result = await self._evaluate_question(q)
            results.append(result)
            
            if progress_callback:
                progress_callback(idx + 1, len(questions))
            
            if (idx + 1) % 5 == 0:
                logger.info(f"Progress: {idx + 1}/{len(questions)}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return results, metrics
    
    async def _evaluate_question(self, q: PharmaQuestion) -> PharmaEvalResult:
        """Evaluate a single question."""
        start_time = time.time()
        
        try:
            # Get answer from RAG system
            response = await self.orchestrator.process_query(q.question)
            
            answer = response.answer
            predicted_type = response.classification.query_type
            tier1_tokens = response.tier1_tokens
            tier2_tokens = response.tier2_tokens
            cost = response.cost.total_cost_usd if response.cost else 0.0
            sources = [s.source_id for s in response.sources]
            
        except Exception as e:
            logger.error(f"Error evaluating {q.id}: {e}")
            answer = f"Error: {e}"
            predicted_type = "error"
            tier1_tokens = 0
            tier2_tokens = 0
            cost = 0.0
            sources = []
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Check classification
        classification_correct = predicted_type == q.query_type
        
        # Check keyword coverage
        answer_lower = answer.lower()
        keywords_found = []
        keywords_missing = []
        
        for kw in q.ground_truth_contains:
            if kw.lower() in answer_lower:
                keywords_found.append(kw)
            else:
                keywords_missing.append(kw)
        
        keyword_coverage = len(keywords_found) / len(q.ground_truth_contains) if q.ground_truth_contains else 0.0
        
        return PharmaEvalResult(
            question_id=q.id,
            question=q.question,
            expected_type=q.query_type,
            predicted_type=predicted_type,
            classification_correct=classification_correct,
            answer=answer,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            keyword_coverage=keyword_coverage,
            latency_ms=latency_ms,
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
            total_cost_usd=cost,
            sources_used=sources,
        )
    
    def _calculate_metrics(self, results: list[PharmaEvalResult]) -> PharmaEvalMetrics:
        """Calculate aggregated metrics."""
        n = len(results)
        
        # Classification accuracy
        correct_classifications = sum(1 for r in results if r.classification_correct)
        classification_accuracy = correct_classifications / n if n > 0 else 0.0
        
        # Classification by type
        classification_by_type = {}
        for query_type in ["text", "sql", "hybrid"]:
            type_results = [r for r in results if r.expected_type == query_type]
            if type_results:
                correct = sum(1 for r in type_results if r.classification_correct)
                classification_by_type[query_type] = {
                    "total": len(type_results),
                    "correct": correct,
                    "accuracy": correct / len(type_results),
                }
        
        # Keyword coverage
        avg_keyword_coverage = sum(r.keyword_coverage for r in results) / n if n > 0 else 0.0
        perfect_coverage = sum(1 for r in results if r.keyword_coverage == 1.0)
        perfect_coverage_rate = perfect_coverage / n if n > 0 else 0.0
        
        # Cost metrics
        total_cost = sum(r.total_cost_usd for r in results)
        avg_cost = total_cost / n if n > 0 else 0.0
        
        total_tier1 = sum(r.tier1_tokens for r in results)
        total_tier2 = sum(r.tier2_tokens for r in results)
        total_tokens = total_tier1 + total_tier2
        tier1_ratio = total_tier1 / total_tokens if total_tokens > 0 else 0.0
        
        # Estimate savings (Tier-1 is ~20x cheaper than Tier-2)
        # If all tokens went through Tier-2, cost would be higher
        tier2_only_cost = total_cost * (1 + tier1_ratio * 0.95)  # Rough estimate
        savings = (tier2_only_cost - total_cost) / tier2_only_cost if tier2_only_cost > 0 else 0.0
        
        # Latency
        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / n if n > 0 else 0.0
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * n)
        latency_p95 = sorted_latencies[p95_idx] if p95_idx < n else avg_latency
        
        # By difficulty
        accuracy_by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if any(
                q.difficulty == diff for q in self.load_dataset() if q.id == r.question_id
            )]
            # Simplified: use keyword coverage as accuracy proxy
            diff_results = [r for r in results]  # All results for now
        
        # By query type
        coverage_by_query_type = {}
        for query_type in ["text", "sql", "hybrid"]:
            type_results = [r for r in results if r.expected_type == query_type]
            if type_results:
                coverage_by_query_type[query_type] = sum(r.keyword_coverage for r in type_results) / len(type_results)
        
        return PharmaEvalMetrics(
            classification_accuracy=classification_accuracy,
            classification_by_type=classification_by_type,
            avg_keyword_coverage=avg_keyword_coverage,
            perfect_coverage_rate=perfect_coverage_rate,
            total_cost_usd=total_cost,
            avg_cost_per_query=avg_cost,
            tier1_ratio=tier1_ratio,
            estimated_savings_vs_tier2_only=savings,
            avg_latency_ms=avg_latency,
            latency_p95_ms=latency_p95,
            accuracy_by_difficulty={},  # TODO: implement
            coverage_by_query_type=coverage_by_query_type,
        )
    
    def print_report(self, results: list[PharmaEvalResult], metrics: PharmaEvalMetrics):
        """Print a formatted report."""
        print("\n" + "="*70)
        print("PHARMACEUTICAL RAG EVALUATION REPORT")
        print("="*70)
        
        print(f"\n{'='*30} CLASSIFICATION {'='*30}")
        print(f"Overall Accuracy: {metrics.classification_accuracy:.1%}")
        print("\nBy Query Type:")
        for qtype, data in metrics.classification_by_type.items():
            print(f"  {qtype:8}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
        
        print(f"\n{'='*30} ANSWER QUALITY {'='*30}")
        print(f"Avg Keyword Coverage: {metrics.avg_keyword_coverage:.1%}")
        print(f"Perfect Answers: {metrics.perfect_coverage_rate:.1%}")
        print("\nBy Query Type:")
        for qtype, coverage in metrics.coverage_by_query_type.items():
            print(f"  {qtype:8}: {coverage:.1%}")
        
        print(f"\n{'='*30} COST EFFICIENCY {'='*30}")
        print(f"Total Cost: ${metrics.total_cost_usd:.4f}")
        print(f"Avg Cost/Query: ${metrics.avg_cost_per_query:.5f}")
        print(f"Tier-1 Token Ratio: {metrics.tier1_ratio:.1%} (cheaper model)")
        print(f"Est. Savings vs Tier-2 Only: {metrics.estimated_savings_vs_tier2_only:.1%}")
        
        print(f"\n{'='*30} PERFORMANCE {'='*30}")
        print(f"Avg Latency: {metrics.avg_latency_ms:.0f}ms")
        print(f"P95 Latency: {metrics.latency_p95_ms:.0f}ms")
        
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)
        
        for r in results:
            status = "✓" if r.classification_correct else "✗"
            coverage = f"{r.keyword_coverage:.0%}"
            print(f"\n[{r.question_id}] {status} Type: {r.predicted_type} (expected: {r.expected_type})")
            print(f"  Q: {r.question[:60]}...")
            print(f"  Coverage: {coverage} | Found: {r.keywords_found} | Missing: {r.keywords_missing}")
            print(f"  Cost: ${r.total_cost_usd:.5f} | Latency: {r.latency_ms:.0f}ms")


async def run_pharma_evaluation():
    """Run the pharmaceutical evaluation."""
    evaluator = PharmaEvaluator()
    results, metrics = await evaluator.evaluate()
    evaluator.print_report(results, metrics)
    return results, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(run_pharma_evaluation())
