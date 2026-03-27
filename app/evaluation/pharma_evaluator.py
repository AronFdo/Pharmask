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
from app.services.retrieval import VectorRetriever
from app.services.synthesis import AnswerSynthesizer
from app.services.cost_calculator import calculate_cost
from app.models import RetrievalResult, SourceReference

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
    variant: str  # "hybrid" | "vector_only"
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
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    
    # Performance
    latency_ms: float
    tier1_tokens: int
    tier2_tokens: int
    total_cost_usd: float
    
    # Sources
    sources_used: list[str]
    used_text_sources: bool
    used_table_sources: bool
    db_type_correct: bool


@dataclass
class PharmaEvalMetrics:
    """Aggregated evaluation metrics."""
    variant: str
    # Classification
    classification_accuracy: float
    classification_by_type: dict[str, dict]
    
    # Answer Quality
    avg_keyword_coverage: float
    perfect_coverage_rate: float  # Questions with 100% keywords
    avg_evidence_f1: float
    perfect_f1_rate: float  # Questions with F1 == 1
    multimodal_avg_evidence_f1: float  # expected_type == "hybrid"
    
    # Cost Efficiency
    total_cost_usd: float
    avg_cost_per_query: float
    cer_cost_div_f1: float  # avg_cost_per_query / avg_evidence_f1 (lower is better)
    tier1_ratio: float  # % of tokens in Tier-1 (cheaper)
    estimated_savings_vs_tier2_only: float
    
    # Performance
    avg_latency_ms: float
    latency_p95_ms: float
    
    # By difficulty
    accuracy_by_difficulty: dict[str, float]
    
    # By query type
    coverage_by_query_type: dict[str, float]
    f1_by_query_type: dict[str, float]

    # Retrieval correctness proxy (did it use expected DB type(s)?)
    db_type_correct_rate: float


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
        self.vector_retriever: Optional[VectorRetriever] = None
        self.synthesizer: Optional[AnswerSynthesizer] = None
    
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
        """Initialize the shared RAG components."""
        if self.orchestrator is None:
            self.orchestrator = RAGOrchestrator()
        if self.classifier is None:
            self.classifier = QueryClassifier()
        if self.vector_retriever is None:
            self.vector_retriever = VectorRetriever()
        if self.synthesizer is None:
            self.synthesizer = AnswerSynthesizer()
        logger.info("PharmaEvaluator initialized (hybrid + vector-only components)")
    
    async def evaluate(
        self,
        questions: Optional[list[PharmaQuestion]] = None,
        progress_callback: Optional[callable] = None,
    ) -> tuple[dict[str, list[PharmaEvalResult]], dict[str, PharmaEvalMetrics]]:
        """
        Run evaluation on pharmaceutical questions.
        
        Returns:
            Tuple of (detailed results, aggregated metrics)
        """
        await self.initialize()
        
        if questions is None:
            questions = self.load_dataset()
        
        logger.info(f"Evaluating {len(questions)} pharmaceutical questions...")
        
        results_by_variant: dict[str, list[PharmaEvalResult]] = {
            "hybrid": [],
            "vector_only": [],
        }
        for idx, q in enumerate(questions):
            hybrid_result = await self._evaluate_question(q, variant="hybrid")
            vector_result = await self._evaluate_vector_only_question(q)
            results_by_variant["hybrid"].append(hybrid_result)
            results_by_variant["vector_only"].append(vector_result)
            
            if progress_callback:
                progress_callback(idx + 1, len(questions))
            
            if (idx + 1) % 5 == 0:
                logger.info(f"Progress: {idx + 1}/{len(questions)}")
        
        # Calculate metrics per variant
        metrics_by_variant: dict[str, PharmaEvalMetrics] = {}
        for variant, variant_results in results_by_variant.items():
            metrics_by_variant[variant] = self._calculate_metrics(variant_results, variant=variant)
        
        return results_by_variant, metrics_by_variant
    
    async def _evaluate_question(self, q: PharmaQuestion, *, variant: str) -> PharmaEvalResult:
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
            used_text_sources = any(s.source_type == "text" for s in response.sources)
            used_table_sources = any(s.source_type == "table" for s in response.sources)
            db_type_correct = self._db_type_matches(
                expected_type=q.query_type,
                used_text=used_text_sources,
                used_table=used_table_sources,
            )
            sources = [s.source_id for s in response.sources]
            
        except Exception as e:
            logger.error(f"Error evaluating {q.id}: {e}")
            answer = f"Error: {e}"
            predicted_type = "error"
            tier1_tokens = 0
            tier2_tokens = 0
            cost = 0.0
            sources = []
            used_text_sources = False
            used_table_sources = False
            db_type_correct = False
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Check classification
        classification_correct = predicted_type == q.query_type
        
        # Compute evidence-match scores (keyword overlap => evidence F1)
        (
            keywords_found,
            keywords_missing,
            keyword_coverage,
            evidence_precision,
            evidence_recall,
            evidence_f1,
        ) = self._compute_evidence_match_scores(
            answer=answer,
            ground_truth_contains=q.ground_truth_contains,
        )
        
        return PharmaEvalResult(
            variant=variant,
            question_id=q.id,
            question=q.question,
            expected_type=q.query_type,
            predicted_type=predicted_type,
            classification_correct=classification_correct,
            answer=answer,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            keyword_coverage=keyword_coverage,
            evidence_precision=evidence_precision,
            evidence_recall=evidence_recall,
            evidence_f1=evidence_f1,
            latency_ms=latency_ms,
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
            total_cost_usd=cost,
            sources_used=sources,
            used_text_sources=used_text_sources,
            used_table_sources=used_table_sources,
            db_type_correct=db_type_correct,
        )

    async def _evaluate_vector_only_question(self, q: PharmaQuestion) -> PharmaEvalResult:
        """Vector-only evaluation variant (no SQL/table retrieval)."""
        if self.vector_retriever is None or self.synthesizer is None or self.classifier is None:
            raise RuntimeError("PharmaEvaluator not initialized")

        start_time = time.time()

        try:
            # Tier-1: classifier to keep routing metrics comparable
            classification, tier1_tokens = await self.classifier.classify(q.question)
            predicted_type = classification.query_type
            classification_correct = predicted_type == q.query_type

            # Vector retrieval only
            vector_result = await self.vector_retriever.retrieve(q.question)
            text_chunks = vector_result.get("chunks", [])
            retrieval_sources = vector_result.get("sources", [])

            retrieval_result = RetrievalResult(
                text_chunks=text_chunks,
                sql_rows=[],
                sources=retrieval_sources,
            )

            # Tier-2: synthesize from text chunks only
            answer, synth_sources, tier2_tokens = await self.synthesizer.synthesize(
                q.question,
                retrieval_result,
            )

            merged_sources = self._merge_sources(retrieval_sources, synth_sources)
            sources = [s.source_id for s in merged_sources]

            used_text_sources = any(s.source_type == "text" for s in merged_sources)
            used_table_sources = any(s.source_type == "table" for s in merged_sources)
            db_type_correct = self._db_type_matches(
                expected_type=q.query_type,
                used_text=used_text_sources,
                used_table=used_table_sources,
            )

            cost = calculate_cost(tier1_tokens=tier1_tokens, tier2_tokens=tier2_tokens).total_cost_usd

        except Exception as e:
            logger.error(f"Vector-only evaluation error {q.id}: {e}")
            answer = f"Error: {e}"
            predicted_type = "error"
            tier1_tokens = 0
            tier2_tokens = 0
            classification_correct = False
            cost = 0.0
            sources = []
            used_text_sources = False
            used_table_sources = False
            db_type_correct = False

        latency_ms = (time.time() - start_time) * 1000

        # Evidence metrics from keyword overlap with generated answer
        (
            keywords_found,
            keywords_missing,
            keyword_coverage,
            evidence_precision,
            evidence_recall,
            evidence_f1,
        ) = self._compute_evidence_match_scores(
            answer=answer,
            ground_truth_contains=q.ground_truth_contains,
        )

        return PharmaEvalResult(
            variant="vector_only",
            question_id=q.id,
            question=q.question,
            expected_type=q.query_type,
            predicted_type=predicted_type,
            classification_correct=classification_correct,
            answer=answer,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            keyword_coverage=keyword_coverage,
            evidence_precision=evidence_precision,
            evidence_recall=evidence_recall,
            evidence_f1=evidence_f1,
            latency_ms=latency_ms,
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
            total_cost_usd=cost,
            sources_used=sources,
            used_text_sources=used_text_sources,
            used_table_sources=used_table_sources,
            db_type_correct=db_type_correct,
        )

    def _compute_evidence_match_scores(
        self,
        *,
        answer: str,
        ground_truth_contains: list[str],
    ) -> tuple[list[str], list[str], float, float, float, float]:
        """
        Evidence-match scoring:
        - Positive evidence = ground-truth evidence strings that appear in the generated answer.
        - Precision is 1 when at least one ground-truth evidence item is found (0 otherwise).
        - Recall is coverage of ground-truth evidence strings.
        - Evidence F1 combines precision + recall.
        """
        answer_lower = (answer or "").lower()
        keywords_found: list[str] = []
        keywords_missing: list[str] = []

        for kw in ground_truth_contains:
            if kw.lower() in answer_lower:
                keywords_found.append(kw)
            else:
                keywords_missing.append(kw)

        total = len(ground_truth_contains)
        if total == 0:
            return keywords_found, keywords_missing, 0.0, 0.0, 0.0, 0.0

        found = len(keywords_found)
        recall = found / total
        precision = 1.0 if found > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Keep keyword_coverage as recall for backwards compatibility.
        keyword_coverage = recall
        return keywords_found, keywords_missing, keyword_coverage, precision, recall, f1

    def _db_type_matches(self, *, expected_type: str, used_text: bool, used_table: bool) -> bool:
        """Check whether retrieval used DB types expected for the question category."""
        if expected_type == "text":
            return used_text and not used_table
        if expected_type == "sql":
            return used_table and not used_text
        if expected_type == "hybrid":
            return used_text and used_table
        return False

    def _merge_sources(
        self,
        a: list[SourceReference],
        b: list[SourceReference],
    ) -> list[SourceReference]:
        """Merge and deduplicate sources by (source_type, source_id)."""
        seen: set[tuple[str, str]] = set()
        merged: list[SourceReference] = []

        for src in a + b:
            key = (src.source_type, src.source_id)
            if key not in seen:
                seen.add(key)
                merged.append(src)

        return merged
    
    def _calculate_metrics(self, results: list[PharmaEvalResult], *, variant: str) -> PharmaEvalMetrics:
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
        
        # Evidence keyword coverage (recall-style)
        avg_keyword_coverage = sum(r.keyword_coverage for r in results) / n if n > 0 else 0.0
        perfect_coverage = sum(1 for r in results if r.keyword_coverage == 1.0)
        perfect_coverage_rate = perfect_coverage / n if n > 0 else 0.0

        # Evidence F1 (thesis-aligned evidence-match F1)
        avg_evidence_f1 = sum(r.evidence_f1 for r in results) / n if n > 0 else 0.0
        perfect_f1 = sum(1 for r in results if r.evidence_f1 == 1.0)
        perfect_f1_rate = perfect_f1 / n if n > 0 else 0.0

        multimodal_results = [r for r in results if r.expected_type == "hybrid"]
        multimodal_avg_evidence_f1 = (
            sum(r.evidence_f1 for r in multimodal_results) / len(multimodal_results)
            if multimodal_results
            else 0.0
        )
        
        # Cost metrics
        total_cost = sum(r.total_cost_usd for r in results)
        avg_cost = total_cost / n if n > 0 else 0.0
        # CER: avg_cost_per_query / avg_F1
        cer_cost_div_f1 = avg_cost / avg_evidence_f1 if avg_evidence_f1 > 0 else float("inf")

        
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
        coverage_by_query_type: dict[str, float] = {}
        f1_by_query_type: dict[str, float] = {}
        for query_type in ["text", "sql", "hybrid"]:
            type_results = [r for r in results if r.expected_type == query_type]
            if type_results:
                coverage_by_query_type[query_type] = sum(r.keyword_coverage for r in type_results) / len(type_results)
                f1_by_query_type[query_type] = sum(r.evidence_f1 for r in type_results) / len(type_results)

        db_type_correct_rate = sum(1 for r in results if r.db_type_correct) / n if n > 0 else 0.0
        
        return PharmaEvalMetrics(
            variant=variant,
            classification_accuracy=classification_accuracy,
            classification_by_type=classification_by_type,
            avg_keyword_coverage=avg_keyword_coverage,
            perfect_coverage_rate=perfect_coverage_rate,
            avg_evidence_f1=avg_evidence_f1,
            perfect_f1_rate=perfect_f1_rate,
            multimodal_avg_evidence_f1=multimodal_avg_evidence_f1,
            total_cost_usd=total_cost,
            avg_cost_per_query=avg_cost,
            cer_cost_div_f1=cer_cost_div_f1,
            tier1_ratio=tier1_ratio,
            estimated_savings_vs_tier2_only=savings,
            avg_latency_ms=avg_latency,
            latency_p95_ms=latency_p95,
            accuracy_by_difficulty={},  # TODO: implement
            coverage_by_query_type=coverage_by_query_type,
            f1_by_query_type=f1_by_query_type,
            db_type_correct_rate=db_type_correct_rate,
        )
    
    def print_report(
        self,
        results_by_variant: dict[str, list[PharmaEvalResult]],
        metrics_by_variant: dict[str, PharmaEvalMetrics],
    ):
        """Print a formatted, thesis-aligned side-by-side report."""
        print("\n" + "=" * 70)
        print("PHARMACEUTICAL RAG EVALUATION REPORT (HYBRID vs VECTOR-ONLY)")
        print("=" * 70)

        for variant in ["hybrid", "vector_only"]:
            metrics = metrics_by_variant[variant]
            print(f"\n{'='*30} {variant.upper()} SUMMARY {'='*30}")

            print("\nClassification:")
            print(f"  Overall Accuracy: {metrics.classification_accuracy:.1%}")
            for qtype, data in metrics.classification_by_type.items():
                print(f"  {qtype:8}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")

            print("\nEvidence match (RQ2 + RQ3):")
            print(f"  Avg Evidence Recall (Keyword Coverage): {metrics.avg_keyword_coverage:.1%}")
            print(f"  Avg Evidence F1: {metrics.avg_evidence_f1:.3f}")
            print(f"  Perfect Evidence Recall: {metrics.perfect_coverage_rate:.1%}")
            print(f"  Perfect Evidence F1: {metrics.perfect_f1_rate:.1%}")
            print(f"  Multimodal subset (expected_type='hybrid') Avg Evidence F1: {metrics.multimodal_avg_evidence_f1:.3f}")
            print(f"  F1 by expected query type: { {k: round(v, 3) for k, v in metrics.f1_by_query_type.items()} }")

            print("\nCost-efficiency (RQ3):")
            print(f"  Avg Cost/Query: ${metrics.avg_cost_per_query:.5f}")
            print(f"  CER = avg_cost_per_query / avg_F1: {metrics.cer_cost_div_f1:.6f}")
            print(f"  Est. Savings vs Tier-2 Only: {metrics.estimated_savings_vs_tier2_only:.1%}")

            print("\nPerformance:")
            print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms")
            print(f"  P95 Latency: {metrics.latency_p95_ms:.0f}ms")

            print("\nRetrieval DB-type evidence proxy (RQ1 optional):")
            print(f"  DB-type correctness rate: {metrics.db_type_correct_rate:.1%}")

        # Quick per-question side-by-side comparison (cost + evidence F1 + routing correctness)
        hybrid_by_qid = {r.question_id: r for r in results_by_variant["hybrid"]}
        vector_by_qid = {r.question_id: r for r in results_by_variant["vector_only"]}

        print("\n" + "=" * 70)
        print("DETAILED RESULTS (per question)")
        print("=" * 70)

        for qid, hres in hybrid_by_qid.items():
            vres = vector_by_qid.get(qid)
            status_h = "✓" if hres.classification_correct else "✗"
            status_v = "✓" if (vres and vres.classification_correct) else "✗"
            print(f"\n[{qid}] Routing: hybrid={status_h}, vector-only={status_v}")
            if vres:
                print(f"  Expected type: {hres.expected_type}")
                print(f"  Predicted types: hybrid={hres.predicted_type}, vector-only={vres.predicted_type}")
                print(f"  Evidence F1: hybrid={hres.evidence_f1:.3f}, vector-only={vres.evidence_f1:.3f}")
                print(f"  Cost/Query: hybrid=${hres.total_cost_usd:.5f}, vector-only=${vres.total_cost_usd:.5f}")

async def run_pharma_evaluation():
    """Run the pharmaceutical evaluation."""
    evaluator = PharmaEvaluator()
    results_by_variant, metrics_by_variant = await evaluator.evaluate()
    evaluator.print_report(results_by_variant, metrics_by_variant)
    return results_by_variant, metrics_by_variant


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(run_pharma_evaluation())
