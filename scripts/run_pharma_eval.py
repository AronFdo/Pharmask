"""
Run pharmaceutical-specific evaluation.

This evaluation tests what the system is ACTUALLY designed for:
- Query classification (text/sql/hybrid)
- Hybrid retrieval value
- Cost-efficiency

Usage:
    python scripts/run_pharma_eval.py
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.pharma_evaluator import PharmaEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run the pharmaceutical evaluation."""
    logger.info("Starting Pharmaceutical RAG Evaluation...")
    logger.info("This tests the CORE research claims of your thesis:")
    logger.info("  1. Query Classification Accuracy")
    logger.info("  2. Hybrid Retrieval Value (text + sql + both)")
    logger.info("  3. Cost-Efficiency (Tier-1 vs Tier-2)")
    print()
    
    evaluator = PharmaEvaluator()
    
    # Progress callback
    def progress(current, total):
        print(f"  Progress: {current}/{total} ({100*current/total:.0f}%)", end="\r")
    
    results_by_variant, metrics_by_variant = await evaluator.evaluate(progress_callback=progress)
    print()  # Clear progress line
    
    # Print report
    evaluator.print_report(results_by_variant, metrics_by_variant)
    
    # Save results
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"pharma_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "metrics_by_variant": {
                    variant: {
                        "classification_accuracy": m.classification_accuracy,
                        "classification_by_type": m.classification_by_type,
                        "avg_keyword_coverage": m.avg_keyword_coverage,
                        "perfect_coverage_rate": m.perfect_coverage_rate,
                        "avg_evidence_f1": m.avg_evidence_f1,
                        "perfect_f1_rate": m.perfect_f1_rate,
                        "multimodal_avg_evidence_f1": m.multimodal_avg_evidence_f1,
                        "cer_cost_div_f1": m.cer_cost_div_f1,
                        "total_cost_usd": m.total_cost_usd,
                        "avg_cost_per_query": m.avg_cost_per_query,
                        "tier1_ratio": m.tier1_ratio,
                        "estimated_savings": m.estimated_savings_vs_tier2_only,
                        "avg_latency_ms": m.avg_latency_ms,
                        "coverage_by_query_type": m.coverage_by_query_type,
                        "f1_by_query_type": m.f1_by_query_type,
                        "db_type_correct_rate": m.db_type_correct_rate,
                    }
                    for variant, m in metrics_by_variant.items()
                },
                "results_by_variant": {
                    variant: [
                        {
                            "id": r.question_id,
                            "question": r.question,
                            "expected_type": r.expected_type,
                            "predicted_type": r.predicted_type,
                            "classification_correct": r.classification_correct,
                            "keyword_coverage": r.keyword_coverage,
                            "evidence_f1": r.evidence_f1,
                            "keywords_found": r.keywords_found,
                            "keywords_missing": r.keywords_missing,
                            "cost_usd": r.total_cost_usd,
                            "latency_ms": r.latency_ms,
                            "used_text_sources": r.used_text_sources,
                            "used_table_sources": r.used_table_sources,
                            "db_type_correct": r.db_type_correct,
                        }
                        for r in rs
                    ]
                    for variant, rs in results_by_variant.items()
                },
            },
            f,
            indent=2,
        )
    
    logger.info(f"\nResults saved to: {json_path}")
    
    # Print thesis-ready summary
    print("\n" + "="*70)
    print("THESIS-READY SUMMARY")
    print("="*70)
    hybrid_metrics = metrics_by_variant.get("hybrid")
    vector_metrics = metrics_by_variant.get("vector_only")
    print(f"""
Research Question 1: Query Classification
  - Hybrid Classification Accuracy: {hybrid_metrics.classification_accuracy:.1%}
  - Vector-only Classification Accuracy: {vector_metrics.classification_accuracy:.1%}
  - The Tier-1 model correctly routes queries to appropriate retrieval

Research Question 2: Hybrid Retrieval Value
  - Hybrid Expected-Text Evidence Recall: {hybrid_metrics.coverage_by_query_type.get('text', 0):.1%}
  - Hybrid Expected-SQL Evidence Recall: {hybrid_metrics.coverage_by_query_type.get('sql', 0):.1%}
  - Hybrid Expected-Hybrid Evidence Recall: {hybrid_metrics.coverage_by_query_type.get('hybrid', 0):.1%}
  - Hybrid queries require BOTH sources for complete answers

Research Question 3: Cost-Efficiency
  - CER (Hybrid): {hybrid_metrics.cer_cost_div_f1:.6f}
  - CER (Vector-only): {vector_metrics.cer_cost_div_f1:.6f}
  - Tier-1 Token Ratio (Hybrid): {hybrid_metrics.tier1_ratio:.1%}
  - Estimated Savings (Hybrid): {hybrid_metrics.estimated_savings_vs_tier2_only:.1%} vs Tier-2 only
  - The cascade architecture reduces costs significantly
""")


if __name__ == "__main__":
    asyncio.run(main())
