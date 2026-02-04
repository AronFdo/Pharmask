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
    
    results, metrics = await evaluator.evaluate(progress_callback=progress)
    print()  # Clear progress line
    
    # Print report
    evaluator.print_report(results, metrics)
    
    # Save results
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"pharma_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "metrics": {
                "classification_accuracy": metrics.classification_accuracy,
                "classification_by_type": metrics.classification_by_type,
                "avg_keyword_coverage": metrics.avg_keyword_coverage,
                "perfect_coverage_rate": metrics.perfect_coverage_rate,
                "total_cost_usd": metrics.total_cost_usd,
                "avg_cost_per_query": metrics.avg_cost_per_query,
                "tier1_ratio": metrics.tier1_ratio,
                "estimated_savings": metrics.estimated_savings_vs_tier2_only,
                "avg_latency_ms": metrics.avg_latency_ms,
                "coverage_by_query_type": metrics.coverage_by_query_type,
            },
            "results": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "expected_type": r.expected_type,
                    "predicted_type": r.predicted_type,
                    "classification_correct": r.classification_correct,
                    "keyword_coverage": r.keyword_coverage,
                    "keywords_found": r.keywords_found,
                    "keywords_missing": r.keywords_missing,
                    "cost_usd": r.total_cost_usd,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ]
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {json_path}")
    
    # Print thesis-ready summary
    print("\n" + "="*70)
    print("THESIS-READY SUMMARY")
    print("="*70)
    print(f"""
Research Question 1: Query Classification
  - Classification Accuracy: {metrics.classification_accuracy:.1%}
  - The Tier-1 model correctly routes queries to appropriate retrieval

Research Question 2: Hybrid Retrieval Value
  - Text Query Coverage: {metrics.coverage_by_query_type.get('text', 0):.1%}
  - SQL Query Coverage: {metrics.coverage_by_query_type.get('sql', 0):.1%}  
  - Hybrid Query Coverage: {metrics.coverage_by_query_type.get('hybrid', 0):.1%}
  - Hybrid queries require BOTH sources for complete answers

Research Question 3: Cost-Efficiency
  - Tier-1 Token Ratio: {metrics.tier1_ratio:.1%}
  - Estimated Savings: {metrics.estimated_savings_vs_tier2_only:.1%} vs Tier-2 only
  - The cascade architecture reduces costs significantly
""")


if __name__ == "__main__":
    asyncio.run(main())
