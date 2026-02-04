"""
Run evaluation using questions based on YOUR ACTUAL DATA.

These questions are derived from the real OTC drugs in your database:
- ZYRTEC, Children's Cough DM, Advil, Nicotine lozenges
- Minoxidil, Wart removers, Hand sanitizers, Sunscreens
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.orchestrator import RAGOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "real_pharma_qa.json"


async def run_evaluation():
    """Run evaluation on real data questions."""
    
    # Load questions
    with open(DATASET_PATH, "r") as f:
        questions = json.load(f)
    
    logger.info(f"Loaded {len(questions)} questions based on YOUR ACTUAL DATA")
    
    # Initialize orchestrator
    orchestrator = RAGOrchestrator()
    
    results = []
    
    # Track by category
    text_results = []
    sql_results = []
    hybrid_results = []
    
    for i, q in enumerate(questions):
        logger.info(f"\n[{i+1}/{len(questions)}] {q['query_type'].upper()}: {q['question'][:50]}...")
        
        try:
            response = await orchestrator.process_query(q['question'])
            
            answer = response.answer
            predicted_type = response.classification.query_type
            
            # Check classification
            classification_correct = predicted_type == q['query_type']
            
            # Check keyword coverage
            answer_lower = answer.lower()
            keywords_found = [kw for kw in q['ground_truth_contains'] if kw.lower() in answer_lower]
            keywords_missing = [kw for kw in q['ground_truth_contains'] if kw.lower() not in answer_lower]
            coverage = len(keywords_found) / len(q['ground_truth_contains'])
            
            result = {
                "id": q["id"],
                "question": q["question"],
                "expected_type": q["query_type"],
                "predicted_type": predicted_type,
                "classification_correct": classification_correct,
                "keyword_coverage": coverage,
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "cost_usd": response.cost.total_cost_usd if response.cost else 0,
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
            }
            
            results.append(result)
            
            # Track by category
            if q['query_type'] == 'text':
                text_results.append(result)
            elif q['query_type'] == 'sql':
                sql_results.append(result)
            else:
                hybrid_results.append(result)
            
            # Print result
            status = "✓" if classification_correct else "✗"
            print(f"  {status} Classified as: {predicted_type} | Coverage: {coverage:.0%}")
            print(f"    Found: {keywords_found}")
            if keywords_missing:
                print(f"    Missing: {keywords_missing}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "error": str(e)
            })
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS - BASED ON YOUR ACTUAL DATA")
    print("="*70)
    
    valid_results = [r for r in results if "error" not in r]
    
    # Overall
    total_correct = sum(1 for r in valid_results if r['classification_correct'])
    total_coverage = sum(r['keyword_coverage'] for r in valid_results) / len(valid_results)
    total_cost = sum(r['cost_usd'] for r in valid_results)
    
    print(f"\nOVERALL:")
    print(f"  Classification Accuracy: {total_correct}/{len(valid_results)} ({total_correct/len(valid_results):.1%})")
    print(f"  Average Keyword Coverage: {total_coverage:.1%}")
    print(f"  Total Cost: ${total_cost:.4f}")
    
    # By category
    print(f"\nBY QUERY TYPE:")
    for name, category_results in [("TEXT", text_results), ("SQL", sql_results), ("HYBRID", hybrid_results)]:
        if category_results:
            cat_correct = sum(1 for r in category_results if r['classification_correct'])
            cat_coverage = sum(r['keyword_coverage'] for r in category_results) / len(category_results)
            print(f"  {name}:")
            print(f"    Classification: {cat_correct}/{len(category_results)} ({cat_correct/len(category_results):.1%})")
            print(f"    Keyword Coverage: {cat_coverage:.1%}")
    
    # Save results
    output_dir = PROJECT_ROOT / "data" / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"real_eval_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_questions": len(questions),
            "classification_accuracy": total_correct / len(valid_results),
            "avg_keyword_coverage": total_coverage,
            "total_cost_usd": total_cost,
            "by_type": {
                "text": {
                    "accuracy": sum(1 for r in text_results if r['classification_correct']) / len(text_results) if text_results else 0,
                    "coverage": sum(r['keyword_coverage'] for r in text_results) / len(text_results) if text_results else 0,
                },
                "sql": {
                    "accuracy": sum(1 for r in sql_results if r['classification_correct']) / len(sql_results) if sql_results else 0,
                    "coverage": sum(r['keyword_coverage'] for r in sql_results) / len(sql_results) if sql_results else 0,
                },
                "hybrid": {
                    "accuracy": sum(1 for r in hybrid_results if r['classification_correct']) / len(hybrid_results) if hybrid_results else 0,
                    "coverage": sum(r['keyword_coverage'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0,
                },
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print sample answers
    print("\n" + "="*70)
    print("SAMPLE ANSWERS")
    print("="*70)
    for r in results[:5]:
        if "error" not in r:
            print(f"\nQ: {r['question']}")
            print(f"A: {r['answer_preview']}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
