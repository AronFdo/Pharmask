"""CLI script to run evaluation of the RAG system against baselines."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.dataset import PubMedQADataset
from app.evaluation.runner import EvaluationRunner
from app.evaluation.report import ReportGenerator
from app.evaluation.baselines import (
    BM25Baseline,
    DirectLLMBaseline,
    HybridRAGAdapter,
    ContextProvidedRAG,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation of Pharmaceutical RAG system against baselines"
    )
    
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        choices=["all", "fair", "hybrid_rag", "bm25", "direct_llm", "context_rag"],
        help="Which systems to evaluate: 'all' (all systems), 'fair' (context-provided comparison), or individual system"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced sampling (equal questions per class)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 questions"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    
    return parser.parse_args()


def get_systems(system_choice: str) -> dict:
    """Get the systems to evaluate based on user choice."""
    all_systems = {
        "hybrid_rag": HybridRAGAdapter(),
        "bm25": BM25Baseline(),
        "direct_llm": DirectLLMBaseline(),
        "context_rag": ContextProvidedRAG(),
    }
    
    # Fair comparison: same context provided to all
    fair_systems = {
        "bm25": BM25Baseline(),
        "context_rag": ContextProvidedRAG(),
        "direct_llm": DirectLLMBaseline(),
    }
    
    if system_choice == "all":
        return all_systems
    elif system_choice == "fair":
        return fair_systems
    else:
        return {system_choice: all_systems[system_choice]}


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Quick mode
    if args.quick:
        args.samples = 10
        logger.info("Quick mode: using 10 samples")
    
    # Load dataset
    logger.info("Loading PubMedQA dataset...")
    dataset = PubMedQADataset()
    
    try:
        dataset.load()
    except FileNotFoundError:
        logger.error(
            "PubMedQA dataset not found. Please run:\n"
            "  python scripts/download_pubmedqa.py"
        )
        sys.exit(1)
    
    # Get questions
    if args.samples:
        if args.balanced:
            # Balanced sampling
            n_per_class = args.samples // 3
            questions = dataset.sample_balanced(n_per_class, seed=args.seed)
            logger.info(f"Using {len(questions)} balanced samples ({n_per_class} per class)")
        else:
            questions = dataset.sample(args.samples, seed=args.seed)
            logger.info(f"Using {len(questions)} random samples")
    else:
        questions = list(dataset)
        logger.info(f"Using all {len(questions)} questions")
    
    # Show label distribution
    dist = dataset.get_label_distribution()
    logger.info(f"Dataset label distribution: {dist}")
    
    # Get systems
    systems = get_systems(args.systems)
    logger.info(f"Evaluating systems: {list(systems.keys())}")
    
    # Setup output
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Create runner
    runner_kwargs = {}
    if output_dir:
        runner_kwargs["output_dir"] = output_dir
    
    runner = EvaluationRunner(systems, **runner_kwargs)
    
    # Progress callback
    def progress(system_name, current, total):
        if current % 10 == 0 or current == total:
            print(f"  [{system_name}] {current}/{total} ({100*current/total:.0f}%)")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    report = await runner.run_evaluation(
        questions=questions,
        save_results=not args.no_save,
        progress_callback=progress,
    )
    
    # Generate reports
    report_gen = ReportGenerator(runner.output_dir)
    
    # Print summary to console
    report_gen.print_summary(report)
    
    # Save markdown report
    if not args.no_save:
        md_path = report_gen.save_markdown(report)
        logger.info(f"Markdown report saved to: {md_path}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
