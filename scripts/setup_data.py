"""
Complete data setup script: Download datasets and run ingestion.

Usage:
    python scripts/setup_data.py [--pmc-limit N] [--skip-pmc] [--dailymed-path PATH]

This script:
1. Downloads PMC-OA dataset from Hugging Face (optional limit)
2. Copies DailyMed files if path provided
3. Runs ingestion pipeline to load into Vector DB and SQL DB
"""

import argparse
import asyncio
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_data(
    pmc_limit: int = None,
    skip_pmc: bool = False,
    dailymed_path: str = None,
    run_ingestion: bool = True,
):
    """
    Set up pharmaceutical data for the RAG system.
    
    Args:
        pmc_limit: Maximum PMC documents to download (None for all)
        skip_pmc: Skip PMC download
        dailymed_path: Path to existing DailyMed download
        run_ingestion: Run ingestion after downloading
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "documents"
    
    # Step 1: Download PMC-OA dataset
    if not skip_pmc:
        logger.info("=" * 60)
        logger.info("Step 1: Downloading PMC Open Access dataset...")
        logger.info("=" * 60)
        
        from download_pmc_dataset import download_pmc_dataset
        
        pmc_dir = data_dir / "pmc"
        count = download_pmc_dataset(
            output_dir=pmc_dir,
            subset="commercial",
            limit=pmc_limit,
            streaming=True,
        )
        logger.info(f"PMC download complete: {count} documents")
    else:
        logger.info("Skipping PMC download")
    
    # Step 2: Set up DailyMed data
    if dailymed_path:
        logger.info("=" * 60)
        logger.info("Step 2: Setting up DailyMed data...")
        logger.info("=" * 60)
        
        dailymed_source = Path(dailymed_path)
        dailymed_dest = data_dir / "dailymed"
        dailymed_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy XML files from prescription and otc folders
        copied = 0
        for subdir in ["prescription", "otc", "homeopathic"]:
            source_subdir = dailymed_source / subdir
            if source_subdir.exists():
                for xml_file in source_subdir.glob("**/*.xml"):
                    dest_file = dailymed_dest / f"{subdir}_{xml_file.name}"
                    if not dest_file.exists():
                        shutil.copy2(xml_file, dest_file)
                        copied += 1
                        if copied % 100 == 0:
                            logger.info(f"Copied {copied} DailyMed files...")
        
        logger.info(f"DailyMed setup complete: {copied} files copied")
    else:
        logger.info("No DailyMed path provided, skipping")
    
    # Step 3: Run ingestion
    if run_ingestion:
        logger.info("=" * 60)
        logger.info("Step 3: Running ingestion pipeline...")
        logger.info("=" * 60)
        
        # Add parent directory to path for imports
        import sys
        sys.path.insert(0, str(base_dir))
        
        from app.services.ingestion import IngestionWorker
        
        worker = IngestionWorker()
        result = await worker.ingest_directory(data_dir)
        
        logger.info("Ingestion complete!")
        logger.info(f"  Documents processed: {result.documents_processed}")
        logger.info(f"  Text chunks created: {result.text_chunks_created}")
        logger.info(f"  Tables extracted: {result.tables_extracted}")
        
        if result.errors:
            logger.warning(f"  Errors: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.warning(f"    - {error}")
    
    logger.info("=" * 60)
    logger.info("Data setup complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Set up pharmaceutical data for the RAG system"
    )
    parser.add_argument(
        "--pmc-limit",
        type=int,
        default=100,  # Default to 100 for quick testing
        help="Maximum PMC documents to download (default: 100, use 0 for all)",
    )
    parser.add_argument(
        "--skip-pmc",
        action="store_true",
        help="Skip PMC dataset download",
    )
    parser.add_argument(
        "--dailymed-path",
        type=str,
        default=None,
        help="Path to existing DailyMed download folder",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion after downloading",
    )
    
    args = parser.parse_args()
    
    # Handle pmc_limit=0 as "all"
    pmc_limit = args.pmc_limit if args.pmc_limit > 0 else None
    
    asyncio.run(setup_data(
        pmc_limit=pmc_limit,
        skip_pmc=args.skip_pmc,
        dailymed_path=args.dailymed_path,
        run_ingestion=not args.skip_ingestion,
    ))


if __name__ == "__main__":
    main()
