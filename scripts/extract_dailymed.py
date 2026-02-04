"""
Extract DailyMed ZIP files to get SPL XML documents.

Usage:
    python scripts/extract_dailymed.py [--data-dir PATH] [--delete-zips]

Options:
    --data-dir PATH    Base data directory (default: ./data/documents)
    --delete-zips      Delete ZIP files after extraction
"""

import argparse
import logging
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_zip(zip_path: Path, output_dir: Path, delete_after: bool = False) -> tuple[bool, str]:
    """
    Extract a single ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract to
        delete_after: Delete ZIP after successful extraction
        
    Returns:
        Tuple of (success, message)
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract only XML files
            xml_files = [f for f in zf.namelist() if f.lower().endswith('.xml')]
            
            if not xml_files:
                return False, f"No XML files in {zip_path.name}"
            
            for xml_file in xml_files:
                # Extract to output directory with a clean filename
                xml_name = Path(xml_file).name
                # Prefix with zip name to avoid collisions
                output_name = f"{zip_path.stem}_{xml_name}"
                output_path = output_dir / output_name
                
                # Extract the file
                with zf.open(xml_file) as source:
                    content = source.read()
                    with open(output_path, 'wb') as target:
                        target.write(content)
        
        if delete_after:
            zip_path.unlink()
        
        return True, f"Extracted {len(xml_files)} XML files from {zip_path.name}"
        
    except zipfile.BadZipFile:
        return False, f"Bad ZIP file: {zip_path.name}"
    except Exception as e:
        return False, f"Error extracting {zip_path.name}: {str(e)}"


def extract_folder(folder_path: Path, delete_zips: bool = False, max_workers: int = 8) -> dict:
    """
    Extract all ZIP files in a folder.
    
    Args:
        folder_path: Path to folder containing ZIPs
        delete_zips: Delete ZIPs after extraction
        max_workers: Number of parallel extraction threads
        
    Returns:
        Statistics dict
    """
    if not folder_path.exists():
        logger.warning(f"Folder does not exist: {folder_path}")
        return {"extracted": 0, "failed": 0, "skipped": 0}
    
    zip_files = list(folder_path.glob("*.zip"))
    
    if not zip_files:
        logger.info(f"No ZIP files in {folder_path}")
        return {"extracted": 0, "failed": 0, "skipped": 0}
    
    logger.info(f"Found {len(zip_files)} ZIP files in {folder_path.name}")
    
    extracted = 0
    failed = 0
    
    # Use thread pool for parallel extraction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_zip, zf, folder_path, delete_zips): zf 
            for zf in zip_files
        }
        
        for future in tqdm(as_completed(futures), total=len(zip_files), desc=f"Extracting {folder_path.name}"):
            success, message = future.result()
            if success:
                extracted += 1
            else:
                failed += 1
                if failed <= 5:  # Only log first 5 errors
                    logger.warning(message)
    
    return {"extracted": extracted, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Extract DailyMed ZIP files to get SPL XML documents"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/documents",
        help="Base data directory (default: ./data/documents)",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete ZIP files after successful extraction",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel extraction threads (default: 8)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Folders to process
    folders = ["prescription", "otc", "other"]
    
    total_stats = {"extracted": 0, "failed": 0}
    
    logger.info("=" * 60)
    logger.info("DailyMed ZIP Extraction")
    logger.info("=" * 60)
    
    for folder_name in folders:
        folder_path = data_dir / folder_name
        
        if folder_path.exists():
            logger.info(f"\nProcessing {folder_name}...")
            stats = extract_folder(folder_path, args.delete_zips, args.workers)
            
            total_stats["extracted"] += stats["extracted"]
            total_stats["failed"] += stats["failed"]
            
            logger.info(f"  {folder_name}: {stats['extracted']} extracted, {stats['failed']} failed")
        else:
            logger.info(f"Skipping {folder_name} (not found)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Complete!")
    logger.info(f"  Total extracted: {total_stats['extracted']}")
    logger.info(f"  Total failed: {total_stats['failed']}")
    logger.info("=" * 60)
    
    if not args.delete_zips:
        logger.info("\nTip: Run with --delete-zips to remove ZIP files after extraction")


if __name__ == "__main__":
    main()
