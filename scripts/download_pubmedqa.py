"""Download PubMedQA dataset from HuggingFace for evaluation."""

import json
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "pubmedqa"


def download_pubmedqa(subset: str = "pqa_labeled", output_dir: Path = DATA_DIR):
    """
    Download PubMedQA dataset from HuggingFace.
    
    Args:
        subset: Dataset subset to download ('pqa_labeled', 'pqa_unlabeled', 'pqa_artificial')
        output_dir: Directory to save the dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading PubMedQA dataset (subset: {subset})...")
    
    try:
        # Load from HuggingFace
        dataset = load_dataset("qiaojin/PubMedQA", subset)
        
        logger.info(f"Dataset loaded. Available splits: {list(dataset.keys())}")
        
        # Process each split
        for split_name, split_data in dataset.items():
            logger.info(f"Processing split '{split_name}' with {len(split_data)} examples...")
            
            # Convert to list of dicts
            examples = []
            for idx, item in enumerate(split_data):
                example = {
                    "id": str(item.get("pubid", idx)),
                    "question": item.get("question", ""),
                    "context": extract_context(item),
                    "long_answer": item.get("long_answer", ""),
                    "final_decision": item.get("final_decision", ""),
                    "meshes": item.get("context", {}).get("meshes", []) if isinstance(item.get("context"), dict) else [],
                }
                examples.append(example)
            
            # Save to JSON
            output_file = output_dir / f"{subset}_{split_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(examples)} examples to {output_file}")
        
        # Create a summary file
        summary = {
            "subset": subset,
            "splits": list(dataset.keys()),
            "total_examples": sum(len(split) for split in dataset.values()),
            "label_distribution": get_label_distribution(dataset),
        }
        
        summary_file = output_dir / f"{subset}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to {summary_file}")
        logger.info(f"Download complete! Label distribution: {summary['label_distribution']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def extract_context(item: dict) -> str:
    """Extract context text from PubMedQA item."""
    context_data = item.get("context", {})
    
    if isinstance(context_data, dict):
        contexts = context_data.get("contexts", [])
        if isinstance(contexts, list):
            return "\n\n".join(contexts)
        return str(contexts)
    elif isinstance(context_data, str):
        return context_data
    elif isinstance(context_data, list):
        return "\n\n".join(str(c) for c in context_data)
    
    return ""


def get_label_distribution(dataset) -> dict:
    """Get distribution of yes/no/maybe labels."""
    distribution = {"yes": 0, "no": 0, "maybe": 0, "unknown": 0}
    
    for split_data in dataset.values():
        for item in split_data:
            label = item.get("final_decision", "").lower()
            if label in distribution:
                distribution[label] += 1
            else:
                distribution["unknown"] += 1
    
    return distribution


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PubMedQA dataset")
    parser.add_argument(
        "--subset",
        type=str,
        default="pqa_labeled",
        choices=["pqa_labeled", "pqa_unlabeled", "pqa_artificial"],
        help="Dataset subset to download (default: pqa_labeled)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Output directory (default: {DATA_DIR})"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_pubmedqa(subset=args.subset, output_dir=output_dir)


if __name__ == "__main__":
    main()
