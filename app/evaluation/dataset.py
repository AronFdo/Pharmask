"""PubMedQA dataset loader for evaluation."""

import json
import logging
import random
from pathlib import Path
from typing import Optional

from .schemas import EvalQuestion

logger = logging.getLogger(__name__)

# Default data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "evaluation" / "pubmedqa"


class PubMedQADataset:
    """
    Loader for PubMedQA evaluation dataset.
    
    Usage:
        dataset = PubMedQADataset()
        questions = dataset.load()
        sample = dataset.sample(n=50)
    """
    
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing PubMedQA JSON files
        """
        self.data_dir = Path(data_dir)
        self._data: list[EvalQuestion] = []
        self._loaded = False
    
    def load(self, subset: str = "pqa_labeled", split: str = "train") -> list[EvalQuestion]:
        """
        Load the dataset from JSON files.
        
        Args:
            subset: Dataset subset ('pqa_labeled', 'pqa_unlabeled', 'pqa_artificial')
            split: Data split ('train', 'test', 'validation')
            
        Returns:
            List of EvalQuestion objects
        """
        if self._loaded:
            return self._data
        
        # Try different file naming patterns
        possible_files = [
            self.data_dir / f"{subset}_{split}.json",
            self.data_dir / f"{subset}_train.json",  # Fallback to train if split not found
            self.data_dir / f"{subset}.json",
        ]
        
        data_file = None
        for f in possible_files:
            if f.exists():
                data_file = f
                break
        
        if not data_file:
            raise FileNotFoundError(
                f"PubMedQA dataset not found in {self.data_dir}. "
                f"Run 'python scripts/download_pubmedqa.py' first."
            )
        
        logger.info(f"Loading PubMedQA from {data_file}")
        
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        self._data = []
        for item in raw_data:
            question = EvalQuestion(
                id=str(item.get("id", "")),
                question=item.get("question", ""),
                context=item.get("context", ""),
                ground_truth=item.get("final_decision", "").lower(),
                long_answer=item.get("long_answer", ""),
                meshes=item.get("meshes", []),
            )
            
            # Only include questions with valid labels
            if question.ground_truth in ("yes", "no", "maybe"):
                self._data.append(question)
            else:
                logger.debug(f"Skipping question {question.id} with invalid label: {question.ground_truth}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._data)} questions from PubMedQA")
        
        return self._data
    
    def sample(self, n: int, seed: Optional[int] = 42) -> list[EvalQuestion]:
        """
        Get a random sample of questions.
        
        Args:
            n: Number of questions to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled EvalQuestion objects
        """
        if not self._loaded:
            self.load()
        
        if n >= len(self._data):
            return self._data.copy()
        
        if seed is not None:
            random.seed(seed)
        
        return random.sample(self._data, n)
    
    def sample_balanced(self, n_per_class: int, seed: Optional[int] = 42) -> list[EvalQuestion]:
        """
        Get a balanced sample with equal questions per class.
        
        Args:
            n_per_class: Number of questions per class (yes/no/maybe)
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled EvalQuestion objects
        """
        if not self._loaded:
            self.load()
        
        if seed is not None:
            random.seed(seed)
        
        # Group by class
        by_class = {"yes": [], "no": [], "maybe": []}
        for q in self._data:
            if q.ground_truth in by_class:
                by_class[q.ground_truth].append(q)
        
        # Sample from each class
        sampled = []
        for label, questions in by_class.items():
            n = min(n_per_class, len(questions))
            sampled.extend(random.sample(questions, n))
        
        # Shuffle the combined sample
        random.shuffle(sampled)
        
        return sampled
    
    def get_label_distribution(self) -> dict[str, int]:
        """Get the distribution of labels in the dataset."""
        if not self._loaded:
            self.load()
        
        distribution = {"yes": 0, "no": 0, "maybe": 0}
        for q in self._data:
            if q.ground_truth in distribution:
                distribution[q.ground_truth] += 1
        
        return distribution
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._data)
    
    def __iter__(self):
        if not self._loaded:
            self.load()
        return iter(self._data)
