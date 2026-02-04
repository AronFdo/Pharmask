"""
PMC Data Pipeline - Process and store PubMed Central research articles.

This script processes PMC-OA research articles and stores them in ChromaDB
for semantic search over pharmaceutical research literature.

Usage:
    # Process PMC JSON files from data/documents/pmc
    python scripts/pmc_pipeline.py --limit 100

    # Process from custom directory
    python scripts/pmc_pipeline.py --source ./data/documents/pmc --limit 500

    # Clear existing data and reprocess
    python scripts/pmc_pipeline.py --clear-db --limit 100

    # Download fresh from Hugging Face and process
    python scripts/pmc_pipeline.py --download --limit 200
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PMCArticle:
    """Structured PMC research article."""
    doc_id: str
    pmid: str = ""
    title: str = ""
    abstract: str = ""
    sections: list = field(default_factory=list)
    tables: str = ""
    figures: str = ""
    citation: str = ""
    license: str = ""
    source_file: str = ""
    
    def get_full_text(self) -> str:
        """Get concatenated full text of the article."""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        for section in self.sections:
            title = section.get("title", "")
            text = section.get("text", "")
            if text:
                if title:
                    parts.append(f"{title}: {text}")
                else:
                    parts.append(text)
        return "\n\n".join(parts)
    
    def get_section_text(self, section_name: str) -> str:
        """Get text from a specific section."""
        section_name_lower = section_name.lower()
        for section in self.sections:
            if section.get("title", "").lower() == section_name_lower:
                return section.get("text", "")
        return ""


class PMCProcessor:
    """Process PMC JSON files into structured articles."""
    
    def parse_json_file(self, file_path: Path) -> Optional[PMCArticle]:
        """Parse a PMC JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._parse_json_data(data, str(file_path))
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_json_data(self, data: dict, source_file: str) -> PMCArticle:
        """Parse JSON data into PMCArticle."""
        article = PMCArticle(
            doc_id=data.get("doc_id", "") or Path(source_file).stem,
            pmid=data.get("pmid", ""),
            citation=data.get("citation", ""),
            license=data.get("license", ""),
            tables=data.get("tables", ""),
            figures=data.get("figures", ""),
            source_file=source_file,
        )
        
        # Extract title from citation if available
        if article.citation:
            # First line is usually the title
            lines = article.citation.split("\n")
            if lines:
                article.title = lines[0].strip()[:500]
        
        # Process sections
        sections = data.get("sections", [])
        for section in sections:
            title = section.get("title", "")
            text = section.get("text", "")
            
            if text and text.strip():
                # Check if this is abstract
                if title.lower() in ["abstract", "front"]:
                    article.abstract = text
                else:
                    article.sections.append({
                        "title": title,
                        "text": text,
                    })
        
        return article


class PMCDatabaseLoader:
    """Load PMC articles into ChromaDB."""
    
    # Collection name for PMC articles
    COLLECTION_NAME = "pmc_articles"
    
    def __init__(self, clear_existing: bool = False):
        from app.db import VectorClient
        from app.config import settings
        
        self.vector_client = VectorClient()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        if clear_existing:
            self._clear_pmc_data()
    
    def _clear_pmc_data(self):
        """Clear existing PMC data from vector DB."""
        logger.info("Clearing existing PMC data from vector DB...")
        try:
            # Get all documents with PMC prefix and delete them
            # For simplicity, we'll just note that ChromaDB doesn't have easy selective delete
            # In production, you might use a separate collection for PMC
            logger.info("Note: Using shared collection - clear manually if needed")
        except Exception as e:
            logger.warning(f"Could not clear PMC data: {e}")
    
    def load_article(self, article: PMCArticle) -> dict:
        """Load a single article into ChromaDB."""
        stats = {"chunks": 0}
        
        try:
            chunks = self._create_chunks(article)
            
            if chunks:
                self.vector_client.add_documents(
                    documents=[c["text"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks],
                    ids=[c["id"] for c in chunks],
                )
                stats["chunks"] = len(chunks)
        except Exception as e:
            logger.error(f"Error loading article {article.doc_id}: {e}")
        
        return stats
    
    def _create_chunks(self, article: PMCArticle) -> list[dict]:
        """Create text chunks for vector storage."""
        chunks = []
        
        # Chunk the abstract
        if article.abstract:
            abstract_chunks = self._chunk_text(
                article.abstract,
                article.doc_id,
                "Abstract",
                article
            )
            chunks.extend(abstract_chunks)
        
        # Chunk each section
        for section in article.sections:
            title = section.get("title", "Content")
            text = section.get("text", "")
            
            if text:
                section_chunks = self._chunk_text(
                    text,
                    article.doc_id,
                    title,
                    article
                )
                chunks.extend(section_chunks)
        
        # If no sections, chunk full text
        if not chunks:
            full_text = article.get_full_text()
            if full_text:
                chunks = self._chunk_text(
                    full_text,
                    article.doc_id,
                    "Content",
                    article
                )
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        section: str,
        article: PMCArticle
    ) -> list[dict]:
        """Split text into overlapping chunks."""
        chunks = []
        
        if not text or not text.strip():
            return chunks
        
        text = text.strip()
        
        if len(text) <= self.chunk_size:
            # Single chunk
            chunks.append({
                "id": f"pmc_{doc_id}_{section}_0",
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "pmid": article.pmid,
                    "title": article.title[:200] if article.title else "",
                    "section": section,
                    "chunk_index": 0,
                    "source_type": "pmc",
                    "source_file": article.source_file,
                }
            })
        else:
            # Multiple chunks with overlap
            chunk_idx = 0
            start = 0
            
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                
                # Try to break at sentence boundary
                if end < len(text):
                    for punct in ['. ', '! ', '? ', '.\n']:
                        boundary = text.rfind(punct, start + int(self.chunk_size * 0.8), end)
                        if boundary != -1:
                            end = boundary + 1
                            break
                
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        "id": f"pmc_{doc_id}_{section}_{chunk_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "doc_id": doc_id,
                            "pmid": article.pmid,
                            "title": article.title[:200] if article.title else "",
                            "section": section,
                            "chunk_index": chunk_idx,
                            "source_type": "pmc",
                            "source_file": article.source_file,
                        }
                    })
                    chunk_idx += 1
                
                start = end - self.chunk_overlap
                if start >= len(text) or end >= len(text):
                    break
        
        return chunks


async def download_pmc_dataset(output_dir: Path, limit: int) -> int:
    """Download PMC dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return 0
    
    logger.info(f"Downloading PMC-OA dataset (limit: {limit})...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ds = load_dataset(
        "TomTBT/pmc_open_access_xml",
        "commercial",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    count = 0
    
    for item in tqdm(ds, desc="Downloading", total=limit):
        if count >= limit:
            break
        
        # Check if has content
        has_content = False
        for field in ['introduction', 'methods', 'results', 'discussion', 'conclusion', 'body', 'front']:
            value = item.get(field)
            if value:
                if isinstance(value, str) and value.strip():
                    has_content = True
                    break
                elif isinstance(value, list) and any(v.strip() if isinstance(v, str) else v for v in value):
                    has_content = True
                    break
        
        if not has_content:
            continue
        
        # Create document
        doc_id = item.get('accession_id', f'doc_{count}')
        doc_id = str(doc_id).replace('/', '_').replace('\\', '_')
        
        doc = {
            'doc_id': doc_id,
            'pmid': _get_text(item.get('pmid', '')),
            'citation': _get_text(item.get('citation', '')),
            'license': _get_text(item.get('license', '')),
            'tables': _get_text(item.get('table', '')),
            'figures': _get_text(item.get('figure', '')),
            'sections': [],
        }
        
        # Add sections
        front = _get_text(item.get('front', ''))
        if front:
            doc['sections'].append({'title': 'Abstract', 'text': front})
        
        for section_name in ['introduction', 'methods', 'results', 'discussion', 'conclusion']:
            content = _get_text(item.get(section_name, ''))
            if content:
                doc['sections'].append({
                    'title': section_name.title(),
                    'text': content,
                })
        
        # Add body if no sections
        if not doc['sections']:
            body = _get_text(item.get('body', ''))
            if body:
                doc['sections'].append({'title': 'Content', 'text': body})
        
        # Save to file
        filepath = output_dir / f"{doc_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        
        count += 1
        
        if count % 50 == 0:
            logger.info(f"Downloaded {count} articles...")
    
    logger.info(f"Downloaded {count} articles to {output_dir}")
    return count


def _get_text(value) -> str:
    """Extract text from a value that might be string, list, or None."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value if v)
    return str(value)


async def run_pipeline(
    source_dir: Optional[Path] = None,
    download: bool = False,
    limit: int = 100,
    clear_db: bool = False,
):
    """Run the PMC data pipeline."""
    
    logger.info("=" * 60)
    logger.info("PMC Research Articles Pipeline")
    logger.info("=" * 60)
    
    # Set default source directory
    if source_dir is None:
        source_dir = Path("./data/documents/pmc")
    
    # Download if requested
    if download:
        await download_pmc_dataset(source_dir, limit)
    
    # Check for files
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {source_dir}")
        logger.info("Run with --download to fetch from Hugging Face")
        return
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Limit files to process
    json_files = json_files[:limit]
    logger.info(f"Processing {len(json_files)} files...")
    
    processor = PMCProcessor()
    loader = PMCDatabaseLoader(clear_existing=clear_db)
    
    articles_processed = 0
    total_chunks = 0
    
    for file_path in tqdm(json_files, desc="Processing articles"):
        article = processor.parse_json_file(file_path)
        
        if article:
            stats = loader.load_article(article)
            articles_processed += 1
            total_chunks += stats["chunks"]
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Articles processed: {articles_processed}")
    logger.info(f"Vector chunks created: {total_chunks}")
    
    # Show database status
    vector_count = loader.vector_client.get_document_count()
    logger.info(f"\nTotal documents in Vector DB: {vector_count}")


def main():
    parser = argparse.ArgumentParser(
        description="PMC Data Pipeline - Process research articles for semantic search"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./data/documents/pmc",
        help="Source directory with PMC JSON files (default: ./data/documents/pmc)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download articles from Hugging Face PMC-OA dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of articles to process (default: 100)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear existing PMC data before processing",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_pipeline(
        source_dir=Path(args.source),
        download=args.download,
        limit=args.limit,
        clear_db=args.clear_db,
    ))


if __name__ == "__main__":
    main()
