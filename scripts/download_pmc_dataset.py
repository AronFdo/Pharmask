"""
Download PMC Open Access XML dataset from Hugging Face and store in data/documents.

The dataset contains pre-parsed sections (introduction, methods, results, etc.)
which we combine into structured XML documents for ingestion.

Usage:
    python scripts/download_pmc_dataset.py [--limit N] [--subset commercial|non_commercial|other]

Options:
    --limit N       Download only first N documents (default: all)
    --subset        Dataset subset: commercial, non_commercial, or other (default: commercial)
"""

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Section fields in the dataset that contain article content
CONTENT_SECTIONS = [
    'introduction',
    'methods', 
    'results',
    'discussion',
    'conclusion',
]

# Additional content fields
ADDITIONAL_FIELDS = [
    'front',      # Front matter (title, abstract, authors)
    'body',       # Full body text
    'back',       # Back matter (references, acknowledgments)
    'table',      # Tables
    'figure',     # Figure captions
    'supplementary',
]


def create_xml_from_sections(item: dict) -> str:
    """
    Create an XML document from the parsed sections.
    
    Args:
        item: Dataset item with section fields
        
    Returns:
        XML string
    """
    accession_id = get_text_content(item.get('accession_id', ''))
    pmid = get_text_content(item.get('pmid', ''))
    
    # Start building XML
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<article>',
        '  <front>',
        f'    <article-id pub-id-type="pmc">{accession_id}</article-id>',
        f'    <article-id pub-id-type="pmid">{pmid}</article-id>',
    ]
    
    # Add front matter if available
    front = get_text_content(item.get('front', ''))
    if front:
        xml_parts.append(f'    <article-meta>{_escape_xml(front)}</article-meta>')
    
    xml_parts.append('  </front>')
    xml_parts.append('  <body>')
    
    # Add main content sections
    for section in CONTENT_SECTIONS:
        content = get_text_content(item.get(section, ''))
        if content and content.strip():
            section_title = section.replace('_', ' ').title()
            xml_parts.append(f'    <sec>')
            xml_parts.append(f'      <title>{section_title}</title>')
            xml_parts.append(f'      <p>{_escape_xml(content)}</p>')
            xml_parts.append(f'    </sec>')
    
    # Add body content if no sections found
    body = get_text_content(item.get('body', ''))
    if body and not any(get_text_content(item.get(s)) for s in CONTENT_SECTIONS):
        xml_parts.append(f'    <sec>')
        xml_parts.append(f'      <title>Content</title>')
        xml_parts.append(f'      <p>{_escape_xml(body)}</p>')
        xml_parts.append(f'    </sec>')
    
    xml_parts.append('  </body>')
    
    # Add tables if available
    tables = get_text_content(item.get('table', ''))
    if tables:
        xml_parts.append('  <floats-group>')
        xml_parts.append(f'    <table-wrap><table>{_escape_xml(tables)}</table></table-wrap>')
        xml_parts.append('  </floats-group>')
    
    # Add back matter
    back = get_text_content(item.get('back', ''))
    if back:
        xml_parts.append('  <back>')
        xml_parts.append(f'    <p>{_escape_xml(back)}</p>')
        xml_parts.append('  </back>')
    
    xml_parts.append('</article>')
    
    return '\n'.join(xml_parts)


def create_json_document(item: dict) -> dict:
    """
    Create a structured JSON document from the dataset item.
    This format is easier to process and preserves all sections.
    
    Args:
        item: Dataset item with section fields
        
    Returns:
        Structured document dict
    """
    doc = {
        'doc_id': get_text_content(item.get('accession_id', '')),
        'pmid': get_text_content(item.get('pmid', '')),
        'license': get_text_content(item.get('license', '')),
        'citation': get_text_content(item.get('citation', '')),
        'retracted': item.get('retracted', False),
        'sections': [],
        'tables': get_text_content(item.get('table', '')),
        'figures': get_text_content(item.get('figure', '')),
    }
    
    # Add front matter as abstract section
    front = get_text_content(item.get('front', ''))
    if front and front.strip():
        doc['sections'].append({
            'title': 'Abstract',
            'text': front,
        })
    
    # Add main content sections
    for section in CONTENT_SECTIONS:
        content = get_text_content(item.get(section, ''))
        if content and content.strip():
            doc['sections'].append({
                'title': section.replace('_', ' ').title(),
                'text': content,
            })
    
    # Add body if no other sections
    if not doc['sections']:
        body = get_text_content(item.get('body', ''))
        if body and body.strip():
            doc['sections'].append({
                'title': 'Content',
                'text': body,
            })
    
    return doc


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    if not text:
        return ''
    return (text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))


def get_text_content(value) -> str:
    """
    Extract text content from a field that might be string, list, or None.
    
    Args:
        value: Field value (string, list of strings, or None)
        
    Returns:
        Concatenated string content
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Join list items with newlines
        return "\n".join(str(item) for item in value if item)
    return str(value)


def has_content(item: dict) -> bool:
    """Check if the item has any meaningful content."""
    # Check main sections
    for section in CONTENT_SECTIONS:
        content = get_text_content(item.get(section))
        if content.strip():
            return True
    
    # Check body
    body = get_text_content(item.get('body'))
    if body.strip():
        return True
    
    # Check front matter
    front = get_text_content(item.get('front'))
    if front.strip():
        return True
    
    return False


def download_pmc_dataset(
    output_dir: Path,
    subset: str = "commercial",
    limit: int = None,
    streaming: bool = True,
    output_format: str = "json",  # "json" or "xml"
):
    """
    Download PMC Open Access dataset from Hugging Face.
    
    The dataset contains pre-parsed sections which we save as structured documents.
    
    Args:
        output_dir: Directory to save files
        subset: Dataset subset (commercial, non_commercial, other)
        limit: Maximum number of documents to download (None for all)
        streaming: Use streaming mode to avoid loading full dataset into memory
        output_format: Output format - "json" (recommended) or "xml"
    """
    from datasets import load_dataset
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading PMC Open Access dataset (subset: {subset})...")
    logger.info("This may take a moment to initialize...")
    
    # Load dataset in streaming mode
    ds = load_dataset(
        "TomTBT/pmc_open_access_xml",
        subset,
        split="train",
        streaming=streaming,
        trust_remote_code=True,
    )
    
    logger.info(f"Saving documents to {output_dir} (format: {output_format})...")
    
    count = 0
    skipped = 0
    errors = 0
    
    iterator = iter(ds)
    
    try:
        for item in tqdm(iterator, desc="Downloading", total=limit):
            if limit and count >= limit:
                break
            
            try:
                # Skip items without content
                if not has_content(item):
                    skipped += 1
                    continue
                
                # Get document ID
                doc_id = item.get('accession_id', f'doc_{count}')
                doc_id = str(doc_id).replace('/', '_').replace('\\', '_').replace(':', '_')
                
                if output_format == "json":
                    # Create JSON document
                    doc = create_json_document(item)
                    filepath = output_dir / f"{doc_id}.json"
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(doc, f, indent=2, ensure_ascii=False)
                else:
                    # Create XML document
                    xml_content = create_xml_from_sections(item)
                    filepath = output_dir / f"{doc_id}.xml"
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(xml_content)
                
                count += 1
                
                # Log progress every 100 documents
                if count % 100 == 0:
                    logger.info(f"Downloaded {count} documents (skipped {skipped} empty)...")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error processing document: {e}")
                if errors > 50:
                    logger.error("Too many errors, stopping...")
                    break
                    
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    
    logger.info(f"Download complete!")
    logger.info(f"  Saved: {count} documents")
    logger.info(f"  Skipped (empty): {skipped}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Location: {output_dir}")
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download PMC Open Access dataset from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/documents/pmc",
        help="Output directory for files (default: ./data/documents/pmc)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["commercial", "non_commercial", "other"],
        default="commercial",
        help="Dataset subset (default: commercial)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to download (default: all)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "xml"],
        default="json",
        help="Output format: json (recommended) or xml (default: json)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (loads full dataset into memory)",
    )
    
    args = parser.parse_args()
    
    download_pmc_dataset(
        output_dir=Path(args.output_dir),
        subset=args.subset,
        limit=args.limit,
        streaming=not args.no_streaming,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
