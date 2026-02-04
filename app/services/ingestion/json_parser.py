"""JSON parser for pre-parsed PMC documents from Hugging Face dataset."""

import json
import logging
from pathlib import Path
from typing import Optional

from .xml_parser import ParsedDocument

logger = logging.getLogger(__name__)


class JSONParser:
    """Parser for JSON documents created from PMC-OA Hugging Face dataset."""
    
    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """
        Parse a JSON file and convert to ParsedDocument.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ParsedDocument with extracted content, or None on error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return self._parse_pmc_json(data, file_path)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return None
    
    def _parse_pmc_json(self, data: dict, file_path: Path) -> ParsedDocument:
        """Parse PMC JSON format from the download script."""
        doc = ParsedDocument(
            doc_id=data.get("doc_id", "") or file_path.stem,
            source_file=str(file_path),
        )
        
        # Extract citation as title if available
        citation = data.get("citation", "")
        if citation:
            # First line of citation is usually title
            doc.title = citation.split("\n")[0][:500]
        
        # Extract sections
        sections = data.get("sections", [])
        for section in sections:
            title = section.get("title", "")
            text = section.get("text", "")
            
            if text and text.strip():
                # Check if this is abstract/front matter
                if title.lower() in ["abstract", "front"]:
                    if not doc.abstract:
                        doc.abstract = text
                else:
                    doc.sections.append({
                        "title": title,
                        "text": text,
                    })
        
        # Extract tables
        tables_text = data.get("tables", "")
        if tables_text and tables_text.strip():
            # Tables are stored as text, create a pseudo-table entry
            doc.tables.append({
                "id": "tables",
                "label": "Tables",
                "caption": "",
                "headers": [],
                "rows": [[tables_text]],  # Store as single cell for now
            })
        
        # Extract metadata
        doc.metadata = {
            "pmid": data.get("pmid", ""),
            "license": data.get("license", ""),
            "retracted": data.get("retracted", False),
        }
        
        return doc
