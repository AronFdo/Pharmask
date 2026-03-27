"""JSON parser for pre-parsed PMC documents from Hugging Face dataset."""

import json
import logging
import re
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
            extracted_tables = self._extract_structured_tables(tables_text)
            if extracted_tables:
                doc.tables.extend(extracted_tables)
            else:
                # Fallback for malformed/unparsable table fragments.
                doc.tables.append({
                    "id": "tables",
                    "label": "Tables",
                    "caption": "",
                    "headers": [],
                    "rows": [[tables_text]],
                })
        
        # Extract metadata
        doc.metadata = {
            "pmid": data.get("pmid", ""),
            "license": data.get("license", ""),
            "retracted": data.get("retracted", False),
        }
        
        return doc

    def _extract_structured_tables(self, tables_text: str) -> list[dict]:
        """
        Parse PMC `tables` XML fragment into structured table entries.

        The source format usually contains one or more `<table-wrap>` elements.
        We only return table entries when at least one data row is extracted.
        """
        try:
            from lxml import etree

            wrapped = f"<root>{tables_text}</root>"
            root = etree.fromstring(wrapped.encode("utf-8"))
        except Exception:
            return []

        tables: list[dict] = []
        table_wraps = root.xpath(".//*[local-name()='table-wrap']")
        for idx, tw in enumerate(table_wraps):
            table_id = tw.get("id", "") or f"table_{idx}"
            caption_text = self._normalize_ws(" ".join(tw.xpath(".//*[local-name()='caption']//text()")))

            header_rows = tw.xpath(
                ".//*[local-name()='table']/*[local-name()='thead']/*[local-name()='tr']"
            )
            headers: list[str] = []
            if header_rows:
                # Prefer the first non-empty header row.
                for hr in header_rows:
                    cells = hr.xpath("./*[local-name()='th' or local-name()='td']")
                    values = [self._normalize_ws("".join(c.itertext())) for c in cells]
                    values = [v for v in values if v]
                    if values:
                        headers = values
                        break

            body_rows = tw.xpath(
                ".//*[local-name()='table']/*[local-name()='tbody']/*[local-name()='tr']"
            )
            if not body_rows:
                # Some tables may omit <tbody>
                body_rows = tw.xpath(
                    ".//*[local-name()='table']/*[local-name()='tr']"
                )

            rows: list[list[str]] = []
            for br in body_rows:
                cells = br.xpath("./*[local-name()='td' or local-name()='th']")
                row_vals = [self._normalize_ws("".join(c.itertext())) for c in cells]
                # Keep row only if at least one non-empty cell exists.
                if any(row_vals):
                    rows.append(row_vals)

            if rows:
                tables.append({
                    "id": table_id,
                    "label": table_id,
                    "caption": caption_text,
                    "headers": headers,
                    "rows": rows,
                    # Provide section for downstream rule-based classifier.
                    "section": caption_text,
                })

        return tables

    def _normalize_ws(self, text: str) -> str:
        """Normalize whitespace and strip table cell text."""
        return re.sub(r"\s+", " ", text or "").strip()
