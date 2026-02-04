"""PDF parser for pharmaceutical documents."""

import pdfplumber
from pathlib import Path
from typing import Optional
import logging
import re

from .xml_parser import ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser:
    """Parser for PDF pharmaceutical documents."""
    
    # Common section headers in pharmaceutical documents
    SECTION_PATTERNS = [
        r"^(INDICATIONS?\s*(AND\s*USAGE)?)",
        r"^(DOSAGE\s*(AND\s*ADMINISTRATION)?)",
        r"^(CONTRAINDICATIONS?)",
        r"^(WARNINGS?\s*(AND\s*PRECAUTIONS)?)",
        r"^(PRECAUTIONS?)",
        r"^(ADVERSE\s*REACTIONS?)",
        r"^(DRUG\s*INTERACTIONS?)",
        r"^(OVERDOSAGE)",
        r"^(CLINICAL\s*PHARMACOLOGY)",
        r"^(DESCRIPTION)",
        r"^(HOW\s*SUPPLIED)",
        r"^(STORAGE)",
        r"^(ABSTRACT)",
        r"^(INTRODUCTION)",
        r"^(METHODS?)",
        r"^(RESULTS?)",
        r"^(DISCUSSION)",
        r"^(CONCLUSION)",
        r"^(REFERENCES?)",
    ]
    
    def __init__(self):
        """Initialize the PDF parser."""
        self._section_re = re.compile(
            "|".join(self.SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
    
    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """
        Parse a PDF file and extract text sections and tables.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content, or None on error
        """
        try:
            doc = ParsedDocument(
                doc_id=file_path.stem,
                source_file=str(file_path),
            )
            
            with pdfplumber.open(str(file_path)) as pdf:
                # Extract text from all pages
                full_text = ""
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            table_data = self._process_table(
                                table, 
                                page_num=page_num + 1,
                                table_idx=table_idx
                            )
                            if table_data:
                                tables.append(table_data)
                
                # Try to extract title from first page
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                    lines = first_page_text.strip().split("\n")
                    if lines:
                        # First non-empty line as title (heuristic)
                        doc.title = lines[0].strip()[:200]
                
                # Parse sections from full text
                doc.sections = self._extract_sections(full_text)
                doc.tables = tables
                
                # If no sections found, use full text as single section
                if not doc.sections and full_text.strip():
                    doc.sections = [{
                        "title": "Content",
                        "text": full_text.strip(),
                    }]
            
            return doc
            
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {e}")
            return None
    
    def _extract_sections(self, text: str) -> list[dict]:
        """
        Extract sections from text based on common headers.
        
        Args:
            text: Full document text
            
        Returns:
            List of section dicts with 'title' and 'text'
        """
        sections = []
        
        # Find all section headers
        matches = list(self._section_re.finditer(text))
        
        if not matches:
            # No section headers found
            return []
        
        # Extract text between section headers
        for i, match in enumerate(matches):
            section_title = match.group().strip()
            start = match.end()
            
            # End is start of next section or end of text
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            
            if section_text:
                sections.append({
                    "title": section_title,
                    "text": section_text,
                })
        
        return sections
    
    def _process_table(
        self, 
        table: list[list], 
        page_num: int,
        table_idx: int
    ) -> Optional[dict]:
        """
        Process a raw table extracted from PDF.
        
        Args:
            table: 2D list of cell values
            page_num: Page number where table was found
            table_idx: Index of table on the page
            
        Returns:
            Table dict with headers and rows
        """
        if not table or len(table) < 2:
            return None
        
        # Clean cell values
        cleaned = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean whitespace and normalize
                    cleaned_row.append(" ".join(str(cell).split()))
            cleaned.append(cleaned_row)
        
        # First row as headers
        headers = cleaned[0]
        rows = cleaned[1:]
        
        # Filter out empty rows
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        
        if not rows:
            return None
        
        return {
            "id": f"table_p{page_num}_{table_idx}",
            "label": f"Table from page {page_num}",
            "caption": "",
            "headers": headers,
            "rows": rows,
        }
