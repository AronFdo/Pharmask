"""XML parser for PMC-OA and DailyMed style documents."""

from lxml import etree
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Parsed document with text sections and tables."""
    doc_id: str
    title: str = ""
    abstract: str = ""
    sections: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_file: str = ""


class XMLParser:
    """Parser for PMC-OA and DailyMed XML documents."""
    
    # Common namespace prefixes
    NAMESPACES = {
        "pmc": "https://jats.nlm.nih.gov/ns/archiving/1.3/",
        "dailymed": "urn:hl7-org:v3",
    }
    
    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """
        Parse an XML file and extract text sections and tables.
        
        Args:
            file_path: Path to the XML file
            
        Returns:
            ParsedDocument with extracted content, or None on error
        """
        try:
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            
            # Detect document type based on root element (case-insensitive)
            root_tag = etree.QName(root).localname if root.tag.startswith("{") else root.tag
            root_tag_lower = root_tag.lower()
            
            # Check namespace for DailyMed detection
            root_ns = etree.QName(root).namespace if root.tag.startswith("{") else ""
            is_hl7 = "hl7-org" in (root_ns or "") or "hl7" in (root_ns or "").lower()
            
            logger.debug(f"Parsing {file_path.name}: root_tag={root_tag}, namespace={root_ns}")
            
            if root_tag_lower == "article":
                # PMC-OA JATS format
                return self._parse_pmc_article(root, file_path)
            elif root_tag_lower == "document" or is_hl7:
                # DailyMed SPL format
                return self._parse_dailymed(root, file_path)
            else:
                # Generic XML
                return self._parse_generic_xml(root, file_path)
                
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            return None
    
    def _parse_pmc_article(self, root: etree._Element, file_path: Path) -> ParsedDocument:
        """Parse PMC-OA JATS format article."""
        doc = ParsedDocument(
            doc_id=self._extract_text(root, ".//article-id[@pub-id-type='pmc']") or file_path.stem,
            source_file=str(file_path),
        )
        
        # Extract title
        doc.title = self._extract_text(root, ".//article-title")
        
        # Extract abstract
        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            doc.abstract = self._get_element_text(abstract_elem)
        
        # Extract body sections
        body = root.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                title_elem = sec.find("title")
                section_title = title_elem.text if title_elem is not None else ""
                section_text = self._get_section_text(sec)
                
                if section_text.strip():
                    doc.sections.append({
                        "title": section_title,
                        "text": section_text,
                    })
        
        # Extract tables
        for table_wrap in root.findall(".//table-wrap"):
            table_data = self._extract_table(table_wrap)
            if table_data:
                doc.tables.append(table_data)
        
        # Extract metadata
        doc.metadata = {
            "journal": self._extract_text(root, ".//journal-title"),
            "pub_date": self._extract_text(root, ".//pub-date/year"),
            "doi": self._extract_text(root, ".//article-id[@pub-id-type='doi']"),
        }
        
        return doc
    
    def _parse_dailymed(self, root: etree._Element, file_path: Path) -> ParsedDocument:
        """Parse DailyMed SPL format."""
        # Handle namespace - try to detect from root
        ns = {"hl7": "urn:hl7-org:v3"}
        
        doc = ParsedDocument(
            doc_id=file_path.stem,
            source_file=str(file_path),
        )
        
        # Try to get document ID from various locations
        doc_id = self._extract_text_ns(root, ".//hl7:id/@root", ns)
        if not doc_id:
            doc_id = root.get("ID") or root.get("id") or file_path.stem
        doc.doc_id = doc_id
        
        # Extract title - try multiple paths
        doc.title = (
            self._extract_text_ns(root, ".//hl7:title", ns) or
            self._find_text_any_ns(root, "title") or
            ""
        )
        
        # Find sections - try with namespace first, then without
        sections = root.findall(".//hl7:component/hl7:section", ns)
        if not sections:
            sections = root.findall(".//{*}component/{*}section")
        if not sections:
            sections = root.findall(".//section")
        
        logger.debug(f"DailyMed {file_path.name}: found {len(sections)} sections")
        
        # Extract structured body sections
        for component in sections:
            # Get section title
            title_elem = component.find("hl7:title", ns)
            if title_elem is None:
                title_elem = component.find("{*}title") or component.find("title")
            section_title = title_elem.text if title_elem is not None else ""
            
            # Get text content
            text_parts = []
            text_elems = component.findall(".//hl7:text", ns)
            if not text_elems:
                text_elems = component.findall(".//{*}text") or component.findall(".//text")
            
            for text_elem in text_elems:
                text_parts.append(self._get_element_text(text_elem))
            
            section_text = "\n".join(text_parts)
            
            if section_text.strip():
                doc.sections.append({
                    "title": section_title,
                    "text": section_text,
                })
            
            # Extract tables from this section - try multiple paths
            tables = component.findall(".//hl7:table", ns)
            if not tables:
                tables = component.findall(".//{*}table") or component.findall(".//table")
            
            for table in tables:
                table_data = self._extract_html_table(table, ns)
                if table_data:
                    table_data["section"] = section_title
                    doc.tables.append(table_data)
        
        logger.debug(f"DailyMed {file_path.name}: extracted {len(doc.sections)} sections, {len(doc.tables)} tables")
        
        # Extract drug metadata
        drug_name = self._extract_text_ns(root, ".//hl7:manufacturedProduct//hl7:name", ns)
        if not drug_name:
            drug_name = self._find_text_any_ns(root, "name")
        
        doc.metadata = {
            "drug_name": drug_name or "",
            "ndc": self._extract_text_ns(root, ".//hl7:code/@code", ns) or "",
        }
        
        return doc
    
    def _find_text_any_ns(self, element: etree._Element, tag_name: str) -> str:
        """Find text from a tag regardless of namespace."""
        # Try with wildcard namespace
        elem = element.find(f".//{{{self.NAMESPACES['dailymed']}}}{tag_name}")
        if elem is not None and elem.text:
            return elem.text
        
        elem = element.find(f".//*[local-name()='{tag_name}']")
        if elem is not None and elem.text:
            return elem.text
            
        elem = element.find(f".//{tag_name}")
        if elem is not None and elem.text:
            return elem.text
        
        return ""
    
    def _parse_generic_xml(self, root: etree._Element, file_path: Path) -> ParsedDocument:
        """Parse generic XML by extracting all text content."""
        doc = ParsedDocument(
            doc_id=file_path.stem,
            source_file=str(file_path),
        )
        
        # Extract all text content
        full_text = self._get_element_text(root)
        if full_text.strip():
            doc.sections.append({
                "title": "Content",
                "text": full_text,
            })
        
        return doc
    
    def _extract_text(self, element: etree._Element, xpath: str) -> str:
        """Extract text from an xpath match."""
        result = element.find(xpath)
        if result is not None:
            return result.text or ""
        return ""
    
    def _extract_text_ns(self, element: etree._Element, xpath: str, ns: dict) -> str:
        """Extract text from an xpath match with namespaces."""
        result = element.find(xpath, ns)
        if result is not None:
            if isinstance(result, str):
                return result
            return result.text or ""
        return ""
    
    def _get_element_text(self, element: etree._Element) -> str:
        """Get all text content from an element, including children."""
        texts = []
        for text in element.itertext():
            cleaned = text.strip()
            if cleaned:
                texts.append(cleaned)
        return " ".join(texts)
    
    def _get_section_text(self, section: etree._Element) -> str:
        """Get text from a section, excluding nested sections and tables."""
        texts = []
        
        for child in section:
            # Skip nested sections and tables
            if child.tag in ["sec", "table-wrap", "table"]:
                continue
            
            # Get text from paragraphs and other elements
            if child.tag == "p" or child.tag == "title":
                text = self._get_element_text(child)
                if text:
                    texts.append(text)
            elif child.text:
                texts.append(child.text.strip())
        
        return "\n".join(texts)
    
    def _extract_table(self, table_wrap: etree._Element) -> Optional[dict]:
        """Extract table data from a table-wrap element."""
        table_id = table_wrap.get("id", "")
        
        # Get caption/label
        label = self._extract_text(table_wrap, ".//label")
        caption = self._extract_text(table_wrap, ".//caption/p") or self._extract_text(table_wrap, ".//caption/title")
        
        # Find the table element
        table = table_wrap.find(".//table")
        if table is None:
            return None
        
        # Extract headers
        headers = []
        thead = table.find(".//thead")
        if thead is not None:
            for th in thead.findall(".//th"):
                headers.append(self._get_element_text(th))
        
        # If no thead, try first row
        if not headers:
            first_row = table.find(".//tr")
            if first_row is not None:
                for cell in first_row:
                    headers.append(self._get_element_text(cell))
        
        # Extract rows
        rows = []
        tbody = table.find(".//tbody")
        row_elements = tbody.findall(".//tr") if tbody is not None else table.findall(".//tr")[1:]
        
        for tr in row_elements:
            row = []
            for td in tr.findall(".//td"):
                row.append(self._get_element_text(td))
            if row:
                rows.append(row)
        
        if not rows:
            return None
        
        return {
            "id": table_id,
            "label": label,
            "caption": caption,
            "headers": headers,
            "rows": rows,
        }
    
    def _extract_html_table(self, table: etree._Element, ns: dict) -> Optional[dict]:
        """Extract table data from HTML-style table in SPL."""
        headers = []
        rows = []
        
        # Try multiple ways to find rows (namespaced and non-namespaced)
        tr_elements = table.findall(".//hl7:tr", ns)
        if not tr_elements:
            tr_elements = table.findall(".//tr")
        if not tr_elements:
            # Try with wildcard namespace
            tr_elements = table.findall(".//{*}tr")
        
        for tr in tr_elements:
            row_data = []
            is_header = False
            
            # Find cells (th or td) with various namespace possibilities
            cells = list(tr)
            for cell in cells:
                cell_tag = etree.QName(cell).localname if cell.tag.startswith("{") else cell.tag
                cell_tag_lower = cell_tag.lower()
                if cell_tag_lower == "th":
                    is_header = True
                if cell_tag_lower in ("th", "td"):
                    row_data.append(self._get_element_text(cell))
            
            if is_header and not headers:
                headers = row_data
            elif row_data:
                rows.append(row_data)
        
        if not rows:
            return None
        
        return {
            "id": "",
            "label": "",
            "caption": "",
            "headers": headers,
            "rows": rows,
        }
