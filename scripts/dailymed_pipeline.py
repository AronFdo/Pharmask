"""
DailyMed Data Pipeline - Download, process, and store pharmaceutical data.

This script creates a complete data pipeline that:
1. Downloads drug data from DailyMed API (or processes local files)
2. Parses and extracts structured information
3. Stores in SQLite and ChromaDB for immediate querying

Usage:
    # Download from DailyMed API and process
    python scripts/dailymed_pipeline.py --download --limit 100

    # Process existing XML files
    python scripts/dailymed_pipeline.py --source ./data/documents/prescription --limit 500

    # Full pipeline with everything
    python scripts/dailymed_pipeline.py --download --limit 1000 --clear-db
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import httpx

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Structured drug information extracted from DailyMed."""
    drug_id: str
    name: str
    generic_name: str = ""
    brand_name: str = ""
    manufacturer: str = ""
    ndc_code: str = ""
    description: str = ""
    indications: list = None
    dosages: list = None
    adverse_reactions: list = None
    interactions: list = None
    warnings: str = ""
    source_file: str = ""
    
    def __post_init__(self):
        self.indications = self.indications or []
        self.dosages = self.dosages or []
        self.adverse_reactions = self.adverse_reactions or []
        self.interactions = self.interactions or []


class DailyMedAPI:
    """Client for DailyMed REST API."""
    
    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
    
    def search_drugs(self, query: str = "", limit: int = 100, page: int = 1) -> list[dict]:
        """Search for drugs via DailyMed API."""
        params = {
            "pagesize": min(limit, 100),
            "page": page,
        }
        if query:
            params["drug_name"] = query
        
        try:
            response = self.client.get(f"{self.BASE_URL}/spls.json", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"API search error: {e}")
            return []
    
    def get_drug_spl(self, setid: str) -> Optional[str]:
        """Get SPL XML content for a drug by setid."""
        try:
            response = self.client.get(f"{self.BASE_URL}/spls/{setid}.xml")
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching SPL {setid}: {e}")
            return None
    
    def get_drug_info(self, setid: str) -> Optional[dict]:
        """Get drug info JSON for a drug by setid."""
        try:
            response = self.client.get(f"{self.BASE_URL}/spls/{setid}.json")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching drug info {setid}: {e}")
            return None
    
    def download_drugs(self, limit: int = 100) -> list[dict]:
        """Download multiple drugs from the API."""
        all_drugs = []
        page = 1
        
        with tqdm(total=limit, desc="Fetching drug list") as pbar:
            while len(all_drugs) < limit:
                batch = self.search_drugs(limit=min(100, limit - len(all_drugs)), page=page)
                if not batch:
                    break
                all_drugs.extend(batch)
                pbar.update(len(batch))
                page += 1
                time.sleep(0.1)  # Rate limiting
        
        return all_drugs[:limit]
    
    def close(self):
        self.client.close()


class DailyMedProcessor:
    """Process DailyMed SPL files and extract structured data."""
    
    # Section code mappings for SPL
    SECTION_CODES = {
        "34066-1": "indications",      # BOXED WARNING
        "34067-9": "indications",      # INDICATIONS & USAGE
        "34068-7": "dosages",          # DOSAGE & ADMINISTRATION
        "34069-5": "dosages",          # HOW SUPPLIED
        "34070-3": "contraindications",
        "34071-1": "warnings",         # WARNINGS
        "34072-9": "adverse_reactions", # ADVERSE REACTIONS
        "34073-7": "interactions",      # DRUG INTERACTIONS
        "34074-5": "interactions",      # DRUG/LABORATORY TEST INTERACTIONS
        "34084-4": "adverse_reactions", # ADVERSE REACTIONS TABLE
        "43685-7": "warnings",          # WARNINGS AND PRECAUTIONS
    }
    
    def __init__(self):
        from lxml import etree
        self.etree = etree
        self.ns = {"hl7": "urn:hl7-org:v3"}
    
    def parse_spl_content(self, xml_content: str, source_file: str = "") -> Optional[DrugInfo]:
        """Parse SPL XML content and extract drug information."""
        try:
            root = self.etree.fromstring(xml_content.encode('utf-8'))
            return self._extract_drug_info(root, source_file)
        except Exception as e:
            logger.error(f"Error parsing SPL: {e}")
            return None
    
    def parse_spl_file(self, file_path: Path) -> Optional[DrugInfo]:
        """Parse an SPL XML file."""
        try:
            tree = self.etree.parse(str(file_path))
            root = tree.getroot()
            return self._extract_drug_info(root, str(file_path))
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_drug_info(self, root, source_file: str) -> DrugInfo:
        """Extract structured drug information from SPL root element."""
        ns = self.ns
        
        # Get document ID
        id_elem = root.find(".//hl7:id", ns)
        drug_id = id_elem.get("root", "") if id_elem is not None else ""
        if not drug_id:
            drug_id = Path(source_file).stem if source_file else "unknown"
        
        # Get drug name
        name_elem = root.find(".//hl7:manufacturedProduct//hl7:name", ns)
        name = name_elem.text if name_elem is not None else ""
        
        # Get manufacturer
        mfr_elem = root.find(".//hl7:representedOrganization/hl7:name", ns)
        manufacturer = mfr_elem.text if mfr_elem is not None else ""
        
        drug = DrugInfo(
            drug_id=drug_id,
            name=name,
            manufacturer=manufacturer,
            source_file=source_file,
        )
        
        # Process sections
        for section in root.findall(".//hl7:section", ns):
            self._process_section(section, drug)
        
        return drug
    
    def _process_section(self, section, drug: DrugInfo):
        """Process a section element and extract relevant data."""
        ns = self.ns
        
        # Get section code
        code_elem = section.find("hl7:code", ns)
        if code_elem is None:
            return
        
        code = code_elem.get("code", "")
        section_type = self.SECTION_CODES.get(code)
        
        if not section_type:
            return
        
        # Get section title
        title_elem = section.find("hl7:title", ns)
        title = title_elem.text if title_elem is not None else ""
        
        # Get section text
        text_elem = section.find("hl7:text", ns)
        text = self._get_text_content(text_elem) if text_elem is not None else ""
        
        # Extract tables from section
        tables = self._extract_tables(section)
        
        # Store based on section type
        if section_type == "indications" and text:
            drug.indications.append({"text": text, "title": title})
        
        elif section_type == "dosages":
            if tables:
                for table in tables:
                    for row in table.get("rows", []):
                        if len(row) >= 2:
                            drug.dosages.append({
                                "form": row[0] if row else "",
                                "strength": row[1] if len(row) > 1 else "",
                                "route": row[2] if len(row) > 2 else "",
                                "frequency": row[3] if len(row) > 3 else "",
                            })
            elif text:
                drug.dosages.append({"description": text, "title": title})
        
        elif section_type == "adverse_reactions":
            if tables:
                for table in tables:
                    for row in table.get("rows", []):
                        if row:
                            drug.adverse_reactions.append({
                                "reaction": row[0],
                                "frequency": row[1] if len(row) > 1 else "",
                            })
            elif text:
                drug.adverse_reactions.append({"text": text})
        
        elif section_type == "interactions":
            if tables:
                for table in tables:
                    for row in table.get("rows", []):
                        if row:
                            drug.interactions.append({
                                "drug": row[0],
                                "description": row[1] if len(row) > 1 else "",
                            })
            elif text:
                drug.interactions.append({"text": text})
        
        elif section_type == "warnings":
            drug.warnings += f"\n{title}:\n{text}" if title else f"\n{text}"
    
    def _get_text_content(self, element) -> str:
        """Get all text content from an element."""
        if element is None:
            return ""
        texts = []
        for text in element.itertext():
            cleaned = text.strip()
            if cleaned:
                texts.append(cleaned)
        return " ".join(texts)
    
    def _extract_tables(self, section) -> list[dict]:
        """Extract tables from a section."""
        ns = self.ns
        tables = []
        
        for table in section.findall(".//hl7:table", ns):
            headers = []
            rows = []
            
            # Find rows
            for tr in table.findall(".//{*}tr"):
                row_data = []
                is_header = False
                
                for cell in tr:
                    tag = self.etree.QName(cell).localname if cell.tag.startswith("{") else cell.tag
                    if tag.lower() == "th":
                        is_header = True
                    if tag.lower() in ("th", "td"):
                        row_data.append(self._get_text_content(cell))
                
                if is_header and not headers:
                    headers = row_data
                elif row_data:
                    rows.append(row_data)
            
            if rows:
                tables.append({"headers": headers, "rows": rows})
        
        return tables


class DatabaseLoader:
    """Load processed drug data into databases."""
    
    def __init__(self, clear_existing: bool = False):
        from app.db import SQLClient, VectorClient
        
        self.sql_client = SQLClient()
        self.vector_client = VectorClient()
        
        if clear_existing:
            self._clear_databases()
    
    def _clear_databases(self):
        """Clear existing data from databases."""
        logger.info("Clearing existing database data...")
        
        # Clear SQL tables
        for table in ["drugs", "indications", "dosages", "adverse_reactions", "interactions"]:
            try:
                self.sql_client.clear_table(table)
            except Exception as e:
                logger.warning(f"Could not clear {table}: {e}")
        
        # Clear vector DB
        try:
            self.vector_client.delete_all()
        except Exception as e:
            logger.warning(f"Could not clear vector DB: {e}")
    
    def load_drug(self, drug: DrugInfo) -> dict:
        """Load a single drug into databases."""
        stats = {"sql_rows": 0, "vector_chunks": 0}
        
        try:
            # Insert into drugs table
            self.sql_client.insert_row("drugs", {
                "id": drug.drug_id,
                "name": drug.name,
                "generic_name": drug.generic_name,
                "brand_name": drug.brand_name,
                "manufacturer": drug.manufacturer,
                "ndc_code": drug.ndc_code,
                "description": drug.description,
                "source_doc": drug.source_file,
            })
            stats["sql_rows"] += 1
        except Exception as e:
            # Drug might already exist
            pass
        
        # Insert indications
        for ind in drug.indications:
            try:
                text = ind.get("text", "") if isinstance(ind, dict) else str(ind)
                if text:
                    self.sql_client.insert_row("indications", {
                        "drug_id": drug.drug_id,
                        "indication": text[:5000],
                        "source_doc": drug.source_file,
                    })
                    stats["sql_rows"] += 1
            except Exception:
                pass
        
        # Insert dosages
        for dos in drug.dosages:
            try:
                if isinstance(dos, dict):
                    self.sql_client.insert_row("dosages", {
                        "drug_id": drug.drug_id,
                        "form": dos.get("form", "")[:100],
                        "strength": dos.get("strength", "")[:100],
                        "route": dos.get("route", "")[:100],
                        "frequency": dos.get("frequency", "")[:255],
                        "source_doc": drug.source_file,
                    })
                    stats["sql_rows"] += 1
            except Exception:
                pass
        
        # Insert adverse reactions
        for ar in drug.adverse_reactions:
            try:
                if isinstance(ar, dict):
                    reaction = ar.get("reaction") or ar.get("text", "")
                    if reaction:
                        self.sql_client.insert_row("adverse_reactions", {
                            "drug_id": drug.drug_id,
                            "reaction": reaction[:1000],
                            "frequency": ar.get("frequency", "")[:50],
                            "severity": ar.get("severity", "")[:50],
                            "source_doc": drug.source_file,
                        })
                        stats["sql_rows"] += 1
            except Exception:
                pass
        
        # Insert interactions
        for inter in drug.interactions:
            try:
                if isinstance(inter, dict):
                    interacting = inter.get("drug") or inter.get("text", "")
                    if interacting:
                        self.sql_client.insert_row("interactions", {
                            "drug_id": drug.drug_id,
                            "interacting_drug": interacting[:255],
                            "description": inter.get("description", "")[:1000],
                            "source_doc": drug.source_file,
                        })
                        stats["sql_rows"] += 1
            except Exception:
                pass
        
        # Add to vector DB
        try:
            chunks = self._create_text_chunks(drug)
            if chunks:
                self.vector_client.add_documents(
                    documents=[c["text"] for c in chunks],
                    metadatas=[c["metadata"] for c in chunks],
                    ids=[c["id"] for c in chunks],
                )
                stats["vector_chunks"] += len(chunks)
        except Exception as e:
            logger.warning(f"Vector DB error for {drug.drug_id}: {e}")
        
        return stats
    
    def _create_text_chunks(self, drug: DrugInfo) -> list[dict]:
        """Create text chunks for vector DB."""
        chunks = []
        
        # Combine all text content
        text_parts = []
        
        if drug.name:
            text_parts.append(f"Drug: {drug.name}")
        if drug.manufacturer:
            text_parts.append(f"Manufacturer: {drug.manufacturer}")
        
        for ind in drug.indications:
            if isinstance(ind, dict) and ind.get("text"):
                text_parts.append(f"Indications: {ind['text']}")
        
        if drug.warnings:
            text_parts.append(f"Warnings: {drug.warnings}")
        
        # Create chunk(s)
        full_text = "\n\n".join(text_parts)
        if full_text:
            # Simple chunking - split if too long
            max_chunk = 2000
            if len(full_text) <= max_chunk:
                chunks.append({
                    "id": f"{drug.drug_id}_0",
                    "text": full_text,
                    "metadata": {
                        "drug_id": drug.drug_id,
                        "drug_name": drug.name,
                        "source_file": drug.source_file,
                    }
                })
            else:
                # Split into chunks
                for i in range(0, len(full_text), max_chunk - 200):
                    chunk_text = full_text[i:i + max_chunk]
                    chunks.append({
                        "id": f"{drug.drug_id}_{i}",
                        "text": chunk_text,
                        "metadata": {
                            "drug_id": drug.drug_id,
                            "drug_name": drug.name,
                            "source_file": drug.source_file,
                            "chunk_index": i // max_chunk,
                        }
                    })
        
        return chunks


async def run_pipeline(
    source_dir: Optional[Path] = None,
    download: bool = False,
    limit: int = 100,
    clear_db: bool = False,
    workers: int = 4,
):
    """Run the complete DailyMed data pipeline."""
    
    logger.info("=" * 60)
    logger.info("DailyMed Data Pipeline")
    logger.info("=" * 60)
    
    processor = DailyMedProcessor()
    loader = DatabaseLoader(clear_existing=clear_db)
    
    drugs_processed = 0
    total_sql_rows = 0
    total_vector_chunks = 0
    
    if download:
        # Download from API
        logger.info(f"Downloading {limit} drugs from DailyMed API...")
        api = DailyMedAPI()
        
        try:
            drug_list = api.download_drugs(limit=limit)
            logger.info(f"Found {len(drug_list)} drugs")
            
            for item in tqdm(drug_list, desc="Processing drugs"):
                setid = item.get("setid")
                if not setid:
                    continue
                
                # Get SPL XML
                xml_content = api.get_drug_spl(setid)
                if not xml_content:
                    continue
                
                # Parse and load
                drug = processor.parse_spl_content(xml_content, f"api:{setid}")
                if drug:
                    stats = loader.load_drug(drug)
                    drugs_processed += 1
                    total_sql_rows += stats["sql_rows"]
                    total_vector_chunks += stats["vector_chunks"]
                
                time.sleep(0.1)  # Rate limiting
        finally:
            api.close()
    
    elif source_dir:
        # Process local files
        logger.info(f"Processing files from {source_dir}...")
        
        xml_files = list(source_dir.glob("**/*.xml"))[:limit]
        logger.info(f"Found {len(xml_files)} XML files")
        
        for file_path in tqdm(xml_files, desc="Processing files"):
            drug = processor.parse_spl_file(file_path)
            if drug:
                stats = loader.load_drug(drug)
                drugs_processed += 1
                total_sql_rows += stats["sql_rows"]
                total_vector_chunks += stats["vector_chunks"]
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Drugs processed: {drugs_processed}")
    logger.info(f"SQL rows inserted: {total_sql_rows}")
    logger.info(f"Vector chunks created: {total_vector_chunks}")
    
    # Show database status
    logger.info("\nDatabase Status:")
    for table in ["drugs", "indications", "dosages", "adverse_reactions", "interactions"]:
        count = loader.sql_client.get_table_row_count(table)
        logger.info(f"  {table}: {count} rows")
    
    vector_count = loader.vector_client.get_document_count()
    logger.info(f"  vector_db: {vector_count} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="DailyMed Data Pipeline - Download and process pharmaceutical data"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download drugs from DailyMed API",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source directory with XML files to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of drugs to process (default: 100)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear existing database before processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    
    args = parser.parse_args()
    
    if not args.download and not args.source:
        # Default to processing prescription folder
        args.source = "./data/documents/prescription"
    
    asyncio.run(run_pipeline(
        source_dir=Path(args.source) if args.source else None,
        download=args.download,
        limit=args.limit,
        clear_db=args.clear_db,
        workers=args.workers,
    ))


if __name__ == "__main__":
    main()
