"""Ingestion worker for processing pharmaceutical documents."""

from pathlib import Path
from typing import Optional
import logging
import hashlib
from datetime import datetime

from app.db import VectorClient, SQLClient
from app.models import IngestionResult
from .xml_parser import XMLParser, ParsedDocument
from .pdf_parser import PDFParser
from .json_parser import JSONParser
from .chunker import TextChunker

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Worker for ingesting pharmaceutical documents into Vector and SQL databases."""
    
    def __init__(self):
        """Initialize the ingestion worker."""
        self.xml_parser = XMLParser()
        self.pdf_parser = PDFParser()
        self.json_parser = JSONParser()
        self.chunker = TextChunker()
        self.vector_client = VectorClient()
        self.sql_client = SQLClient()
    
    async def ingest_directory(self, directory: Path) -> IngestionResult:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            IngestionResult with processing statistics
        """
        result = IngestionResult()
        
        if not directory.exists():
            result.errors.append(f"Directory does not exist: {directory}")
            return result
        
        # Find all supported files
        xml_files = list(directory.glob("**/*.xml"))
        pdf_files = list(directory.glob("**/*.pdf"))
        json_files = list(directory.glob("**/*.json"))
        
        all_files = xml_files + pdf_files + json_files
        
        if not all_files:
            result.errors.append(f"No XML, PDF, or JSON files found in {directory}")
            return result
        
        logger.info(f"Found {len(xml_files)} XML, {len(pdf_files)} PDF, {len(json_files)} JSON files")
        
        # Process each file
        for file_path in all_files:
            try:
                doc = await self._process_file(file_path)
                if doc:
                    chunks_created, tables_extracted = await self._load_document(doc)
                    result.documents_processed += 1
                    result.text_chunks_created += chunks_created
                    result.tables_extracted += tables_extracted
                else:
                    result.errors.append(f"Failed to parse: {file_path}")
            except Exception as e:
                result.errors.append(f"Error processing {file_path}: {str(e)}")
                logger.exception(f"Error processing {file_path}")
        
        result.timestamp = datetime.now()
        return result
    
    async def ingest_file(self, file_path: Path) -> IngestionResult:
        """
        Ingest a single document file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            IngestionResult with processing statistics
        """
        result = IngestionResult()
        
        if not file_path.exists():
            result.errors.append(f"File does not exist: {file_path}")
            return result
        
        try:
            doc = await self._process_file(file_path)
            if doc:
                chunks_created, tables_extracted = await self._load_document(doc)
                result.documents_processed = 1
                result.text_chunks_created = chunks_created
                result.tables_extracted = tables_extracted
            else:
                result.errors.append(f"Failed to parse: {file_path}")
        except Exception as e:
            result.errors.append(f"Error processing {file_path}: {str(e)}")
            logger.exception(f"Error processing {file_path}")
        
        result.timestamp = datetime.now()
        return result
    
    async def _process_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a file based on its extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".xml":
            return self.xml_parser.parse_file(file_path)
        elif suffix == ".pdf":
            return self.pdf_parser.parse_file(file_path)
        elif suffix == ".json":
            return self.json_parser.parse_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return None
    
    async def _load_document(self, doc: ParsedDocument) -> tuple[int, int]:
        """
        Load a parsed document into Vector and SQL databases.
        
        Args:
            doc: Parsed document
            
        Returns:
            Tuple of (chunks_created, tables_extracted)
        """
        chunks_created = 0
        tables_extracted = 0
        
        # Load text chunks to Vector DB
        chunks_created = await self._load_text_chunks(doc)
        
        # Load tables to SQL DB
        tables_extracted = await self._load_tables(doc)
        
        return chunks_created, tables_extracted
    
    async def _load_text_chunks(self, doc: ParsedDocument) -> int:
        """Load text chunks from document into Vector DB."""
        documents = []
        metadatas = []
        ids = []
        
        # Add abstract if present
        if doc.abstract:
            chunks = list(self.chunker.chunk_text(
                doc.abstract,
                source_doc=doc.doc_id,
                section="Abstract"
            ))
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(doc.doc_id, "abstract", chunk.chunk_index)
                documents.append(chunk.text)
                metadatas.append({
                    "source_doc": doc.doc_id,
                    "title": doc.title,
                    "section": "Abstract",
                    "chunk_index": chunk.chunk_index,
                    "source_file": doc.source_file,
                })
                ids.append(chunk_id)
        
        # Add sections
        for section in doc.sections:
            chunks = list(self.chunker.chunk_text(
                section["text"],
                source_doc=doc.doc_id,
                section=section["title"]
            ))
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(
                    doc.doc_id, 
                    section["title"], 
                    chunk.chunk_index
                )
                documents.append(chunk.text)
                metadatas.append({
                    "source_doc": doc.doc_id,
                    "title": doc.title,
                    "section": section["title"],
                    "chunk_index": chunk.chunk_index,
                    "source_file": doc.source_file,
                })
                ids.append(chunk_id)
        
        # Load to Vector DB
        if documents:
            self.vector_client.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        
        return len(documents)
    
    async def _load_tables(self, doc: ParsedDocument) -> int:
        """Load tables from document into SQL DB."""
        tables_loaded = 0
        
        for table in doc.tables:
            try:
                loaded = await self._classify_and_load_table(table, doc)
                if loaded:
                    tables_loaded += 1
            except Exception as e:
                logger.error(f"Error loading table {table.get('id')}: {e}")
        
        return tables_loaded
    
    async def _classify_and_load_table(self, table: dict, doc: ParsedDocument) -> bool:
        """
        Classify a table and load it into the appropriate SQL table.
        
        Uses header keywords to determine table type.
        """
        headers = [h.lower() for h in table.get("headers", [])]
        rows = table.get("rows", [])
        caption = table.get("caption", "").lower()
        section = table.get("section", "").lower()
        
        if not rows:
            return False
        
        # Try to classify based on headers and context
        if any(kw in " ".join(headers) for kw in ["dose", "dosage", "strength", "form", "route"]):
            return self._load_dosage_table(table, doc)
        elif any(kw in " ".join(headers) for kw in ["adverse", "reaction", "side effect", "event"]):
            return self._load_adverse_reactions_table(table, doc)
        elif any(kw in " ".join(headers) for kw in ["interaction", "drug", "contraindicated"]):
            return self._load_interactions_table(table, doc)
        elif any(kw in section for kw in ["indication", "use"]):
            return self._load_indications_from_table(table, doc)
        elif any(kw in section for kw in ["adverse", "reaction"]):
            return self._load_adverse_reactions_table(table, doc)
        elif any(kw in section for kw in ["dosage", "administration"]):
            return self._load_dosage_table(table, doc)
        else:
            # Generic table - store as text in vector DB for now
            return False
    
    def _load_dosage_table(self, table: dict, doc: ParsedDocument) -> bool:
        """Load a dosage table into the dosages SQL table."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        # Map headers to columns
        header_map = {}
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if "form" in h_lower or "formulation" in h_lower:
                header_map["form"] = i
            elif "strength" in h_lower or "dose" in h_lower or "concentration" in h_lower:
                header_map["strength"] = i
            elif "route" in h_lower:
                header_map["route"] = i
            elif "frequency" in h_lower or "schedule" in h_lower:
                header_map["frequency"] = i
        
        dosage_rows = []
        for row in rows:
            dosage = {
                "drug_id": doc.doc_id,
                "source_doc": doc.source_file,
            }
            for col_name, idx in header_map.items():
                if idx < len(row):
                    dosage[col_name] = row[idx]
            
            # If no mapped columns, use first columns as form/strength
            if not header_map and len(row) >= 2:
                dosage["form"] = row[0]
                dosage["strength"] = row[1]
            
            if dosage.get("form") or dosage.get("strength"):
                dosage_rows.append(dosage)
        
        if dosage_rows:
            self.sql_client.insert_rows("dosages", dosage_rows)
            return True
        return False
    
    def _load_adverse_reactions_table(self, table: dict, doc: ParsedDocument) -> bool:
        """Load an adverse reactions table into the adverse_reactions SQL table."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        # Map headers to columns
        header_map = {}
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if "reaction" in h_lower or "event" in h_lower or "effect" in h_lower:
                header_map["reaction"] = i
            elif "frequency" in h_lower or "incidence" in h_lower or "%" in h:
                header_map["frequency"] = i
            elif "severity" in h_lower or "serious" in h_lower:
                header_map["severity"] = i
        
        reaction_rows = []
        for row in rows:
            reaction = {
                "drug_id": doc.doc_id,
                "source_doc": doc.source_file,
            }
            
            if "reaction" in header_map:
                idx = header_map["reaction"]
                if idx < len(row):
                    reaction["reaction"] = row[idx]
            elif row:
                # First column as reaction
                reaction["reaction"] = row[0]
            
            for col_name in ["frequency", "severity"]:
                if col_name in header_map:
                    idx = header_map[col_name]
                    if idx < len(row):
                        reaction[col_name] = row[idx]
            
            if reaction.get("reaction"):
                reaction_rows.append(reaction)
        
        if reaction_rows:
            self.sql_client.insert_rows("adverse_reactions", reaction_rows)
            return True
        return False
    
    def _load_interactions_table(self, table: dict, doc: ParsedDocument) -> bool:
        """Load a drug interactions table into the interactions SQL table."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        # Map headers
        header_map = {}
        for i, h in enumerate(headers):
            h_lower = h.lower()
            if "drug" in h_lower or "medication" in h_lower:
                header_map["interacting_drug"] = i
            elif "type" in h_lower or "mechanism" in h_lower:
                header_map["interaction_type"] = i
            elif "description" in h_lower or "effect" in h_lower or "clinical" in h_lower:
                header_map["description"] = i
            elif "severity" in h_lower:
                header_map["severity"] = i
        
        interaction_rows = []
        for row in rows:
            interaction = {
                "drug_id": doc.doc_id,
                "source_doc": doc.source_file,
            }
            
            if "interacting_drug" in header_map:
                idx = header_map["interacting_drug"]
                if idx < len(row):
                    interaction["interacting_drug"] = row[idx]
            elif row:
                interaction["interacting_drug"] = row[0]
            
            for col_name in ["interaction_type", "description", "severity"]:
                if col_name in header_map:
                    idx = header_map[col_name]
                    if idx < len(row):
                        interaction[col_name] = row[idx]
            
            if interaction.get("interacting_drug"):
                interaction_rows.append(interaction)
        
        if interaction_rows:
            self.sql_client.insert_rows("interactions", interaction_rows)
            return True
        return False
    
    def _load_indications_from_table(self, table: dict, doc: ParsedDocument) -> bool:
        """Load indications from a table into the indications SQL table."""
        rows = table.get("rows", [])
        
        indication_rows = []
        for row in rows:
            if row and row[0]:
                indication_rows.append({
                    "drug_id": doc.doc_id,
                    "indication": " ".join(str(cell) for cell in row if cell),
                    "source_doc": doc.source_file,
                })
        
        if indication_rows:
            self.sql_client.insert_rows("indications", indication_rows)
            return True
        return False
    
    def _generate_chunk_id(self, doc_id: str, section: str, chunk_index: int) -> str:
        """Generate a unique ID for a text chunk."""
        content = f"{doc_id}:{section}:{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
