"""Text chunking utilities for document processing."""

from typing import Generator
from dataclasses import dataclass

from app.config import settings


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    chunk_index: int
    source_doc: str
    section: str = ""
    start_char: int = 0
    end_char: int = 0


class TextChunker:
    """Chunker for splitting text into overlapping chunks."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        source_doc: str,
        section: str = "",
    ) -> Generator[TextChunk, None, None]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            source_doc: Source document identifier
            section: Section name within the document
            
        Yields:
            TextChunk objects
        """
        if not text or not text.strip():
            return
        
        # Clean the text
        text = text.strip()
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # Text is small enough to be a single chunk
            yield TextChunk(
                text=text,
                chunk_index=0,
                source_doc=source_doc,
                section=section,
                start_char=0,
                end_char=text_length,
            )
            return
        
        # Split into overlapping chunks
        chunk_index = 0
        start = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at a sentence or word boundary
            if end < text_length:
                # Look for sentence boundary (. ! ?)
                boundary = self._find_boundary(text, start, end)
                if boundary > start:
                    end = boundary
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                yield TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    source_doc=source_doc,
                    section=section,
                    start_char=start,
                    end_char=end,
                )
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break
            
            # Ensure we make progress
            if start <= 0 or (end >= text_length and chunk_text):
                break
    
    def _find_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find a good boundary point (sentence or word) near the end position.
        
        Args:
            text: Full text
            start: Start position of current chunk
            end: Proposed end position
            
        Returns:
            Adjusted end position at a boundary
        """
        # Look for sentence boundaries in the last 20% of the chunk
        search_start = max(start, end - int(self.chunk_size * 0.2))
        
        # Look for sentence ending punctuation
        for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            pos = text.rfind(punct, search_start, end)
            if pos != -1:
                return pos + 1  # Include the punctuation
        
        # Fall back to word boundary (space)
        space_pos = text.rfind(' ', search_start, end)
        if space_pos != -1:
            return space_pos
        
        # No good boundary found, use original end
        return end
    
    def chunk_sections(
        self,
        sections: list[dict],
        source_doc: str,
    ) -> Generator[TextChunk, None, None]:
        """
        Chunk multiple sections from a document.
        
        Args:
            sections: List of dicts with 'title' and 'text' keys
            source_doc: Source document identifier
            
        Yields:
            TextChunk objects from all sections
        """
        for section in sections:
            title = section.get("title", "")
            text = section.get("text", "")
            
            yield from self.chunk_text(text, source_doc, section=title)
