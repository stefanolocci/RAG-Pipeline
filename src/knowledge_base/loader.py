"""
Document loader for Open RAG knowledge base.

This module provides functionality to load and manage documents
from the documents directory for use in RAG retrieval.
Supports both TXT and PDF file formats.

Example:
    >>> from open_RAG.src.knowledge_base import DocumentLoader
    >>> loader = DocumentLoader(file_type="txt")
    >>> docs = loader.load_all()
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from open_RAG.src.config import DOCUMENTS_DIR


# Configure logging
logger = logging.getLogger(__name__)


# Type alias for file types
FileType = Literal["txt", "pdf", "both"]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Extracted text content.
        
    Raises:
        ImportError: If pdfplumber is not installed.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF support. "
            "Install with: pip install pdfplumber"
        )
    
    text_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return ""
    
    return "\n".join(text_parts)


class DocumentLoader:
    """
    Loader for documents in the knowledge base.
    
    Provides methods to load text documents (TXT and/or PDF) from the
    documents directory and prepare them for use in retrieval systems.
    
    Attributes:
        documents_dir: Path to the documents directory.
        file_type: Type of files to load ("txt", "pdf", or "both").
    
    Example:
        >>> loader = DocumentLoader(file_type="pdf")
        >>> documents = loader.load_all()
        >>> print(f"Loaded {len(documents)} documents")
    """
    
    def __init__(
        self,
        documents_dir: Optional[Path | str] = None,
        file_type: FileType = "txt",
    ):
        """
        Initialize the document loader.
        
        Args:
            documents_dir: Path to documents directory. Defaults to config.
            file_type: Type of files to load:
                - "txt": Load only .txt files
                - "pdf": Load only .pdf files
                - "both": Load both .txt and .pdf files
        """
        self.documents_dir = Path(documents_dir) if documents_dir else DOCUMENTS_DIR
        self.file_type = file_type
        logger.info(f"Document loader initialized: {self.documents_dir} (type: {file_type})")
    
    def _get_file_patterns(self) -> list[str]:
        """Get file glob patterns based on file_type setting."""
        if self.file_type == "txt":
            return ["*.txt"]
        elif self.file_type == "pdf":
            return ["*.pdf"]
        else:  # "both"
            return ["*.txt", "*.pdf"]
    
    def _read_file(self, filepath: Path) -> str:
        """
        Read content from a file (TXT or PDF).
        
        Args:
            filepath: Path to the file.
            
        Returns:
            File content as string.
        """
        if filepath.suffix.lower() == ".pdf":
            return extract_text_from_pdf(filepath)
        else:
            return filepath.read_text(encoding='utf-8')
    
    def load_all(self) -> list[dict[str, str]]:
        """
        Load all documents from the directory.
        
        Returns:
            List of dictionaries with 'filename', 'content', and 'type' keys.
        """
        documents = []
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            return documents
        
        # Collect files based on file_type
        files = []
        for pattern in self._get_file_patterns():
            files.extend(self.documents_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} files matching {self.file_type}")
        
        for filepath in files:
            try:
                content = self._read_file(filepath).strip()
                if content:
                    documents.append({
                        'filename': filepath.name,
                        'content': content,
                        'type': filepath.suffix.lower().lstrip('.'),
                    })
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_texts(self) -> list[str]:
        """
        Load all document texts (content only).
        
        Returns:
            List of document content strings.
        """
        documents = self.load_all()
        return [doc['content'] for doc in documents]
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the directory.
        
        Returns:
            Number of matching files in the documents directory.
        """
        if not self.documents_dir.exists():
            return 0
        
        count = 0
        for pattern in self._get_file_patterns():
            count += len(list(self.documents_dir.glob(pattern)))
        return count
