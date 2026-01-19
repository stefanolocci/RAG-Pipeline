"""
OpenRAG - RAG Generator using HuggingFace open-source models.

This module provides RAG generation using open-source LLMs like Gemma, Llama, Mistral.

Example:
    >>> from open_RAG.src.retrievers import FAISSRetriever
    >>> from open_RAG.src.generators import OpenRAG
    >>> retriever = FAISSRetriever(documents_dir="./documents", file_extension="pdf")
    >>> rag = OpenRAG(retriever, model_name="meta-llama/Llama-3.2-1B")
    >>> result = rag.query("What is prompt injection?")
"""

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from open_RAG.src.retrievers.base import BaseRetriever, create_prompt
from open_RAG.src.config import (
    DEFAULT_LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    REPETITION_PENALTY,
    get_hf_token,
)


logger = logging.getLogger(__name__)


class OpenRAG:
    """
    RAG Generator using HuggingFace open-source models.
    
    Combines a retriever with a HuggingFace LLM for RAG-based generation.
    
    Attributes:
        retriever: Document retriever instance (FAISS or BM25).
        model_name: HuggingFace model name.
        tokenizer: Model tokenizer.
        model: The language model.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        model_name: str = DEFAULT_LLM_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        device: Optional[str] = None,
    ):
        """
        Initialize OpenRAG.
        
        Args:
            retriever: A retriever instance (FAISSRetriever or BM25Retriever).
            model_name: HuggingFace model name.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            device: Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Auto-detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load LLM
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the HuggingFace model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        hf_token = get_hf_token()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Original model loading pattern that worked before refactoring
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Move to device (for non-CUDA where device_map isn't used)
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the loaded LLM.
        
        Args:
            prompt: The input prompt with context.
            
        Returns:
            Generated text response.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def query(self, question: str, k: Optional[int] = None) -> dict:
        """
        Perform RAG: retrieve relevant docs and generate response.
        
        Args:
            question: The user's question.
            k: Number of documents to retrieve.
            
        Returns:
            Dictionary with 'response', 'retrieved_docs', 'sources', and 'prompt'.
        """
        # Retrieve context
        context = self.retriever.get_context(question, k)
        
        # Create prompt
        prompt = create_prompt(question, context["documents"])
        
        # Generate response
        response = self.generate(prompt)
        
        return {
            "response": response,
            "retrieved_docs": context["documents"],
            "sources": context["sources"],
            "prompt": prompt,
        }
    
    def unload_model(self) -> None:
        """Unload the model and clear memory."""
        logger.info("Unloading model...")
        
        # Delete model and tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear MPS cache if available (macOS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except AttributeError:
                # Some older pytorch versions might not have empty_cache for mps
                pass
                
        logger.info("Model unloaded")
