#!/usr/bin/env python3
"""
Open RAG - Main Entry Point

This module provides a command-line interface for running RAG queries
using either open-source LLMs (HuggingFace) or OpenAI's API.

Usage:
    # HuggingFace model with FAISS
    python main.py --method faiss --generator huggingface "Your question"
    
    # OpenAI with BM25
    python main.py --method bm25 --generator openai "Your question"

Example:
    $ python main.py --method faiss -e pdf "What are the main prompt injection techniques?"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Open RAG - Query documents using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The question or query to ask"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["faiss", "bm25"],
        default="faiss",
        help="Retrieval method (default: faiss)"
    )
    
    parser.add_argument(
        "--generator", "-g",
        choices=["huggingface", "openai"],
        default="huggingface",
        help="Generator to use: huggingface (open-source) or openai (default: huggingface)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. For HuggingFace: google/gemma-2b-it, meta-llama/Llama-3.2-1B. For OpenAI: gpt-4o-mini, gpt-4"
    )
    
    parser.add_argument(
        "--documents", "-d",
        type=str,
        default=None,
        help="Path to documents directory"
    )
    
    parser.add_argument(
        "--extension", "-e",
        type=str,
        default="txt",
        help="File extension for documents without dot (default: txt)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use (default: auto). Use 'cpu' if you get segfaults on MPS"
    )
    
    args = parser.parse_args()
    
    # Fix for segmentation fault on macOS/Linux with HuggingFace
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # CRITICAL: Import torch before any other heavy libraries (like faiss)
    # This prevents OpenMP runtime conflicts that cause segfaults
    import torch
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    rag = None
    
    try:
        # Create retriever
        if args.method == "faiss":
            from open_RAG.src.retrievers.faiss_retriever import FAISSRetriever
            retriever = FAISSRetriever(
                documents_dir=args.documents,
                file_extension=args.extension,
                top_k=args.top_k,
            )
        else:
            from open_RAG.src.retrievers.bm25_retriever import BM25Retriever
            retriever = BM25Retriever(
                documents_dir=args.documents,
                file_extension=args.extension,
                top_k=args.top_k,
            )
        
        # Create generator
        if args.generator == "openai":
            from open_RAG.src.generators import GPT_RAG
            model = args.model or "gpt-4o-mini"
            rag = GPT_RAG(
                retriever=retriever,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            from open_RAG.src.generators import OpenRAG
            model = args.model or "google/gemma-2b-it"
            device = None if args.device == "auto" else args.device
            rag = OpenRAG(
                retriever=retriever,
                model_name=model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )
        
        logger.info(f"Running {args.method.upper()} + {args.generator.upper()} with model: {model}")
        
        # Run query
        result = rag.query(args.query, k=args.top_k)
        
        # Print results
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(result['response'])
        
        # Show retrieved documents section
        print("\n" + "="*60)
        print(f"RETRIEVED PASSAGES ({len(result['retrieved_docs'])}):")
        print("="*60)
        
        if result['retrieved_docs']:
            unique_sources = list(dict.fromkeys(result.get('sources', [])))
            print(f"\nUnique source files: {', '.join(unique_sources)}")
            print("-"*60)
            
            for i, (doc, source) in enumerate(zip(result['retrieved_docs'], result.get('sources', ['unknown'] * len(result['retrieved_docs']))), 1):
                print(f"\n--- Chunk {i} (from: {source}) ---")
                print(doc)
        else:
            print("\nNo documents retrieved.")
        
        print("\n" + "="*60)

    finally:
        # Explicit cleanup to prevent memory leaks/semaphores issues
        if rag and hasattr(rag, 'unload_model'):
            logger.info("Unloading model and cleaning up resources...")
            rag.unload_model()


if __name__ == "__main__":
    main()
