#!/usr/bin/env python3
"""
CLI script to bulk-ingest PDFs/TXTs from data/ directory and build FAISS index with mxbai-embed-large-v1.
Usage: python ingest.py
"""

import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class MxbaiEmbeddings:
    """Optimized embeddings wrapper using mxbai-embed-large-v1."""
    
    def __init__(self):
        print("Loading mxbai-embed-large-v1 model...")
        self.model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        self.model.max_seq_length = 512  # Optimize for performance
        print("Model loaded successfully!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching."""
        # Process in smaller batches for memory efficiency
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            all_embeddings.extend(batch_embeddings.tolist())
            
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0].tolist()


def load_documents_from_directory(data_dir: str):
    """Load all PDF and TXT files from the data directory."""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return documents
    
    # Load PDF files
    pdf_files = list(data_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")
    
    # Load TXT files
    txt_files = list(data_path.glob("*.txt"))
    for txt_file in txt_files:
        print(f"Loading TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file.name
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
    
    return documents


def main():
    """Main function to process documents and build FAISS index."""
    # Load environment variables
    load_dotenv()
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir}/ directory.")
        print("Please add your PDF and TXT files to the data/ directory and run this script again.")
        return
    
    # Check if data directory has files
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    txt_files = list(data_path.glob("*.txt"))
    
    if not pdf_files and not txt_files:
        print("No PDF or TXT files found in data/ directory.")
        print("Please add your files to the data/ directory and run this script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files and {len(txt_files)} TXT files.")
    
    # Load documents
    print("\nLoading documents...")
    documents = load_documents_from_directory(data_dir)
    
    if not documents:
        print("No documents could be loaded. Please check your files.")
        return
    
    print(f"Loaded {len(documents)} document chunks.")
    
    # Split documents
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\nFile:", "\n\n", "\n", " ", ""]
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")
    
    # Remove existing index if it exists
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        print(f"\nRemoving existing {index_dir}/ directory...")
        shutil.rmtree(index_dir)
    
    # Create embeddings
    print("\nInitializing mxbai-embed-large-v1 embeddings...")
    try:
        embeddings = MxbaiEmbeddings()
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return
    
    # Build FAISS index with optimized processing
    print("Building FAISS index with mxbai-embed-large-v1...")
    batch_size = 32  # Larger batch size for local processing
    
    try:
        # Process in batches for memory efficiency
        vectorstore = None
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            if vectorstore is None:
                # Create initial index
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Add to existing index
                batch_store = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_store)
        
        # Save the index
        print(f"\nSaving FAISS index to {index_dir}/...")
        vectorstore.save_local(index_dir)
        
        # Save backend information
        with open(os.path.join(index_dir, "backend.txt"), "w") as f:
            f.write("mxbai")
        
        print(f"✅ Successfully built FAISS index with {len(texts)} documents using mxbai-embed-large-v1!")
        print(f"Index saved to {index_dir}/")
        print("You can now run 'streamlit run main.py' to use the app.")
        
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return


if __name__ == "__main__":
    main()