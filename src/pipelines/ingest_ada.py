import os
import re
import pdfplumber
import chromadb
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT_DIR / "data" / "raw" / "standards-of-care-2026.pdf"
DB_PATH = ROOT_DIR / "data" / "vector_db" / "ada_standards_chroma"

def clean_text(text: str) -> str:
    # Basic text cleanup
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, page_num: int, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Chunks text by roughly size."""
    # Split by periods for sentence-like chunking
    sentences = re.split(r'(?<=\.)\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({"text": current_chunk.strip(), "page": page_num})
            # Overlap: keep the last sentence for context
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
            
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "page": page_num})
        
    return chunks

def ingest_pdf():
    print(f"Loading PDF from: {PDF_PATH}")
    if not PDF_PATH.exists():
        print("Error: PDF file not found!")
        return

    all_chunks = []
    
    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages found: {total_pages}. Starting extraction...")
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text = clean_text(text)
                page_chunks = chunk_text(text, page_num=i+1)
                all_chunks.extend(page_chunks)
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{total_pages} pages...")
                
    print(f"Extraction complete. Total chunks generated: {len(all_chunks)}")
    
    # Initialize Persistent ChromaDB
    print(f"Initializing ChromaDB at {DB_PATH}...")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    # Reset collection if exists to avoid duplicates during re-runs
    try:
        client.delete_collection("ada_standards_of_care")
    except Exception:
        pass
        
    collection = client.create_collection(name="ada_standards_of_care")
    
    print("Adding chunks to Vector Store (this may take a minute depending on the embedding model)...")
    
    # Prepare batch insertion
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"page": c["page"], "source": "standards-of-care-2026.pdf"} for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    # Chroma handles batching, but let's chunk it to avoid max payload issues
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Inserted {end}/{len(documents)} chunks...")
        
    print("Ingestion complete. Vector DB is ready!")

if __name__ == "__main__":
    ingest_pdf()
