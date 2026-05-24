import os
from typing import List, Dict, Any
from pathlib import Path

# Try to import chromadb
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

class LocalADAVectorStore:
    """
    A lightweight, modular Vector Database for ADA Standards of Care guidelines.
    Uses ChromaDB to query a persistent vector store populated from the actual PDF.
    """
    def __init__(self):
        self.use_fallback = not (HAS_CHROMADB)
        self.chroma_client = None
        self.collection = None
        
        # Paths
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.db_path = self.root_dir / "data" / "vector_db" / "ada_standards_chroma"
        
        if not self.use_fallback:
            try:
                if not self.db_path.exists():
                    print(f"[VectorDB] Warning: DB path {self.db_path} does not exist. Please run ingest_ada.py first.")
                    self.use_fallback = True
                else:
                    self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
                    # Check if collection exists
                    collections = [c.name for c in self.chroma_client.list_collections()]
                    if "ada_standards_of_care" in collections:
                        self.collection = self.chroma_client.get_collection(name="ada_standards_of_care")
                        print("[VectorDB] ChromaDB initialized successfully from disk.")
                    else:
                        print("[VectorDB] Collection not found. Please run ingest_ada.py first.")
                        self.use_fallback = True
            except Exception as e:
                print(f"[VectorDB] ChromaDB init warning: {e}. Ensure you have run ingest_ada.py. Falling back.")
                self.use_fallback = True
                
        if self.use_fallback:
            print("[VectorDB] Pure Python fallback initialized (will return empty results).")

    def query(self, query_text: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """Queries the vector database for matching ADA Guidelines."""
        if not self.use_fallback and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                formatted = []
                if results and "documents" in results and results["documents"] and results["documents"][0]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    ids = results["ids"][0]
                    for i in range(len(docs)):
                        formatted.append({
                            "id": ids[i],
                            "section": f"Page {metas[i].get('page', 'Unknown')}",
                            "text": docs[i]
                        })
                return formatted
            except Exception as e:
                print(f"[VectorDB] Query error: {e}")
                pass
                
        # Fallback return empty list if DB is not populated
        return []
