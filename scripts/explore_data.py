"""Explore the database contents to understand what data we have."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pharma.db"

def explore_sql():
    """Explore SQL database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("=== TABLES ===")
    for t in tables:
        print(t[0])
    
    print("\n=== SAMPLE DATA FROM EACH TABLE ===")
    for table in ['drugs', 'dosages', 'adverse_reactions', 'interactions', 'indications']:
        try:
            cursor.execute(f'SELECT * FROM {table} LIMIT 5')
            rows = cursor.fetchall()
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f"\n--- {table} ({count} rows) ---")
            cursor.execute(f'PRAGMA table_info({table})')
            cols = [c[1] for c in cursor.fetchall()]
            print(f"Columns: {cols}")
            for row in rows:
                print(row)
        except Exception as e:
            print(f"{table}: {e}")
    
    # Get unique drug names
    print("\n=== UNIQUE DRUGS ===")
    cursor.execute("SELECT DISTINCT name FROM drugs LIMIT 30")
    drugs = cursor.fetchall()
    for d in drugs:
        print(f"  - {d[0]}")
    
    conn.close()

def explore_vector():
    """Explore vector database."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        chroma_path = PROJECT_ROOT / "data" / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("pharmaceutical_texts")
        
        count = collection.count()
        print(f"\n=== VECTOR DB ===")
        print(f"Total documents: {count}")
        
        # Sample some documents
        results = collection.get(limit=10, include=['documents', 'metadatas'])
        print("\n=== SAMPLE DOCUMENTS ===")
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\n--- Document {i+1} ---")
            print(f"Metadata: {meta}")
            print(f"Text (first 200 chars): {doc[:200]}...")
    except Exception as e:
        print(f"Vector DB error: {e}")

if __name__ == "__main__":
    explore_sql()
    explore_vector()
