"""
Test script to verify XML parsing and table extraction.

Usage:
    python scripts/test_ingestion.py [--file PATH] [--dir PATH] [--limit N]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ingestion.xml_parser import XMLParser
from app.services.ingestion.json_parser import JSONParser
from app.db import SQLClient

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_file(file_path: Path):
    """Test parsing a single file."""
    print(f"\n{'='*60}")
    print(f"Testing: {file_path.name}")
    print('='*60)
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".xml":
        parser = XMLParser()
        doc = parser.parse_file(file_path)
    elif suffix == ".json":
        parser = JSONParser()
        doc = parser.parse_file(file_path)
    else:
        print(f"Unsupported file type: {suffix}")
        return None
    
    if doc is None:
        print("FAILED: Could not parse document")
        return None
    
    print(f"\nDocument ID: {doc.doc_id}")
    print(f"Title: {doc.title[:100]}..." if len(doc.title) > 100 else f"Title: {doc.title}")
    print(f"Sections: {len(doc.sections)}")
    for i, sec in enumerate(doc.sections[:5]):
        print(f"  [{i+1}] {sec['title'][:50]}: {len(sec['text'])} chars")
    if len(doc.sections) > 5:
        print(f"  ... and {len(doc.sections) - 5} more sections")
    
    print(f"\nTables: {len(doc.tables)}")
    for i, table in enumerate(doc.tables[:3]):
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        section = table.get('section', 'Unknown')
        print(f"  [{i+1}] Section: {section[:30]}")
        print(f"      Headers: {headers[:5]}...")
        print(f"      Rows: {len(rows)}")
    if len(doc.tables) > 3:
        print(f"  ... and {len(doc.tables) - 3} more tables")
    
    print(f"\nMetadata: {doc.metadata}")
    
    return doc


def test_directory(dir_path: Path, limit: int = 5):
    """Test parsing files from a directory."""
    xml_files = list(dir_path.glob("**/*.xml"))[:limit]
    json_files = list(dir_path.glob("**/*.json"))[:limit]
    
    all_files = xml_files + json_files
    
    print(f"Found {len(xml_files)} XML files, {len(json_files)} JSON files")
    print(f"Testing first {len(all_files)} files...\n")
    
    results = {
        "success": 0,
        "failed": 0,
        "total_sections": 0,
        "total_tables": 0,
    }
    
    for file_path in all_files:
        doc = test_single_file(file_path)
        if doc:
            results["success"] += 1
            results["total_sections"] += len(doc.sections)
            results["total_tables"] += len(doc.tables)
        else:
            results["failed"] += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Files parsed: {results['success']}")
    print(f"Files failed: {results['failed']}")
    print(f"Total sections: {results['total_sections']}")
    print(f"Total tables: {results['total_tables']}")
    
    return results


def check_sql_tables():
    """Check SQL database tables."""
    print(f"\n{'='*60}")
    print("SQL DATABASE STATUS")
    print('='*60)
    
    try:
        client = SQLClient()
        tables = client.get_all_tables()
        
        print(f"Tables: {tables}")
        for table in tables:
            count = client.get_table_row_count(table)
            print(f"  {table}: {count} rows")
            
            # Show sample data
            if count > 0:
                sample = client.execute_query(f"SELECT * FROM {table} LIMIT 2")
                for row in sample:
                    print(f"    Sample: {dict(list(row.items())[:3])}...")
    except Exception as e:
        print(f"Error accessing SQL database: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test XML parsing and table extraction")
    parser.add_argument("--file", type=str, help="Test a single file")
    parser.add_argument("--dir", type=str, default="./data/documents", help="Test files from directory")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to test")
    parser.add_argument("--check-sql", action="store_true", help="Check SQL database status")
    
    args = parser.parse_args()
    
    if args.file:
        test_single_file(Path(args.file))
    else:
        test_directory(Path(args.dir), args.limit)
    
    if args.check_sql:
        check_sql_tables()


if __name__ == "__main__":
    main()
