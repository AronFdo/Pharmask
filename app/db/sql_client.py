"""SQLite client for structured table storage and retrieval."""

from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, Float, Integer, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from typing import Optional, Any
from pathlib import Path

from app.config import settings


class SQLClient:
    """Client for SQLite database operations."""
    
    _instance: Optional["SQLClient"] = None
    _engine = None
    _session_factory = None
    _metadata: Optional[MetaData] = None
    
    def __new__(cls):
        """Singleton pattern for SQL client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize SQLite client."""
        if self._engine is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the SQLite engine and session factory."""
        # Ensure database directory exists
        db_path = Path(settings.sqlite_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=settings.debug,
        )
        
        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)
        
        # Initialize metadata
        self._metadata = MetaData()
        
        # Create core tables
        self._create_tables()
    
    def _create_tables(self):
        """Create the core pharmaceutical tables."""
        # Drug information table
        Table(
            "drugs",
            self._metadata,
            Column("id", String(100), primary_key=True),
            Column("name", String(255), nullable=False),
            Column("generic_name", String(255)),
            Column("brand_name", String(255)),
            Column("manufacturer", String(255)),
            Column("ndc_code", String(50)),
            Column("description", Text),
            Column("source_doc", String(255)),
        )
        
        # Drug indications table
        Table(
            "indications",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("drug_id", String(100), nullable=False),
            Column("indication", Text, nullable=False),
            Column("source_doc", String(255)),
        )
        
        # Drug dosages table
        Table(
            "dosages",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("drug_id", String(100), nullable=False),
            Column("form", String(100)),
            Column("strength", String(100)),
            Column("route", String(100)),
            Column("frequency", String(255)),
            Column("source_doc", String(255)),
        )
        
        # Adverse reactions table
        Table(
            "adverse_reactions",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("drug_id", String(100), nullable=False),
            Column("reaction", Text, nullable=False),
            Column("frequency", String(50)),
            Column("severity", String(50)),
            Column("source_doc", String(255)),
        )
        
        # Drug interactions table
        Table(
            "interactions",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("drug_id", String(100), nullable=False),
            Column("interacting_drug", String(255), nullable=False),
            Column("interaction_type", String(100)),
            Column("description", Text),
            Column("severity", String(50)),
            Column("source_doc", String(255)),
        )
        
        # Create all tables
        self._metadata.create_all(self._engine)
    
    @property
    def engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine
    
    def get_session(self):
        """Get a new database session."""
        return self._session_factory()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """
        Execute a SQL query and return results as list of dicts.
        
        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            List of result rows as dictionaries
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
    
    def insert_row(self, table_name: str, data: dict) -> None:
        """
        Insert a row into a table.
        
        Args:
            table_name: Name of the table
            data: Dictionary of column: value pairs
        """
        table = self._metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' does not exist")
        
        with self.get_session() as session:
            session.execute(table.insert().values(**data))
            session.commit()
    
    def insert_rows(self, table_name: str, rows: list[dict]) -> int:
        """
        Insert multiple rows into a table.
        
        Args:
            table_name: Name of the table
            rows: List of dictionaries with column: value pairs
            
        Returns:
            Number of rows inserted
        """
        if not rows:
            return 0
            
        table = self._metadata.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' does not exist")
        
        with self.get_session() as session:
            session.execute(table.insert(), rows)
            session.commit()
        
        return len(rows)
    
    def get_table_schema(self, table_name: str) -> list[dict]:
        """
        Get the schema of a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column info dictionaries
        """
        inspector = inspect(self._engine)
        columns = inspector.get_columns(table_name)
        return [{"name": c["name"], "type": str(c["type"])} for c in columns]
    
    def get_all_tables(self) -> list[str]:
        """Get list of all table names."""
        inspector = inspect(self._engine)
        return inspector.get_table_names()
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        result = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
        return result[0]["count"] if result else 0
    
    def clear_table(self, table_name: str) -> None:
        """Delete all rows from a table."""
        with self.get_session() as session:
            session.execute(text(f"DELETE FROM {table_name}"))
            session.commit()
