"""
Batch document ingestion (small groups) for quicker rebuilds.

This script is meant for rebuilding `data/pharma_dailymed_p1.db` and
`data/chroma_dailymed_p1` in manageable chunks, and supports XML/PDF/JSON.

Example:
  # Rebuild from scratch (first 100 files)
  python scripts/batch_ingest_dailymed.py --clear --batch-size 100 --offset 0

  # Next batch (next 100 files)
  python scripts/batch_ingest_dailymed.py --batch-size 100 --offset 100
"""

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path


def _normalize_exts(raw_exts: list[str]) -> list[str]:
    exts: list[str] = []
    for ext in raw_exts:
        normalized = ext.strip().lower()
        if not normalized:
            continue
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        exts.append(normalized)
    return sorted(set(exts))


def _collect_files(source_dir: Path, recursive: bool, file_exts: list[str]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for p in source_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in file_exts:
            files.append(p.resolve())
    return sorted(files)


def _set_env_if_provided(args: argparse.Namespace) -> None:
    # Important: config/settings are instantiated at import-time.
    if args.chroma_persist_dir:
        os.environ["CHROMA_PERSIST_DIR"] = args.chroma_persist_dir
    if args.sqlite_db_path:
        os.environ["SQLITE_DB_PATH"] = args.sqlite_db_path
    if args.debug is not None:
        os.environ["DEBUG"] = "true" if args.debug else "false"


async def _ingest_batch(
    *,
    source_dir: Path,
    batch_size: int,
    offset: int,
    clear: bool,
    manifest_path: Path | None,
    recursive: bool,
    chroma_clear_method: str,
    file_exts: list[str],
) -> None:
    from app.db import SQLClient, VectorClient
    from app.services.ingestion import IngestionWorker

    # Clear only when offset==0 to keep later runs additive/idempotent.
    if clear and offset == 0:
        sql_client = SQLClient()
        for table in ["drugs", "indications", "dosages", "adverse_reactions", "interactions"]:
            try:
                sql_client.clear_table(table)
            except Exception:
                pass

        if chroma_clear_method == "wipe-persist-dir":
            # Dropping/recreating the Chroma collection can be slow for very large
            # corpora; wiping the persisted directory is usually much faster.
            persist_dir = Path(os.environ.get("CHROMA_PERSIST_DIR", "data/chroma"))
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            # Create a fresh Chroma collection
            VectorClient()
        else:
            VectorClient().delete_all()

    def _build_manifest_entries() -> list[Path]:
        return _collect_files(source_dir, recursive, file_exts)

    matched_files: list[Path]
    if manifest_path:
        if manifest_path.exists():
            matched_files = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            # If manifest is stale/empty (e.g., previously generated for XML only),
            # rebuild it using the current extension filter.
            if not matched_files:
                matched_files = _build_manifest_entries()
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text("\n".join(str(p) for p in matched_files), encoding="utf-8")
        else:
            matched_files = _build_manifest_entries()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("\n".join(str(p) for p in matched_files), encoding="utf-8")
    else:
        matched_files = _build_manifest_entries()
    if not matched_files:
        pretty_exts = ", ".join(file_exts)
        raise SystemExit(f"No matching files ({pretty_exts}) found under: {source_dir}")

    batch = matched_files[offset : offset + batch_size]
    if not batch:
        raise SystemExit(
            f"No files in range offset={offset}, batch_size={batch_size}. Total files: {len(matched_files)}"
        )

    worker = IngestionWorker()
    total_errors = 0
    processed = 0

    for i, file_path in enumerate(batch, start=1):
        res = await worker.ingest_file(file_path)
        processed += res.documents_processed
        total_errors += len(res.errors)

        if i == 1 or (processed > 0 and i % 10 == 0):
            print(
                f"[{offset + i}/{len(matched_files)}] processed={processed} "
                f"text_chunks={res.text_chunks_created} tables={res.tables_extracted} errors_in_file={len(res.errors)}"
            )
            if res.errors:
                for e in res.errors[:3]:
                    print(f"  - {e}")

    # Show updated counts for sanity checks.
    sql_client = SQLClient()
    vector_client = VectorClient()
    print("---- Batch summary ----")
    print(f"Files ingested in this run: {len(batch)}")
    print(f"Documents processed: {processed}")
    print(f"Total errors: {total_errors}")
    print(f"Vector chunk count: {vector_client.get_document_count()}")
    for table in ["drugs", "indications", "dosages", "adverse_reactions", "interactions"]:
        try:
            print(f"SQL rows ({table}): {sql_client.get_table_row_count(table)}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest documents into vector+SQL stores.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/documents/dailymed_unpacked/prescription",
        help="Directory containing source documents.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Number of files per run.")
    parser.add_argument("--offset", type=int, default=0, help="Start index into the sorted matched file list.")
    parser.add_argument("--clear", action="store_true", help="Clear SQL+Chroma before ingesting (only when offset==0).")
    parser.add_argument("--chroma-persist-dir", type=str, default=None, help="Overrides CHROMA_PERSIST_DIR env var.")
    parser.add_argument("--sqlite-db-path", type=str, default=None, help="Overrides SQLITE_DB_PATH env var.")
    parser.add_argument("--debug", type=int, choices=[0, 1], default=None, help="Set DEBUG=true/false.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest file for stable file ordering.")
    parser.add_argument("--no-recursive", action="store_true", help="Only match files directly under --source-dir.")
    parser.add_argument(
        "--file-ext",
        action="append",
        default=None,
        help="File extension to include (repeatable). Examples: --file-ext xml --file-ext json",
    )
    parser.add_argument(
        "--chroma-clear-method",
        type=str,
        default="vector-reset",
        choices=["vector-reset", "wipe-persist-dir"],
        help="How to clear Chroma when --clear is used.",
    )

    args = parser.parse_args()
    file_exts = _normalize_exts(args.file_ext or [".xml"])
    if not file_exts:
        raise SystemExit("No valid file extensions provided via --file-ext.")

    _set_env_if_provided(args)

    # Add project root to python path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    source_dir = Path(args.source_dir)
    asyncio.run(
        _ingest_batch(
            source_dir=source_dir,
            batch_size=args.batch_size,
            offset=args.offset,
            clear=bool(args.clear),
            manifest_path=Path(args.manifest) if args.manifest else None,
            recursive=not args.no_recursive,
            chroma_clear_method=args.chroma_clear_method,
            file_exts=file_exts,
        )
    )


if __name__ == "__main__":
    main()

