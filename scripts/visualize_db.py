"""
Generate an HTML report to visualize pharma.db contents.

Usage:
    python scripts/visualize_db.py [--db PATH] [--out PATH]

Output:
    Opens or saves an HTML file with table summaries, row counts, and sample data.
"""

import argparse
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "pharma.db"
DEFAULT_OUT = PROJECT_ROOT / "data" / "evaluation" / "pharma_db_report.html"
MAX_SAMPLE_ROWS = 50


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row[0] for row in cur.fetchall()]


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]


def get_table_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"PRAGMA table_info([{table}])").fetchall()


def get_sample_rows(conn: sqlite3.Connection, table: str, limit: int) -> list[tuple]:
    cur = conn.execute(f"SELECT * FROM [{table}] LIMIT {limit}")
    return cur.fetchall()


def get_column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [info[1] for info in get_table_info(conn, table)]


def escape_html(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def truncate(s: str, max_len: int = 80) -> str:
    if s is None or not isinstance(s, str):
        return str(s) if s is not None else ""
    return s[:max_len] + "..." if len(s) > max_len else s


def generate_html(db_path: Path, out_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    tables = get_table_names(conn)

    # Gather stats and samples
    stats = []
    samples = {}
    for table in tables:
        count = get_row_count(conn, table)
        stats.append((table, count))
        cols = get_column_names(conn, table)
        rows = get_sample_rows(conn, table, MAX_SAMPLE_ROWS)
        samples[table] = (cols, rows)

    conn.close()

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Pharma DB Report</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #1a1a2e; color: #eee; }",
        "h1 { color: #e94560; }",
        "h2 { color: #0f3460; background: #eaeaea; color: #16213e; padding: 0.5rem 1rem; border-radius: 6px; }",
        ".summary { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }",
        ".card { background: #16213e; padding: 1rem 1.5rem; border-radius: 8px; min-width: 140px; }",
        ".card .count { font-size: 1.5rem; font-weight: bold; color: #e94560; }",
        ".card .label { font-size: 0.9rem; color: #aaa; }",
        "table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }",
        "th, td { border: 1px solid #333; padding: 0.5rem 0.75rem; text-align: left; }",
        "th { background: #0f3460; color: #eee; }",
        "tr:nth-child(even) { background: #16213e; }",
        "td { max-width: 320px; overflow: hidden; text-overflow: ellipsis; }",
        ".meta { color: #888; font-size: 0.85rem; margin-bottom: 1rem; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Pharma DB Visualization</h1>",
        f"<p class='meta'>Database: <code>{escape_html(str(db_path))}</code></p>",
        "<div class='summary'>",
    ]

    for table, count in stats:
        html_parts.append(
            f"<div class='card'><div class='count'>{count}</div><div class='label'>{escape_html(table)}</div></div>"
        )
    html_parts.append("</div>")

    stats_dict = dict(stats)
    for table in tables:
        cols, rows = samples[table]
        count = stats_dict[table]
        html_parts.append(f"<h2>{escape_html(table)} ({count} rows)</h2>")
        html_parts.append("<table><thead><tr>")
        for c in cols:
            html_parts.append(f"<th>{escape_html(c)}</th>")
        html_parts.append("</tr></thead><tbody>")
        for row in rows:
            html_parts.append("<tr>")
            for i, val in enumerate(row):
                cell = truncate(escape_html(str(val) if val is not None else ""), 120)
                html_parts.append(f"<td title='{escape_html(str(val) or '')}'>{cell}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        if count > MAX_SAMPLE_ROWS:
            html_parts.append(f"<p class='meta'>Showing first {MAX_SAMPLE_ROWS} of {count} rows.</p>")

    html_parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Report written to: {out_path}")
    print("Open this file in a browser to visualize the database.")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization of pharma.db")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to SQLite database")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output HTML path")
    args = parser.parse_args()
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        return
    generate_html(args.db, args.out)


if __name__ == "__main__":
    main()
