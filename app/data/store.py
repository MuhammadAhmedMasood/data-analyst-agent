"""Data layer: load CSVs into DuckDB and describe them for the LLM."""
import re
import sys
from pathlib import Path

import duckdb


def clean_table_name(raw: str) -> str:
    """'Sales Data 2024' -> 'sales_data_2024' (safe to use in SQL)."""
    name = re.sub(r"\W+", "_", raw.strip().lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name

def ensure_utf8(path: Path) -> Path:
    """Return a UTF-8 version of the file, converting from Windows-1252 if needed."""
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        return path
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        fixed = path.with_name(f"{path.stem}.utf8{path.suffix}")
        fixed.write_text(text, encoding="utf-8")
        return fixed

class DataStore:
    """One in-memory DuckDB database holding the user's tables."""

    def __init__(self) -> None:
        self.con = duckdb.connect(":memory:")
        self.tables: list[str] = []

    def load_csv(self, path: str | Path) -> str:
        path = Path(path)
        name = clean_table_name(path.stem)
        self.con.execute(f'DROP TABLE IF EXISTS "{name}"')
        utf8_path = ensure_utf8(path)
        self.con.read_csv(str(utf8_path)).create(name)
        if name not in self.tables:
            self.tables.append(name)
        return name

    def describe(self, table: str, n_samples: int = 3) -> str:
        if table not in self.tables:
            raise ValueError(f"Unknown table '{table}'. Available: {self.tables}")

        n_rows = self.con.sql(f'SELECT COUNT(*) FROM "{table}"').fetchall()[0][0]
        stats = self.con.sql(f'SUMMARIZE "{table}"').df()
        sample = self.con.sql(f'SELECT * FROM "{table}" LIMIT {n_samples}').df()

        lines = [f"Table: {table} ({n_rows:,} rows, {len(stats)} columns)", "Columns:"]
        for row in stats.itertuples():
            lines.append(
                f"  - {row.column_name} ({row.column_type}): "
                f"~{row.approx_unique} unique, null% {row.null_percentage}, "
                f"range {_short(row.min)} .. {_short(row.max)}"
            )
        lines.append("Sample rows:")
        lines.append(sample.to_string(index=False))
        return "\n".join(lines)


def _short(value, limit: int = 25) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    store = DataStore()
    for csv_path in sys.argv[1:]:
        table = store.load_csv(csv_path)
        print(store.describe(table))
        print()