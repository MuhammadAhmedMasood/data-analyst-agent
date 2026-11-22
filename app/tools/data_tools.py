"""LangChain tools that let the agent inspect and query the user's data."""
import duckdb
from langchain_core.tools import tool

from app.data.store import DataStore
from app.sandbox.runner import run_code

MAX_ROWS = 50


def validate_select(con: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    """Return an error message if the SQL is not a single SELECT, else None."""
    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as e:
        return f"SQL syntax error: {e}"
    if len(statements) != 1:
        return f"Send exactly one statement, got {len(statements)}."
    if statements[0].type != duckdb.StatementType.SELECT:
        return f"Only SELECT queries are allowed, got {statements[0].type.name}."
    return None


def make_data_tools(store: DataStore) -> list:
    """Build the tools for one DataStore (one user session)."""

    @tool
    def get_schema(table: str | None = None) -> str:
        """Describe the available tables: columns, types, null %, value ranges
        and sample rows. Pass a table name for one table, or nothing for all.
        Call this before writing SQL."""
        if not store.tables:
            return "No tables loaded yet."
        names = [table] if table else store.tables
        try:
            return "\n\n".join(store.describe(name) for name in names)
        except ValueError as e:
            return f"Error: {e}"

    @tool
    def run_sql(query: str) -> str:
        """Run one read-only DuckDB SELECT query and return the result as a table.
        Wrap column names containing spaces or symbols in double quotes,
        e.g. SELECT "Order Date" FROM superstore. Results are capped at 50 rows,
        so aggregate (GROUP BY, SUM, COUNT) rather than selecting raw rows."""
        query = query.strip().rstrip(";")
        error = validate_select(store.con, query)
        if error:
            return f"Error: {error}"
        try:
            df = store.con.sql(f"SELECT * FROM ({query}) LIMIT {MAX_ROWS + 1}").df()
        except duckdb.Error as e:
            return f"Error: {e}"

        truncated = len(df) > MAX_ROWS
        shown = df.head(MAX_ROWS).copy()
        text_cols = shown.select_dtypes(exclude="number").columns
        shown[text_cols] = shown[text_cols].fillna("NULL")
        text = shown.to_string(index=False, na_rep="NULL")
        if truncated:
            text += f"\n(showing first {MAX_ROWS} rows only; aggregate to see everything)"
        return text
        
    @tool
    def run_python(code: str) -> str:
        """Run Python for analysis that is awkward in SQL (statistics, cleaning,
        correlations, reshaping). pandas is imported as pd and numpy as np.
        Every table is preloaded as a pandas DataFrame with the same name as
        the table, e.g. superstore. Only printed output is returned, so use
        print(). Each call starts fresh: variables do not carry over."""
        return run_code(code, store.data_dir)

    return [get_schema, run_sql, run_python]