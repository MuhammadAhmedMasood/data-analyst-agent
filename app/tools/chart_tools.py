"""Chart tool: run a SELECT query and plot the result with matplotlib."""
from typing import Literal

import duckdb
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from langchain_core.tools import tool

from app.data.store import DataStore
from app.tools.data_tools import validate_select

MAX_CHART_ROWS = 1000


def make_chart_tools(store: DataStore) -> list:

    @tool
    def make_chart(
        query: str,
        kind: Literal["bar", "line", "scatter", "hist"],
        x: str,
        y: str | None = None,
        title: str = "",
    ) -> str:
        """Plot the result of a DuckDB SELECT query and save it as a PNG.
        Shape the data in SQL first (e.g. GROUP BY for a bar chart).
        x and y are column names from the query result. y is required
        except for 'hist', which plots the distribution of x."""
        query = query.strip().rstrip(";")
        error = validate_select(store.con, query)
        if error:
            return f"Error: {error}"
        try:
            df = store.con.sql(f"SELECT * FROM ({query}) LIMIT {MAX_CHART_ROWS}").df()
        except duckdb.Error as e:
            return f"Error: {e}"

        needed = [x] if kind == "hist" else [x, y]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            return f"Error: columns {missing} not in query result {list(df.columns)}."

        fig, ax = plt.subplots(figsize=(8, 5))
        if kind == "bar":
            ax.bar(df[x].astype(str), df[y])
        elif kind == "line":
            ax.plot(df[x], df[y], marker="o")
        elif kind == "scatter":
            ax.scatter(df[x], df[y], alpha=0.6)
        else:
            ax.hist(df[x].dropna(), bins=30)

        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y or "count")
        fig.autofmt_xdate()
        fig.tight_layout()

        path = store.chart_dir / f"chart_{len(list(store.chart_dir.glob('*.png'))) + 1}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return f"Chart saved to {path.name} ({kind} of {y or x} by {x}, {len(df)} rows)."

    return [make_chart]