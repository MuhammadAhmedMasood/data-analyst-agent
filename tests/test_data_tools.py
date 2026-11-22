import pytest

from app.data.store import DataStore
from app.tools.data_tools import make_data_tools




@pytest.fixture
def tools():
    store = DataStore()
    store.load_csv("data/samples/Superstore.csv")
    get_schema, run_sql, _ = make_data_tools(store)
    return get_schema, run_sql


def test_get_schema_lists_columns(tools):
    get_schema, _ = tools
    out = get_schema.invoke({})
    assert "superstore" in out
    assert "Order Date (DATE)" in out


def test_get_schema_unknown_table(tools):
    get_schema, _ = tools
    out = get_schema.invoke({"table": "nope"})
    assert out.startswith("Error")


def test_run_sql_aggregate(tools):
    _, run_sql = tools
    out = run_sql.invoke({"query": "SELECT COUNT(*) AS n FROM superstore"})
    assert "9994" in out


def test_run_sql_quoted_columns(tools):
    _, run_sql = tools
    out = run_sql.invoke(
        {"query": 'SELECT "Region", ROUND(SUM("Sales"), 2) AS total FROM superstore GROUP BY 1 ORDER BY 1;'}
    )
    assert "West" in out and "Error" not in out


@pytest.mark.parametrize("bad", [
    "DROP TABLE superstore",
    "DELETE FROM superstore",
    "SELECT 1; DROP TABLE superstore",
])
def test_run_sql_blocks_writes(tools, bad):
    _, run_sql = tools
    assert run_sql.invoke({"query": bad}).startswith("Error")


def test_run_sql_bad_column_returns_error(tools):
    _, run_sql = tools
    out = run_sql.invoke({"query": "SELECT not_a_column FROM superstore"})
    assert out.startswith("Error")


def test_run_sql_truncates(tools):
    _, run_sql = tools
    out = run_sql.invoke({"query": "SELECT * FROM superstore"})
    assert "showing first 50 rows" in out

def test_run_sql_shows_nulls_as_null(tools):
    _, run_sql = tools
    out = run_sql.invoke({"query": "SELECT NULL AS region, 1 AS n"})
    assert "NULL" in out and "NaN" not in out