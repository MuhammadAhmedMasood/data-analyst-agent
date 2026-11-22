import pytest

from app.data.store import DataStore
from app.sandbox.runner import run_code
from app.tools.chart_tools import make_chart_tools
from app.tools.data_tools import make_data_tools


@pytest.fixture
def store():
    s = DataStore()
    s.load_csv("data/samples/Superstore.csv")
    return s


def test_run_python_sees_tables(store):
    run_python = make_data_tools(store)[2]
    out = run_python.invoke({"code": "print(len(superstore))"})
    assert out == "9994"


def test_run_python_dates_are_datetimes(store):
    run_python = make_data_tools(store)[2]
    out = run_python.invoke({"code": 'print(superstore["Order Date"].dt.year.min())'})
    assert out == "2014"


def test_run_python_returns_traceback(store):
    run_python = make_data_tools(store)[2]
    out = run_python.invoke({"code": "print(no_such_table)"})
    assert out.startswith("Error") and "NameError" in out


def test_run_python_needs_print(store):
    run_python = make_data_tools(store)[2]
    assert "printed nothing" in run_python.invoke({"code": "x = 1"})


def test_run_code_timeout(store):
    out = run_code("while True: pass", store.data_dir, timeout=2)
    assert "longer than 2" in out


def test_make_chart_bar(store):
    make_chart = make_chart_tools(store)[0]
    out = make_chart.invoke({
        "query": 'SELECT "Region", SUM("Sales") AS total FROM superstore GROUP BY 1',
        "kind": "bar", "x": "Region", "y": "total", "title": "Sales by region",
    })
    assert out.startswith("Chart saved")
    assert len(list(store.chart_dir.glob("*.png"))) == 1


def test_make_chart_bad_column(store):
    make_chart = make_chart_tools(store)[0]
    out = make_chart.invoke({
        "query": 'SELECT "Region" FROM superstore', "kind": "bar", "x": "Region", "y": "total",
    })
    assert out.startswith("Error") and "total" in out