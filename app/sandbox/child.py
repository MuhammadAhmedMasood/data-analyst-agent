"""Runs inside a separate Python process. Loads every table as a DataFrame,
then executes the agent's code (read from stdin)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

namespace = {"pd": pd, "np": np}
for parquet_file in sorted(Path(".").glob("*.parquet")):
    namespace[parquet_file.stem] = pq.read_table(parquet_file).to_pandas(date_as_object=False)

code = sys.stdin.read()
exec(compile(code, "<agent_code>", "exec"), namespace)