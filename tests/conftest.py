"""Tests run on fake data in a temp folder, so they never touch your real data/ or models/."""
import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="stock_analyzer_test_")
os.environ["STOCKS_DATA_DIR"] = str(Path(_tmp) / "data")
os.environ["STOCKS_MODELS_DIR"] = str(Path(_tmp) / "models")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
