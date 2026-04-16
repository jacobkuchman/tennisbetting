import pandas as pd
import pytest

from src.data.loader import validate_time_order
from src.data.real_ingestion import _validate_date_bounds


def test_validate_time_order_passes():
    df = pd.DataFrame({"match_date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    validate_time_order(df)


def test_validate_time_order_raises():
    df = pd.DataFrame({"match_date": pd.to_datetime(["2024-01-02", "2024-01-01"])})
    with pytest.raises(ValueError):
        validate_time_order(df)


def test_impossible_date_raises():
    s = pd.Series(pd.to_datetime(["1950-01-01"]))
    with pytest.raises(ValueError):
        _validate_date_bounds(s)
