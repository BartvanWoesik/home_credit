import pytest
from data_pipeline.pipes import get_mode, get_last
import pandas as pd
import numpy as np


def test_get_mode():
    assert get_mode([1, 2, 2, 3, 4]) == 2
    assert get_mode(["a", "b", "b", "c"]) == "b"
    assert get_mode([]) is None
    assert get_mode(None) is None
    assert get_mode([None, None, None]) is None
    assert get_mode([1, 1, 2, 2]) in [1, 2]
    assert get_mode(pd.DataFrame([1, 2, 2, 4])) == 2


def test_get_mode_2_col():
    with pytest.raises(AssertionError):
        get_mode(
            pd.DataFrame({"a": [1, 2, 2, 4], "b": [1, 1, 1, 1], "C": [2, 2, 2, 2]})
        )


def test_get_mode_with_nan():
    assert pd.isna(get_mode([np.nan, np.nan, np.nan]))
    assert get_mode([1, 2, 2, np.nan]) == 2


def test_get_last():
    assert get_last([1, 2, 3, 4]) == 4
    assert get_last(["a", "b", "c"]) == "c"
    assert get_last([]) is None
    assert get_last(None) is None
    assert get_last([None, None, None]) is None
    assert get_last(pd.DataFrame([1, 2, 2, 4])) == 4


def test_get_last_with_generator():
    assert get_last(range(10)) == 9
    assert get_last((i for i in range(10))) == 9


def test_get_last_with_string():
    assert get_last("hello") == "o"
