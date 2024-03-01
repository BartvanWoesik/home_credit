import numpy as np
from evaluate.scorers import gini_score, check_scorer_input, kaggle_score

import pytest


def test_gini_score():
    # Test case 1: Perfect prediction
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert gini_score(y_true, y_pred) == 1.0

    # Test case 2: Random prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(gini_score(y_true, y_pred), 0.0)

    # Test case 3: Inverted prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    assert np.isclose(gini_score(y_true, y_pred), -1.0)


def test_kaggle_score():
    # Test case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert kaggle_score(y_true, y_pred) == 0.0

    # Test case 2: Random prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    assert kaggle_score(y_true, y_pred) == -0.5

    # Test case 3: All positive prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert kaggle_score(y_true, y_pred) == -0.5

    # Test case 4: All negative prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert kaggle_score(y_true, y_pred) == -0.5

    # Test case 5: Empty prediction
    y_true = np.array([])
    y_pred = np.array([])
    assert kaggle_score(y_true, y_pred) == 0.0


def test_scorer_empty_prediction():
    # Test case 4: Empty prediction
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(AssertionError):
        check_scorer_input(y_true, y_pred)


def test_scorer_wrong_type_int():
    # Test case 5: Wrong type of input
    y_true = np.array([1, 1, 0, 0])
    y_pred = 5
    try:
        check_scorer_input(y_true, y_pred)
        assert False, "Expected TypeError"
    except TypeError:
        assert True


def test_scorer_wrong_type_str():
    # Test case 5: Wrong type of input
    y_true = np.array([1, 1, 0, 0])
    y_pred = "12"
    try:
        check_scorer_input(y_true, y_pred)
        assert False, "Expected TypeError"
    except TypeError:
        assert True
