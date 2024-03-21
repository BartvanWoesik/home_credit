import polars as pl


def last(col):
    return (pl.col(col)).last()


def mean(col):
    return (pl.col(col)).mean()


def max(col):
    return (pl.col(col)).max()


def mode(col):
    return (pl.col(col)).mode().last()


def min(col):
    return (pl.col(col)).min()