"""
Common and shared pipeline steps.
"""

import pandas as pd

from billys.checkpoint import save

def show(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print the dataframe as a side effect and return it.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        The dataframe itself without changes.
    """
    print(df)
    return df


def skip(x):
    """
    The identity function.
    """
    return x


def dump(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dump the dataframe on file as a side effect with :func:`billys.checkpoint.save`,
    and returns the dataframe without changes.

    Parameters
    ----------
    df
        The dataset as a dataframe.

    Returns
    -------
    df
        The dataframe without changes.
    """
    filename = save('dump_ocr', df)
    print(f'Dumped object into {filename}')
    return df
