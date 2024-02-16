import pandas as pd
import numpy as np  

def get_last(iterable):
    """
    Get the last value in an iterable.

    Parameters:
        iterable (iterable): The iterable.

    Returns:
        object: The last value in the iterable, or None if the iterable is empty.
    """
    if iterable is None:
        return None
    
    if isinstance(iterable, (pd.DataFrame)):
        iterable = pd.Series(iterable.iloc[0])
    
    try:
        return next(reversed(list(iterable)))
    except StopIteration:
        return None

import pandas as pd

def get_mode(iterable):
    """
    Get the mode of an iterable.

    Parameters:
        iterable (iterable): The iterable.

    Returns:
        object: The mode of the iterable, or None if the iterable is empty.
    """
    if iterable is None:
        return None
    if isinstance(iterable, (pd.DataFrame)):
        series = pd.Series(iterable.iloc[0])
    else:
        series = pd.Series(iterable)   
    series = series.dropna()  
    mode_element = series.mode()
    return mode_element[0] if not mode_element.empty else None