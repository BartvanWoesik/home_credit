import pandas as pd
from copy import deepcopy
import inspect


def safe(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        cp_args = deepcopy(args)
        cp_kwargs = deepcopy(kwargs)
        res = fn(self, *cp_args, **cp_kwargs)
        return res

    return wrapper
  

class PandasDataPipeline:
    def __init__(
        self,
        steps,
        name: str = "pipeline",
    ) -> None:
        self.steps = steps
        self.name = name
  

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for step_number, step in enumerate(self.steps, start=1):
            for step_number, step in enumerate(self.steps, start=1):
                if isinstance(step, tuple):
                    # If step is a tuple, assume it's (description, function)
                    _, step_func = step
                else:
                    step_func = step

            # Check if step_func expects a pandas DataFrame as its argument
            if not self._function_accepts_dataframe(step_func):
                raise TypeError(f"The step function at step {step_number} does not accept a pandas DataFrame as an argument.")
            
            # Apply the step
            df = step_func(df)

        return df
    


    def _function_accepts_dataframe(self, func):
        """Check if first argument op function expects pd.DataFrame  """
        sig = inspect.signature(func)
        params = sig.parameters.values()
        first_param = next(iter(params), None)
        return first_param and first_param.annotation is pd.DataFrame

    @safe
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply(df)