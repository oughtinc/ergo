from typing import Union

import jax.numpy as np
import numpy as onp
import pandas as pd

ArrayLikes = [pd.DataFrame, pd.Series, np.ndarray, np.DeviceArray, onp.ndarray]

ArrayLikeType = Union[pd.DataFrame, pd.Series, np.ndarray, np.DeviceArray, onp.ndarray]
