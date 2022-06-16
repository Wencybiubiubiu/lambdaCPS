from typing import Annotated

import numpy as np

Vec2 = Annotated[np.ndarray, 2]
Vec3 = Annotated[np.ndarray, 3]
Vec4 = Annotated[np.ndarray, 4]
Vec = np.ndarray
