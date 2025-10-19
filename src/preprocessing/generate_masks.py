import numpy as np
from typing import Dict, List
from shapely.geometry import Polygon
from ..data_loader.vector_to_raster import rasterize_plan
def build_mask_from_plan(polygons_by_class: Dict[str, List[Polygon]], size=(512,512)) -> np.ndarray:
    """
    Wrapper that produces a single-channel integer mask from vector polygons.
    0=background,1=wall,2=door,3=window
    """
    mask, meta = rasterize_plan(polygons_by_class, size=size)
    return mask