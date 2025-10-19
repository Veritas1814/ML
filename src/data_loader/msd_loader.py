import pandas as pd
import shapely.wkt
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Dict, Any
def load_msd_csv(path: str, plan_id_field: str = 'plan_id') -> List[Dict[str, Any]]:
    """Load MSD CSV and group rows by plan_id producing polygons for wall/door/window.

    Returns list of plans in same format as load_resplan_pkl.
    """
    df = pd.read_csv(path)
    plans = {}

    for _, row in df.iterrows():
        pid = row.get(plan_id_field, None)
        if pd.isna(pid):
            continue
        pid = int(pid)
        if pid not in plans:
            plans[pid] = {'id': pid, 'polygons': {'wall': [], 'door': [], 'window': []}, 'meta': {}}
        geom_wkt = row.get('geom')
        if pd.isna(geom_wkt):
            continue
        try:
            geom = shapely.wkt.loads(geom_wkt)
        except Exception:
            continue
        et = str(row.get('entity_type', '')).lower()
        st = str(row.get('entity_subtype', '')).lower()
        key = 'other'
        if 'wall' in st or 'wall' in et:
            key = 'wall'
        elif 'door' in st or 'door' in et:
            key = 'door'
        elif 'window' in st or 'window' in et:
            key = 'window'
        if key in plans[pid]['polygons']:
            # flatten multipolygons
            if hasattr(geom, 'geoms'):
                for g in geom.geoms:
                    plans[pid]['polygons'][key].append(g)
            else:
                plans[pid]['polygons'][key].append(geom)
    return list(plans.values())