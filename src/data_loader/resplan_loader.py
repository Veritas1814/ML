import pickle
from typing import List, Dict, Any
from shapely.geometry import shape, mapping, Polygon, MultiPolygon


def load_resplan_pkl(path: str) -> List[Dict[str, Any]]:
    """Load ResPlan .pkl file and return a list of plan dicts.

    Output format (per plan):
    {
        'id': int or str,
        'polygons': {   
            'wall': [shapely.geometry],
            'door': [shapely.geometry],
            'window': [shapely.geometry]
        },
        'meta': {...}
    }
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    plans = []
    # Data format varies between dumps; try to be robust
    if isinstance(data, dict):
        # maybe a mapping of id -> item
        iter_data = data.values()
    else:
        iter_data = data

    for idx, entry in enumerate(iter_data):
        # entry is expected to be a dict-like with 'wall','door','window' keys
        plan = {'id': getattr(entry, 'get', lambda k, d=None: None)('id', idx), 'polygons': {'wall': [], 'door': [], 'window': []}, 'meta': {}}
        # Accept different key types
        for key in ('wall', 'walls'):
            if key in entry:
                plan['polygons']['wall'] = list(entry[key].geoms) if hasattr(entry[key], 'geoms') else [entry[key]]
                break
        for key in ('door', 'doors'):
            if key in entry:
                plan['polygons']['door'] = list(entry[key].geoms) if hasattr(entry[key], 'geoms') else [entry[key]]
                break
        for key in ('window', 'windows'):
            if key in entry:
                plan['polygons']['window'] = list(entry[key].geoms) if hasattr(entry[key], 'geoms') else [entry[key]]
                break
        # copy any other metadata
        for m in ('wall_depth','id','unitType','area'):
            if m in entry:
                plan['meta'][m] = entry[m]
        plans.append(plan)
    return plans