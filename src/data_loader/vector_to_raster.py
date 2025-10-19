import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform
from typing import Tuple, Dict, List


def _world_bounds(polygons_by_class):
    # Ensure polygons_by_class values are iterables
    minx = min((poly.bounds[0] for polys in polygons_by_class.values() if isinstance(polys, (list, tuple)) for poly in polys), default=0.0)
    miny = min((poly.bounds[1] for polys in polygons_by_class.values() if isinstance(polys, (list, tuple)) for poly in polys), default=0.0)
    maxx = max((poly.bounds[2] for polys in polygons_by_class.values() if isinstance(polys, (list, tuple)) for poly in polys), default=1.0)
    maxy = max((poly.bounds[3] for polys in polygons_by_class.values() if isinstance(polys, (list, tuple)) for poly in polys), default=1.0)
    return minx, miny, maxx, maxy


def rasterize_plan(polygons_by_class: Dict[str, List[Polygon]], size: Tuple[int, int] = (512,512), padding: float = 0.05) -> Tuple[np.ndarray, Dict]:
    """Rasterize polygons into a multi-class integer mask.

    Returns (mask, meta) where mask.shape == (H,W) and values are 0=bg,1=wall,2=door,3=window.
    meta contains the transform used to map world coords -> pixel coords: (minx,miny,maxx,maxy,width,height)
    """
    W, H = size
    minx, miny, maxx, maxy = _world_bounds(polygons_by_class)
    w = maxx - minx or 1.0
    h = maxy - miny or 1.0
    padx = w * padding
    pady = h * padding
    minx -= padx; miny -= pady; maxx += padx; maxy += pady
    w = maxx - minx; h = maxy - miny

    def world_to_px(x, y):
        px = int((x - minx) / w * (W - 1) + 0.5)
        py = int((maxy - y) / h * (H - 1) + 0.5)
        return px, py

    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # draw walls first
    for poly in polygons_by_class.get('wall', []):
        if poly is None:
            continue
        if hasattr(poly, 'geoms'):
            parts = poly.geoms
        else:
            parts = [poly]
        for part in parts:
            exterior = [world_to_px(x, y) for x, y in part.exterior.coords]
            draw.polygon(exterior, fill=1)
            for interior in part.interiors:
                draw.polygon([world_to_px(x,y) for x,y in interior.coords], fill=0)

    mask_np = np.array(mask, dtype=np.uint8)
    # doors overwrite walls
    door_mask = Image.new('L', (W,H), 0)
    draw_d = ImageDraw.Draw(door_mask)
    for poly in polygons_by_class.get('door', []):
        if poly is None: continue
        parts = poly.geoms if hasattr(poly, 'geoms') else [poly]
        for part in parts:
            draw_d.polygon([world_to_px(x,y) for x,y in part.exterior.coords], fill=1)
    door_np = np.array(door_mask, dtype=np.uint8)

    window_mask = Image.new('L', (W,H), 0)
    draw_w = ImageDraw.Draw(window_mask)
    for poly in polygons_by_class.get('window', []):
        if poly is None: continue
        parts = poly.geoms if hasattr(poly, 'geoms') else [poly]
        for part in parts:
            draw_w.polygon([world_to_px(x,y) for x,y in part.exterior.coords], fill=1)
    window_np = np.array(window_mask, dtype=np.uint8)

    # combine with priority: door/window > wall
    out = np.zeros((H, W), dtype=np.uint8)
    out[mask_np == 1] = 1
    out[window_np == 1] = 3
    out[door_np == 1] = 2

    meta = {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy, 'width': W, 'height': H}
    return out, meta


def render_floorplan_image(polygons_by_class: Dict[str, List[Polygon]], size=(512,512), bg=255) -> Image.Image:
    """Render a synthetic grayscale floorplan image (no furniture) from polygons.

    Walls drawn as thick strokes; doors/windows as shapes. Useful for supervised learning when original raster images are not provided.
    """
    from PIL import ImageFilter
    mask, meta = rasterize_plan(polygons_by_class, size=size)
    W, H = size
    img = Image.new('L', (W,H), bg)
    draw = ImageDraw.Draw(img)

    # draw walls as dark thick lines based on wall mask via dilation effect
    wall_mask = Image.fromarray((mask==1).astype('uint8')*255)
    wall_mask = wall_mask.filter(ImageFilter.MaxFilter(3))
    img.paste(0, mask=wall_mask)

    # draw windows as lighter gray
    window_mask = Image.fromarray((mask==3).astype('uint8')*255)
    img.paste(100, mask=window_mask)

    # draw doors as rectangles (black) — they will be overwritten where appropriate
    door_mask = Image.fromarray((mask==2).astype('uint8')*255)
    img.paste(0, mask=door_mask)

    return img