from typing import Tuple
def pixel_to_world(px: int, py: int, meta: dict) -> Tuple[float, float]:
    minx = meta['minx']; miny = meta['miny']; maxx = meta['maxx']; maxy = meta['maxy']
    W = meta['width']; H = meta['height']
    x = minx + (px / (W - 1)) * (maxx - minx)
    y = maxy - (py / (H - 1)) * (maxy - miny)
    return x, y


def world_to_pixel(x: float, y: float, meta: dict) -> Tuple[int, int]:
    minx = meta['minx']; miny = meta['miny']; maxx = meta['maxx']; maxy = meta['maxy']
    W = meta['width']; H = meta['height']
    px = int((x - minx) / (maxx - minx) * (W - 1) + 0.5)
    py = int((maxy - y) / (maxy - miny) * (H - 1) + 0.5)
    return px, py