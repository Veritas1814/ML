import numpy as np
import nbtlib
from .block_palette import BLOCK_PALETTE

def export_schematic(voxel_grid, filename='floorplan.sch'):
    height, width, depth = voxel_grid.shape
    blocks = bytearray()
    data = bytearray()

    for y in range(height):
        for z in range(depth):
            for x in range(width):
                block_id = voxel_grid[y, z, x]
                blocks.append(block_id & 0xFF)
                data.append(0)

    schematic = nbtlib.File({
        'Schematic': nbtlib.Compound({
            'Width': nbtlib.Short(width),
            'Height': nbtlib.Short(height),
            'Length': nbtlib.Short(depth),
            'Blocks': nbtlib.ByteArray(blocks),
            'Data': nbtlib.ByteArray(data),
            'Materials': nbtlib.String('Alpha')
        })
    })
    schematic.save(filename)
