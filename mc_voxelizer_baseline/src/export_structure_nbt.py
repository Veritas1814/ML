from typing import List, Tuple
import os
import nbtlib

def write_structure_nbt(voxels: List[Tuple[int,int,int]], out_dir: str,
                        block_id: str = "minecraft:stone",
                        filename: str = "floorplan.nbt") -> str:
    if not voxels:
        raise ValueError("No voxels to export")
    min_x = min(v[0] for v in voxels); min_y = min(v[1] for v in voxels); min_z = min(v[2] for v in voxels)
    vox = [(x-min_x,y-min_y,z-min_z) for (x,y,z) in voxels]
    sx=max(v[0] for v in vox)+1; sy=max(v[1] for v in vox)+1; sz=max(v[2] for v in vox)+1
    palette = [nbtlib.Compound({"Name": nbtlib.String(block_id)})]
    blocks = [nbtlib.Compound({"pos": nbtlib.List[nbtlib.Int]([nbtlib.Int(x),nbtlib.Int(y),nbtlib.Int(z)]),
                                "state": nbtlib.Int(0)}) for (x,y,z) in vox]
    root = nbtlib.Compound({
        "size": nbtlib.List[nbtlib.Int]([nbtlib.Int(sx), nbtlib.Int(sy), nbtlib.Int(sz)]),
        "palette": nbtlib.List[nbtlib.Compound](palette),
        "palette_max": nbtlib.Int(len(palette)),
        "entities": nbtlib.List[nbtlib.Compound]([]),
        "blocks": nbtlib.List[nbtlib.Compound](blocks),
        "DataVersion": nbtlib.Int(3955)
    })
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    nbtlib.File(root).save(out_path, gzipped=True)
    return out_path
