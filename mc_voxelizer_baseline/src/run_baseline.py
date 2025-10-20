import argparse, os, cv2, numpy as np, shutil
from pathlib import Path
from parse_floorplan import segment_floorplan, downsample_mask
from preprocess import clean_mask
from voxelize import mask_to_voxels
from export_mcfunction import write_datapack
from export_structure_nbt import write_structure_nbt

def load_mask_from_input(input_path: str, threshold: int, voxel_size: int,
                         flip_ud: bool, flip_lr: bool, rot90: int):
    p = Path(input_path)
    if p.suffix.lower() == ".npy":
        m = np.load(str(p))
        if m.ndim == 3:
            m = (m.sum(axis=-1) > 0).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        mask = m.astype(bool)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read image: {input_path}")
        mask = segment_floorplan(img, threshold=threshold)

    if rot90 % 4 != 0:
        mask = np.rot90(mask, k=rot90)
    if flip_ud:
        mask = np.flipud(mask)
    if flip_lr:
        mask = np.fliplr(mask)

    mask = downsample_mask(mask, voxel_size=voxel_size)
    return mask

def deploy_to_world_nbt(nbt_path: str, world_name: str, name: str):
    home = Path.home()
    world_dir = home / "Library" / "Application Support" / "minecraft" / "saves" / world_name
    gen_mc = world_dir / "generated" / "minecraft" / "structures"
    gen_mc.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nbt_path, gen_mc / f"{name}.nbt")

    pack_dir = world_dir / "datapacks" / "floorplan_struct_pack"
    (pack_dir / "data" / "floorplan" / "structures").mkdir(parents=True, exist_ok=True)
    with open(pack_dir / "pack.mcmeta", "w", encoding="utf-8") as f:
        f.write('{"pack":{"pack_format":12,"description":"Floorplan structures"}}')
    shutil.copy2(nbt_path, pack_dir / "data" / "floorplan" / "structures" / f"{name}.nbt")
    return str(gen_mc / f"{name}.nbt")

def maybe_preview(mask_raw: np.ndarray, mask_final: np.ndarray, out_png: Path | None):
    if out_png is None:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(mask_raw, cmap="gray"); plt.title("input mask"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(mask_final, cmap="gray"); plt.title("cleaned+downsampled"); plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to PNG/JPG or MSD .npy mask")
    ap.add_argument("--voxel_size", type=int, default=3)
    ap.add_argument("--height", type=int, default=3)
    ap.add_argument("--threshold", type=int, default=180)
    ap.add_argument("--out_dir", default="out_pack")
    ap.add_argument("--namespace", default="floorplan")
    ap.add_argument("--function_name", default="build")
    ap.add_argument("--export", choices=["mcfunction","structure_nbt"], default="structure_nbt")
    ap.add_argument("--block", default="minecraft:stone")
    ap.add_argument("--deploy_world", default="")
    ap.add_argument("--structure_name", default="floorplan_build")
    ap.add_argument("--min_component", type=int, default=120)
    ap.add_argument("--close_k", type=int, default=5)
    ap.add_argument("--open_k", type=int, default=0)
    ap.add_argument("--skeleton", action="store_true")
    ap.add_argument("--thicken", type=int, default=2)
    ap.add_argument("--flip_ud", action="store_true", default=True)
    ap.add_argument("--flip_lr", action="store_true", default=False)
    ap.add_argument("--rot90", type=int, default=0)
    ap.add_argument("--preview", default="")
    args = ap.parse_args()

    mask_loaded = load_mask_from_input(args.input, args.threshold, args.voxel_size,
                                       args.flip_ud, args.flip_lr, args.rot90)
    mask_clean = clean_mask(mask_loaded,
                            min_component=args.min_component,
                            close_k=args.close_k,
                            open_k=args.open_k,
                            do_skeleton=args.skeleton,
                            thicken=args.thicken)
    if args.preview:
        maybe_preview(mask_loaded, mask_clean, Path(args.preview))

    voxels = mask_to_voxels(mask_clean.astype(bool), height=args.height, y_base=0)

    if args.export == "mcfunction":
        pack_dir = write_datapack(voxels, out_dir=args.out_dir, namespace=args.namespace, function_name=args.function_name)
        print(f"Datapack generated at: {pack_dir}")
        print(f"In-game: /reload  then  /function {args.namespace}:{args.function_name}")
    else:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_file = write_structure_nbt(voxels, out_dir=args.out_dir, block_id=args.block,
                                       filename=f"{args.namespace}_{args.function_name}.nbt")
        print(f"Structure NBT generated: {out_file}")
        if args.deploy_world:
            deploy_to_world_nbt(out_file, args.deploy_world, args.structure_name)
            print(f"In-game: /reload  then  /place template floorplan:{args.structure_name}")

if __name__ == "__main__":
    main()
