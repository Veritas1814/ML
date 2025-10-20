How to run:

```
cd mc_voxelizer_baseline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/run_baseline.py --input [input path for npy] --voxel_size [scale, the larger the value the smaller in MC] --height [height] --export structure_nbt --block [block id] --out_dir [dir with output] --namespace [datapack namespace] --deploy_world [minecraft world] --structure_name [name structure] --min_component [adjusting] --close_k [adjusting] --skeleton --thicken [adjusting] --preview [preview file]
```
