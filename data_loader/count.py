import os
from resplan_loader import load_resplan_pkl

RAW_DIR = 'C:/Users/matvi/Code/ML/data/raw'
pkl_path = os.path.join(RAW_DIR, 'ResPlan.pkl')

print("Loading floorplans...")
floorplans = load_resplan_pkl(pkl_path)
total = len(floorplans)
print(f"Total number of floorplans: {total}")

# Print some additional stats
valid = sum(1 for fp in floorplans if fp is not None)
print(f"Valid floorplans: {valid}")
print(f"Invalid/empty floorplans: {total - valid}")