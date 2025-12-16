
import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def analyze_raw_data():
    # Path to the reorganized data (biofilm folder has all the biofilm images)
    # We use the reorganized folder because it contains the paired/curated dataset
    data_dir = Path("data/raw_data_reorganized/biofilm")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    print(f"Scanning directory: {data_dir}\n")

    files = sorted(list(data_dir.glob("*.tif")))
    total_files = len(files)
    
    print(f"Total images found: {total_files}\n")

    treatments = []
    dates = []
    treatment_date_map = defaultdict(list)
    
    image_shapes = Counter()
    image_dtypes = Counter()
    
    # Sample one image for detailed inspection
    sample_image_path = files[0] if files else None
    
    for f in files:
        parts = f.stem.split('_')
        # Expected format: biofilm_{Treatment...}_{Date}_{XY}
        # We parsed from back because treatment can contain underscores
        
        # The last part is XY number
        xy_part = parts[-1]
        
        # The second to last part is Date (MM-DD-YYYY)
        date_part = parts[-2]
        
        # Everything from index 1 to -2 is Treatment
        # index 0 is 'biofilm'
        if len(parts) < 4:
            print(f"Warning: Unexpected filename format: {f.name}")
            continue
            
        treatment_part = "_".join(parts[1:-2])
        
        treatments.append(treatment_part)
        dates.append(date_part)
        treatment_date_map[treatment_part].append(date_part)
        
        # Check dimensions for a few images to ensure consistency (or all if fast enough)
        # Reading all might be slow, let's read the first one of each treatment/date combo?
        # Or just read the first one found overall.
        pass

    # Analyze sample image
    if sample_image_path:
        img = cv2.imread(str(sample_image_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            print("=== Image Properties (Sample) ===")
            print(f"Filename: {sample_image_path.name}")
            print(f"Shape: {img.shape}")
            print(f"Dtype: {img.dtype}")
            print(f"Min/Max value: {img.min()} / {img.max()}")
            print("================================\n")
        else:
             print("Error: Could not read sample image.\n")

    # Distribution Analysis
    treatment_counts = Counter(treatments)
    date_counts = Counter(dates)
    
    print("=== Distribution by Treatment ===")
    for t, count in treatment_counts.most_common():
        print(f"{t:<20}: {count}")
    print("\n")
    
    print("=== Distribution by Date ===")
    for d, count in date_counts.most_common():
        print(f"{d:<20}: {count}")
    print("\n")

    print("=== Detailed Breakdown (Treatment x Date) ===")
    # Create a matrix or list
    all_treatments = sorted(treatment_counts.keys())
    all_dates = sorted(date_counts.keys())
    
    # Header
    print(f"{'Treatment':<20} | " + " | ".join([f"{d:<10}" for d in all_dates]) + " | Total")
    print("-" * (20 + 3 + (13 * len(all_dates)) + 8))
    
    for t in all_treatments:
        row_counts = []
        t_dates = treatment_date_map[t]
        t_date_counts = Counter(t_dates)
        
        for d in all_dates:
            c = t_date_counts.get(d, 0)
            row_counts.append(c)
            
        row_str = f"{t:<20} | " + " | ".join([f"{str(c):<10}" for c in row_counts]) + f" | {sum(row_counts)}"
        print(row_str)

if __name__ == "__main__":
    analyze_raw_data()

