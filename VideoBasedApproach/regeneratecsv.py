import os
import pandas as pd
from tqdm import tqdm

# =========================
# PATH CONFIG
# =========================
CSV_PATH = "reconstructed_sequences.csv"
LABELS_DIR = "Dataset/ModifiedAUDataset/labels"
OUTPUT_CSV = "final_matched_output.csv"

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Normalize column name if needed
df.columns = [col.lower() for col in df.columns]

# Try to detect image column automatically
if 'frame_name' in df.columns:
    img_col = 'frame_name'
elif 'image' in df.columns:
    img_col = 'image'
else:
    raise Exception("No image column found in CSV")

# =========================
# OUTPUT STORAGE
# =========================
final_data = []

# =========================
# PROCESS EACH IMAGE
# =========================
for idx, row in tqdm(df.iterrows(), total=len(df)):

    frame_name = row[img_col]

    # remove extension if exists
    base_name = os.path.splitext(frame_name)[0]
    label_path = os.path.join(LABELS_DIR, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"Label not found for: {frame_name}")
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # =========================
    # FIRST LINE: flight height
    # =========================
    flight_height_line = lines[0].strip()
    
    try:
        flight_height = float(flight_height_line.split(":")[1])
    except:
        flight_height = None

    # =========================
    # REMAINING: OBJECTS
    # =========================
    for obj_line in lines[1:]:
        parts = obj_line.strip().split()

        if len(parts) < 11:
            continue

        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
        class_label = parts[8]
        difficulty = int(parts[9])
        heading_angle = float(parts[10])

        final_data.append({
            "frame_name": frame_name,
            "sequence_index": idx,
            "flight_height": flight_height,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "x3": x3, "y3": y3,
            "x4": x4, "y4": y4,
            "class_label": class_label,
            "difficulty": difficulty,
            "heading_angle": heading_angle
        })

# =========================
# SAVE FINAL CSV
# =========================
final_df = pd.DataFrame(final_data)
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nFinal CSV saved at: {OUTPUT_CSV}")
print(f"Total annotations: {len(final_df)}")