import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATASET_PATH = 'Dataset/DRASHTI-HaOBB' 
INFO_CSV = 'Dataset/DRASHTI-HaOBB/DRASHTI-HaOBB_framewise_info.csv'
OUTPUT_BASE = 'Outputs'
BATCH_SIZE = 64
# Ensure output directory exists
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Step 1: Load and Filter Metadata
df = pd.read_csv(INFO_CSV)
# Filter out augmented samples to keep original sequences
df_clean = df[df['augmented'] == 'No'].copy()

# Step 2: Identify Scenes
# Based on filename patterns, the first part usually identifies the location/scene
def get_scene_id(frame_name):
    # Split by underscore or take first 4 characters
    return frame_name.split('_')[0]

df_clean['scene_id'] = df_clean['frame_name'].apply(get_scene_id)
df_clean['file_path'] = df_clean.apply(lambda row: os.path.join(DATASET_PATH, 'images', row['split'], f"{row['frame_name']}.jpg"), axis=1)

print(f"Total original frames: {len(df_clean)}")
print(f"Unique scenes detected: {df_clean['scene_id'].nunique()}")

# Step 3: Prepare Label Features
# Using the vehicle count columns to define the 'profile' of a frame
label_columns = [
    'Auto3WCargo', 'AutoRicksaw', 'Bus', 'Container', 'Mixer', 
    'MotorCycle', 'PickUp', 'SUV', 'Sedan', 'Tanker', 'Tipper', 
    'Trailer', 'Truck', 'Van'
]

# Standardize counts to give equal weight to rare vs common vehicles
scaler = StandardScaler()
df_clean[label_columns] = scaler.fit_transform(df_clean[label_columns])

# Step 4: Reconstruct Sequence per Scene
def reconstruct_scene_sequence(scene_df):
    if len(scene_df) < 2:
        return scene_df
    
    feats = scene_df[label_columns].values
    remaining_indices = list(range(1, len(feats)))
    current_idx = 0
    sequence = [0]
    
    while remaining_indices:
        current_feat = feats[current_idx].reshape(1, -1)
        search_feats = feats[remaining_indices]
        
        # Calculate distance based on object distribution (labels)
        dists = np.linalg.norm(search_feats - current_feat, axis=1)
        best_match_local_idx = np.argmin(dists)
        
        current_idx = remaining_indices.pop(best_match_local_idx)
        sequence.append(current_idx)
        
    return scene_df.iloc[sequence].reset_index(drop=True)

# Step 5: Process and Generate Separate Videos
unique_scenes = df_clean['scene_id'].unique()

for scene in unique_scenes:
    print(f"\n--- Processing Scene: {scene} ---")
    scene_df = df_clean[df_clean['scene_id'] == scene].copy()
    
    if len(scene_df) == 0:
        continue
        
    # Order the frames based on label similarity
    ordered_scene_df = reconstruct_scene_sequence(scene_df)
    
    # Save scene-specific CSV
    csv_name = f"sequence_scene_{scene}.csv"
    ordered_scene_df[['frame_name', 'split', 'scene_id']].to_csv(os.path.join(OUTPUT_BASE, csv_name), index=False)
    
    # Generate Video for the Scene
    video_path = os.path.join(OUTPUT_BASE, f"video_scene_{scene}.mp4")
    
    # Try to load the first image to get dimensions
    test_img = cv2.imread(ordered_scene_df.iloc[0]['file_path'])
    if test_img is None:
        print(f"Warning: Could not read image {ordered_scene_df.iloc[0]['file_path']}. skipping video.")
        continue
        
    height, width, _ = test_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height)) # 10 FPS for better visibility
    
    for _, row in tqdm(ordered_scene_df.iterrows(), total=len(ordered_scene_df), desc=f"Writing {scene}"):
        img = cv2.imread(row['file_path'])
        if img is not None:
            # Optional: Add frame name text to the video
            cv2.putText(img, f"Scene: {scene} | Frame: {row['frame_name']}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video.write(img)
            
    video.release()
    print(f"Saved: {video_path}")

print("\nAll sequences processed successfully.")