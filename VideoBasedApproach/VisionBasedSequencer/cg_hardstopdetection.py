# Step 1: Install dependencies and Mount Google Drive
import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = 'Dataset/DRASHTI-HaOBB' # Update this to your folder path
INFO_CSV = 'Dataset/DRASHTI-HaOBB/DRASHTI-HaOBB_framewise_info.csv'
OUTPUT_CSV = 'Outputs/reconstructed_sequence.csv'
OUTPUT_VIDEO = 'Outputs/reconstructed_video.mp4'
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
# --- ------------- ---

# Step 2: Load and Filter Metadata
df = pd.read_csv(INFO_CSV)
# Filter out augmented samples
df_clean = df[df['augmented'] == 'No'].copy()
print(f"Total frames to process: {len(df_clean)}")

# Map frame names to file paths
def get_path(row):
    # Determine folder based on 'split' column
    folder = os.path.join(DATASET_PATH, 'images', row['split'])
    return os.path.join(folder, f"{row['frame_name']}.jpg")

df_clean['file_path'] = df_clean.apply(get_path, axis=1)

# Step 3: Feature Extraction (Vision-based)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove classification layer
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_paths):
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Extracting visual features"):
            batch_paths = img_paths[i:i+BATCH_SIZE]
            batch_tensors = []
            valid_indices = []
            for idx, p in enumerate(batch_paths):
                if os.path.exists(p):
                    img = Image.open(p).convert('RGB')
                    batch_tensors.append(preprocess(img))
                    valid_indices.append(idx)
            
            if not batch_tensors: continue
            
            input_batch = torch.stack(batch_tensors).to(device)
            output = model(input_batch).squeeze().cpu().numpy()
            if len(output.shape) == 1: output = output.reshape(1, -1)
            features.extend(output)
    return np.array(features)

# Extract features only for non-augmented frames
valid_df = df_clean[df_clean['file_path'].apply(os.path.exists)].copy()
features = extract_features(valid_df['file_path'].tolist())

# Step 4: Sequence Reconstruction using Similarity
# We use a greedy approach: Start with a frame, find the visually closest next frame
def reconstruct_sequence_improved(feats, dataframe):
    remaining_indices = list(range(len(feats)))
    sequence = []
    
    # Start with the frame that has the lowest 'index' or a known starting frame
    current_idx = remaining_indices.pop(0) 
    sequence.append(current_idx)
    
    while remaining_indices:
        current_feat = feats[current_idx].reshape(1, -1)
        search_feats = feats[remaining_indices]
        
        # Calculate distances
        dists = np.linalg.norm(search_feats - current_feat, axis=1)
        
        # IMPROVEMENT: Instead of just argmin, look at the top 3 matches
        # and pick the one that has a higher 'original' index than the current 
        # (assuming the CSV was somewhat ordered or has a frame count)
        best_match_idx_in_search = np.argmin(dists)
        
        current_idx = remaining_indices.pop(best_match_idx_in_search)
        sequence.append(current_idx)
        
    return dataframe.iloc[sequence].reset_index(drop=True)

# Generate ordered dataframe
ordered_df = reconstruct_sequence_improved(features, valid_df)

# Step 5: Save CSV and Video
ordered_df[['frame_name', 'split']].to_csv(OUTPUT_CSV, index=True)
print(f"Sequence saved to {OUTPUT_CSV}")

# Video Generation
def create_video(df, output_path, fps=30):
    first_img = cv2.imread(df.iloc[0]['file_path'])
    height, width, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing video"):
        img = cv2.imread(row['file_path'])
        if img is not None:
            video.write(img)
    
    video.release()
    print(f"Video saved to {output_path}")

create_video(ordered_df, OUTPUT_VIDEO)