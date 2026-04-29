import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import BallTree
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = 'Dataset/DRASHTI-HaOBB' 
INFO_CSV = 'Dataset/DRASHTI-HaOBB/DRASHTI-HaOBB_framewise_info.csv'
OUTPUT_CSV = 'Outputs/reconstructed_sequence.csv'
OUTPUT_VIDEO = 'Outputs/reconstructed_video.mp4'
BATCH_SIZE = 128  
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 4   
# --- ------------- ---

class FrameDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            return torch.zeros(3, 224, 224)

def reconstruct_single_video(feats, dataframe):
    """Reconstructs the order for a single video group."""
    if len(feats) <= 1:
        return dataframe

    tree = BallTree(feats)
    remaining_indices = set(range(len(feats)))
    sequence = []
    
    # Start with the first frame available in this subset
    current_idx = 0 
    sequence.append(current_idx)
    remaining_indices.remove(current_idx)
    
    while remaining_indices:
        # Search for the nearest neighbor within this specific video's feature set
        dist, ind = tree.query(feats[current_idx].reshape(1, -1), k=min(10, len(remaining_indices) + 1))
        found = False
        for neighbor_idx in ind[0]:
            if neighbor_idx in remaining_indices:
                current_idx = neighbor_idx
                found = True
                break
        
        if not found:
            current_idx = next(iter(remaining_indices))

        sequence.append(current_idx)
        remaining_indices.remove(current_idx)
            
    return dataframe.iloc[sequence].reset_index(drop=True)

def create_video(df, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if df.empty: return
    
    first_img = cv2.imread(df.iloc[0]['file_path'])
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing video"):
        img = cv2.imread(row['file_path'])
        if img is not None: 
            video.write(img)
    video.release()

if __name__ == '__main__':
    # 1. Data Preparation
    df = pd.read_csv(INFO_CSV)
    
    # Ensure 'video_name' exists; if your CSV uses a different name like 'video_id', change it here
    VIDEO_COL = 'video_name' if 'video_name' in df.columns else 'video_id'
    
    df_clean = df[df['augmented'] == 'No'].copy()

    def get_path(row):
        folder = os.path.join(DATASET_PATH, 'images', row['split'])
        return os.path.join(folder, f"{row['frame_name']}.jpg")

    df_clean['file_path'] = df_clean.apply(get_path, axis=1)
    valid_df = df_clean[df_clean['file_path'].apply(os.path.exists)].copy().reset_index(drop=True)
    
    print(f"Total valid frames: {len(valid_df)} across {valid_df[VIDEO_COL].nunique()} videos.")

    # 2. Feature Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) 
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FrameDataset(valid_df['file_path'].tolist(), preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting visual features"):
            batch = batch.to(device)
            output = model(batch).squeeze()
            if len(output.shape) == 1: output = output.unsqueeze(0)
            all_features.append(output.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    # 3. Multi-Video Reconstruction
    final_ordered_dfs = []
    
    # Group by video to prevent frames from jumping between different recordings
    video_groups = valid_df.groupby(VIDEO_COL)
    
    for video_id, group_df in tqdm(video_groups, desc="Processing individual videos"):
        indices = group_df.index.tolist()
        group_features = all_features[indices]
        
        # Reconstruct sequence for THIS video only
        ordered_group = reconstruct_single_video(group_features, group_df)
        final_ordered_dfs.append(ordered_group)

    # Combine all reconstructed videos back together
    full_ordered_df = pd.concat(final_ordered_dfs).reset_index(drop=True)
    
    # 4. Save Results
    os.makedirs('Outputs', exist_ok=True)
    full_ordered_df.to_csv(OUTPUT_CSV, index=True)
    create_video(full_ordered_df, OUTPUT_VIDEO)
    
    print(f"Success! Processed {len(full_ordered_df)} frames into {OUTPUT_VIDEO}")