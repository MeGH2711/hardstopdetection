import os
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = 'Dataset/DRASHTI-HaOBB' 
INFO_CSV = 'Dataset/DRASHTI-HaOBB/DRASHTI-HaOBB_framewise_info.csv'
OUTPUT_CSV = 'Outputs/VisionBasedSequencer_V4/reconstructed_sequences.csv'
BATCH_SIZE = 128  
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 4   
RECONSTRUCTION_THRESHOLD = 1.0 
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
        except Exception:
            return torch.zeros(3, 224, 224)

def reconstruct_multi_video_sequences(feats, threshold=1.0):
    feats = normalize(feats)
    tree = BallTree(feats)
    remaining_indices = set(range(len(feats)))
    
    all_videos = []
    current_idx = 0 
    
    with tqdm(total=len(feats), desc="Segmenting Videos") as pbar:
        while remaining_indices:
            current_video_indices = []
            if current_idx not in remaining_indices:
                current_idx = next(iter(remaining_indices))
            
            while True:
                current_video_indices.append(current_idx)
                remaining_indices.remove(current_idx)
                pbar.update(1)
                
                if not remaining_indices:
                    break
                    
                dist, ind = tree.query(feats[current_idx].reshape(1, -1), k=min(10, len(remaining_indices) + 1))
                
                found_next = False
                for d, neighbor_idx in zip(dist[0], ind[0]):
                    if neighbor_idx in remaining_indices:
                        if d < threshold:
                            current_idx = neighbor_idx
                            found_next = True
                            break
                
                if not found_next:
                    break
            
            all_videos.append(current_video_indices)
                    
    return all_videos

if __name__ == '__main__':
    # Step 1: Data Setup
    df = pd.read_csv(INFO_CSV)
    df_clean = df[df['augmented'] == 'No'].copy()

    def get_path(row):
        folder = os.path.join(DATASET_PATH, 'images', row['split'])
        return os.path.join(folder, f"{row['frame_name']}.jpg")

    df_clean['file_path'] = df_clean.apply(get_path, axis=1)
    valid_df = df_clean[df_clean['file_path'].apply(os.path.exists)].copy().reset_index(drop=True)
    print(f"Processing {len(valid_df)} frames...")

    # Step 2: Feature Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) 
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FrameDataset(valid_df['file_path'].tolist(), preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting visual features"):
            batch = batch.to(device)
            output = model(batch).squeeze()
            if len(output.shape) == 1: output = output.unsqueeze(0)
            features.append(output.cpu().numpy())

    features = np.concatenate(features, axis=0)

    # Step 3: Reconstruction and CSV Export
    video_segments = reconstruct_multi_video_sequences(features, threshold=RECONSTRUCTION_THRESHOLD)
    
    # Create a mapping of frame index to Video ID
    reconstructed_data = []
    for video_id, indices in enumerate(video_segments):
        for order, idx in enumerate(indices):
            row = valid_df.iloc[idx].to_dict()
            row['video_id'] = video_id
            row['sequence_order'] = order
            reconstructed_data.append(row)

    output_df = pd.DataFrame(reconstructed_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Complete! CSV saved to: {OUTPUT_CSV}")