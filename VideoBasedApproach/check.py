import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def create_vehicle_video(csv_path, image_folder, output_path, fps=10):
    # Load the processed data
    df = pd.read_csv(csv_path)
    
    # Sort by sequence_index to ensure the video plays in order
    frames = df.sort_values('sequence_index')['frame_name'].unique()
    
    # Get the resolution from the first image to initialize VideoWriter
    sample_img_path = os.path.join(image_folder, f"{frames[0]}.jpg")
    if not os.path.exists(sample_img_path):
        print(f"Error: Could not find image {sample_img_path}. Check path or extension.")
        return

    sample_img = cv2.imread(sample_img_path)
    height, width, layers = sample_img.shape
    
    # Initialize Video Writer (using XVID or MP4V)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating video: {output_path}...")
    
    for frame_name in tqdm(frames):
        img_path = os.path.join(image_folder, f"{frame_name}.jpg")
        frame_img = cv2.imread(img_path)
        
        if frame_img is None:
            continue
            
        # Filter rows for the current frame
        frame_data = df[df['frame_name'] == frame_name]
        
        for _, row in frame_data.iterrows():
            # Get corners (x1,y1 to x4,y4)
            pts = np.array([
                [row['x1'], row['y1']],
                [row['x2'], row['y2']],
                [row['x3'], row['y3']],
                [row['x4'], row['y4']]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw the rotated bounding box
            cv2.polylines(frame_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Prepare label (Class and Velocity)
            label = f"{row['class_label']}"
            if pd.notnull(row['velocity_px_sec']):
                label += f" | {row['velocity_px_sec']:.1f} px/s"
            
            # Add text label near the center or top-left corner
            cv2.putText(frame_img, label, (int(row['x1']), int(row['y1']) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        video.write(frame_img)
    
    video.release()
    print("Video saved successfully!")

# Usage
create_vehicle_video(
    csv_path='processed_vehicle_velocity.csv',
    image_folder='Dataset/ModifiedAUDataset/images',
    output_path='vehicle_tracking_video.mp4',
    fps=10 # Adjust FPS based on your dataset requirements
)