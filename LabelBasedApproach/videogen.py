import pandas as pd
import cv2
import os

def generate_videos(csv_path, base_dataset_path, output_dir, fps=10):
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_ids = df['video_id'].unique()
    print(f"Found {len(video_ids)} unique video sequences.")

    for video_id in video_ids:
        video_df = df[df['video_id'] == video_id].sort_values(by='sequence_index')
        
        # FIX: Clean the path to ensure we aren't doubling up on folder names
        frame_paths = []
        for p in video_df['image_path']:
            # Replace backslashes and join with the base path
            clean_path = p.replace('\\', '/')
            full_path = os.path.join(base_dataset_path, clean_path)
            frame_paths.append(full_path)
        
        if not frame_paths:
            continue

        print(f"Processing {video_id} ({len(frame_paths)} frames)...")
        video_writer = None
        
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                # Print the exact path being checked to help you debug
                print(f"Warning: File not found at: {os.path.abspath(frame_path)}")
                continue
            
            frame = cv2.imread(frame_path)
            if frame is None: continue

            if video_writer is None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = os.path.join(output_dir, f"{video_id}.mp4")
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            video_writer.write(frame)

        if video_writer:
            video_writer.release()
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    # If you are running the script FROM INSIDE 'ModifiedAUDataset':
    # The 'image_path' in CSV is 'images\0Vsj_JQGe_ATD0.jpg'
    # So base_dataset_path should just be '.' (current directory)
    
    CSV_FILE = 'outputs/frame_video_sequence.csv'
    DATASET_ROOT = '.'  # Point to the current folder where 'images/' exists
    OUTPUT_FOLDER = 'generated_videos'
    FRAME_RATE = 30 
    
    generate_videos(CSV_FILE, DATASET_ROOT, OUTPUT_FOLDER, FRAME_RATE)